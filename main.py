import argparse
import sys
import threading
import time
import tkinter as tk
import traceback
import linecache
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Any, Optional
import queue
import subprocess
import os
from dotenv import load_dotenv
import requests
from settings_manager import SettingsManager, open_settings_dialog
from grouped_download_manager import GroupedDownloadManager, FileSelectionDialog
from grouped_download_gui import GroupedDownloadManagerTab
from model_library import ModelLibraryTab
from mcp_config import open_mcp_config
from splash_screen import SplashManager
from mcp_tab import MCPTab
from model_converter import ModelConverterTab
from chess_tab import ChessTab
from chat_templates import get_template_manager, ChatTemplateDialog

# from finetune_tab import FineTuneTab  # Temporarily disabled for debugging

# Try to import torch for GPU functionality
try:
    import torch
except ImportError:
    torch = None

# Load from your custom env file
load_dotenv("HUGGINGFACE.env")

# Global execution tracer
_trace_enabled = False
_trace_filters = ['llm_runtime', 'main.py', 'autogptq', 'transformers']

def execution_tracer(frame, event, arg):
    """Trace every line of code execution during model loading"""
    if not _trace_enabled:
        return
    
    if event == 'line':
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        
        # Only trace files we care about
        if any(filter_term in filename for filter_term in _trace_filters):
            try:
                line = linecache.getline(filename, lineno).strip()
                short_filename = filename.split('/')[-1] if '/' in filename else filename.split('\\')[-1]
                print(f"[TRACE] {short_filename}:{lineno} | {line}")
            except:
                pass
    
    return execution_tracer

def start_tracing():
    """Start execution tracing"""
    global _trace_enabled
    _trace_enabled = True
    print("[TRACE] Execution tracing STARTED")

def stop_tracing():
    """Stop execution tracing"""
    global _trace_enabled
    _trace_enabled = False
    print("[TRACE] Execution tracing STOPPED")

# Access the key
hf_key = os.getenv("HF_API_KEY")

# Embedded llama.cpp (self-contained, no external daemons)
try:
    from llama_cpp import Llama


except Exception:
    print(
        "The 'llama-cpp-python' package is required. Please install dependencies with: pip install -r requirements.txt",
        file=sys.stderr)
    raise

# ChessGPT support
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
    MIN_TRANSFORMERS_VERSION = '4.25.1'
    if transformers.__version__ < MIN_TRANSFORMERS_VERSION:
        print(
            f"Warning: transformers version {transformers.__version__} may not be compatible. Recommended: {MIN_TRANSFORMERS_VERSION}+")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Chess mode will not work. Install with: pip install torch transformers")

_llama_cache = {
    "key": None,  # (model_path, lora_path, n_ctx, n_gpu_layers, n_threads)
    "llm": None,
}

# ChessGPT model cache
_chessgpt_cache = {
    "tokenizer": None,
    "model": None,
    "loaded": False
}

# Serialize llama native calls when sharing one instance across threads
_LLAMA_LOCK = threading.RLock()

# Default system prompt and anti-echo stop tokens
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer the user's question directly. "
    "Do not repeat or paraphrase the user's prompt; provide only your answer."
)
# Conservative defaults to reduce echo
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 256
# Add common model chat-template markers to stop tokens to avoid template echoes like [INST] <<SYS>> ...
STOP_TOKENS = [
    "\nUser:", "\nYou:", "User:", "You:",
    "<|user|>", "<|assistant|>", "<|eot_id|>", "<|eom_id|>",
    "[INST]", "[/INST]", "<<SYS>>", "</SYS>", "<</SYS>>", "</SYS>>"
]


def _strip_echo_from_response(text: str, last_user_prompt: Optional[str]) -> str:
    try:
        s = text or ""
        # Remove the last user prompt if the model echoed it at the start
        if last_user_prompt:
            lu = (last_user_prompt or "").strip()
            if lu and s.strip().startswith(lu):
                # Cut the first occurrence
                idx = s.find(lu)
                if idx == 0:
                    s = s[len(lu):]
        # Remove common stop tokens/templates that may leak
        for tok in STOP_TOKENS:
            s = s.replace(tok, " ")
        # Clean up repeated whitespace
        s = " ".join(s.split())
        return s.strip()
    except Exception:
        return text or ""


# Provide safe fallbacks so static analysis / early references don't error.
# If real implementations are defined later they will overwrite these.
try:
    _extract_gguf_metadata  # type: ignore
except NameError:
    def _extract_gguf_metadata(path: str, key: str) -> Optional[str]:
        # Non-invasive fallback: best-effort no-op that returns None
        return None

try:
    _extract_gguf_int_metadata  # type: ignore
except NameError:
    def _extract_gguf_int_metadata(path: str, key: str) -> Optional[int]:
        # Non-invasive fallback: best-effort no-op that returns None
        return None


def _is_gguf_model(path: str) -> bool:
    if not path:
        return False
    p = path.strip().strip('"')
    return os.path.isfile(p) and p.lower().endswith('.gguf')


def _is_valid_model(path: str) -> bool:
    """Check if the given path is a valid model file of any supported format."""
    if not path:
        return False

    p = path.strip().strip('"')

    # Check if it might be a HuggingFace repo ID
    if "/" in p and not os.path.exists(p):
        return True  # Let the loader validate it

    if not os.path.exists(p):
        return False

    # Check for supported file extensions
    p_lower = p.lower()
    
    # GGUF format (llama.cpp)
    if p_lower.endswith('.gguf'):
        return True
    
    # SafeTensors format (HuggingFace)
    if p_lower.endswith('.safetensors'):
        return True
    
    # PyTorch format
    if p_lower.endswith('.bin') or p_lower.endswith('.pt') or p_lower.endswith('.pth'):
        return True
    
    # GPTQ quantized models
    if 'gptq' in p_lower and (p_lower.endswith('.safetensors') or p_lower.endswith('.bin')):
        return True
    
    # AWQ quantized models  
    if 'awq' in p_lower and (p_lower.endswith('.safetensors') or p_lower.endswith('.bin')):
        return True
    
    # EXL2 format
    if p_lower.endswith('.exl2'):
        return True
    
    # Check if it's a directory with model files
    if os.path.isdir(p):
        # Check for standard HuggingFace structure
        config_path = os.path.join(p, "config.json")
        if os.path.exists(config_path):
            return True
        
        # Check for GPTQ models
        if any(f for f in os.listdir(p) if 'gptq' in f.lower() and (f.endswith('.safetensors') or f.endswith('.bin'))):
            return True
            
        # Check for AWQ models
        if any(f for f in os.listdir(p) if 'awq' in f.lower() and (f.endswith('.safetensors') or f.endswith('.bin'))):
            return True
            
        # Check for any model files
        if any(f for f in os.listdir(p) if f.endswith(('.gguf', '.safetensors', '.bin', '.pt', '.pth', '.exl2'))):
            return True

    return False


def _get_llama(model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0, lora_path: Optional[str] = None,
               n_threads: Optional[int] = None) -> "Llama":
    mp = model_path.strip().strip('"')
    if n_threads is None or n_threads <= 0:
        n_threads = max(1, os.cpu_count() or 1)

    # include file modification time so cache invalidates when model file changes
    try:
        mtime = os.path.getmtime(mp)
    except Exception:
        mtime = None

    # Normalize inputs to ints for cache key
    try:
        n_ctx_int = int(n_ctx) if n_ctx is not None else 0
    except Exception:
        n_ctx_int = 0
    try:
        n_gpu_int = int(n_gpu_layers) if n_gpu_layers is not None else 0
    except Exception:
        n_gpu_int = 0

    # include n_ctx and n_gpu_layers in key so different contexts create separate instances
    key = (mp, lora_path, n_ctx_int, n_gpu_int, int(n_threads), mtime)
    if _llama_cache["llm"] is not None and _llama_cache["key"] == key:
        return _llama_cache["llm"]

    # Pass the requested context size and gpu layers to Llama so it uses the correct capacity.
    # If the caller passes 0 for n_ctx, the underlying library will use the model's trained n_ctx.
    print(f"[GGUF_DEBUG] Loading GGUF model with n_ctx={n_ctx_int}, n_gpu_layers={n_gpu_int}")
    llm = Llama(
        model_path=mp,
        n_ctx=n_ctx_int,
        n_gpu_layers=n_gpu_int,
        lora_path=lora_path,
        n_threads=n_threads,
        verbose=False,
    )
    print(f"[GGUF_DEBUG] GGUF model loaded successfully with GPU layers: {n_gpu_int}")
    _llama_cache["key"] = key
    _llama_cache["llm"] = llm
    return llm


def _get_chessgpt():
    """Get the already-loaded ChessGPT GGUF model."""
    # The ChessGPT GGUF is already loaded as the main model
    # We don't need to download anything - just return a flag
    # The actual model is accessed through _get_llama()
    return None, None  # Return None since we're using the GGUF version


def _run_chessgpt_prompt(prompt: str, model_path: str = None, on_chunk: Optional[callable] = None,
                         max_tokens: int = 128) -> str:
    """Run a prompt through ChessGPT GGUF model using the ChessGPT conversation format."""
    try:
        # Format prompt for ChessGPT conversation style
        # Add explicit instruction to return only the move
        chess_prompt = f"A friendly, helpful chat between some humans.<|endoftext|>Human 0: {prompt}\nRespond with ONLY the chess move in UCI format (like e2e4).<|endoftext|>Human 1:"

        # Use the already-loaded GGUF model with proper context size
        # ChessGPT was trained on 2048 context
        llm = _get_llama(model_path, n_ctx=2048, n_gpu_layers=32)

        # Generate response using the GGUF model with lower temperature for more deterministic moves
        with _LLAMA_LOCK:
            response = llm(
                chess_prompt,
                max_tokens=20,  # Reduced - we only need a move
                temperature=0.3,  # Lower temperature for more deterministic chess moves
                top_p=0.9,
                top_k=40,
                echo=False,
                stop=["<|endoftext|>", "Human 0:", "Human 1:", "\n"]
            )

        output_str = response['choices'][0]['text'].strip()

        print(f"[CHESS DEBUG] ChessGPT raw response: '{output_str}'")

        # Stream output if callback provided
        if on_chunk:
            for char in output_str:
                on_chunk(char)

        return output_str

    except Exception as e:
        print(f"ChessGPT GGUF generation failed: {e}")
        raise


def run_prompt(model_path: str, prompt: str, stream: bool, n_ctx: int = 4096, n_gpu_layers: int = 0,
               lora_path: Optional[str] = None, on_chunk: Optional[callable] = None, n_threads: Optional[int] = None,
               max_tokens: Optional[int] = None, history: Optional[List[Dict[str, Any]]] = None,
               cancel_event: Optional[threading.Event] = None, chess_mode: bool = False) -> str:
    # Use ChessGPT GGUF if chess mode is enabled
    if chess_mode:
        try:
            return _run_chessgpt_prompt(prompt, model_path=model_path, on_chunk=on_chunk, max_tokens=max_tokens or 128)
        except Exception as e:
            print(f"ChessGPT GGUF failed, falling back to regular model: {e}")
            # Fall through to regular model with limited context
            n_ctx = min(n_ctx, 2048)  # Limit context for chess to avoid overflow

    llm = _get_llama(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, lora_path=lora_path, n_threads=n_threads)
    # Build messages from prior history plus the new user message
    messages: List[Dict[str, Any]] = []
    if not history or (history and history[0].get("role") != "system"):
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    max_new_tokens = DEFAULT_MAX_TOKENS if (max_tokens is None or int(max_tokens) <= 0) else int(max_tokens)
    out_parts: List[str] = []
    if stream:
        with _LLAMA_LOCK:
            for part in llm.create_chat_completion(messages=messages, stream=True, stop=STOP_TOKENS,
                                                   temperature=DEFAULT_TEMPERATURE, repeat_penalty=1.2,
                                                   max_tokens=max_new_tokens):
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    chunk = part["choices"][0]["delta"].get("content", "")
                except Exception:
                    chunk = part.get("choices", [{}])[0].get("message", {}).get("content", "")
                if chunk:
                    out_parts.append(chunk)
                    if on_chunk:
                        on_chunk(chunk)
                if cancel_event is not None and cancel_event.is_set():
                    break
        result = "".join(out_parts)
        # Try to remove prompt echo/templates using the last user prompt
        last_user = prompt
        return _strip_echo_from_response(result, last_user)
    else:
        with _LLAMA_LOCK:
            res = llm.create_chat_completion(messages=messages, stop=STOP_TOKENS, temperature=DEFAULT_TEMPERATURE,
                                             repeat_penalty=1.2, max_tokens=max_new_tokens)
        raw = res.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        # Remove common echoes: use the prompt as the last user message
        return _strip_echo_from_response(raw, prompt)


def chat_stream(model_path: str, messages: List[Dict[str, Any]], n_ctx: int = 4096, n_gpu_layers: int = 0,
                lora_path: Optional[str] = None, on_chunk: Optional[callable] = None, n_threads: Optional[int] = None,
                max_tokens: Optional[int] = None, cancel_event: Optional[threading.Event] = None,
                chess_mode: bool = False, chat_template: Optional[str] = None, session_id: Optional[str] = None) -> str:
    # Use ChessGPT if chess mode is enabled
    if chess_mode:
        try:
            # Extract the last user message for ChessGPT
            last_user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "")
                    break
            if last_user_message:
                return _run_chessgpt_prompt(last_user_message, on_chunk=on_chunk, max_tokens=max_tokens or 128)
        except Exception as e:
            print(f"ChessGPT failed, falling back to regular model: {e}")
            # Fall through to regular model

    # Ensure a system message exists at the start
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + list(messages)
    collected: List[str] = []
    max_new_tokens = DEFAULT_MAX_TOKENS if (max_tokens is None or int(max_tokens) <= 0) else int(max_tokens)

    # Use appropriate loader based on model type
    if _is_gguf_model(model_path):
        # Use existing GGUF loader
        llm = _get_llama(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, lora_path=lora_path, n_threads=n_threads)
        with _LLAMA_LOCK:
            for part in llm.create_chat_completion(messages=messages, stream=True, stop=STOP_TOKENS,
                                                   temperature=DEFAULT_TEMPERATURE, repeat_penalty=1.2,
                                                   max_tokens=max_new_tokens):
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    delta = part["choices"][0]["delta"].get("content", "")
                except Exception:
                    delta = part.get("choices", [{}])[0].get("message", {}).get("content", "")
                if delta:
                    collected.append(delta)
                    if on_chunk:
                        on_chunk(delta)
                if cancel_event is not None and cancel_event.is_set():
                    break
    else:
        # Use unified loader for other model types - reuse cached model
        from llm_runtime import GenerateConfig
        
        # Check if model is already loaded in cache (from _on_load_model)
        if not hasattr(chat_stream, '_unified_model_cache') or chat_stream._unified_model_cache is None:
            raise RuntimeError("Model not loaded. Please load a model first using the 'Load Model' button.")
        
        llm = chat_stream._unified_model_cache
        print("DEBUG: Using cached unified model")

        # Convert messages to prompt format using chat template
        if chat_template and chat_template != "None":
            # Use the selected chat template
            from chat_templates import get_template_manager
            template_manager = get_template_manager()
            prompt = template_manager.format_conversation(chat_template, messages, add_generation_prompt=True)
            print(f"DEBUG: Using chat template '{chat_template}'")
        else:
            # Fallback to simple format for backward compatibility
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
            print("DEBUG: Using fallback User:/Assistant: format")
        
        print(f"DEBUG: Generated prompt: '{prompt}'")

        # Get appropriate stop tokens from template or use defaults
        if chat_template and chat_template != "None":
            from chat_templates import get_template_manager
            template_manager = get_template_manager()
            template_stop_tokens = template_manager.get_stop_tokens(chat_template)
            stop_tokens = template_stop_tokens if template_stop_tokens else STOP_TOKENS
        else:
            stop_tokens = STOP_TOKENS
        
        # Generate with unified API and KV caching
        cfg = GenerateConfig(max_tokens=max_new_tokens, temperature=DEFAULT_TEMPERATURE, top_p=0.9, stop=stop_tokens)
        # Use instance-specific session ID to maintain conversation continuity while preventing cross-chat contamination
        if session_id is None:
            session_id = "default"  # Fallback for CLI usage
        
        # Get session info before generation
        if hasattr(llm, 'get_session_info'):
            session_info = llm.get_session_info(session_id)
            print(f"[KV_CACHE] Pre-generation session info: {session_info}")
        
        print(f"[CHAT] Starting streaming generation with KV caching enabled")
        generation_start = time.time()
        token_count = 0
        
        for delta in llm.stream(prompt, cfg=cfg, session_id=session_id):
            if cancel_event is not None and cancel_event.is_set():
                break
            if delta:
                collected.append(delta)
                token_count += 1
                if on_chunk:
                    on_chunk(delta)
            if cancel_event is not None and cancel_event.is_set():
                break
        
        generation_time = time.time() - generation_start
        print(f"[KV_CACHE] Generated {token_count} tokens in {generation_time:.2f}s ({token_count/generation_time:.1f} tokens/s)")
        
        # Get session info after generation
        if hasattr(llm, 'get_session_info'):
            session_info = llm.get_session_info(session_id)
            print(f"[KV_CACHE] Post-generation session info: {session_info}")
        
        # Get KV cache statistics if available
        if hasattr(llm, 'get_kv_cache_stats'):
            cache_stats = llm.get_kv_cache_stats()
            print(f"[KV_CACHE] Cache statistics: {cache_stats}")
    result = "".join(collected)
    # Find the last user message in provided messages and attempt to strip echoes
    last_user = None
    try:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
    except Exception:
        last_user = None
    return _strip_echo_from_response(result, last_user)


# ---------------- GUI (Embedded only) -----------------
class EmbeddedGUI:
    def __init__(self, root: tk.Tk):
        print("[APP_DEBUG] EmbeddedGUI.__init__() started")
        self.root = root
        self.root.title("DarkHal 2.0 - AI Model Management Platform")

        # Set window icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "assets", "Halico.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            pass

        # Set minimum window size
        self.root.minsize(1000, 700)

        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Initialize grouped download manager
        max_concurrent = self.settings_manager.get('download_settings.max_concurrent_downloads', 3)
        self.download_manager = GroupedDownloadManager(max_concurrent=max_concurrent)
        
        # Initialize agent mode attributes
        self.agent_enabled = False
        self.dhal_agent = None

        # Create menu bar
        self._create_menu_bar()

        # Load settings and initialize variables
        self.model_var = tk.StringVar(value=self.settings_manager.get('paths.last_model_path', ''))
        self.stream_var = tk.BooleanVar(value=self.settings_manager.get('model_settings.stream_by_default', True))
        self.n_ctx_var = tk.IntVar(value=self.settings_manager.get('model_settings.default_n_ctx', 4096))
        # Set default GPU layers - use higher default if GPU is available and auto-config is enabled
        default_gpu_layers = self.settings_manager.get('model_settings.default_n_gpu_layers', 0)
        if default_gpu_layers == 0 and self.settings_manager.get('model_settings.auto_gpu', True):
            # If auto-GPU is enabled and no custom default is set, use a reasonable default for GPU systems
            try:
                import torch
                if torch.cuda.is_available():
                    default_gpu_layers = 16  # Reasonable default for most 7B models
            except:
                pass
        self.n_gpu_layers_var = tk.IntVar(value=default_gpu_layers)
        self.lora_var = tk.StringVar(value=self.settings_manager.get('paths.last_lora_path', ''))
        self.model_status_var = tk.StringVar(value="[not loaded]")
        self.max_tokens_var = tk.IntVar(
            value=self.settings_manager.get('model_settings.default_max_tokens', DEFAULT_MAX_TOKENS))
        self.chess_mode_var = tk.BooleanVar(value=self.settings_manager.get('model_settings.chess_mode', False))
        self.agent_mode_var = tk.BooleanVar(value=False)  # Agent mode always starts disabled for safety
        
        # Advanced loading options
        self.quantization_var = tk.StringVar(value=self.settings_manager.get('model_settings.quantization', 'none'))
        self.device_strategy_var = tk.StringVar(value=self.settings_manager.get('model_settings.device_strategy', 'auto'))
        self.chat_template_var = tk.StringVar(value="None")
        self.gpu_memory_limit_var = tk.DoubleVar(value=self.settings_manager.get('model_settings.gpu_memory_limit', 6.0))
        
        # Sampling parameters
        self.temperature_var = tk.DoubleVar(value=self.settings_manager.get('model_settings.temperature', 0.7))
        self.top_p_var = tk.DoubleVar(value=self.settings_manager.get('model_settings.top_p', 0.9))
        self.repetition_penalty_var = tk.DoubleVar(value=self.settings_manager.get('model_settings.repetition_penalty', 1.1))
        self.no_repeat_ngram_size_var = tk.IntVar(value=self.settings_manager.get('model_settings.no_repeat_ngram_size', 0))
        self.min_p_var = tk.DoubleVar(value=self.settings_manager.get('model_settings.min_p', 0.0))
        self.typical_p_var = tk.DoubleVar(value=self.settings_manager.get('model_settings.typical_p', 1.0))

        # UI queue for thread-safe widget updates from worker threads
        self._ui_queue: "queue.SimpleQueue[callable]" = queue.SimpleQueue()
        self.root.after(30, self._drain_ui_queue)

        # Reset loaded status when key settings change
        try:
            self.model_var.trace_add("write", lambda *a: self._mark_model_unloaded())
            self.n_ctx_var.trace_add("write", lambda *a: self._mark_model_unloaded())
            self.n_gpu_layers_var.trace_add("write", lambda *a: self._mark_model_unloaded())
            self.lora_var.trace_add("write", lambda *a: self._mark_model_unloaded())
            self.chess_mode_var.trace_add("write", lambda *a: self._mark_model_unloaded())
            
            # Advanced loading options - also mark model as unloaded and save settings
            self.quantization_var.trace_add("write", lambda *a: self._on_advanced_setting_changed())
            self.device_strategy_var.trace_add("write", lambda *a: self._on_advanced_setting_changed())
            self.gpu_memory_limit_var.trace_add("write", lambda *a: self._on_advanced_setting_changed())
            
            # Sampling parameters - save settings when changed
            self.temperature_var.trace_add("write", lambda *a: self._on_sampling_setting_changed())
            self.top_p_var.trace_add("write", lambda *a: self._on_sampling_setting_changed())
            self.repetition_penalty_var.trace_add("write", lambda *a: self._on_sampling_setting_changed())
            self.no_repeat_ngram_size_var.trace_add("write", lambda *a: self._on_sampling_setting_changed())
            self.min_p_var.trace_add("write", lambda *a: self._on_sampling_setting_changed())
            self.typical_p_var.trace_add("write", lambda *a: self._on_sampling_setting_changed())
        except Exception:
            try:
                self.model_var.trace("w", lambda *a: self._mark_model_unloaded())
                self.n_ctx_var.trace("w", lambda *a: self._mark_model_unloaded())
                self.n_gpu_layers_var.trace("w", lambda *a: self._mark_model_unloaded())
                self.lora_var.trace("w", lambda *a: self._mark_model_unloaded())
                self.chess_mode_var.trace("w", lambda *a: self._mark_model_unloaded())
            except Exception:
                pass

        # Local models support with settings
        self.models_dir_var = tk.StringVar(value=self.settings_manager.get('paths.models_directory', './models'))
        self.local_model_var = tk.StringVar()
        self._local_model_paths: Dict[str, str] = {}

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)

        # Single Run tab
        self.run_frame = ttk.Frame(nb)
        nb.add(self.run_frame, text="Run")
        self._build_run_tab(self.run_frame)


        # Model Library tab
        self.library_frame = ttk.Frame(nb)
        nb.add(self.library_frame, text="Model Library")
        self.library_tab = ModelLibraryTab(self.library_frame, self.settings_manager)

        # Model Converter tab
        self.converter_frame = ttk.Frame(nb)
        nb.add(self.converter_frame, text="Model Converter")
        self.converter_tab = ModelConverterTab(self.converter_frame, self.settings_manager)

        # Chess tab
        self.chess_frame = ttk.Frame(nb)
        nb.add(self.chess_frame, text="Chess")
        self.chess_tab = ChessTab(self.chess_frame, self.settings_manager)


        # Fine Tune tab - temporarily disabled for debugging
        # self.finetune_frame = ttk.Frame(nb)
        # nb.add(self.finetune_frame, text="Fine Tune")
        # self.finetune_tab = FineTuneTab(self.finetune_frame, self.settings_manager)




        # Initialize local models list if a folder is preset
        if self.models_dir_var.get():
            self._refresh_local_models()

        self.chat_history: List[Dict[str, Any]] = []
        self._current_cancel: Optional[threading.Event] = None
        # Initialize unique session ID for KV cache isolation
        import uuid
        self._session_id = f"chat_session_{uuid.uuid4().hex[:8]}"
        
        # Initialize chat template manager
        self.template_manager = get_template_manager()
        self._refresh_chat_templates()

        # Apply window size from settings
        width = self.settings_manager.get('ui_preferences.window_width', 1200)
        height = self.settings_manager.get('ui_preferences.window_height', 700)
        self.root.geometry(f"{width}x{height}")

        # Save window size on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Model...", command=self._browse_gguf)
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self._open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Output", command=lambda: self.output_text.delete('1.0', tk.END))
        edit_menu.add_command(label="Clear Chat History", command=self._clear_chat_history)
        edit_menu.add_command(label="Clear KV Cache", command=self._clear_kv_cache)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="HuggingFace Downloader", command=self._open_hf_downloader)
        tools_menu.add_command(label="Downloads Manager", command=self._open_downloads_manager)
        tools_menu.add_command(label="MCP Server", command=self._open_mcp_server)
        tools_menu.add_command(label="Resource Monitor", command=self._open_resource_monitor)
        
        # Debug submenu
        debug_menu = tk.Menu(tools_menu, tearoff=0)
        tools_menu.add_separator()
        tools_menu.add_cascade(label="Debug", menu=debug_menu)
        debug_menu.add_command(label="Inspect Model Devices", command=self._inspect_model_devices)
        tools_menu.add_command(label="Refresh Local Models", command=self._refresh_local_models)
        tools_menu.add_separator()
        tools_menu.add_command(label="Clear Completed Downloads", command=self._clear_completed_downloads)
        tools_menu.add_command(label="MCP Server Config", command=self._open_mcp_config)

        # Agents menu
        agents_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Agents", menu=agents_menu)
        
        # DarkHal submenu
        darkhal_menu = tk.Menu(agents_menu, tearoff=0)
        agents_menu.add_cascade(label="DarkHal", menu=darkhal_menu)
        darkhal_menu.add_command(label="Dhal", command=self._open_dhal_agent)
        darkhal_menu.add_command(label="Agent Dev Kit (ADK)", command=self._open_adk)
        
        # Metasploit option
        agents_menu.add_command(label="Metasploit", command=self._open_metasploit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _open_settings(self):
        """Open the settings dialog."""
        open_settings_dialog(self.root, self.settings_manager)
        # Reload settings after dialog closes
        self._reload_settings()

    def _reload_settings(self):
        """Reload settings after they've been changed."""
        # Update variables from settings
        self.n_ctx_var.set(self.settings_manager.get('model_settings.default_n_ctx', 4096))
        self.n_gpu_layers_var.set(self.settings_manager.get('model_settings.default_n_gpu_layers', 0))
        self.max_tokens_var.set(self.settings_manager.get('model_settings.default_max_tokens', DEFAULT_MAX_TOKENS))
        self.stream_var.set(self.settings_manager.get('model_settings.stream_by_default', True))

        # Reload HF API if token settings changed
        if hasattr(self, 'hf_api'):
            try:
                from hf_downloader import HuggingFaceAPI
                api_key = None
                organization = None

                if not self.settings_manager.get('api.use_env_token', True):
                    api_key = self.settings_manager.get('api.huggingface_token', '').strip()

                if self.settings_manager.get('api.use_organization', False):
                    organization = self.settings_manager.get('api.organization', '').strip()

                self.hf_api = HuggingFaceAPI(api_key=api_key, organization=organization)
            except Exception:
                pass

    def _clear_chat_history(self):
        """Clear the chat history and invalidate KV cache session."""
        self.chat_history.clear()
        self.output_text.delete('1.0', tk.END)
        
        # Clear KV cache for the current session to prevent contamination
        try:
            if hasattr(chat_stream, '_unified_model_cache') and chat_stream._unified_model_cache:
                llm = chat_stream._unified_model_cache
                if hasattr(llm, 'clear_session_cache'):
                    llm.clear_session_cache(self._session_id)
                    print(f"[KV_CACHE] Cleared cache for session {self._session_id}")
        except Exception as e:
            print(f"[KV_CACHE] Error clearing session cache: {e}")
        
        # Generate new session ID for fresh conversation
        import uuid
        self._session_id = f"chat_session_{uuid.uuid4().hex[:8]}"
        
        self._append_output("[Chat history cleared - KV cache reset]\n")
    
    def _refresh_chat_templates(self):
        """Refresh the chat template dropdown with available templates"""
        try:
            template_names = ["None"] + self.template_manager.get_template_names()
            self.chat_template_combo['values'] = template_names
            
            # Set to "None" if current selection is not available
            current = self.chat_template_var.get()
            if current not in template_names:
                self.chat_template_var.set("None")
        except Exception as e:
            print(f"Error refreshing chat templates: {e}")
    
    def _load_chat_template(self):
        """Load chat templates from file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Chat Templates",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                parent=self.root
            )
            
            if filename:
                # Load templates from the selected file
                import json
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                loaded_count = 0
                for name, template_data in data.items():
                    try:
                        from chat_templates import ChatTemplate
                        template = ChatTemplate(**template_data)
                        if self.template_manager.add_template(template):
                            loaded_count += 1
                        else:
                            # Template exists, ask if user wants to update
                            if messagebox.askyesno("Template Exists", 
                                                 f"Template '{name}' already exists. Update it?"):
                                self.template_manager.update_template(template)
                                loaded_count += 1
                    except Exception as e:
                        print(f"Error loading template '{name}': {e}")
                
                self._refresh_chat_templates()
                messagebox.showinfo("Templates Loaded", f"Successfully loaded {loaded_count} template(s)")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading templates: {e}")
    
    def _add_chat_template(self):
        """Add a new chat template"""
        try:
            dialog = ChatTemplateDialog(self.root)
            self.root.wait_window(dialog.dialog)
            
            if dialog.result:
                template = dialog.result
                if self.template_manager.add_template(template):
                    self._refresh_chat_templates()
                    self.chat_template_var.set(template.name)
                    messagebox.showinfo("Success", f"Template '{template.name}' added successfully")
                else:
                    # Template exists, ask if user wants to update
                    if messagebox.askyesno("Template Exists", 
                                         f"Template '{template.name}' already exists. Update it?"):
                        self.template_manager.update_template(template)
                        self._refresh_chat_templates()
                        messagebox.showinfo("Success", f"Template '{template.name}' updated successfully")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error adding template: {e}")
    
    def _clear_kv_cache(self):
        """Clear the KV cache for the current chat session."""
        try:
            # Clear cache for unified models (HuggingFace)
            if hasattr(chat_stream, '_unified_model_cache') and chat_stream._unified_model_cache:
                llm = chat_stream._unified_model_cache
                if hasattr(llm, 'clear_session_cache'):
                    llm.clear_session_cache(self._session_id)
                    self._append_output(f"[KV Cache cleared for session {self._session_id}]\n")
                    
                    # Show cache statistics after clearing
                    if hasattr(llm, 'get_kv_cache_stats'):
                        cache_stats = llm.get_kv_cache_stats()
                        self._append_output(f"[Cache Stats] Active sessions: {cache_stats.get('active_sessions', 0)}\n")
                else:
                    self._append_output("[KV Cache clear not supported for this model]\n")
            else:
                self._append_output("[No cached model found - KV cache already clear]\n")
        except Exception as e:
            self._append_output(f"[Error clearing KV cache: {e}]\n")

    def _open_hf_downloader(self):
        """Open standalone HuggingFace downloader window."""
        try:
            from hf_downloader import HuggingFaceDownloaderGUI
            downloader_window = tk.Toplevel(self.root)
            HuggingFaceDownloaderGUI(downloader_window)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open HuggingFace downloader: {e}")

    def _show_about(self):
        """Show about dialog."""
        about_text = (
            "LLM_Train - Advanced Local Model Manager\n\n"
            "A comprehensive local GGUF model runner with cloud integration\n\n"
            "Features:\n"
            "• Run local GGUF models with optimized performance\n"
            "• Search and download from HuggingFace Hub\n"
            "• Advanced download manager with pause/resume/retry\n"
            "• Model Library with smart scanning and indexing\n"
            "• Multi-model MCP server for Claude integration\n"
            "• Organization support for HuggingFace teams\n"
            "• Chat and single prompt modes\n"
            "• Customizable settings and preferences\n"
            "• Optimized USB/SSD write speeds\n\n"
            "Powered by llama-cpp-python and MCP protocol"
        )
        messagebox.showinfo("About", about_text)

    def _clear_completed_downloads(self):
        """Clear completed downloads from download manager."""
        if hasattr(self, 'download_tab'):
            self.download_tab._clear_completed()

    def _open_mcp_config(self):
        """Open MCP server configuration."""
        try:
            open_mcp_config(self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open MCP configuration: {e}")
    
    def _open_resource_monitor(self):
        """Open Resource Monitor in a new window."""
        resource_window = tk.Toplevel(self.root)
        resource_window.title("Resource Monitor")
        resource_window.geometry("800x600")
        resource_window.transient(self.root)
        
        # Build resource monitor content in the new window
        self._build_resource_tab(resource_window)
    
    def _open_downloads_manager(self):
        """Open Downloads Manager in a new window."""
        downloads_window = tk.Toplevel(self.root)
        downloads_window.title("Downloads Manager")
        downloads_window.geometry("900x700")
        downloads_window.transient(self.root)
        
        # Create download manager in the new window
        from download_manager_tab import GroupedDownloadManagerTab
        GroupedDownloadManagerTab(downloads_window, self.download_manager)
    
    def _open_mcp_server(self):
        """Open MCP Server in a new window."""
        mcp_window = tk.Toplevel(self.root)
        mcp_window.title("MCP Server")
        mcp_window.geometry("800x600")
        mcp_window.transient(self.root)
        
        # Create MCP server tab in the new window
        MCPTab(mcp_window, self.settings_manager)
    
    def _open_dhal_agent(self):
        """Open Dhal Dark Agent in a new window."""
        dhal_window = tk.Toplevel(self.root)
        dhal_window.title("Dhal - Dark Agent")
        dhal_window.geometry("1000x700")
        dhal_window.transient(self.root)
        
        # Create Dark Agent tab in the new window with proper main_app reference
        from dark_agent import DarkAgentTab
        dark_agent_tab = DarkAgentTab(dhal_window, self.settings_manager, self)
        # Ensure the main_app reference is properly set
        dark_agent_tab.main_app = self
    
    def _open_adk(self):
        """Open Agent Development Kit in a new window."""
        adk_window = tk.Toplevel(self.root)
        adk_window.title("Agent Development Kit (ADK)")
        adk_window.geometry("900x600")
        adk_window.transient(self.root)
        
        # Create ADK interface
        adk_frame = ttk.Frame(adk_window)
        adk_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(adk_frame, text="Agent Development Kit", font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Label(adk_frame, text="Advanced tools for creating and managing AI agents").pack(pady=5)
        ttk.Label(adk_frame, text="Coming Soon...", font=("Arial", 12, "italic")).pack(pady=20)
    
    def _open_metasploit(self):
        """Open Metasploit interface in a new window."""
        metasploit_window = tk.Toplevel(self.root)
        metasploit_window.title("Metasploit")
        metasploit_window.geometry("1000x700")
        metasploit_window.transient(self.root)
        
        # Create Metasploit interface
        from pentestgpt import PentestGPTTab
        PentestGPTTab(metasploit_window, self.settings_manager, self)

    def _inspect_model_devices(self):
        """Open device inspection dialog"""
        from tools.inspect_devices import inspect_loaded_model, inspect_model_devices
        
        # Check if we have a loaded model
        current_model_path = self.model_var.get()
        if not current_model_path or current_model_path == "Select a model...":
            tk.messagebox.showwarning("No Model", "Please load a model first.")
            return
        
        # Create inspection window
        inspect_window = tk.Toplevel(self.root)
        inspect_window.title("Model Device Inspection")
        inspect_window.geometry("800x600")
        inspect_window.configure(bg='#2b2b2b')
        
        # Create text widget with scrollbar
        frame = tk.Frame(inspect_window, bg='#2b2b2b')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(frame, bg='#1e1e1e', fg='#ffffff', font=('Consolas', 10), wrap=tk.WORD)
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add inspection results
        text_widget.insert(tk.END, "Inspecting model devices...\n\n")
        text_widget.update()
        
        try:
            # Check if we have a loaded unified model
            if hasattr(chat_stream, '_unified_model_cache') and chat_stream._unified_model_cache:
                result = inspect_loaded_model(chat_stream._unified_model_cache.model)
            else:
                # Fallback: load and inspect the model path
                result = inspect_model_devices(current_model_path)
            
            text_widget.delete('1.0', tk.END)
            text_widget.insert(tk.END, result)
        except Exception as e:
            text_widget.delete('1.0', tk.END)
            text_widget.insert(tk.END, f"Error inspecting model:\n{str(e)}")
        
        text_widget.config(state=tk.DISABLED)

    def _on_closing(self):
        """Handle window closing event."""
        # Save current window size
        self.settings_manager.set('ui_preferences.window_width', self.root.winfo_width())
        self.settings_manager.set('ui_preferences.window_height', self.root.winfo_height())

        # Save other current values
        if self.model_var.get():
            self.settings_manager.set('paths.last_model_path', self.model_var.get())
        if self.lora_var.get():
            self.settings_manager.set('paths.last_lora_path', self.lora_var.get())

        self.settings_manager.save_settings()
        self.root.destroy()

    def _build_run_tab(self, frame: ttk.Frame):
        # Create notebook for Run sub-tabs
        run_notebook = ttk.Notebook(frame)
        run_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Chat sub-tab
        self.chat_frame = ttk.Frame(run_notebook)
        run_notebook.add(self.chat_frame, text="Chat")
        self._build_chat_subtab(self.chat_frame)
        
        # Model Settings sub-tab
        self.model_settings_frame = ttk.Frame(run_notebook)
        run_notebook.add(self.model_settings_frame, text="Model Settings")
        self._build_model_settings_tab(self.model_settings_frame)
    
    def _build_chat_subtab(self, frame: ttk.Frame):
        # Model loading section
        model_frame = ttk.LabelFrame(frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_var, width=50)
        self.model_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.browse_model_btn = ttk.Button(model_frame, text="Browse Model", command=self._browse_gguf)
        self.browse_model_btn.grid(row=0, column=2, padx=2)
        self.browse_folder_btn = ttk.Button(model_frame, text="Browse Folder", command=self._browse_folder)
        self.browse_folder_btn.grid(row=0, column=3, padx=2)
        self.load_model_btn = ttk.Button(model_frame, text="Load Model", command=self._on_load_unload_model)
        self.load_model_btn.grid(row=0, column=4, padx=5)

        # Chat Template row
        ttk.Label(model_frame, text="Chat Template:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.chat_template_combo = ttk.Combobox(model_frame, textvariable=self.chat_template_var, 
                                               values=["None"], state="readonly", width=20)
        self.chat_template_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(5, 0))
        ttk.Button(model_frame, text="Load", command=self._load_chat_template).grid(row=1, column=2, padx=2, pady=(5, 0))
        ttk.Button(model_frame, text="Add", command=self._add_chat_template).grid(row=1, column=3, padx=2, pady=(5, 0))

        status_frame = ttk.Frame(model_frame)
        status_frame.grid(row=2, column=0, columnspan=5, sticky=tk.W, pady=(5, 0))
        ttk.Label(status_frame, textvariable=self.model_status_var).pack(side=tk.LEFT)
        ttk.Label(status_frame, text=" | Supports: GGUF, SafeTensors, GPTQ, AWQ, EXL2, PyTorch", 
                 font=('Arial', 8), foreground='gray').pack(side=tk.LEFT, padx=(10, 0))

        # Configure grid weights for resizing
        model_frame.grid_columnconfigure(1, weight=1)

        # Options section
        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        ttk.Checkbutton(options_frame, text="Chess Mode (ChessGPT)", variable=self.chess_mode_var,
                        command=self._on_chess_mode_changed).pack(side=tk.LEFT)
        ttk.Checkbutton(options_frame, text="Stream Output", variable=self.stream_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Agent Mode controls
        self.agent_mode_var = tk.BooleanVar(value=False)
        agent_btn = ttk.Checkbutton(options_frame, text="🤖 Agent Mode (SYSTEM ACCESS)", 
                                    variable=self.agent_mode_var,
                                    command=self._on_agent_mode_changed)
        agent_btn.pack(side=tk.LEFT, padx=(20, 0))
        
        # Initialize agent handler
        self.agent_handler = None
        self._init_agent_mode()

        mid = ttk.Frame(frame)
        mid.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        ttk.Label(mid, text="Prompt / Chat Input:").pack(anchor=tk.W)
        self.prompt_text = tk.Text(mid, height=6)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, padx=8, pady=4)
        self.send_btn = ttk.Button(btns, text="Send (Chat)", command=self._on_chat)
        self.send_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Clear Output", command=lambda: self.output_text.delete('1.0', tk.END)).pack(side=tk.LEFT,
                                                                                                           padx=6)

        out = ttk.Frame(frame)
        out.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        ttk.Label(out, text="Output:").pack(anchor=tk.W)
        self.output_text = tk.Text(out, height=12)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _add_tooltip(self, widget, text):
        """Add a tooltip to a widget"""
        def create_tooltip(widget, text):
            def on_enter(event):
                # Prevent multiple tooltips
                if hasattr(widget, 'tooltip') and widget.tooltip:
                    return
                    
                try:
                    tooltip = tk.Toplevel()
                    tooltip.wm_overrideredirect(True)
                    x = widget.winfo_rootx() + 20
                    y = widget.winfo_rooty() + 20
                    tooltip.wm_geometry(f"+{x}+{y}")
                    label = tk.Label(tooltip, text=text, background="lightyellow", 
                                   relief="solid", borderwidth=1, font=("Arial", "9", "normal"))
                    label.pack()
                    widget.tooltip = tooltip
                except:
                    # Ignore tooltip creation errors
                    pass
                
            def on_leave(event):
                try:
                    if hasattr(widget, 'tooltip') and widget.tooltip:
                        widget.tooltip.destroy()
                        widget.tooltip = None
                except:
                    # Ignore tooltip destruction errors
                    pass
                    
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
        
        create_tooltip(widget, text)

    def _build_model_settings_tab(self, frame: ttk.Frame):
        # Context and GPU settings
        ctx_frame = ttk.LabelFrame(frame, text="Context & GPU Settings", padding="10")
        ctx_frame.pack(fill=tk.X, padx=8, pady=8)

        # Auto-config checkbox
        self.auto_context_var = tk.BooleanVar(value=self.settings_manager.get('model_settings.auto_context', True))
        self.auto_context_check = ttk.Checkbutton(
            ctx_frame, 
            text="Auto-configure context size based on model",
            variable=self.auto_context_var,
            command=self._on_auto_context_changed
        )
        self.auto_context_check.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        self._add_tooltip(self.auto_context_check, "Automatically use the model's trained context size (n_ctx_train) for optimal performance.\nDisable to manually set context size.")

        ttk.Label(ctx_frame, text="Context Length (n_ctx):").grid(row=1, column=0, sticky=tk.W)
        self.n_ctx_spin = ttk.Entry(ctx_frame, textvariable=self.n_ctx_var, width=15)
        self.n_ctx_spin.grid(row=1, column=1, sticky=tk.W, padx=5)
        self._add_tooltip(self.n_ctx_spin, "Maximum number of tokens the model can process at once.\nHigher values use more memory but allow longer conversations.")
        ttk.Label(ctx_frame, text="tokens").grid(row=1, column=2, sticky=tk.W)
        
        # Disable manual entry if auto-config is enabled
        if self.auto_context_var.get():
            self.n_ctx_spin.configure(state='disabled')

        # Auto-GPU config checkbox
        self.auto_gpu_var = tk.BooleanVar(value=self.settings_manager.get('model_settings.auto_gpu', True))
        self.auto_gpu_check = ttk.Checkbutton(
            ctx_frame, 
            text="Auto-configure GPU layers for optimal performance",
            variable=self.auto_gpu_var,
            command=self._on_auto_gpu_changed
        )
        self.auto_gpu_check.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        self._add_tooltip(self.auto_gpu_check, "Automatically set GPU layers based on your VRAM and model size.\nDisable to manually set GPU layers.")

        ttk.Label(ctx_frame, text="GPU Layers (n_gpu_layers):").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.n_gpu_spin = ttk.Entry(ctx_frame, textvariable=self.n_gpu_layers_var, width=15)
        self.n_gpu_spin.grid(row=3, column=1, sticky=tk.W, padx=5, pady=(5, 0))
        self._add_tooltip(self.n_gpu_spin, "Number of model layers to offload to GPU.\nHigher values improve speed but use more VRAM.\nUse 0 for CPU-only.")
        ttk.Label(ctx_frame, text="layers").grid(row=3, column=2, sticky=tk.W, pady=(5, 0))
        
        # Disable manual entry if auto-config is enabled
        if self.auto_gpu_var.get():
            self.n_gpu_spin.configure(state='disabled')

        # LoRA settings
        lora_frame = ttk.LabelFrame(frame, text="LoRA Adapter Settings", padding="10")
        lora_frame.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(lora_frame, text="LoRA Path (optional):").grid(row=0, column=0, sticky=tk.W)
        self.lora_entry = ttk.Entry(lora_frame, textvariable=self.lora_var, width=60)
        self.lora_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self._add_tooltip(self.lora_entry, "Path to LoRA (Low-Rank Adaptation) adapter file.\nLoRA adapters fine-tune model behavior without changing base weights.\nLeave empty if not using LoRA.")
        self.lora_btn = ttk.Button(lora_frame, text="Browse", command=self._browse_lora)
        self.lora_btn.grid(row=0, column=2, padx=5)
        lora_frame.grid_columnconfigure(1, weight=1)

        # Generation settings
        gen_frame = ttk.LabelFrame(frame, text="Generation Settings", padding="10")
        gen_frame.pack(fill=tk.X, padx=8, pady=8)

        # Max tokens (renamed for clarity)
        ttk.Label(gen_frame, text="Max New Tokens (n_predict):").grid(row=0, column=0, sticky=tk.W)
        self.max_tokens_spin = tk.Spinbox(gen_frame, from_=16, to=8192, increment=16, textvariable=self.max_tokens_var,
                                          width=15)
        self.max_tokens_spin.grid(row=0, column=1, sticky=tk.W, padx=5)
        self._add_tooltip(self.max_tokens_spin, "Maximum number of new tokens to generate.\nHigher values allow longer responses but take more time.")
        
        # Temperature
        ttk.Label(gen_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        temp_spin = tk.Spinbox(gen_frame, from_=0.0, to=2.0, increment=0.1, 
                              textvariable=self.temperature_var, width=15, format="%.1f")
        temp_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        self._add_tooltip(temp_spin, "Controls randomness in generation.\n0.0 = deterministic, 1.0 = balanced, 2.0 = very creative.\nLower values for factual tasks, higher for creative tasks.")
        
        # Top P
        ttk.Label(gen_frame, text="Top P:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        top_p_spin = tk.Spinbox(gen_frame, from_=0.0, to=1.0, increment=0.1, 
                               textvariable=self.top_p_var, width=15, format="%.1f")
        top_p_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        self._add_tooltip(top_p_spin, "Nucleus sampling parameter.\nOnly consider tokens in the top P probability mass.\n0.9 is typical, lower values for more focused responses.")
        
        # Repetition Penalty
        ttk.Label(gen_frame, text="Repetition Penalty:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        rep_pen_spin = tk.Spinbox(gen_frame, from_=0.5, to=2.0, increment=0.1, 
                                 textvariable=self.repetition_penalty_var, width=15, format="%.1f")
        rep_pen_spin.grid(row=3, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        self._add_tooltip(rep_pen_spin, "Penalty for repeating tokens.\n1.0 = no penalty, >1.0 = discourage repetition.\n1.1 is typical, higher values reduce repetition more.")
        
        # No Repeat N-gram Size
        ttk.Label(gen_frame, text="No Repeat N-gram Size:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        ngram_spin = tk.Spinbox(gen_frame, from_=0, to=10, increment=1, 
                               textvariable=self.no_repeat_ngram_size_var, width=15)
        ngram_spin.grid(row=4, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        self._add_tooltip(ngram_spin, "Prevent repeating N-grams (sequences of N tokens).\n0 = disabled, 2-4 = typical values.\nHigher values prevent more repetitive patterns.")
        
        # Min P
        ttk.Label(gen_frame, text="Min P:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        min_p_spin = tk.Spinbox(gen_frame, from_=0.0, to=1.0, increment=0.01, 
                               textvariable=self.min_p_var, width=15, format="%.2f")
        min_p_spin.grid(row=5, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        self._add_tooltip(min_p_spin, "Minimum probability threshold.\nTokens below this probability are excluded.\n0.0 = disabled, 0.05 = typical value.")
        
        # Typical P
        ttk.Label(gen_frame, text="Typical P:").grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
        typical_p_spin = tk.Spinbox(gen_frame, from_=0.0, to=1.0, increment=0.1, 
                                   textvariable=self.typical_p_var, width=15, format="%.1f")
        typical_p_spin.grid(row=6, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        self._add_tooltip(typical_p_spin, "Typical sampling parameter.\nFocuses on tokens with 'typical' information content.\n1.0 = disabled, 0.95 = typical value.")

        # Advanced Loading Options
        advanced_frame = ttk.LabelFrame(frame, text="Advanced Loading Options", padding="10")
        advanced_frame.pack(fill=tk.X, padx=8, pady=8)
        
        # Quantization options
        ttk.Label(advanced_frame, text="Quantization:").grid(row=0, column=0, sticky=tk.W)
        self.quantization_combo = ttk.Combobox(advanced_frame, textvariable=self.quantization_var, 
                                         values=["none", "4bit", "8bit", "gptq", "awq", "exl2"], state="readonly", width=20)
        self.quantization_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        self._add_tooltip(self.quantization_combo, "Reduce model memory usage by using lower precision.\nnone = full precision\n4bit/8bit = bitsandbytes quantization\ngptq/awq/exl2 = specialized quantization formats")
        ttk.Label(advanced_frame, text="(auto-detected for pre-quantized models)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        
        # Device strategy
        ttk.Label(advanced_frame, text="Device Strategy:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.device_combo = ttk.Combobox(advanced_frame, textvariable=self.device_strategy_var,
                                   values=["auto", "force_gpu", "balanced_split", "cpu_only"], 
                                   state="readonly", width=20)
        self.device_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        self._add_tooltip(self.device_combo, "How to distribute model across devices.\nauto = automatic distribution\nforce_gpu = all on GPU\nbalanced_split = split between CPU/GPU\ncpu_only = CPU only")
        ttk.Label(advanced_frame, text="(balanced_split for large models)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # GPU memory limit
        ttk.Label(advanced_frame, text="GPU Memory Limit:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.gpu_mem_spin = tk.Spinbox(advanced_frame, from_=1.0, to=24.0, increment=0.5, 
                                 textvariable=self.gpu_memory_limit_var, width=15, format="%.1f")
        self.gpu_mem_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        self._add_tooltip(self.gpu_mem_spin, "Maximum GPU memory to use (in GB).\nUsed with balanced_split strategy.\nSet below your GPU's total VRAM to leave room for other applications.")
        ttk.Label(advanced_frame, text="GB (for balanced_split)").grid(row=2, column=2, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        # Local models browser
        local_frame = ttk.LabelFrame(frame, text="Local Models Browser", padding="10")
        local_frame.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(local_frame, text="Local Models:").grid(row=0, column=0, sticky=tk.W)
        self.local_models_combo = ttk.Combobox(local_frame, textvariable=self.local_model_var, width=50,
                                               state="readonly")
        self.local_models_combo.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.local_models_combo.bind("<<ComboboxSelected>>", self._on_local_model_selected)
        ttk.Button(local_frame, text="Folder...", command=self._choose_models_folder).grid(row=0, column=2, padx=5)
        ttk.Button(local_frame, text="Refresh", command=self._refresh_local_models).grid(row=0, column=3, padx=5)
        local_frame.grid_columnconfigure(1, weight=1)

    def _build_resource_tab(self, frame: ttk.Frame):
        # GPU Information
        gpu_frame = ttk.LabelFrame(frame, text="GPU Information", padding="10")
        gpu_frame.pack(fill=tk.X, padx=8, pady=8)

        self.gpu_info_var = tk.StringVar(value="Checking GPU...")
        self.gpu_memory_var = tk.StringVar(value="Memory: Unknown")
        self.gpu_usage_var = tk.StringVar(value="Usage: Unknown")

        ttk.Label(gpu_frame, textvariable=self.gpu_info_var).grid(row=0, column=0, sticky=tk.W, columnspan=3)
        ttk.Label(gpu_frame, textvariable=self.gpu_memory_var).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(gpu_frame, textvariable=self.gpu_usage_var).grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Button(gpu_frame, text="Test GPU", command=self._test_gpu).grid(row=1, column=2, padx=5, rowspan=2)

        # CPU Information
        cpu_frame = ttk.LabelFrame(frame, text="CPU Information", padding="10")
        cpu_frame.pack(fill=tk.X, padx=8, pady=8)

        self.cpu_info_var = tk.StringVar(value="Detecting CPU...")
        self.cpu_usage_var = tk.StringVar(value="Usage: Unknown")
        self.ram_usage_var = tk.StringVar(value="RAM: Unknown")

        ttk.Label(cpu_frame, textvariable=self.cpu_info_var).grid(row=0, column=0, sticky=tk.W, columnspan=2)
        ttk.Label(cpu_frame, textvariable=self.cpu_usage_var).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Label(cpu_frame, textvariable=self.ram_usage_var).grid(row=2, column=0, sticky=tk.W, pady=(5, 0))

        # Resource monitoring controls
        controls_frame = ttk.LabelFrame(frame, text="Monitoring Controls", padding="10")
        controls_frame.pack(fill=tk.X, padx=8, pady=8)

        self.monitor_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, text="Enable Real-time Monitoring", variable=self.monitor_var,
                        command=self._toggle_monitoring).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Refresh Now", command=self._refresh_resources).pack(side=tk.LEFT, padx=(20, 0))

        # Initialize resource monitoring
        self._initialize_resource_monitoring()

    # ---------------- HuggingFace Browser Tab -----------------
    def _build_hf_tab(self, frame: ttk.Frame):
        # Import the new HuggingFace downloader module
        try:
            from hf_downloader import HuggingFaceAPI
            # Configure API based on settings
            api_key = None
            organization = None

            if not self.settings_manager.get('api.use_env_token', True):
                api_key = self.settings_manager.get('api.huggingface_token', '').strip()

            if self.settings_manager.get('api.use_organization', False):
                organization = self.settings_manager.get('api.organization', '').strip()

            self.hf_api = HuggingFaceAPI(api_key=api_key, organization=organization)
        except ImportError:
            ttk.Label(frame,
                      text="hf_downloader module not found. Please ensure hf_downloader.py is in the same directory.").pack(
                padx=8, pady=8)
            return
        except ValueError as e:
            ttk.Label(frame, text=f"API Key Error: {e}").pack(padx=8, pady=8)
            return
        except Exception as e:
            ttk.Label(frame, text=f"Error initializing HuggingFace API: {e}").pack(padx=8, pady=8)
            return

        # Search bar with dropdown (using settings defaults)
        search_row = ttk.Frame(frame)
        search_row.pack(fill=tk.X, padx=8, pady=8)
        self.hf_search_query = tk.StringVar()
        self.hf_search_type = tk.StringVar(
            value=self.settings_manager.get('search_preferences.default_search_type', 'Models'))

        # Search entry
        self.hf_search_entry = ttk.Entry(search_row, textvariable=self.hf_search_query, width=60)
        self.hf_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.hf_search_entry.bind("<Return>", lambda e: self._hf_search())

        # Search type dropdown
        self.hf_type_combo = ttk.Combobox(search_row, values=["Models", "Datasets"],
                                          textvariable=self.hf_search_type, state="readonly", width=15)
        self.hf_type_combo.pack(side=tk.LEFT, padx=(10, 0))

        # Search button
        ttk.Button(search_row, text="Search", command=self._hf_search).pack(side=tk.LEFT, padx=(10, 0))

        # Results area with enhanced columns
        results_frame = ttk.Frame(frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Create treeview with new column structure
        cols = ("creator", "name", "description", "keywords", "size", "metadata")
        self.hf_tree = ttk.Treeview(results_frame, columns=cols, show="headings", height=15)

        # Define column headings and widths
        self.hf_tree.heading("creator", text="Creator")
        self.hf_tree.heading("name", text="Name")
        self.hf_tree.heading("description", text="Description")
        self.hf_tree.heading("keywords", text="Keywords")
        self.hf_tree.heading("size", text="Size")
        self.hf_tree.heading("metadata", text="Metadata")

        self.hf_tree.column("creator", width=120)
        self.hf_tree.column("name", width=200)
        self.hf_tree.column("description", width=250)
        self.hf_tree.column("keywords", width=150)
        self.hf_tree.column("size", width=80)
        self.hf_tree.column("metadata", width=150)

        self.hf_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.hf_tree.bind("<Double-Button-1>", self._hf_download_selected)

        # Scrollbars
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.hf_tree.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.hf_tree.xview)
        self.hf_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Filter footer with checkboxes
        filter_frame = ttk.Frame(frame)
        filter_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)

        # Initialize filters based on default sort preference
        default_sort = self.settings_manager.get('search_preferences.default_sort', 'downloads')
        self.filter_most_downloaded = tk.BooleanVar(value=(default_sort == 'downloads'))
        self.filter_most_liked = tk.BooleanVar(value=(default_sort == 'likes'))
        self.filter_size = tk.BooleanVar(value=(default_sort == 'lastModified'))

        ttk.Checkbutton(filter_frame, text="Most Downloaded",
                        variable=self.filter_most_downloaded).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(filter_frame, text="Most Liked",
                        variable=self.filter_most_liked).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(filter_frame, text="Size",
                        variable=self.filter_size).pack(side=tk.LEFT, padx=(10, 0))

        # Download button on the right
        ttk.Button(filter_frame, text="Download Selected",
                   command=self._hf_download_selected).pack(side=tk.RIGHT, padx=5)

        # Status area for HF tab
        self.hf_status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.hf_status_var).pack(fill=tk.X, padx=8, pady=(0, 8))

        # Holder for last results (to act on selection)
        self._hf_results: List[Dict[str, Any]] = []

    def _hf_set_status(self, text: str):
        try:
            self.hf_status_var.set(text)
        except Exception:
            pass

    def _format_bytes(self, n: Optional[int]) -> str:
        try:
            if not n or n <= 0:
                return "-"
            units = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            f = float(n)
            while f >= 1024 and i < len(units) - 1:
                f /= 1024.0
                i += 1
            return f"{f:.1f} {units[i]}"
        except Exception:
            return "-"

    def _format_number(self, num: int) -> str:
        """Format large numbers with K, M suffixes."""
        if num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        return str(num)

    def _hf_search(self):
        query = (self.hf_search_query.get() or "").strip()
        search_type = self.hf_search_type.get()

        # Determine sort parameter based on filters
        sort = "downloads"
        if self.filter_most_liked.get() and not self.filter_most_downloaded.get():
            sort = "likes"
        elif self.filter_size.get() and not self.filter_most_downloaded.get() and not self.filter_most_liked.get():
            sort = "lastModified"

        self._hf_set_status("Searching ...")
        self.hf_tree.delete(*self.hf_tree.get_children())
        self._hf_results = []

        threading.Thread(target=self._hf_perform_search_thread, args=(search_type, query, sort), daemon=True).start()

    def _hf_perform_search_thread(self, search_type: str, query: str, sort: str):
        try:
            # Use the new API
            if search_type == "Models":
                results = self.hf_api.search_models(query, limit=50, sort=sort)
            else:
                results = self.hf_api.search_datasets(query, limit=50, sort=sort)

            rows = []
            packed = []

            for item in results:
                try:
                    # Extract fields based on search type
                    if search_type == "Models":
                        repo_id = item.get("modelId", item.get("id", ""))
                        pipeline_tag = item.get("pipeline_tag", "")
                        tags = item.get("tags", [])
                        keywords = ", ".join(tags[:3]) if tags else pipeline_tag
                        description = item.get("description", "")
                    else:
                        repo_id = item.get("id", "")
                        task_ids = item.get("cardData", {}).get("task_ids", [])
                        keywords = ", ".join(task_ids[:3]) if task_ids else "dataset"
                        card_data = item.get("cardData", {})
                        description = card_data.get("description", card_data.get("summary", ""))

                    creator = repo_id.split("/")[0] if "/" in repo_id else ""
                    name = repo_id.split("/")[1] if "/" in repo_id else repo_id

                    # Truncate description
                    if len(description) > 100:
                        description = description[:97] + "..."

                    # Calculate size
                    size_bytes = 0
                    siblings = item.get("siblings", [])
                    for sibling in siblings:
                        if isinstance(sibling, dict):
                            size = sibling.get("size", 0)
                            if isinstance(size, (int, float)):
                                size_bytes += size

                    size_str = self._format_bytes(size_bytes) if size_bytes > 0 else "-"

                    # Get metadata
                    metadata_parts = []
                    downloads = item.get("downloads", 0)
                    likes = item.get("likes", 0)

                    if downloads > 0:
                        metadata_parts.append(f"↓{self._format_number(downloads)}")
                    if likes > 0:
                        metadata_parts.append(f"♥{self._format_number(likes)}")

                    if search_type == "Models":
                        library = item.get("library_name", "")
                        if library:
                            metadata_parts.append(library)

                    metadata = " | ".join(metadata_parts)

                    # Prepare row
                    rows.append((creator, name, description, keywords, size_str, metadata))
                    packed.append({
                        "type": search_type,
                        "repo_id": repo_id,
                    })
                except Exception:
                    continue

            def apply_rows():
                try:
                    for row in rows:
                        self.hf_tree.insert("", tk.END, values=row)
                    self._hf_results = packed
                    self._hf_set_status(f"Found {len(rows)} {search_type.lower()}")
                except Exception:
                    pass

            self._enqueue_ui(apply_rows)
        except Exception as e:
            self._enqueue_ui(lambda: self._hf_set_status(f"Search error: {e}"))

    def _hf_download_selected(self, event=None):
        """Download the selected model or dataset using grouped download manager."""
        selection = self.hf_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select an item to download")
            return

        item = self.hf_tree.item(selection[0])
        values = item['values']

        if len(values) < 2:
            return

        creator = values[0]
        name = values[1]
        repo_id = f"{creator}/{name}" if creator else name

        # Ask for download location (use default from settings)
        initial_dir = self.settings_manager.get('paths.downloads_directory', './downloads')
        download_dir = filedialog.askdirectory(
            title="Select Download Directory",
            initialdir=initial_dir
        )
        if not download_dir:
            return

        self._hf_set_status(f"Fetching file list for {repo_id}...")

        def prepare_downloads():
            try:
                # Get file list
                files = self.hf_api.get_model_files(repo_id)

                if not files:
                    self._enqueue_ui(lambda: self._append_output(f"[Error] No files found for {repo_id}\n"))
                    self._enqueue_ui(lambda: self._hf_set_status("No files found"))
                    return

                # Always show file selection dialog for user choice
                self._enqueue_ui(lambda: self._show_file_selection_dialog(repo_id, files, download_dir))

            except Exception as e:
                self._enqueue_ui(lambda: self._append_output(f"[Download error] {e}\n"))
                self._enqueue_ui(lambda: self._hf_set_status("Download preparation failed"))

        threading.Thread(target=prepare_downloads, daemon=True).start()

    def _show_file_selection_dialog(self, repo_id: str, files: List[Dict], download_dir: str):
        """Show dialog to select files for download using the new FileSelectionDialog."""
        try:
            # Use the new FileSelectionDialog
            dialog = FileSelectionDialog(
                parent=self.root,
                repo_id=repo_id,
                files=files,
                title=f"Select Files to Download - {repo_id}"
            )

            result, selected_files = dialog.show()

            if result == 'download' and selected_files:
                # Create download group
                group_name = f"{repo_id.split('/')[-1] if '/' in repo_id else repo_id}"
                group_description = f"Files from {repo_id}"

                group_id = self.download_manager.create_download_group(
                    repo_id=repo_id,
                    name=group_name,
                    description=group_description
                )

                # Add selected files to the group
                download_count = 0
                for filename, file_info in selected_files:
                    url = f"{self.hf_api.base_url}/{repo_id}/resolve/main/{filename}"
                    save_path = os.path.join(download_dir, repo_id.replace("/", "_"), filename)

                    self.download_manager.add_file_to_group(
                        group_id=group_id,
                        filename=filename,
                        url=url,
                        save_path=save_path,
                        headers=self.hf_api.headers,
                        selected=True
                    )
                    download_count += 1

                self._hf_set_status(f"Added {download_count} file(s) to download queue")
                self._append_output(f"Created download group '{group_name}' with {download_count} files\n")

                # Switch to downloads tab to show the new group
                self.notebook.select(self.downloads_frame)
            else:
                self._hf_set_status("Download cancelled")

        except Exception as e:
            self._append_output(f"[Error] Failed to show file selection dialog: {e}\n")
            self._hf_set_status("Error showing file selection")

    def _browse_gguf(self):
        initial_dir = self.settings_manager.get('paths.models_directory', '.')
        path = filedialog.askopenfilename(
            title="Select Model (GGUF, Safetensors, GPTQ, AWQ, EXL2)",
            initialdir=initial_dir,
            filetypes=[
                ("All Model files", "*.gguf;*.safetensors;*.bin;*.pt;*.pth;*.exl2"),
                ("GGUF files", "*.gguf"),
                ("SafeTensors files", "*.safetensors"),
                ("PyTorch files", "*.bin;*.pt;*.pth"),
                ("GPTQ models", "*gptq*.safetensors;*gptq*.bin"),
                ("AWQ models", "*awq*.safetensors;*awq*.bin"),
                ("EXL2 files", "*.exl2"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.model_var.set(path)
            # Save last model path
            self.settings_manager.set('paths.last_model_path', path)
            self.settings_manager.save_settings()

    def _browse_folder(self):
        initial_dir = self.settings_manager.get('paths.models_directory', '.')
        path = filedialog.askdirectory(
            title="Select Model Directory (HuggingFace format)",
            initialdir=initial_dir
        )
        if path:
            self.model_var.set(path)
            # Save last model path
            self.settings_manager.set('paths.last_model_path', path)
            self.settings_manager.save_settings()

    def _browse_lora(self):
        path = filedialog.askopenfilename(title="Select LoRA/adapter file", filetypes=[("All files", "*.*")])
        if path:
            self.lora_var.set(path)

    def _mark_model_unloaded(self):
        """Mark model as unloaded and clear references"""
        # Clear model references
        if hasattr(self, 'current_model'):
            self.current_model = None
        if hasattr(chat_stream, '_unified_model_cache'):
            chat_stream._unified_model_cache = None
        
        # Clear any cached models
        global _llama_cache
        _llama_cache["key"] = None
        _llama_cache["llm"] = None
        
        # Clear agent reference
        if hasattr(self, 'dhal_agent'):
            self.dhal_agent = None
            
        # Update UI
        self.model_status_var.set("[not loaded]")
        self._update_load_button_text()
        
        # Clear chat history since model is unloaded
        self.chat_history = []
    
    def _disable_model_settings(self):
        """Disable model settings that would unload the model"""
        try:
            # Disable model path entry and browse buttons
            if hasattr(self, 'model_entry'):
                self.model_entry.configure(state='disabled')
            if hasattr(self, 'browse_model_btn'):
                self.browse_model_btn.configure(state='disabled')
            if hasattr(self, 'browse_folder_btn'):
                self.browse_folder_btn.configure(state='disabled')
            
            # Disable settings that would trigger model reload
            if hasattr(self, 'n_ctx_spin'):
                self.n_ctx_spin.configure(state='disabled')
            if hasattr(self, 'n_gpu_spin'):
                self.n_gpu_spin.configure(state='disabled')
            if hasattr(self, 'lora_entry'):
                self.lora_entry.configure(state='disabled')
            if hasattr(self, 'lora_btn'):
                self.lora_btn.configure(state='disabled')
            if hasattr(self, 'quantization_combo'):
                self.quantization_combo.configure(state='disabled')
            if hasattr(self, 'device_combo'):
                self.device_combo.configure(state='disabled')
            if hasattr(self, 'gpu_mem_spin'):
                self.gpu_mem_spin.configure(state='disabled')
        except Exception as e:
            print(f"Error disabling model settings: {e}")
    
    def _enable_model_settings(self):
        """Re-enable model settings"""
        try:
            # Re-enable model path entry and browse buttons
            if hasattr(self, 'model_entry'):
                self.model_entry.configure(state='normal')
            if hasattr(self, 'browse_model_btn'):
                self.browse_model_btn.configure(state='normal')
            if hasattr(self, 'browse_folder_btn'):
                self.browse_folder_btn.configure(state='normal')
            
            # Re-enable settings
            if hasattr(self, 'n_ctx_spin'):
                self.n_ctx_spin.configure(state='normal')
            if hasattr(self, 'n_gpu_spin'):
                self.n_gpu_spin.configure(state='normal')
            if hasattr(self, 'lora_entry'):
                self.lora_entry.configure(state='normal')
            if hasattr(self, 'lora_btn'):
                self.lora_btn.configure(state='normal')
            if hasattr(self, 'quantization_combo'):
                self.quantization_combo.configure(state='readonly')
            if hasattr(self, 'device_combo'):
                self.device_combo.configure(state='readonly')
            if hasattr(self, 'gpu_mem_spin'):
                self.gpu_mem_spin.configure(state='normal')
        except Exception as e:
            print(f"Error enabling model settings: {e}")

    def _init_agent_mode(self):
        """Initialize agent mode handler"""
        try:
            # Import agent components
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent_dhal'))
            from agent_dhal.hal import create_dhal, DhalConfig, HalModelClient
            
            self.agent_enabled = True
            self.dhal_agent = None
        except Exception as e:
            print(f"[Agent Mode] Could not import Hal components: {e}")
            self.agent_enabled = False
            self.agent_mode_var.set(False)
    
    def _on_agent_mode_changed(self):
        """Handle agent mode toggle"""
        if self.agent_mode_var.get():
            # Show warning
            result = messagebox.askyesno(
                "⚠️ Enable Agent Mode",
                "WARNING: Agent mode gives the AI UNRESTRICTED access to:\n\n"
                "• Your file system (read/write/delete)\n"
                "• Shell commands (PowerShell, Bash, CMD)\n"
                "• Mouse and keyboard control\n"
                "• Python code execution\n"
                "• Network requests\n"
                "• System settings\n\n"
                "The AI can control your computer completely!\n\n"
                "Only enable if you trust the model and understand the risks.\n\n"
                "Continue?",
                icon='warning'
            )
            
            if not result:
                self.agent_mode_var.set(False)
                return
                
            # Initialize agent if needed
            if self.agent_enabled and not self.dhal_agent:
                try:
                    # Check if we have a loaded model
                    if not (hasattr(self, 'current_model') and self.current_model) and \
                       not (hasattr(chat_stream, '_unified_model_cache') and chat_stream._unified_model_cache):
                        # No model loaded - show error
                        messagebox.showerror(
                            "No Model Loaded",
                            "Please load a model first before enabling Agent Mode.\n\n"
                            "Agent Mode requires a loaded language model to function."
                        )
                        self.agent_mode_var.set(False)
                        return
                    
                    # Use the existing loaded model
                    from agent_dhal.hal import create_dhal
                    
                    # Create a wrapper for the existing local model
                    class LocalModelClient:
                        def __init__(self, model):
                            self.llm_model = model
                            
                        async def create_chat_completion(self, messages, **kwargs):
                            # Convert messages to prompt for local model
                            prompt = ""
                            for msg in messages:
                                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                    role = msg.role
                                    content = msg.content
                                else:
                                    role = msg.get('role', 'user')
                                    content = msg.get('content', '')
                                
                                if role == "system":
                                    prompt += f"System: {content}\n"
                                elif role == "user":
                                    prompt += f"User: {content}\n"
                                elif role == "assistant":
                                    prompt += f"Assistant: {content}\n"
                            
                            prompt += "Assistant: "
                            
                            # Generate response using local model
                            from llm_runtime import GenerateConfig
                            cfg = GenerateConfig(
                                max_tokens=kwargs.get('max_tokens', 800),
                                temperature=kwargs.get('temperature', 0.7)
                            )
                            response = self.llm_model.generate(prompt, cfg)
                            
                            # Return in expected format
                            return type('Response', (), {
                                'choices': [type('Choice', (), {
                                    'message': type('Message', (), {
                                        'content': response
                                    })()
                                })()]
                            })()
                    
                    # Use whichever model is available
                    if hasattr(self, 'current_model') and self.current_model:
                        model_client = LocalModelClient(self.current_model)
                    else:
                        model_client = LocalModelClient(chat_stream._unified_model_cache)
                    
                    # Create the agent with the local model
                    self.dhal_agent = create_dhal(
                        name="Hal",
                        system_message="You are Hal with full system access. You can execute any command the user requests. Always explain what you're doing before executing commands.",
                        model="local",  # Just use "local" as identifier
                        model_client=model_client  # Pass the wrapped local model
                    )
                    
                    self._append_output("\n[AGENT MODE ACTIVATED] AI has full system control using local model\n")
                    self._append_output("Available tools: file operations, shell commands, Python execution, mouse/keyboard control\n\n")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to initialize agent: {e}")
                    self.agent_mode_var.set(False)
        else:
            self._append_output("\n[AGENT MODE DEACTIVATED] Normal chat mode\n\n")
    
    def _on_auto_context_changed(self):
        """Handle auto-context checkbox changes."""
        auto_context = self.auto_context_var.get()
        self.settings_manager.set('model_settings.auto_context', auto_context)
        self.settings_manager.save_settings()
        
        if auto_context:
            # Disable manual context entry
            if hasattr(self, 'n_ctx_spin'):
                self.n_ctx_spin.configure(state='disabled')
            
            # Try to auto-detect context from currently selected model
            model_path = self.model_var.get()
            if model_path and _is_gguf_model(model_path):
                try:
                    detected_n_ctx = _extract_gguf_int_metadata(model_path, "n_ctx_train") or \
                                   _extract_gguf_int_metadata(model_path, "n_ctx")
                    if detected_n_ctx:
                        self.n_ctx_var.set(detected_n_ctx)
                        self._append_output_threadsafe(f"[Auto-detected context size: {detected_n_ctx} tokens]\n")
                except Exception as e:
                    print(f"Could not auto-detect context size: {e}")
        else:
            # Enable manual context entry
            if hasattr(self, 'n_ctx_spin'):
                self.n_ctx_spin.configure(state='normal')

    def _on_auto_gpu_changed(self):
        """Handle auto-GPU checkbox changes."""
        auto_gpu = self.auto_gpu_var.get()
        self.settings_manager.set('model_settings.auto_gpu', auto_gpu)
        self.settings_manager.save_settings()
        
        if auto_gpu:
            # Disable manual GPU entry
            if hasattr(self, 'n_gpu_spin'):
                self.n_gpu_spin.configure(state='disabled')
            
            # Auto-detect optimal GPU layers
            model_path = self.model_var.get()
            if model_path and self._has_gpu():
                try:
                    optimal_layers = self._calculate_optimal_gpu_layers(model_path)
                    if optimal_layers > 0:
                        self.n_gpu_layers_var.set(optimal_layers)
                        self._append_output_threadsafe(f"[Auto-configured GPU layers: {optimal_layers}]\n")
                except Exception as e:
                    print(f"Could not auto-configure GPU layers: {e}")
        else:
            # Enable manual GPU entry
            if hasattr(self, 'n_gpu_spin'):
                self.n_gpu_spin.configure(state='normal')

    def _on_chess_mode_changed(self):
        """Handle chess mode checkbox changes."""
        chess_mode = self.chess_mode_var.get()
        self.settings_manager.set('model_settings.chess_mode', chess_mode)
        self.settings_manager.save_settings()
        
        if chess_mode:
            # Auto-configure for ChessGPT model
            messagebox.showinfo(
                "Chess Mode Enabled",
                "Chess Mode enabled! This will use the ChessGPT model for chess-specific conversations.\n\n"
                "Make sure you have the Waterhorse/chessgpt-chat-v1 model downloaded or use the HuggingFace browser to get it."
            )
            # Mark model as unloaded since we're switching modes
            self._mark_model_unloaded()
        else:
            # Reset to normal mode
            self._mark_model_unloaded()
    
    def _on_advanced_setting_changed(self):
        """Handle advanced loading settings changes."""
        # Save the settings
        self.settings_manager.set('model_settings.quantization', self.quantization_var.get())
        self.settings_manager.set('model_settings.device_strategy', self.device_strategy_var.get())
        self.settings_manager.set('model_settings.gpu_memory_limit', self.gpu_memory_limit_var.get())
        self.settings_manager.save_settings()
        
        # Mark model as unloaded since these settings affect loading
        self._mark_model_unloaded()
    
    def _on_sampling_setting_changed(self):
        """Handle sampling parameter changes."""
        # Save the sampling parameters
        self.settings_manager.set('model_settings.temperature', self.temperature_var.get())
        self.settings_manager.set('model_settings.top_p', self.top_p_var.get())
        self.settings_manager.set('model_settings.repetition_penalty', self.repetition_penalty_var.get())
        self.settings_manager.set('model_settings.no_repeat_ngram_size', self.no_repeat_ngram_size_var.get())
        self.settings_manager.set('model_settings.min_p', self.min_p_var.get())
        self.settings_manager.set('model_settings.typical_p', self.typical_p_var.get())
        self.settings_manager.save_settings()

    # Enqueue a callable to run on the Tk main thread
    def _enqueue_ui(self, fn):
        try:
            self._ui_queue.put_nowait(fn)
        except Exception:
            pass

    # Periodically drain UI queue
    def _drain_ui_queue(self):
        try:
            while True:
                fn = self._ui_queue.get_nowait()
                try:
                    fn()
                except Exception:
                    pass
        except queue.Empty:
            pass
        finally:
            self.root.after(30, self._drain_ui_queue)

    def _set_status_threadsafe(self, text: str):
        self._enqueue_ui(lambda: self.model_status_var.set(text))

    def _append_output_threadsafe(self, text: str):
        self._enqueue_ui(lambda t=text: self._append_output(t))

    def _set_running(self, running: bool):
        def _apply():
            try:
                state_run = "disabled" if running else "normal"
                state_stop = "normal" if running else "disabled"
                if hasattr(self, "generate_btn"):
                    self.generate_btn.configure(state=state_run)
                if hasattr(self, "send_btn"):
                    self.send_btn.configure(state=state_run)
                if hasattr(self, "stop_btn"):
                    self.stop_btn.configure(state=state_stop)
            except Exception:
                pass

        self._enqueue_ui(_apply)

    def _on_stop(self):
        try:
            if getattr(self, "_current_cancel", None) is not None:
                self._current_cancel.set()
        except Exception:
            pass

    def _on_load_unload_model(self):
        """Handle both load and unload based on current state"""
        if self._is_model_loaded():
            self._on_unload_model()
        else:
            self._on_load_model()

    def _is_model_loaded(self):
        """Check if a model is currently loaded"""
        return (hasattr(self, 'current_model') and self.current_model is not None) or \
               (hasattr(chat_stream, '_unified_model_cache') and chat_stream._unified_model_cache is not None)

    def _on_unload_model(self):
        """Unload the currently loaded model"""
        print("[APP_DEBUG] _on_unload_model() called")
        
        # Confirm before unloading
        result = messagebox.askyesno(
            "Unload Model",
            "Are you sure you want to unload the current model?\n\n"
            "This will clear the chat history and free GPU/CPU memory.",
            icon='question'
        )
        
        if not result:
            return
        
        # Clear model references
        if hasattr(self, 'current_model'):
            self.current_model = None
        if hasattr(chat_stream, '_unified_model_cache'):
            chat_stream._unified_model_cache = None
        
        # Clear any cached models
        global _llama_cache
        _llama_cache["key"] = None
        _llama_cache["llm"] = None
        
        # Clear agent reference
        if hasattr(self, 'dhal_agent'):
            self.dhal_agent = None
        
        # Update UI
        self._set_status_threadsafe("[not loaded]")
        self._update_load_button_text()
        self._append_output_threadsafe("[Model unloaded]\n")
        
        # Re-enable model settings
        self._enable_model_settings()
        
        # Clear chat history since model is unloaded
        self.chat_history = []
        
        print("[APP_DEBUG] Model unloaded successfully")

    def _update_load_button_text(self):
        """Update the load button text based on model state"""
        if self._is_model_loaded():
            self.load_model_btn.configure(text="Unload Model")
        else:
            self.load_model_btn.configure(text="Load Model")

    def _on_load_model(self):
        print("[APP_DEBUG] _on_load_model() called")
        start_tracing()  # Start detailed execution tracing
        model = self.model_var.get().strip()
        print(f"[APP_DEBUG] Model path: '{model}'")
        if not _is_valid_model(model):
            print("[APP_DEBUG] Invalid model detected")
            stop_tracing()
            messagebox.showerror("Load Model",
                                 "Please select a valid model file (GGUF, Safetensors, or HuggingFace repo).")
            return
        n_ctx = self.n_ctx_var.get()
        n_gpu = self.n_gpu_layers_var.get()
        lora = self.lora_var.get().strip() or None
        
        # Auto-detect optimal settings before loading if enabled
        if _is_gguf_model(model):
            # Auto-configure context size
            print(f"[CONTEXT_DEBUG] Auto-context enabled: {self.auto_context_var.get()}")
            if self.auto_context_var.get():
                try:
                    print(f"[CONTEXT_DEBUG] Attempting to extract context metadata from: {model}")
                    n_ctx_train = _extract_gguf_int_metadata(model, "n_ctx_train")
                    n_ctx_fallback = _extract_gguf_int_metadata(model, "n_ctx")
                    print(f"[CONTEXT_DEBUG] n_ctx_train = {n_ctx_train}, n_ctx = {n_ctx_fallback}")
                    detected_n_ctx = n_ctx_train or n_ctx_fallback
                    print(f"[CONTEXT_DEBUG] detected_n_ctx = {detected_n_ctx}")
                    if detected_n_ctx:
                        n_ctx = detected_n_ctx
                        self.n_ctx_var.set(n_ctx)  # Update the UI
                        print(f"[CONTEXT_DEBUG] Setting context size to {n_ctx}")
                        self._append_output_threadsafe(
                            f"[Auto-configuring context size to {n_ctx} tokens (model's trained capacity)]\n")
                    else:
                        print(f"[CONTEXT_DEBUG] No context metadata found, using default: {n_ctx}")
                except Exception as e:
                    print(f"Could not auto-detect context size: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[CONTEXT_DEBUG] Auto-context disabled, using manual setting: {n_ctx}")
            
            # Auto-configure GPU layers
            if self.auto_gpu_var.get():
                print(f"[GPU_DEBUG] Auto-GPU enabled, checking GPU availability...")
                if self._has_gpu():
                    print(f"[GPU_DEBUG] GPU detected, calculating optimal layers for model: {model}")
                    try:
                        optimal_layers = self._calculate_optimal_gpu_layers(model)
                        print(f"[GPU_DEBUG] Calculated optimal GPU layers: {optimal_layers}")
                        if optimal_layers > 0:
                            n_gpu = optimal_layers
                            self.n_gpu_layers_var.set(n_gpu)  # Update the UI
                            self._append_output_threadsafe(
                                f"[Auto-configuring GPU layers to {n_gpu} for optimal performance]\n")
                        else:
                            print(f"[GPU_DEBUG] Optimal layers = 0, not updating n_gpu")
                    except Exception as e:
                        print(f"Could not auto-configure GPU layers: {e}")
                else:
                    print(f"[GPU_DEBUG] No GPU detected, keeping CPU-only mode")
            else:
                print(f"[GPU_DEBUG] Auto-GPU disabled, using manual setting: {n_gpu}")
        
        self._set_status_threadsafe("[loading...]")
        
        # Create loading popup
        loading_popup = tk.Toplevel(self.root)
        loading_popup.title("Loading Model")
        loading_popup.geometry("400x150")
        loading_popup.resizable(False, False)
        loading_popup.transient(self.root)
        loading_popup.grab_set()
        
        # Center the popup
        loading_popup.update_idletasks()
        x = (loading_popup.winfo_screenwidth() // 2) - (loading_popup.winfo_width() // 2)
        y = (loading_popup.winfo_screenheight() // 2) - (loading_popup.winfo_height() // 2)
        loading_popup.geometry(f"+{x}+{y}")
        
        # Add loading message
        tk.Label(loading_popup, text="Loading Model...", font=("Arial", 12, "bold")).pack(pady=10)
        model_name = os.path.basename(model) if os.path.exists(model) else model
        tk.Label(loading_popup, text=model_name, font=("Arial", 10)).pack(pady=5)
        
        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(loading_popup, variable=progress_var, maximum=100, length=350, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start(10)
        
        # Status label
        status_label = tk.Label(loading_popup, text="Initializing...", font=("Arial", 9))
        status_label.pack(pady=5)
        
        # Disable model settings while loading
        self._disable_model_settings()

        def _run():
            try:
                print("[APP_DEBUG] _run() started in loading thread")
                # Load model using appropriate loader
                if _is_gguf_model(model):
                    print("[APP_DEBUG] Detected GGUF model, using _get_llama()")
                    print(f"[APP_DEBUG] GGUF loading parameters: n_ctx={n_ctx}, n_gpu_layers={n_gpu}, lora={lora}")
                    # Use existing GGUF loading logic
                    gguf_model = _get_llama(model, n_ctx=n_ctx, n_gpu_layers=n_gpu, lora_path=lora)
                    # Store as current_model for agent integration
                    self.current_model = gguf_model
                else:
                    print("[APP_DEBUG] Non-GGUF model detected, using unified loader")
                    # Use unified model loader for other formats and cache it
                    from llm_runtime import load_model
                    print("[APP_DEBUG] Imported load_model from llm_runtime")
                    
                    # Get advanced loading options
                    quantization = self.quantization_var.get()
                    device_strategy = self.device_strategy_var.get()
                    gpu_memory_limit = self.gpu_memory_limit_var.get()
                    print(f"[APP_DEBUG] Advanced options: quantization={quantization}, device_strategy={device_strategy}, gpu_memory_limit={gpu_memory_limit}")
                    
                    print(f"[APP_DEBUG] Calling load_model() with: model='{model}', device='auto'")
                    
                    # Ensure quantization is properly passed
                    load_kwargs = {
                        'n_ctx': n_ctx, 
                        'n_gpu_layers': n_gpu,
                        'device_strategy': device_strategy,
                        'gpu_memory_limit': gpu_memory_limit,
                        'device': "auto"
                    }
                    
                    # Only pass quantization if it's not 'none'
                    if quantization and quantization != 'none':
                        load_kwargs['quantization'] = quantization
                        print(f"[QUANTIZATION_DEBUG] Using quantization: {quantization}")
                    
                    unified_model = load_model(model, **load_kwargs)
                    print("[APP_DEBUG] load_model() completed successfully")
                    
                    # Cache the loaded model for chat function to reuse
                    chat_stream._unified_model_cache = unified_model
                    # Store as current_model for agent integration
                    self.current_model = unified_model
                    
                    # Warm up the model if supported
                    if hasattr(unified_model, 'warm_up_model'):
                        self._append_output_threadsafe("[Warming up model for optimal performance...]\n")
                        warmup_stats = unified_model.warm_up_model()
                        if warmup_stats.get('status') == 'success':
                            self._append_output_threadsafe(f"[Model warmed up in {warmup_stats['warmup_time']:.2f}s]\n")
                        else:
                            self._append_output_threadsafe(f"[Model warmup failed: {warmup_stats.get('error', 'unknown')}]\n")
                    
                    # Get model info if available
                    if hasattr(unified_model, 'get_model_info'):
                        model_info = unified_model.get_model_info()
                        self._append_output_threadsafe(f"[Model Info] {model_info.get('model_name', 'Unknown')}: {model_info.get('total_parameters', 'Unknown')} parameters\n")
                        self._append_output_threadsafe(f"[KV Cache] Enabled - Max context: {model_info.get('max_position_embeddings', 'Unknown')} tokens\n")

                # best-effort: try to detect language metadata from the GGUF file
                lang = None
                try:
                    lang = _extract_gguf_metadata(model, "language") or _extract_gguf_metadata(model, "lang")
                except Exception:
                    lang = None

                # Show context info for user awareness
                try:
                    detected_n_ctx = _extract_gguf_int_metadata(model, "n_ctx_train") or _extract_gguf_int_metadata(
                        model, "n_ctx")
                except Exception:
                    detected_n_ctx = None

                if detected_n_ctx and detected_n_ctx != n_ctx and not self.auto_context_var.get():
                    self._append_output_threadsafe(
                        f"[Model's trained context: {detected_n_ctx} tokens, using requested: {n_ctx} tokens]\n")
                    if detected_n_ctx > n_ctx:
                        self._append_output_threadsafe(
                            f"[Note: Enable 'Auto-configure context size' for optimal performance]\n")

                # Auto-configure optimal settings
                self._auto_configure_model_settings(model, detected_n_ctx)

                if lang:
                    self._set_status_threadsafe(f"[loaded] ({lang})")
                    self._append_output_threadsafe(f"[Model language detected: {lang}]\n")
                else:
                    self._set_status_threadsafe("[loaded]")
                
                # Update button text to "Unload Model"
                self._enqueue_ui(self._update_load_button_text)
                
                # Close loading popup on success
                self._enqueue_ui(lambda: loading_popup.destroy())
                
            except Exception as e:
                self._set_status_threadsafe("[error]")
                self._append_output_threadsafe(f"[Load Error] {e}\n")
                # Update button text back to "Load Model" on error
                self._enqueue_ui(self._update_load_button_text)
                # Close loading popup on error
                self._enqueue_ui(lambda: loading_popup.destroy())
                # Re-enable model settings on error
                self._enqueue_ui(self._enable_model_settings)
            finally:
                stop_tracing()  # Stop tracing when loading completes or fails

        threading.Thread(target=_run, daemon=True).start()

    def _auto_configure_model_settings(self, model_path, detected_n_ctx=None):
        """Auto-configure optimal GPU layers based on model and system resources"""
        try:
            # Auto-configure GPU layers based on available VRAM
            if self._has_gpu():
                optimal_gpu_layers = self._calculate_optimal_gpu_layers(model_path)
                if optimal_gpu_layers != self.n_gpu_layers_var.get():
                    self.n_gpu_layers_var.set(optimal_gpu_layers)
                    self._append_output_threadsafe(f"[Auto-configured GPU layers to {optimal_gpu_layers}]\n")
        except Exception as e:
            self._append_output_threadsafe(f"[Auto-config warning: {e}]\n")

    def _calculate_optimal_gpu_layers(self, model_path):
        """Calculate optimal number of GPU layers based on model size and available VRAM"""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            
            # Get available VRAM
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            # Reserve some VRAM for the system (1GB buffer)
            available_vram_gb = max(0, total_vram_gb - 1.0)
            
            # Detect model size from path or filename
            model_name = model_path.lower()
            
            # More comprehensive model size detection
            if any(x in model_name for x in ['1b', '1.5b']):
                # 1-1.5B models: ~0.5GB per layer, ~32 layers total
                layers_per_gb = 6
                max_layers = 32
            elif any(x in model_name for x in ['3b', '3.8b']):
                # 3B models: ~0.75GB per layer, ~32 layers total  
                layers_per_gb = 4
                max_layers = 32
            elif any(x in model_name for x in ['7b', '8b']):
                # 7-8B models: ~1GB per layer, ~32 layers total
                layers_per_gb = 3
                max_layers = 32
            elif any(x in model_name for x in ['13b', '14b']):
                # 13-14B models: ~1.5GB per layer, ~40 layers total
                layers_per_gb = 2
                max_layers = 40
            elif any(x in model_name for x in ['30b', '33b', '34b']):
                # 30-34B models: ~2.5GB per layer, ~60 layers total
                layers_per_gb = 1.2
                max_layers = 60
            elif any(x in model_name for x in ['65b', '70b']):
                # 65-70B models: ~4GB per layer, ~80 layers total
                layers_per_gb = 0.8
                max_layers = 80
            else:
                # Unknown size - conservative estimate
                layers_per_gb = 2
                max_layers = 32
            
            # Calculate optimal layers based on available VRAM
            optimal_layers = int(available_vram_gb * layers_per_gb)
            
            # Cap at the model's actual layer count
            optimal_layers = min(optimal_layers, max_layers)
            
            # Ensure at least some layers go to GPU if we have VRAM
            if available_vram_gb >= 2.0 and optimal_layers < 1:
                optimal_layers = 1
            
            return max(0, optimal_layers)
            
        except Exception as e:
            print(f"Error calculating GPU layers: {e}")
            return 0

    def _has_gpu(self):
        """Check if GPU is available for acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _initialize_resource_monitoring(self):
        """Initialize resource monitoring components"""
        self._refresh_resources()

    def _test_gpu(self):
        """Test GPU functionality by running a small inference"""

        def test():
            try:
                import torch
                if not torch.cuda.is_available():
                    self.gpu_info_var.set("No GPU detected")
                    return

                # Basic GPU test
                device = torch.device("cuda:0")
                test_tensor = torch.randn(1000, 1000).to(device)
                result = torch.matmul(test_tensor, test_tensor)
                torch.cuda.synchronize()

                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_info_var.set(f"GPU Test PASSED: {gpu_name}")
                self._append_output_threadsafe("[GPU test completed successfully]\n")

            except Exception as e:
                self.gpu_info_var.set(f"GPU Test FAILED: {e}")
                self._append_output_threadsafe(f"[GPU test failed: {e}]\n")

        threading.Thread(target=test, daemon=True).start()

    def _refresh_resources(self):
        """Refresh resource usage information"""

        def refresh():
            try:
                import psutil

                # CPU Info
                cpu_count = psutil.cpu_count(logical=False)
                cpu_count_logical = psutil.cpu_count(logical=True)
                self.cpu_info_var.set(f"CPU: {cpu_count} cores ({cpu_count_logical} threads)")

                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage_var.set(f"CPU Usage: {cpu_percent:.1f}%")

                # RAM Usage
                memory = psutil.virtual_memory()
                ram_gb_used = memory.used / (1024 ** 3)
                ram_gb_total = memory.total / (1024 ** 3)
                self.ram_usage_var.set(f"RAM: {ram_gb_used:.1f}GB / {ram_gb_total:.1f}GB ({memory.percent:.1f}%)")

                # GPU Info
                try:
                    import torch
                    self.gpu_info_var.set(f"PyTorch version: {torch.__version__}")

                    if torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                        gpu_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                        gpu_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                        cuda_version = torch.version.cuda

                        self.gpu_info_var.set(f"GPU: {gpu_name} (CUDA {cuda_version}) - {gpu_count} device(s)")
                        self.gpu_memory_var.set(
                            f"VRAM: {gpu_allocated:.1f}GB allocated, {gpu_reserved:.1f}GB reserved / {gpu_memory:.1f}GB total")
                        self.gpu_usage_var.set(f"GPU Usage: {(gpu_allocated / gpu_memory) * 100:.1f}%")
                    else:
                        # More detailed error info
                        cuda_available = hasattr(torch.backends, 'cuda') and torch.backends.cuda.is_built()
                        self.gpu_info_var.set(f"No CUDA GPU available (CUDA built: {cuda_available})")
                        self.gpu_memory_var.set("VRAM: N/A - Check CUDA installation")
                        self.gpu_usage_var.set("GPU Usage: N/A")

                except ImportError as e:
                    self.gpu_info_var.set(f"PyTorch not available: {e}")
                    self.gpu_memory_var.set("VRAM: Install PyTorch with CUDA support")
                    self.gpu_usage_var.set("GPU Usage: Unknown")
                except Exception as e:
                    self.gpu_info_var.set(f"GPU detection error: {e}")
                    self.gpu_memory_var.set(f"VRAM: Error - {str(e)}")
                    self.gpu_usage_var.set("GPU Usage: Error")

            except Exception as e:
                self.cpu_info_var.set(f"Error: {e}")

        threading.Thread(target=refresh, daemon=True).start()

    def _toggle_monitoring(self):
        """Toggle real-time resource monitoring"""
        if self.monitor_var.get():
            self._start_monitoring()
        else:
            self._stop_monitoring()

    def _start_monitoring(self):
        """Start real-time monitoring loop"""

        def monitor_loop():
            while self.monitor_var.get():
                self._refresh_resources()
                time.sleep(2)  # Update every 2 seconds

        if not hasattr(self, '_monitor_thread') or not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            self._monitor_thread.start()

    def _stop_monitoring(self):
        """Stop real-time monitoring"""
        # Thread will stop on next iteration when monitor_var.get() returns False
        pass

    def _choose_models_folder(self):
        initial_dir = self.settings_manager.get('paths.models_directory', '.')
        folder = filedialog.askdirectory(title="Select models folder", initialdir=initial_dir)
        if folder:
            self.models_dir_var.set(folder)
            # Save to settings
            self.settings_manager.set('paths.models_directory', folder)
            self.settings_manager.save_settings()
            self._refresh_local_models()

    def _refresh_local_models(self):
        folder = (self.models_dir_var.get() or "").strip()
        self._local_model_paths.clear()
        values: List[str] = []
        if folder and os.path.isdir(folder):
            try:
                for name in os.listdir(folder):
                    name_lower = name.lower()
                    # Check for all supported model formats
                    if (name_lower.endswith((".gguf", ".safetensors", ".bin", ".pt", ".pth", ".exl2")) or
                        ('gptq' in name_lower and name_lower.endswith(('.safetensors', '.bin'))) or
                        ('awq' in name_lower and name_lower.endswith(('.safetensors', '.bin')))):
                        full = os.path.join(folder, name)
                        display = name
                        self._local_model_paths[display] = full
                        values.append(display)
            except Exception:
                pass
        self.local_models_combo["values"] = values
        # keep selection if still present
        current_display = self.local_model_var.get()
        if current_display not in values:
            self.local_model_var.set(values[0] if values else "")

    def _on_local_model_selected(self, event=None):
        display = self.local_model_var.get()
        path = self._local_model_paths.get(display)
        if path:
            self.model_var.set(path)

    def _append_output(self, text: str):
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    def _on_generate(self):
        model = self.model_var.get().strip()
        prompt = self.prompt_text.get('1.0', tk.END).strip()
        if not _is_valid_model(model):
            messagebox.showerror("Model", "Please select a valid model file.")
            return
        if not prompt:
            messagebox.showinfo("Generate", "Please enter a prompt.")
            return
        self.output_text.delete('1.0', tk.END)
        n_ctx = self.n_ctx_var.get()
        n_gpu = self.n_gpu_layers_var.get()
        lora = self.lora_var.get().strip() or None

        # Retain memory by recording the user turn
        self.chat_history.append({"role": "user", "content": prompt})

        cancel = threading.Event()
        self._current_cancel = cancel
        self._set_running(True)

        def run():
            try:
                content = run_prompt(
                    model,
                    prompt,
                    self.stream_var.get(),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu,
                    lora_path=lora,
                    on_chunk=self._append_output_threadsafe,
                    n_threads=None,
                    max_tokens=self.max_tokens_var.get(),
                    history=self.chat_history,
                    cancel_event=cancel,
                    chess_mode=self.chess_mode_var.get(),
                )
                # Record assistant turn for future context
                self.chat_history.append({"role": "assistant", "content": content})
                self._append_output_threadsafe("\n" if not cancel.is_set() else "\n[stopped]\n")
            except Exception as e:
                self._append_output_threadsafe(f"\n[Error] {e}\n")
            finally:
                self._set_running(False)
                self._current_cancel = None

        threading.Thread(target=run, daemon=True).start()

    def _on_chat(self):
        print("DEBUG: _on_chat called")
        model = self.model_var.get().strip()
        user = self.prompt_text.get('1.0', tk.END).strip()
        print(f"DEBUG: model='{model}', user='{user}'")
        
        # Check for agent mode
        if self.agent_mode_var.get():
            # Use simple agent mode
            self._handle_agent_chat_simple(user)
            return
        
        if not _is_valid_model(model):
            print("DEBUG: Invalid model")
            messagebox.showerror("Model", "Please select a valid model file.")
            return
        if not user:
            print("DEBUG: No user input")
            messagebox.showinfo("Chat", "Please enter a message.")
            return
        print("DEBUG: Starting chat processing")
        n_ctx = self.n_ctx_var.get()
        n_gpu = self.n_gpu_layers_var.get()
        lora = self.lora_var.get().strip() or None

        self.chat_history.append({"role": "user", "content": user})
        self._append_output(f"You: {user}\nAssistant: ")

        cancel = threading.Event()
        self._current_cancel = cancel
        self._set_running(True)

        def run():
            try:
                print("DEBUG: Calling chat_stream")
                content = chat_stream(
                    model,
                    self.chat_history,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu,
                    lora_path=lora,
                    on_chunk=self._append_output_threadsafe,
                    n_threads=None,
                    max_tokens=self.max_tokens_var.get(),
                    cancel_event=cancel,
                    chess_mode=self.chess_mode_var.get(),
                    chat_template=self.chat_template_var.get(),
                    session_id=self._session_id,
                )
                print(f"DEBUG: Got response: '{content}'")
                self.chat_history.append({"role": "assistant", "content": content})
                self._append_output_threadsafe("\n" if not cancel.is_set() else "\n[stopped]\n")
            except Exception as e:
                print(f"DEBUG: Chat error: {e}")
                self._append_output_threadsafe(f"\n[Error] {e}\n")
            finally:
                self._set_running(False)
                self._current_cancel = None

        threading.Thread(target=run, daemon=True).start()
    
    def _handle_agent_chat_simple(self, user_message: str):
        """Simple agent mode that directly executes commands"""
        if not user_message:
            messagebox.showinfo("Chat", "Please enter a message.")
            return
        
        self._append_output(f"You: {user_message}\n")
        self._set_running(True)
        
        # Create agent activity popup
        agent_popup = tk.Toplevel(self.root)
        agent_popup.title("Agent Activity Monitor")
        agent_popup.geometry("500x300")
        agent_popup.resizable(True, True)
        agent_popup.transient(self.root)
        
        # Center the popup
        agent_popup.update_idletasks()
        x = (agent_popup.winfo_screenwidth() // 2) - (agent_popup.winfo_width() // 2)
        y = (agent_popup.winfo_screenheight() // 2) - (agent_popup.winfo_height() // 2)
        agent_popup.geometry(f"+{x}+{y}")
        
        # Add activity display
        tk.Label(agent_popup, text="🤖 Agent Activity Monitor", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Activity log
        log_frame = tk.Frame(agent_popup)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        activity_log = tk.Text(log_frame, height=15, bg='#1e1e1e', fg='#00ff00', font=('Consolas', 9))
        scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=activity_log.yview)
        activity_log.configure(yscrollcommand=scrollbar.set)
        
        activity_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Close button
        tk.Button(agent_popup, text="Close", command=agent_popup.destroy).pack(pady=5)
        
        def log_activity(message):
            """Add message to activity log"""
            try:
                activity_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
                activity_log.see(tk.END)
                activity_log.update()
            except:
                pass
        
        def run_simple_agent():
            try:
                # Import simple agent
                from simple_agent_mode import SimpleAgentExecutor
                agent = SimpleAgentExecutor(log_callback=log_activity)
                
                log_activity("Agent initialized, analyzing request...")
                
                # Use AI model to intelligently determine what actions to take
                log_activity(f"AI analyzing request: {user_message}")
                
                # Generate intelligent response using the loaded model
                if self.current_model:
                    try:
                        # Create a comprehensive system prompt for the agent
                        agent_system_prompt = f"""You are an AI assistant with system access. Analyze the user's request and provide the exact Windows commands needed.

User Request: "{user_message}"

For virus scanning, use Windows Defender PowerShell commands:
- Get-MpComputerStatus: Check antivirus status
- Start-MpScan -ScanType QuickScan: Quick virus scan
- Start-MpScan -ScanType FullScan: Full system scan
- Update-MpSignature: Update virus definitions

Respond with ONLY the PowerShell command(s) needed, one per line:

"""

                        # Get AI response for dynamic command generation
                        from llm_runtime import GenerateConfig
                        cfg = GenerateConfig(max_tokens=800, temperature=0.1)
                        
                        try:
                            # Check if this is a GGUF model (llama-cpp-python)
                            if hasattr(self.current_model, 'create_completion'):
                                # Use llama-cpp-python's native method with proper parameters
                                log_activity("Using GGUF model native completion method")
                                completion = self.current_model.create_completion(
                                    prompt=agent_system_prompt,
                                    max_tokens=800,
                                    temperature=0.1,
                                    stop=["\n\n", "Human:", "User:"],
                                    echo=False
                                )
                                ai_response = completion['choices'][0]['text'].strip()
                                
                            else:
                                # Use unified runtime method
                                log_activity("Using unified runtime method")
                                raw_response = self.current_model.generate(agent_system_prompt, cfg)
                                
                                # Handle different response types
                                if isinstance(raw_response, str):
                                    ai_response = raw_response
                                elif hasattr(raw_response, '__iter__') and not isinstance(raw_response, str):
                                    # It's a generator or iterable, collect tokens
                                    tokens = []
                                    for token in raw_response:
                                        if isinstance(token, str):
                                            tokens.append(token)
                                        else:
                                            tokens.append(str(token))
                                    ai_response = ''.join(tokens)
                                else:
                                    # Fallback: convert to string
                                    ai_response = str(raw_response)
                            
                            log_activity(f"AI generated action plan: {ai_response}")
                            
                        except Exception as gen_error:
                            log_activity(f"Error generating AI response: {gen_error}")
                            import traceback
                            log_activity(f"Full error traceback: {traceback.format_exc()}")
                            ai_response = "Get-MpComputerStatus; Start-MpScan -ScanType QuickScan"  # Fallback command
                        
                        # Execute the AI's action plan - treat as PowerShell commands
                        if ai_response.strip():
                            # Split into individual commands and execute each as PowerShell
                            commands = [cmd.strip() for cmd in ai_response.strip().split('\n') if cmd.strip()]
                            results = []
                            for cmd in commands:
                                if cmd and not cmd.startswith('#'):  # Skip comments
                                    self._append_output_threadsafe(f"[Executing] {cmd}\n")
                                    result = agent.tools["powershell"](cmd)
                                    results.append(f"Command: {cmd}\nResult: {result}")
                            
                            combined_result = "\n\n".join(results)
                        else:
                            combined_result = "No commands generated"
                        
                        result = combined_result
                        self._append_output_threadsafe(f"[AGENT]: {result}\n")
                        
                    except Exception as e:
                        log_activity(f"Error in AI command generation: {e}")
                        # Simple fallback - just pass the request to the agent for basic parsing
                        result = agent.process_request(user_message, f"The user wants: {user_message}")
                        self._append_output_threadsafe(f"[AGENT]: {result}\n")
                else:
                    # No model loaded - basic fallback processing
                    log_activity("No model loaded, using basic command processing")
                    result = agent.process_request(user_message, f"Please help with: {user_message}")
                    self._append_output_threadsafe(f"[AGENT]: {result}\n")
                
                log_activity("Agent processing completed")
                
            except Exception as e:
                log_activity(f"Agent error: {e}")
                self._append_output_threadsafe(f"\n[Agent Error]: {e}\n")
                import traceback
                traceback.print_exc()
            finally:
                log_activity("Agent task completed")
                self._set_running(False)
                # Close agent popup after 5 seconds
                self.root.after(5000, lambda: agent_popup.destroy())
        
        threading.Thread(target=run_simple_agent, daemon=True).start()
    
    def _handle_agent_chat(self, user_message: str):
        """Handle chat in agent mode with full system access"""
        if not user_message:
            messagebox.showinfo("Chat", "Please enter a message.")
            return
            
        self._append_output(f"You: {user_message}\n[AGENT]: ")
        self._set_running(True)
        
        def run_agent():
            try:
                if not self.dhal_agent:
                    # Use the already loaded model if available
                    if hasattr(chat_stream, '_unified_model_cache') and chat_stream._unified_model_cache:
                        # Use the cached model directly
                        from agent_dhal.hal import Dhal, DhalConfig
                        
                        # Create a simple pass-through model client that uses our cached model
                        class CachedModelClient:
                            def __init__(self, cached_model):
                                self.model = cached_model
                                
                            async def create_chat_completion(self, messages, **kwargs):
                                # Convert messages to prompt
                                prompt = ""
                                for msg in messages:
                                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                        role = msg.role
                                        content = msg.content
                                    else:
                                        role = msg.get('role', 'user')
                                        content = msg.get('content', '')
                                    
                                    if role == "system":
                                        prompt += f"System: {content}\n\n"
                                    elif role == "user":
                                        prompt += f"User: {content}\n\n"
                                    elif role == "assistant":
                                        prompt += f"Assistant: {content}\n\n"
                                
                                prompt += "Assistant: "
                                
                                # Generate response using cached model
                                try:
                                    from llm_runtime import GenerateConfig
                                    cfg = GenerateConfig(
                                        max_tokens=kwargs.get('max_tokens', 2000),
                                        temperature=kwargs.get('temperature', 0.7)
                                    )
                                    response = self.model.generate(prompt, cfg)
                                except:
                                    # Fallback for models without GenerateConfig
                                    response = self.model.generate(prompt)
                                
                                # Create response object
                                class CompletionResponse:
                                    def __init__(self, content):
                                        self.content = content
                                        self.function_calls = None
                                
                                return CompletionResponse(response)
                            
                            def is_available(self):
                                return True
                        
                        config = DhalConfig(
                            name="Hal",
                            system_message="You are Hal with full system access. You can execute any command the user requests. Always explain what you're doing before executing commands.",
                            model=self.model_var.get() or "cached"
                        )
                        
                        model_client = CachedModelClient(chat_stream._unified_model_cache)
                        self.dhal_agent = Dhal(config, model_client)
                    else:
                        # Fallback to creating new agent
                        from agent_dhal.hal import create_dhal
                        self.dhal_agent = create_dhal(
                            name="Hal",
                            system_message="You are Hal with full system access. You can execute any command the user requests. Always explain what you're doing before executing commands.",
                            model=self.model_var.get() or "gpt-4"
                        )
                
                # Process message through agent
                import asyncio
                
                async def process():
                    # Create mock context
                    class MockContext:
                        def __init__(self):
                            self.agent_id = "user"
                    
                    response = await self.dhal_agent.handle_user_message(user_message, MockContext())
                    return response
                
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response = loop.run_until_complete(process())
                    self._append_output_threadsafe(response + "\n")
                finally:
                    loop.close()
                    
            except Exception as e:
                self._append_output_threadsafe(f"\n[Agent Error] {e}\n")
                import traceback
                traceback.print_exc()
            finally:
                self._set_running(False)
        
        threading.Thread(target=run_agent, daemon=True).start()


# ---------------- CLI (Embedded only) -----------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedded llama.cpp app (no external daemons). Runs local GGUF models via llama-cpp-python.")
    p.add_argument("--gui", action="store_true", help="Launch the GUI.")
    p.add_argument("--model", required=False, help="Path to a local GGUF model file.")
    p.add_argument("--prompt",
                   help="Single prompt to generate a response for. If omitted with no --gui, starts interactive chat mode.")
    p.add_argument("--stream", action="store_true", help="Stream output tokens for single-prompt mode.")
    p.add_argument("--n_ctx", type=int, default=4096, help="Context window size (default: 4096)")
    p.add_argument("--n_gpu_layers", type=int, default=0, help="GPU layers to offload (default: 0 = CPU)")
    p.add_argument("--lora", help="Optional LoRA/adapter file path to apply.")
    return p.parse_args(argv)


def launch_main_gui(acceleration_type=None):
    """Launch the main DarkHal 2.0 GUI application with hardware acceleration"""
    if acceleration_type:
        print(f"Launching DarkHal 2.0 with {acceleration_type.upper()} acceleration...")
        # Set default GPU layers based on acceleration type
        if acceleration_type == 'cuda':
            # Use high GPU offloading for CUDA
            os.environ['DARKHAL_DEFAULT_GPU_LAYERS'] = '32'
        elif acceleration_type == 'intel':
            # Moderate GPU offloading for Intel GPU
            os.environ['DARKHAL_DEFAULT_GPU_LAYERS'] = '16'
        elif acceleration_type == 'cpu':
            # No GPU offloading for CPU-only mode
            os.environ['DARKHAL_DEFAULT_GPU_LAYERS'] = '0'

    root = tk.Tk()
    app = EmbeddedGUI(root)
    root.mainloop()


def main(argv: List[str]) -> int:
    print(f"[APP_DEBUG] main() called with argv: {argv}")
    args = parse_args(argv)
    print(f"[APP_DEBUG] Parsed args: {args}")

    # Default to GUI when no CLI-specific args are provided, or when --gui is passed
    if args.gui or (not args.model and not args.prompt):
        print("[APP_DEBUG] Starting GUI mode")
        # Show splash screen then launch main app
        splash_manager = SplashManager(main_app_callback=launch_main_gui)
        splash_manager.show_splash_and_launch()
        return 0

    # CLI mode requires a GGUF model path
    if not args.model or not _is_gguf_model(args.model):
        print("Please provide --model pointing to a local .gguf file (or run with --gui).", file=sys.stderr)
        return 2

    if args.prompt:
        out = run_prompt(args.model, args.prompt, args.stream, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                         lora_path=(args.lora or None))
        print(out)
    else:
        # Interactive chat
        messages: List[Dict[str, Any]] = []
        print("Starting interactive chat. Type 'exit' or 'quit' to leave.")
        while True:
            try:
                user = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if user.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if not user:
                continue
            messages.append({"role": "user", "content": user})
            try:
                print("Assistant> ", end="", flush=True)

                def _print_chunk(s: str):
                    print(s, end="", flush=True)

                assistant_content = chat_stream(args.model, messages, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                                                lora_path=(args.lora or None), on_chunk=_print_chunk, chat_template=None, session_id=None)
                print()
                messages.append({"role": "assistant", "content": assistant_content})
            except Exception as e:
                print(f"\n[Error] {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
