from typing import Any, Iterator, List, Optional, Tuple, Dict
import os
import time
from llm_runtime.types import UnifiedModel, GenerateConfig
from llm_runtime.util_chat import apply_chat_template
from llm_runtime.chat_session import ChatSession, get_session_manager
from llm_runtime.device_utils import device_for_hf

class _HFUnified:
    def __init__(self, src: str, **kwargs: Any):
        print(f"[HF_DEBUG] _HFUnified.__init__() called with src='{src}', kwargs={kwargs}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            print("[HF_DEBUG] Successfully imported torch and transformers")
        except ImportError:
            print("[HF_DEBUG] Failed to import torch or transformers")
            raise ImportError("transformers, torch, and accelerate are required for HF models. Install with: pip install transformers torch accelerate safetensors")
        
        self.torch = torch
        self.TextIteratorStreamer = TextIteratorStreamer
        
        # Normalize device for HuggingFace
        self.device = device_for_hf(kwargs.get("device"))
        print(f"Loading HF model from: {src}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            src, 
            use_fast=True,
            trust_remote_code=kwargs.get("trust_remote_code", True),
            token=kwargs.get("token")
        )
        
        # Prepare quantization config if requested
        quantization_config = self._prepare_quantization_config(kwargs)
        
        # Prepare device mapping
        device_map, max_memory = self._prepare_device_mapping(kwargs, self.device)
        
        # Load model with advanced options
        load_kwargs = {
            "torch_dtype": kwargs.get("torch_dtype", "auto"),
            "device_map": device_map,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
            "low_cpu_mem_usage": True,
            "token": kwargs.get("token")
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            print(f"Using quantization: {kwargs.get('quantization', 'none')}")
            # When quantization is enabled, force device_map to "auto" to avoid device format conflicts
            load_kwargs["device_map"] = "auto"
            print(f"[DEBUG] Forcing device_map='auto' for quantization compatibility")
        
        if max_memory:
            load_kwargs["max_memory"] = max_memory
            print(f"Memory limits: {max_memory}")
        
        if kwargs.get("offload_folder"):
            load_kwargs["offload_folder"] = kwargs.get("offload_folder")
            print(f"Offloading to: {kwargs.get('offload_folder')}")
        
        self.model = AutoModelForCausalLM.from_pretrained(src, **load_kwargs)
        
        # Set pad token if not present
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            
        # Initialize chat session support with context from kwargs
        self.session_manager = get_session_manager()
        self.current_session = None
        
        # Store context size for session creation
        self.n_ctx = kwargs.get('n_ctx', 4096)

    def _prepare_quantization_config(self, kwargs: Any):
        """Prepare quantization configuration"""
        quantization = kwargs.get("quantization", "none")
        
        if quantization == "none":
            return None
        
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            print("Warning: BitsAndBytesConfig not available, skipping quantization")
            return None
        
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch.bfloat16,
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            print(f"Warning: Unknown quantization type '{quantization}', skipping")
            return None
    
    def _prepare_device_mapping(self, kwargs: Any, hf_device: str):
        """Prepare device mapping and memory limits"""
        device_strategy = kwargs.get("device_strategy", "auto")
        gpu_memory_limit = kwargs.get("gpu_memory_limit", None)
        
        device_map = "auto"
        max_memory = None
        
        if device_strategy == "force_gpu":
            device_map = {"": hf_device}
        elif device_strategy == "balanced_split":
            # Use memory limits for balanced CPU/GPU split
            if gpu_memory_limit:
                # For quantization, use integer device index; for non-quantization, use string
                if kwargs.get("quantization", "none") != "none":
                    gpu_device = 0 if hf_device.startswith("cuda:") else 0
                else:
                    gpu_device = hf_device if hf_device.startswith("cuda:") else "cuda:0"
                max_memory = {
                    gpu_device: f"{gpu_memory_limit}GiB",
                    "cpu": "48GiB",  # Large CPU limit
                }
            device_map = "auto"
        elif device_strategy == "cpu_only":
            device_map = {"": "cpu"}
        else:  # auto
            if gpu_memory_limit:
                # For quantization, use integer device index; for non-quantization, use string  
                if kwargs.get("quantization", "none") != "none":
                    gpu_device = 0 if hf_device.startswith("cuda:") else 0
                else:
                    gpu_device = hf_device if hf_device.startswith("cuda:") else "cuda:0"
                max_memory = {
                    gpu_device: f"{gpu_memory_limit}GiB",
                    "cpu": "32GiB",
                }
            device_map = "auto"
        
        return device_map, max_memory

    def _build_eos_ids(self, stop) -> Optional[List[int]]:
        """Convert stop strings to token IDs"""
        if not stop: 
            return None
        
        ids = []
        for s in stop:
            # Handle single character stops
            if len(s) == 1:
                tid = self.tok.convert_tokens_to_ids(s)
                if tid is not None: 
                    ids.append(tid)
        return ids or None

    def generate(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> str:
        """Generate text - uses KV cache if session_id provided, otherwise fallback to standard generation"""
        # Set defaults for None values
        if cfg.temperature is None:
            cfg.temperature = 0.7  # Default temperature
        if cfg.top_p is None:
            cfg.top_p = 0.95  # Default top_p
        if cfg.max_tokens is None:
            cfg.max_tokens = 500  # Default max tokens
            
        session_id = kwargs.get('session_id', 'default')  # Use 'default' session if none specified
        if session_id:
            return self.generate_with_cache(prompt, session_id, cfg, **kwargs)
        
        # Fallback to standard generation without KV cache
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        class MultiStringStop(StoppingCriteria):
            def __init__(self, toks, stops): 
                self.toks, self.stops = toks, stops or []
            
            def __call__(self, input_ids, scores, **_):
                text = self.toks.decode(input_ids[0], skip_special_tokens=True)
                return any(s in text for s in self.stops)

        with self.torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_tokens,
                do_sample=cfg.temperature is not None and cfg.temperature > 0.0,
                temperature=cfg.temperature if cfg.temperature is not None and cfg.temperature > 0.0 else None,
                top_p=cfg.top_p,
                eos_token_id=self._build_eos_ids(cfg.stop),
                stopping_criteria=StoppingCriteriaList([MultiStringStop(self.tok, cfg.stop)]) if cfg.stop else None,
                pad_token_id=self.tok.pad_token_id,
            )
        
        # Decode only the new tokens
        generated_text = self.tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text

    def stream(self, prompt: str, cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> Iterator[str]:
        """Stream text generation - uses KV cache if session_id provided, otherwise fallback to standard streaming"""
        session_id = kwargs.pop('session_id', 'default')  # Remove from kwargs to avoid duplicate
        if session_id:
            yield from self.stream_with_cache(prompt, session_id, cfg, **kwargs)
            return
        
        # Fallback to standard streaming without KV cache
        import threading
        
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        streamer = self.TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        
        def _worker():
            with self.torch.no_grad():
                self.model.generate(
                    **enc,
                    max_new_tokens=cfg.max_tokens,
                    do_sample=cfg.temperature is not None and cfg.temperature > 0.0,
                    temperature=cfg.temperature if cfg.temperature is not None and cfg.temperature > 0.0 else None,
                    top_p=cfg.top_p,
                    eos_token_id=self._build_eos_ids(cfg.stop),
                    streamer=streamer,
                    pad_token_id=self.tok.pad_token_id,
                )
        
        t = threading.Thread(target=_worker)
        t.start()
        
        for text in streamer:
            yield text

    def tokenize(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def detokenize(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)
    
    def get_session(self, session_id: str = "default") -> ChatSession:
        """Get or create a chat session for KV caching"""
        # Create session config with the correct context size
        from llm_runtime.chat_session import ChatSessionConfig
        session_config = ChatSessionConfig(max_context_length=self.n_ctx)
        return self.session_manager.get_session(session_id, config=session_config)
    
    def _prefill_phase(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor', 
                      cfg: GenerateConfig) -> Tuple['torch.Tensor', Any]:
        """Prefill phase: process the full prompt and return logits + past_key_values"""
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
        return outputs.logits, outputs.past_key_values
    
    def _prefill_incremental(self, new_input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor',
                           past_key_values: Any, cfg: GenerateConfig) -> Tuple['torch.Tensor', Any]:
        """Incremental prefill: process only new tokens with existing KV cache"""
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        return outputs.logits, outputs.past_key_values
    
    def _decode_step(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor',
                    past_key_values: Any, cfg: GenerateConfig) -> Tuple['torch.Tensor', Any]:
        """Single decode step: generate next token with KV cache"""
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,  # Should be shape [1, 1] for single token
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        return outputs.logits, outputs.past_key_values
    
    def _sample_token(self, logits: 'torch.Tensor', cfg: GenerateConfig) -> int:
        """Sample next token from logits based on generation config with optimized sampling"""
        # Get logits for the last token
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature scaling
        if cfg.temperature is not None and cfg.temperature != 1.0 and cfg.temperature > 0:
            next_token_logits = next_token_logits / cfg.temperature
        
        # Apply top-p (nucleus) sampling if specified
        if cfg.temperature is not None and cfg.temperature > 0.0 and cfg.top_p is not None and cfg.top_p < 1.0:
            sorted_logits, sorted_indices = self.torch.sort(next_token_logits, descending=True)
            cumulative_probs = self.torch.cumsum(self.torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > cfg.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
        
        if cfg.temperature is not None and cfg.temperature > 0.0:
            # Sample from distribution
            probs = self.torch.softmax(next_token_logits, dim=-1)
            next_token = self.torch.multinomial(probs, num_samples=1)
        else:
            # Greedy sampling (deterministic)
            next_token = self.torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        return next_token.item()
    
    def _should_stop(self, token_id: int, generated_text: str, cfg: GenerateConfig) -> bool:
        """Check if generation should stop"""
        # Check for EOS token
        if token_id == self.tok.eos_token_id:
            return True
        
        # Check for custom stop strings
        if cfg.stop:
            for stop_str in cfg.stop:
                if stop_str in generated_text:
                    return True
        
        return False
    
    def generate_with_cache(self, prompt: str, session_id: str = "default", 
                           cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> str:
        """Generate text with persistent KV cache using manual generation loop"""
        # Set defaults for None values
        if cfg.temperature is None:
            cfg.temperature = 0.7  # Default temperature
        if cfg.top_p is None:
            cfg.top_p = 0.95  # Default top_p
        if cfg.max_tokens is None:
            cfg.max_tokens = 500  # Default max tokens
            
        session = self.get_session(session_id)
        
        # Tokenize input
        inputs = self.tok(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        generated_tokens = []
        max_new_tokens = cfg.max_tokens
        
        # Check if we need to invalidate cache with proper token validation
        if session.should_invalidate(prompt, self.tok):
            session.invalidate_cache()
            print(f"[KV_CACHE] Cache invalidated for session {session_id}")
        
        # Determine if we need prefill phase
        if session.past_key_values is None:
            print(f"[KV_CACHE] Running prefill phase for {len(input_ids[0])} tokens")
            # Prefill phase: process full prompt
            logits, past_key_values = self._prefill_phase(input_ids, attention_mask, cfg)
            
            # Update session cache
            session.update_cache(past_key_values, input_ids[0].tolist(), prompt)
            
            # Sample first token
            next_token_id = self._sample_token(logits, cfg)
            generated_tokens.append(next_token_id)
            
            # Update input_ids and attention_mask for decode phase
            current_length = input_ids.shape[1]
        else:
            print(f"[KV_CACHE] Using cached KV state, context length: {session.context_length}")
            # Use cached state
            past_key_values = session.past_key_values
            current_length = session.context_length
            
            # For cached state, we still need to process the new part of the prompt if it exists
            cached_length = len(session.cached_input_ids)
            if input_ids.shape[1] > cached_length:
                print(f"[KV_CACHE] Processing {input_ids.shape[1] - cached_length} new tokens (incremental prefill)")
                new_tokens = input_ids[:, cached_length:]
                
                # Process only the new tokens with existing KV cache
                # Create attention mask that covers both cached and new tokens
                total_length = cached_length + new_tokens.shape[1]
                extended_attention = self.torch.ones((1, total_length), device=self.model.device)
                
                logits, past_key_values = self._prefill_incremental(new_tokens, extended_attention, past_key_values, cfg)
                session.update_cache(past_key_values, input_ids[0].tolist(), prompt)
                current_length = input_ids.shape[1]
            else:
                # No new tokens to process - get initial logits for generation
                # Use a forward pass with the last token to get proper logits distribution
                last_token = self.torch.tensor([[session.cached_input_ids[-1]]], device=self.model.device)
                total_length = current_length + 1
                next_attention = self.torch.ones((1, total_length), device=self.model.device)
                
                logits, past_key_values = self._decode_step(last_token, next_attention, past_key_values, cfg)
            
            # Sample first new token
            next_token_id = self._sample_token(logits, cfg)
            generated_tokens.append(next_token_id)
        
        # Decode phase: generate tokens one by one with optimized attention management
        for step in range(max_new_tokens - 1):  # -1 because we already generated first token
            # Check stop conditions
            generated_text = self.tok.decode(generated_tokens, skip_special_tokens=True)
            if self._should_stop(next_token_id, generated_text, cfg):
                break
            
            # Prepare inputs for next step - only pass the new token
            next_input = self.torch.tensor([[next_token_id]], device=self.model.device)
            total_length = current_length + len(generated_tokens) + 1
            next_attention = self.torch.ones((1, total_length), device=self.model.device)
            
            # Generate next token using cached KV states
            logits, past_key_values = self._decode_step(next_input, next_attention, past_key_values, cfg)
            next_token_id = self._sample_token(logits, cfg)
            generated_tokens.append(next_token_id)
        
        # Update session with final state - include all processed tokens
        final_input_ids = input_ids[0].tolist() + generated_tokens
        session.update_cache(past_key_values, final_input_ids, prompt + self.tok.decode(generated_tokens, skip_special_tokens=True))
        
        # Decode generated tokens
        generated_text = self.tok.decode(generated_tokens, skip_special_tokens=True)
        print(f"[KV_CACHE] Generated {len(generated_tokens)} tokens with KV cache")
        
        return generated_text
    
    def stream_with_cache(self, prompt: str, session_id: str = "default",
                         cfg: GenerateConfig = GenerateConfig(), **kwargs: Any) -> Iterator[str]:
        """Stream generation with persistent KV cache"""
        session = self.get_session(session_id)
        
        # Tokenize input
        inputs = self.tok(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        generated_tokens = []
        max_new_tokens = cfg.max_tokens
        
        # Check if we need to invalidate cache with proper token validation
        if session.should_invalidate(prompt, self.tok):
            session.invalidate_cache()
            print(f"[KV_CACHE] Cache invalidated for session {session_id}")
        
        # Determine if we need prefill phase
        if session.past_key_values is None:
            print(f"[KV_CACHE] Streaming prefill phase for {len(input_ids[0])} tokens")
            # Prefill phase: process full prompt
            logits, past_key_values = self._prefill_phase(input_ids, attention_mask, cfg)
            
            # Update session cache
            session.update_cache(past_key_values, input_ids[0].tolist(), prompt)
            
            # Sample first token
            next_token_id = self._sample_token(logits, cfg)
            generated_tokens.append(next_token_id)
            current_length = input_ids.shape[1]
            
            # Yield first token
            first_text = self.tok.decode([next_token_id], skip_special_tokens=True)
            if first_text:
                yield first_text
        else:
            print(f"[KV_CACHE] Streaming with cached KV state, context length: {session.context_length}")
            # Use cached state - optimized logic similar to generate_with_cache
            past_key_values = session.past_key_values
            current_length = session.context_length
            
            # Handle new tokens in prompt with incremental prefill
            cached_length = len(session.cached_input_ids)
            if input_ids.shape[1] > cached_length:
                print(f"[KV_CACHE] Streaming incremental prefill for {input_ids.shape[1] - cached_length} new tokens")
                new_tokens = input_ids[:, cached_length:]
                
                # Process only new tokens with existing KV cache
                total_length = cached_length + new_tokens.shape[1]
                extended_attention = self.torch.ones((1, total_length), device=self.model.device)
                
                logits, past_key_values = self._prefill_incremental(new_tokens, extended_attention, past_key_values, cfg)
                session.update_cache(past_key_values, input_ids[0].tolist(), prompt)
                current_length = input_ids.shape[1]
            else:
                # No new tokens - use last cached token for initial generation
                last_token = self.torch.tensor([[session.cached_input_ids[-1]]], device=self.model.device)
                total_length = current_length + 1
                next_attention = self.torch.ones((1, total_length), device=self.model.device)
                
                logits, past_key_values = self._decode_step(last_token, next_attention, past_key_values, cfg)
            
            # Sample first token
            next_token_id = self._sample_token(logits, cfg)
            generated_tokens.append(next_token_id)
            
            # Yield first token
            first_text = self.tok.decode([next_token_id], skip_special_tokens=True)
            if first_text:
                yield first_text
        
        # Decode phase: stream tokens one by one with optimized KV cache usage
        for step in range(max_new_tokens - 1):
            # Check stop conditions
            generated_text = self.tok.decode(generated_tokens, skip_special_tokens=True)
            if self._should_stop(next_token_id, generated_text, cfg):
                break
            
            # Prepare inputs for next step - efficient single token processing
            next_input = self.torch.tensor([[next_token_id]], device=self.model.device)
            total_length = current_length + len(generated_tokens) + 1
            next_attention = self.torch.ones((1, total_length), device=self.model.device)
            
            # Generate next token using cached KV states
            logits, past_key_values = self._decode_step(next_input, next_attention, past_key_values, cfg)
            next_token_id = self._sample_token(logits, cfg)
            generated_tokens.append(next_token_id)
            
            # Yield the new token immediately
            token_text = self.tok.decode([next_token_id], skip_special_tokens=True)
            if token_text:
                yield token_text
        
        # Update session with final state - include all processed tokens
        final_input_ids = input_ids[0].tolist() + generated_tokens
        full_generated_text = self.tok.decode(generated_tokens, skip_special_tokens=True)
        session.update_cache(past_key_values, final_input_ids, prompt + full_generated_text)
        
        print(f"[KV_CACHE] Streamed {len(generated_tokens)} tokens with persistent KV cache")
    
    def clear_session_cache(self, session_id: str = "default") -> None:
        """Clear KV cache for a specific session"""
        session = self.get_session(session_id)
        session.invalidate_cache()
        print(f"[DEBUG] Cleared cache for session {session_id}")
    
    def get_session_info(self, session_id: str = "default") -> Dict[str, Any]:
        """Get information about a chat session"""
        session = self.get_session(session_id)
        return session.get_cache_info()
    
    def add_conversation_turn(self, user_message: str, assistant_message: str, 
                            session_id: str = "default") -> None:
        """Add a complete conversation turn to the session history"""
        session = self.get_session(session_id)
        session.add_message("user", user_message)
        session.add_message("assistant", assistant_message)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded model"""
        try:
            # Basic model info
            info = {
                "model_name": getattr(self.model.config, 'name_or_path', 'unknown'),
                "model_type": self.model.config.model_type,
                "vocab_size": self.model.config.vocab_size,
                "device": str(self.device),
                "dtype": str(self.model.dtype),
                "supports_kv_cache": True,
                "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'unknown'),
                "torch_compile_enabled": hasattr(self.model, '_orig_mod')
            }
            
            # Memory info
            if self.torch.cuda.is_available() and str(self.device) != 'cpu':
                info.update({
                    "gpu_memory_allocated": f"{self.torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                    "gpu_memory_reserved": f"{self.torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                    "gpu_memory_total": f"{self.torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                })
            
            # Model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info.update({
                "total_parameters": f"{total_params:,}",
                "trainable_parameters": f"{trainable_params:,}",
                "model_size_mb": f"{total_params * 4 / 1024**2:.2f} MB"  # Assuming float32
            })
            
            return info
        except Exception as e:
            return {"error": f"Could not get model info: {e}", "supports_kv_cache": True}
    
    def get_kv_cache_stats(self) -> Dict[str, Any]:
        """Get KV cache statistics across all sessions"""
        try:
            sessions = self.session_manager.get_all_sessions()
            stats = {
                "total_sessions": len(sessions),
                "active_sessions": 0,
                "total_cached_tokens": 0,
                "memory_usage_estimate": "0 MB"
            }
            
            for session_id in sessions:
                session = self.session_manager.get_session(session_id)
                if session.past_key_values is not None:
                    stats["active_sessions"] += 1
                    stats["total_cached_tokens"] += session.context_length
            
            # Rough estimate of KV cache memory usage
            # Each token in KV cache roughly uses: hidden_size * num_layers * 2 (key + value) * 4 bytes (float32)
            if stats["total_cached_tokens"] > 0:
                try:
                    hidden_size = self.model.config.hidden_size
                    num_layers = self.model.config.num_hidden_layers
                    memory_bytes = stats["total_cached_tokens"] * hidden_size * num_layers * 2 * 4
                    stats["memory_usage_estimate"] = f"{memory_bytes / 1024**2:.2f} MB"
                except:
                    pass
            
            return stats
        except Exception as e:
            return {"error": f"Could not get KV cache stats: {e}"}
    
    def warm_up_model(self, test_prompt: str = "Hello") -> Dict[str, float]:
        """Warm up the model and measure performance metrics"""
        try:
            print("[KV_CACHE] Warming up model...")
            start_time = time.time()
            
            # Simple generation to warm up CUDA kernels
            cfg = GenerateConfig(max_tokens=5, temperature=0.0)
            _ = self.generate(test_prompt, cfg=cfg, session_id="warmup")
            
            warmup_time = time.time() - start_time
            
            # Clean up warmup session
            self.clear_session_cache("warmup")
            
            return {
                "warmup_time": warmup_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "warmup_time": 0.0,
                "status": "failed",
                "error": str(e)
            }

class HFTransformersLoader:
    name = "hf"
    
    def can_load(self, source: str, **kwargs: Any) -> bool:
        # Accept HF repo-id or local dir with config.json (covers .safetensors)
        if "/" in source and not os.path.exists(source):  # repo-id like "microsoft/DialoGPT-medium"
            return True
        
        # Check if it's a directory with config.json
        if os.path.isdir(source) and os.path.exists(os.path.join(source, "config.json")):
            return True
            
        # If it's a single .safetensors file, look for config.json in the same directory
        if source.lower().endswith('.safetensors'):
            parent_dir = os.path.dirname(source)
            return os.path.exists(os.path.join(parent_dir, "config.json"))
        
        # Also support .bin files (PyTorch checkpoints)
        if source.lower().endswith('.bin'):
            parent_dir = os.path.dirname(source)
            return os.path.exists(os.path.join(parent_dir, "config.json"))
            
        return False
    
    def load(self, source: str, **kwargs: Any) -> UnifiedModel:
        return _HFUnified(source, **kwargs)