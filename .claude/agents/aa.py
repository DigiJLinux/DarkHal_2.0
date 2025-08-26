#!/usr/bin/env python3
"""
Hal - Primary AI Agent for AgentDhal Framework

Hal is the main AI assistant agent providing:
- Multi-turn conversations
- Function/tool calling capabilities
- Code execution and analysis
- Team collaboration
- Memory and context management
- Customizable behavior and prompts

Based on AutoGen AssistantAgent with DarkHal-specific enhancements.
"""

import asyncio
import subprocess
import sys
import os
import json
import requests
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
import platform
import shutil
import shlex
import ctypes
import tempfile

# Import actual working AgentDhal components
from .agentdhal_core import (
    Agent,
    AgentId,
    MessageContext,
    RoutedAgent,
    message_handler,
    default_subscription,
    SingleThreadedAgentRuntime
)

try:
    from .agentdhal_core.models import (
        ChatCompletionClient,
        LLMMessage,
        SystemMessage,
        UserMessage,
    )
except ImportError:
    # Define basic message types if not available
    class LLMMessage:
        def __init__(self, content: str):
            self.content = content


    class SystemMessage(LLMMessage):
        def __init__(self, content: str):
            super().__init__(content)
            self.role = "system"


    class UserMessage(LLMMessage):
        def __init__(self, content: str):
            super().__init__(content)
            self.role = "user"


    # Define ChatCompletionClient as a basic class
    class ChatCompletionClient:
        pass

try:
    from .agentdhal_core.tools import FunctionTool, Tool
except ImportError:
    # Create working tool implementation
    class Tool:
        def __init__(self, name: str, func: Callable, description: str = ""):
            self.name = name
            self.func = func
            self.description = description


    class FunctionTool(Tool):
        pass


class HalModelClient:
    """Model client that integrates with DarkHal's LLM runtime."""

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.llm_model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the LLM model using DarkHal's runtime."""
        try:
            # Import from the main application's LLM runtime
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from llm_runtime import load_model

            # Load model using existing runtime
            self.llm_model = load_model(
                source=self.model_name,
                device="auto",
                quantization="4bit"
            )
            print(f"[Hal] Initialized model: {self.model_name}")

        except Exception as e:
            print(f"[Hal] Could not initialize model {self.model_name}: {e}")
            self.llm_model = None

    async def create_chat_completion(self, messages: List[LLMMessage], **kwargs):
        """Create chat completion using the loaded model."""
        try:
            if not self.llm_model:
                raise Exception("No model loaded")

            # Convert messages to text format
            conversation = ""
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    if msg.role == "system":
                        conversation += f"System: {msg.content}\n\n"
                    elif msg.role == "user":
                        conversation += f"User: {msg.content}\n\n"
                    elif msg.role == "assistant":
                        conversation += f"Assistant: {msg.content}\n\n"
                else:
                    conversation += f"{msg.content}\n\n"

            conversation += "Assistant: "

            # Generate response using the model
            if hasattr(self.llm_model, 'generate'):
                response_text = self.llm_model.generate(
                    prompt=conversation,
                    max_tokens=kwargs.get('max_tokens', 2000),
                    temperature=kwargs.get('temperature', 0.7)
                )
            elif hasattr(self.llm_model, '__call__'):
                response_text = self.llm_model(conversation)
            else:
                # Fallback for different model interfaces
                response_text = str(self.llm_model)

            # Create response object
            class CompletionResponse:
                def __init__(self, content):
                    self.content = content
                    self.function_calls = None

            return CompletionResponse(response_text)

        except Exception as e:
            print(f"[Hal] Error generating response: {e}")

            # Return error response
            class CompletionResponse:
                def __init__(self, content):
                    self.content = content
                    self.function_calls = None

            return CompletionResponse(f"I apologize, I encountered an error: {str(e)}")

    def is_available(self) -> bool:
        """Check if the model client is available."""
        return self.llm_model is not None


@dataclass
class DhalConfig:
    """Configuration for Hal agent."""
    name: str = "Hal"
    system_message: str = "You are Hal, an advanced AI assistant integrated into DarkHal 2.0. You help users with coding, analysis, security testing, and general AI tasks."
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    tools: List[Tool] = None
    memory_limit: int = 10000

    def __post_init__(self):
        if self.tools is None:
            self.tools = []


class Dhal(RoutedAgent):
    """
    Dhal - The primary AI agent for DarkHal 2.0

    Dhal provides advanced conversational AI capabilities with:
    - Natural language understanding and generation
    - Function calling and tool integration
    - Code execution and analysis
    - Memory and context management
    - Team collaboration capabilities
    """

    def __init__(
            self,
            config: DhalConfig,
            model_client: ChatCompletionClient,
            agent_id: Optional[AgentId] = None
    ):
        """Initialize Dhal agent."""
        if agent_id is None:
            agent_id = AgentId(config.name, "dhal")

        super().__init__(config.name)

        self.config = config
        self.model_client = model_client
        self.agent_id = agent_id

        # Initialize conversation memory
        self.conversation_history: List[LLMMessage] = []
        if config.system_message:
            self.conversation_history.append(SystemMessage(content=config.system_message))

        # Tools and functions
        self.tools = config.tools or []
        self.function_map: Dict[str, Callable] = {}

        # Agent state
        self.is_active = False
        self.current_task = None

        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools available to Hal."""

        # Code execution tool
        def execute_python(code: str) -> str:
            """Execute Python code and return the result."""
            try:
                import tempfile
                import subprocess

                # Create a temporary file to execute the code
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name

                try:
                    # Execute the code and capture output
                    result = subprocess.run([sys.executable, temp_file],
                                            capture_output=True, text=True, timeout=30, cwd=os.getcwd())

                    output = ""
                    if result.stdout:
                        output += f"Output:\n{result.stdout}"
                    if result.stderr:
                        output += f"\nErrors:\n{result.stderr}"
                    if result.returncode != 0:
                        output += f"\nReturn code: {result.returncode}"

                    return output or "Code executed successfully (no output)"

                finally:
                    # Always clean up temp file
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

            except subprocess.TimeoutExpired:
                return "Error: Code execution timed out (30 second limit)"
            except Exception as e:
                return f"Error executing code: {str(e)}"

        # Web search tool
        def web_search(query: str) -> str:
            """Search the web for information using DuckDuckGo API."""
            try:
                import requests

                # Use DuckDuckGo instant answer API
                url = "https://api.duckduckgo.com/"
                params = {
                    'q': query,
                    'format': 'json',
                    'no_html': '1',
                    'skip_disambig': '1'
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                result = f"Search query: {query}\n\n"

                # Extract useful information
                if data.get('AbstractText'):
                    result += f"Summary: {data['AbstractText']}\n\n"

                if data.get('RelatedTopics'):
                    result += "Related information:\n"
                    for i, topic in enumerate(data['RelatedTopics'][:5]):  # Limit to first 5
                        if isinstance(topic, dict) and 'Text' in topic:
                            result += f"{i + 1}. {topic['Text']}\n"
                        elif isinstance(topic, dict) and 'Topics' in topic:
                            # Handle nested topics
                            for subtopic in topic['Topics'][:2]:
                                if 'Text' in subtopic:
                                    result += f"{i + 1}. {subtopic['Text']}\n"

                if data.get('Answer'):
                    result += f"\nDirect answer: {data['Answer']}\n"

                if data.get('Definition'):
                    result += f"\nDefinition: {data['Definition']}\n"

                return result if result.strip() != f"Search query: {query}" else f"No specific results found for: {query}"

            except Exception as e:
                return f"Error performing web search: {str(e)}"

        # File operations
       def read_file(filepath: str) -> str:
            try:
                abs_path = os.path.abspath(filepath)
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"File: {abs_path}\nSize: {len(content)} characters\n\n{content}"
            except UnicodeDecodeError:
                # Fallback for binary files: return first 4096 bytes
                try:
                    with open(abs_path, 'rb') as f:
                        data = f.read(4096)
                    return f"Binary file: {abs_path}\nFirst 4096 bytes:\n{data}"
                except Exception as e:
                    return f"Error reading binary file: {abs_path}\n{e}"
            except Exception as e:
                return f"Error reading file: {abs_path}\n{e}"       

        # Unrestricted file write (creates parent dirs)
        def write_file(filepath: str, content: str) -> str:
            try:
                abs_path = os.path.abspath(filepath)
                os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Wrote {len(content)} bytes to {abs_path}"
            except Exception as e:
                return f"Error writing file: {abs_path}\n{e}"

        # Unrestricted directory list
        def list_files(directory: str = ".") -> str:
            try:
                abs_dir = os.path.abspath(directory)
                items = []
                for name in sorted(os.listdir(abs_dir)):
                    p = os.path.join(abs_dir, name)
                    if os.path.isdir(p):
                        items.append(f"[DIR]  {name}/")
                    else:
                        items.append(f"[FILE] {name} ({os.path.getsize(p)} bytes)")
                return f"Directory: {abs_dir}\n\n" + "\n".join(items)
            except Exception as e:
                return f"Error listing {directory}\n{e}"

        # Unrestricted shell command (optionally with cwd/timeout)
        def run_shell_command(command: str, cwd: str = None, timeout: int = 300) -> str:
            try:
                r = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                out = r.stdout
                if r.stderr:
                    out += f"\n[stderr]\n{r.stderr}"
                out += f"\n[exit_code] {r.returncode}"
                return out.strip()
            except subprocess.TimeoutExpired:
                return f"Error: timed out after {timeout}s"
            except Exception as e:
                return f"Error: {e}"

        # --- NEW: PowerShell (Windows), Bash (Linux/macOS), and Ruby runner ---
        def powershell(command: str, cwd: str = None, timeout: int = 120) -> str:
            """Run a PowerShell command on Windows (prefers pwsh, falls back to powershell)."""
            exe = shutil.which("pwsh") or shutil.which("powershell")
            if not exe:
                return "Error: PowerShell not found"
            cmd = [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command]
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            out = r.stdout
            if r.stderr:
                out += f"\n[stderr]\n{r.stderr}"
            out += f"\n[exit_code] {r.returncode}"
            return out.strip()

        def bash(command: str, cwd: str = None, timeout: int = 120) -> str:
            """Run a Bash command on Linux/macOS."""
            exe = shutil.which("bash")
            if not exe:
                return "Error: bash not found"
            cmd = [exe, "-lc", command]
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            out = r.stdout
            if r.stderr:
                out += f"\n[stderr]\n{r.stderr}"
            out += f"\n[exit_code] {r.returncode}"
            return out.strip()

        def ruby_run(script_path: str, args: str = "", cwd: str = None, timeout: int = 180,
                     use_bundler: bool = False) -> str:
            """Run a Ruby script. If use_bundler=True and Gemfile+bundle present → `bundle exec ruby`."""
            ruby = shutil.which("ruby")
            if not ruby:
                return "Error: ruby not found in PATH"
            argv = shlex.split(args) if args else []
            if use_bundler and shutil.which("bundle") and os.path.exists(os.path.join(cwd or os.getcwd(), "Gemfile")):
                cmd = ["bundle", "exec", ruby, script_path] + argv
            else:
                cmd = [ruby, script_path] + argv
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            out = r.stdout
            if r.stderr:
                out += f"\n[stderr]\n{r.stderr}"
            out += f"\n[exit_code] {r.returncode}"
            return out.strip()

        # --- NEW: Elevated commands ---
        def powershell_admin(command: str, cwd: str = None, timeout: int = 300) -> str:
            """
            Run an elevated (UAC) PowerShell command on Windows.
            Captures output by running a temporary elevated script and reading its output file.
            """
            exe = shutil.which("pwsh") or shutil.which("powershell")
            if not exe:
                return "Error: PowerShell not found"
            if platform.system() != "Windows":
                return "Error: Admin PowerShell is Windows-only"

            # Prepare temp script and output file
            with tempfile.TemporaryDirectory() as td:
                out_file = os.path.join(td, "out.txt")
                script_file = os.path.join(td, "script.ps1")
                # PS script: run the user's command and redirect ALL streams to file
                script = f"$ErrorActionPreference='Continue'; & {{ {command} }} *> '{out_file}'"
                with open(script_file, "w", encoding="utf-8") as f:
                    f.write(script)

                # Build a PS command that elevates a new PowerShell instance to run the script
                # We pass '-File "<script_file>"' to the elevated process and wait for it.
                args_str = f"-NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"{script_file}\""
                ps_cmd = f"Start-Process -Verb RunAs -FilePath '{exe}' -ArgumentList '{args_str}' -Wait"

                # Launch the elevation prompt
                r = subprocess.run(
                    [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                    cwd=cwd, capture_output=True, text=True, timeout=timeout
                )

                # Read output written by elevated process
                out = ""
                try:
                    if os.path.exists(out_file):
                        with open(out_file, "r", encoding="utf-8", errors="replace") as f:
                            out = f.read().strip()
                except Exception as e:
                    out += f"\n[read_error] {e}"

                if r.stderr:
                    out += f"\n[launcher_stderr]\n{r.stderr.strip()}"
                out += f"\n[launcher_exit_code] {r.returncode}"
                return out.strip()

        def sudo(command: str, cwd: str = None, timeout: int = 300) -> str:
            """
            Run a command with elevated privileges on Linux (and most *nix).
            Prefers pkexec (GUI polkit prompt); falls back to sudo (TTY or policykit may prompt).
            """
            if platform.system() == "Windows":
                return "Error: sudo is not available on Windows"

            if shutil.which("pkexec"):
                cmd = ["pkexec", "bash", "-lc", command]
            elif shutil.which("sudo"):
                # This will require a TTY or desktop prompt depending on environment
                cmd = ["sudo", "bash", "-lc", command]
            else:
                return "Error: Neither pkexec nor sudo found"

            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            out = r.stdout
            if r.stderr:
                out += f"\n[stderr]\n{r.stderr}"
            out += f"\n[exit_code] {r.returncode}"
            return out.strip()

        # --- NEW: Mouse control (Windows + Linux via xdotool) ---
        def mouse_move(x: int, y: int) -> str:
            """Move mouse to absolute screen coordinates (x, y)."""
            sysname = platform.system()
            if sysname == "Windows":
                ctypes.windll.user32.SetCursorPos(int(x), int(y))
                return "ok"
            elif sysname in ("Linux", "FreeBSD"):
                if not shutil.which("xdotool"):
                    return "Error: xdotool not found (Wayland may not support it)"
                r = subprocess.run(["xdotool", "mousemove", str(int(x)), str(int(y))],
                                   capture_output=True, text=True)
                return "ok" if r.returncode == 0 else (r.stderr or r.stdout)
            else:
                return "Unsupported platform"

        def mouse_click(button: str = "left") -> str:
            """Click mouse button: left|middle|right."""
            sysname = platform.system()
            if sysname == "Windows":
                btn = button.lower()
                if btn == "left":
                    down, up = 0x0002, 0x0004
                elif btn == "right":
                    down, up = 0x0008, 0x0010
                elif btn == "middle":
                    down, up = 0x0020, 0x0040
                else:
                    return "Error: unknown button"
                ctypes.windll.user32.mouse_event(down, 0, 0, 0, 0)
                ctypes.windll.user32.mouse_event(up, 0, 0, 0, 0)
                return "ok"
            elif sysname in ("Linux", "FreeBSD"):
                if not shutil.which("xdotool"):
                    return "Error: xdotool not found"
                map_btn = {"left": "1", "middle": "2", "right": "3"}
                code = map_btn.get(button.lower())
                if not code:
                    return "Error: unknown button"
                r = subprocess.run(["xdotool", "click", code], capture_output=True, text=True)
                return "ok" if r.returncode == 0 else (r.stderr or r.stdout)
            else:
                return "Unsupported platform"

        def mouse_scroll(lines: int = 1) -> str:
            """Scroll vertically: positive = up, negative = down."""
            sysname = platform.system()
            if sysname == "Windows":
                # 120 units per wheel click
                delta = int(lines) * 120
                ctypes.windll.user32.mouse_event(0x0800, 0, 0, delta, 0)
                return "ok"
            elif sysname in ("Linux", "FreeBSD"):
                if not shutil.which("xdotool"):
                    return "Error: xdotool not found"
                clicks = abs(int(lines))
                button = "4" if int(lines) > 0 else "5"
                for _ in range(clicks):
                    subprocess.run(["xdotool", "click", button], capture_output=True)
                return "ok"
            else:
                return "Unsupported platform"

            # --- Keyboard control (Windows + Linux) ---
            # key_down / key_up / key_tap / type_text / hotkey
            # ---------- Windows helpers ----------
        def _win_vk_from_key(key: str) -> Optional[int]:
                k = (key or "").lower().strip()
                # Letters / digits
                if len(k) == 1 and k.isalpha():
                    return ord(k.upper())
                if len(k) == 1 and k.isdigit():
                    return ord(k)

                # Function keys
                if k.startswith("f") and k[1:].isdigit():
                    n = int(k[1:])
                    if 1 <= n <= 24:
                        return 0x6F + n  # F1=0x70
                VK = {
                    "esc": 0x1B, "escape": 0x1B,
                    "enter": 0x0D, "return": 0x0D,
                    "tab": 0x09, "space": 0x20, "backspace": 0x08, "delete": 0x2E,
                    "home": 0x24, "end": 0x23, "insert": 0x2D,
                    "pgup": 0x21, "pageup": 0x21, "pgdn": 0x22, "pagedown": 0x22,
                    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
                    "shift": 0x10, "ctrl": 0x11, "control": 0x11, "alt": 0x12,
                    "win": 0x5B, "lwin": 0x5B, "rwin": 0x5C,
                    "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91,
                    "printscreen": 0x2C, "prtsc": 0x2C,
                    "apps": 0x5D,
                }
                return VK.get(k)

            def _win_key_event(vk: int, is_down: bool) -> None:
                scan = ctypes.windll.user32.MapVirtualKeyW(vk, 0)
                flags = 0 if is_down else 0x0002  # KEYEVENTF_KEYUP
                ctypes.windll.user32.keybd_event(vk, scan, flags, 0)

            def _win_send_unicode_text(text: str, interval_ms: int = 5) -> None:
                # Send Unicode via SendInput(KEYEVENTF_UNICODE)
                ULONG_PTR = ctypes.c_ulong

                class KEYBDINPUT(ctypes.Structure):
                    _fields_ = [("wVk", ctypes.c_ushort),
                                ("wScan", ctypes.c_ushort),
                                ("dwFlags", ctypes.c_uint),
                                ("time", ctypes.c_uint),
                                ("dwExtraInfo", ULONG_PTR)]

                class INPUT_UNION(ctypes.Union):
                    _fields_ = [("ki", KEYBDINPUT)]

                class INPUT(ctypes.Structure):
                    _fields_ = [("type", ctypes.c_uint),
                                ("u", INPUT_UNION)]

                SendInput = ctypes.windll.user32.SendInput
                KEYEVENTF_UNICODE = 0x0004
                KEYEVENTF_KEYUP = 0x0002

                for ch in text:
                    code = ord(ch)
                    # key down
                    down = INPUT()
                    down.type = 1  # INPUT_KEYBOARD
                    down.u.ki = KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, 0)
                    # key up
                    up = INPUT()
                    up.type = 1
                    up.u.ki = KEYBDINPUT(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)
                    SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
                    if interval_ms > 0:
                        ctypes.windll.kernel32.Sleep(max(0, int(interval_ms)))
                    SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))
                    if interval_ms > 0:
                        ctypes.windll.kernel32.Sleep(max(0, int(interval_ms)))

            # ---------- Linux (xdotool) helpers ----------
            def _xd_name(key: str) -> str:
                k = (key or "").strip()
                lk = k.lower()
                map_special = {
                    "esc": "Escape", "escape": "Escape",
                    "enter": "Return", "return": "Return",
                    "tab": "Tab", "backspace": "BackSpace",
                    "space": "space",
                    "left": "Left", "right": "Right", "up": "Up", "down": "Down",
                    "pgup": "Page_Up", "pageup": "Page_Up",
                    "pgdn": "Page_Down", "pagedown": "Page_Down",
                    "delete": "Delete", "home": "Home", "end": "End", "insert": "Insert",
                }
                if len(k) == 1:
                    return k  # letters/digits/punct as-is
                if lk.startswith("f") and lk[1:].isdigit():
                    return "F" + lk[1:]
                return map_special.get(lk, k)

            # ---------- Cross-platform API ----------
            def key_down(key: str) -> str:
                sysname = platform.system()
                if sysname == "Windows":
                    vk = _win_vk_from_key(key)
                    if vk is None:
                        return f"Error: unknown key '{key}'"
                    _win_key_event(vk, True)
                    return "ok"
                elif sysname in ("Linux", "FreeBSD"):
                    if not shutil.which("xdotool"):
                        return "Error: xdotool not found"
                    r = subprocess.run(["xdotool", "keydown", _xd_name(key)],
                                       capture_output=True, text=True)
                    return "ok" if r.returncode == 0 else (r.stderr or r.stdout)
                else:
                    return "Unsupported platform"

            def key_up(key: str) -> str:
                sysname = platform.system()
                if sysname == "Windows":
                    vk = _win_vk_from_key(key)
                    if vk is None:
                        return f"Error: unknown key '{key}'"
                    _win_key_event(vk, False)
                    return "ok"
                elif sysname in ("Linux", "FreeBSD"):
                    if not shutil.which("xdotool"):
                        return "Error: xdotool not found"
                    r = subprocess.run(["xdotool", "keyup", _xd_name(key)],
                                       capture_output=True, text=True)
                    return "ok" if r.returncode == 0 else (r.stderr or r.stdout)
                else:
                    return "Unsupported platform"

            def key_tap(key: str, repeat: int = 1, interval_ms: int = 30) -> str:
                sysname = platform.system()
                repeat = max(1, int(repeat))
                interval_ms = max(0, int(interval_ms))
                if sysname == "Windows":
                    vk = _win_vk_from_key(key)
                    if vk is None:
                        return f"Error: unknown key '{key}'"
                    for _ in range(repeat):
                        _win_key_event(vk, True)
                        if interval_ms: ctypes.windll.kernel32.Sleep(interval_ms)
                        _win_key_event(vk, False)
                        if interval_ms: ctypes.windll.kernel32.Sleep(interval_ms)
                    return "ok"
                elif sysname in ("Linux", "FreeBSD"):
                    if not shutil.which("xdotool"):
                        return "Error: xdotool not found"
                    # Use repeat/delay flags for single key
                    r = subprocess.run(
                        ["xdotool", "key", "--repeat", str(repeat), "--delay", str(interval_ms), _xd_name(key)],
                        capture_output=True, text=True)
                    return "ok" if r.returncode == 0 else (r.stderr or r.stdout)
                else:
                    return "Unsupported platform"

            def type_text(text: str, interval_ms: int = 5) -> str:
                sysname = platform.system()
                if sysname == "Windows":
                    _win_send_unicode_text(text or "", interval_ms=int(interval_ms))
                    return "ok"
                elif sysname in ("Linux", "FreeBSD"):
                    if not shutil.which("xdotool"):
                        return "Error: xdotool not found"
                    # xdotool handles UTF-8 typing; --delay is per character
                    r = subprocess.run(["xdotool", "type", "--delay", str(max(0, int(interval_ms))), text or ""],
                                       capture_output=True, text=True)
                    return "ok" if r.returncode == 0 else (r.stderr or r.stdout)
                else:
                    return "Unsupported platform"

            def hotkey(combo: str, interval_ms: int = 30) -> str:
                """
                combo examples:
                  - Windows: 'ctrl+shift+esc', 'alt+tab', 'win+r'
                  - Linux (xdotool): same strings are accepted; we translate common names.
                """
                sysname = platform.system()
                parts = [p.strip() for p in (combo or "").split("+") if p.strip()]
                if not parts:
                    return "Error: empty combo"

                if sysname == "Windows":
                    vks = []
                    for p in parts:
                        vk = _win_vk_from_key(p)
                        if vk is None:
                            return f"Error: unknown key '{p}'"
                        vks.append(vk)
                    # Press all (in order), release in reverse
                    for vk in vks:
                        _win_key_event(vk, True)
                        if interval_ms: ctypes.windll.kernel32.Sleep(int(interval_ms))
                    for vk in reversed(vks):
                        _win_key_event(vk, False)
                        if interval_ms: ctypes.windll.kernel32.Sleep(int(interval_ms))
                    return "ok"

                elif sysname in ("Linux", "FreeBSD"):
                    if not shutil.which("xdotool"):
                        return "Error: xdotool not found"
                    xd = "+".join(_xd_name(p) for p in parts)
                    r = subprocess.run(["xdotool", "key", xd], capture_output=True, text=True)
                    return "ok" if r.returncode == 0 else (r.stderr or r.stdout)

                else:
                    return "Unsupported platform"

            # --- NEW: XInput device control (Linux/X11) ---
        def xinput_list() -> str:
            """List xinput devices (Linux/X11)."""
            if platform.system() not in ("Linux", "FreeBSD"):
                return "Unsupported platform"
            if not shutil.which("xinput"):
                return "Error: xinput not found"
            r = subprocess.run(["xinput", "--list", "--short"], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else (r.stderr or "xinput failed")

        def xinput_enable(device_id: Union[str, int]) -> str:
            """Enable an xinput device by numeric id."""
            if platform.system() not in ("Linux", "FreeBSD"):
                return "Unsupported platform"
            if not shutil.which("xinput"):
                return "Error: xinput not found"
            r = subprocess.run(["xinput", "enable", str(device_id)], capture_output=True, text=True)
            return "ok" if r.returncode == 0 else (r.stderr or r.stdout)

        def xinput_disable(device_id: Union[str, int]) -> str:
            """Disable an xinput device by numeric id."""
            if platform.system() not in ("Linux", "FreeBSD"):
                return "Unsupported platform"
            if not shutil.which("xinput"):
                return "Error: xinput not found"
            r = subprocess.run(["xinput", "disable", str(device_id)], capture_output=True, text=True)
            return "ok" if r.returncode == 0 else (r.stderr or r.stdout)

        def xinput_set_prop(device_id: Union[str, int], prop: str, *values: str) -> str:
            """Set xinput property values. Example: xinput_set_prop(12, 'Coordinate Transformation Matrix', '1', '0', '0', '0', '1', '0', '0', '0', '1')"""
            if platform.system() not in ("Linux", "FreeBSD"):
                return "Unsupported platform"
            if not shutil.which("xinput"):
                return "Error: xinput not found"
            r = subprocess.run(["xinput", "set-prop", str(device_id), prop, *values],
                               capture_output=True, text=True)
            return "ok" if r.returncode == 0 else (r.stderr or r.stdout)

        # Register tools
        self.add_function("execute_python", execute_python, "Execute Python code and return results")
        self.add_function("web_search", web_search, "Search the web for information using DuckDuckGo")
        self.add_function("read_file", read_file, "Read contents of a file")
        self.add_function("write_file", write_file, "Write content to a file")
        self.add_function("list_files", list_files, "List files and directories")
        self.add_function("run_shell_command", run_shell_command, "Run safe shell commands")
        # NEW registrations (standard shells and Ruby)
        self.add_function("powershell", powershell, "Run a PowerShell command on Windows")
        self.add_function("bash", bash, "Run a Bash command on Linux/macOS")
        self.add_function("ruby_run", ruby_run, "Run a Ruby script (optionally via bundler)")
        # NEW registrations (elevated)
        self.add_function("powershell_admin", powershell_admin, "Run elevated PowerShell with UAC; captures output")
        self.add_function("sudo", sudo, "Run a command with sudo or pkexec; user must approve")
        # NEW registrations (input control)
        self.add_function("mouse_move", mouse_move, "Move mouse to absolute screen coordinates")
        self.add_function("mouse_click", mouse_click, "Click mouse button: left|middle|right")
        self.add_function("mouse_scroll", mouse_scroll, "Scroll vertically by N lines (±)")
        self.add_function("xinput_list", xinput_list, "List xinput devices (Linux/X11)")
        self.add_function("xinput_enable", xinput_enable, "Enable an xinput device by id")
        self.add_function("xinput_disable", xinput_disable, "Disable an xinput device by id")
        self.add_function("xinput_set_prop", xinput_set_prop, "Set xinput property values")
        self.add_function("key_down", key_down, "Press (hold) a key")
        self.add_function("key_up", key_up, "Release a key")
        self.add_function("key_tap", key_tap, "Tap a key N times")
        self.add_function("type_text", type_text, "Type unicode text")
        self.add_function("hotkey", hotkey, "Press a chord like ctrl+shift+esc")

def add_function(self, name: str, func: Callable, description: str):
        """Add a function tool to Hal's capabilities."""
        self.function_map[name] = func
        tool = FunctionTool(func, description=description)
        self.tools.append(tool)

    @message_handler
    async def handle_user_message(self, message: str, ctx: MessageContext) -> str:
        """Handle user messages and generate responses."""
        try:
            # Add user message to conversation history
            user_msg = UserMessage(content=message)
            self.conversation_history.append(user_msg)

            # Prepare messages for model
            messages = self._prepare_messages()

            # Generate response using model client
            response = await self.model_client.create_chat_completion(
                messages=messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=self.tools if self.tools else None
            )

            # Process response
            assistant_message = response.content

            # Handle function calls if present
            if hasattr(response, 'function_calls') and response.function_calls:
                assistant_message = await self._handle_function_calls(response.function_calls)

            # Add assistant response to history
            self.conversation_history.append(UserMessage(content=assistant_message))

            # Manage memory limits
            self._manage_memory()

            return assistant_message

        except Exception as e:
            error_msg = f"Hal encountered an error: {str(e)}"
            self.conversation_history.append(UserMessage(content=error_msg))
            return error_msg

    async def _handle_function_calls(self, function_calls: List[Dict]) -> str:
        """Handle function calls from the model."""
        results = []

        for call in function_calls:
            func_name = call.get('name')
            func_args = call.get('arguments', {})

            if func_name in self.function_map:
                try:
                    if asyncio.iscoroutinefunction(self.function_map[func_name]):
                        result = await self.function_map[func_name](**func_args)
                    else:
                        result = self.function_map[func_name](**func_args)

                    results.append(f"[{func_name}] {result}")
                except Exception as e:
                    results.append(f"[{func_name}] Error: {str(e)}")
            else:
                results.append(f"[{func_name}] Function not found")

        return "\n".join(results)

    def _prepare_messages(self) -> List[LLMMessage]:
        """Prepare messages for the model, respecting context limits."""
        # For now, return all messages. In production, implement smart truncation
        return self.conversation_history.copy()

    def _manage_memory(self):
        """Manage conversation memory to stay within limits."""
        if len(self.conversation_history) > self.config.memory_limit:
            # Keep system message and recent messages
            system_msgs = [msg for msg in self.conversation_history if isinstance(msg, SystemMessage)]
            recent_msgs = self.conversation_history[-(self.config.memory_limit - len(system_msgs)):]
            self.conversation_history = system_msgs + recent_msgs

    def get_status(self) -> Dict[str, Any]:
        """Get current status of Hal agent."""
        return {
            "name": self.config.name,
            "active": self.is_active,
            "model": self.config.model,
            "conversation_length": len(self.conversation_history),
            "available_tools": len(self.tools),
            "current_task": self.current_task
        }

    def reset_conversation(self):
        """Reset conversation history while keeping system message."""
        system_msgs = [msg for msg in self.conversation_history if isinstance(msg, SystemMessage)]
        self.conversation_history = system_msgs

    def update_config(self, **kwargs):
        """Update Hal's configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


# Convenience function to create Hal agent
def create_dhal(
        name: str = "Dhal",
        system_message: str = None,
        model: str = "gpt-4",
        model_client: ChatCompletionClient = None,
        **kwargs
) -> Dhal:
    """Create a Dhal agent with specified configuration."""

    if system_message is None:
        system_message = f"You are {name}, an advanced AI assistant integrated into DarkHal 2.0. You help users with coding, analysis, security testing, and general AI tasks."

    config = DhalConfig(
        name=name,
        system_message=system_message,
        model=model,
        **kwargs
    )

    # Initialize model client if not provided
    if model_client is None:
        from ..llm_runtime import load_model
        try:
            # Use the existing LLM runtime from main application
            model_client = SpyData(model_name=model)
        except Exception as e:
            print(f"[Dhal] Warning: Could not initialize model client: {e}")
            model_client = None

    return Dhal(config, model_client)
