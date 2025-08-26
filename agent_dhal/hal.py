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
    # Create working tool implementation compatible with add_function(func, description)
    class FunctionTool:
        def __init__(self, func: Callable, description: str = ""):
            self.func = func
            self.description = description
    Tool = FunctionTool  # alias for typing


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
        """Create chat completion using the loaded model (supports streaming)."""
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

            # Callbacks and options
            stream = bool(kwargs.get('stream', False))
            on_token = kwargs.get('on_token')
            on_complete = kwargs.get('on_complete')
            on_error = kwargs.get('on_error')
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 2000)

            # Try to build a config if available (optional)
            cfg = None
            try:
                from llm_runtime import GenerateConfig  # type: ignore
                cfg = GenerateConfig(
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except Exception:
                cfg = None

            full_text = ""

            # Streaming if supported
            if stream and hasattr(self.llm_model, 'stream'):
                try:
                    iterator = self.llm_model.stream(conversation, cfg) if cfg is not None else self.llm_model.stream(conversation)
                    for chunk in iterator:
                        if chunk:
                            full_text += str(chunk)
                            if on_token:
                                try:
                                    on_token("request-1", str(chunk))
                                except Exception:
                                    pass
                    if on_complete:
                        try:
                            on_complete("request-1", full_text, {"finish_reason": "stop"})
                        except Exception:
                            pass
                except Exception as se:
                    if on_error:
                        try:
                            on_error("request-1", se)
                        except Exception:
                            pass
                    raise

            else:
                # Non-streaming path
                if hasattr(self.llm_model, 'generate'):
                    full_text = self.llm_model.generate(conversation, cfg) if cfg is not None else self.llm_model.generate(conversation)
                elif hasattr(self.llm_model, '__call__'):
                    full_text = self.llm_model(conversation)
                else:
                    # Fallback for different model interfaces
                    full_text = str(self.llm_model)
                if on_complete:
                    try:
                        on_complete("request-1", full_text, {"finish_reason": "stop"})
                    except Exception:
                        pass

            # Create response object
            class CompletionResponse:
                def __init__(self, content):
                    self.content = content
                    self.function_calls = None

            return CompletionResponse(full_text)

        except Exception as e:
            print(f"[Hal] Error generating response: {e}")

            # Return error response
            class CompletionResponse:
                def __init__(self, content):
                    self.content = content
                    self.function_calls = None

            if kwargs.get('on_error'):
                try:
                    kwargs['on_error']("request-1", e)
                except Exception:
                    pass
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
    tools: List[FunctionTool] = None  # type: ignore
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

        # Unrestricted file read
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

        # --- Mouse control (Windows + Linux via xdotool) ---
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

        # =========================
        #  WINDOWS XINPUT (DirectX)
        # =========================
        # Tools: xinput_list, xinput_get_state, xinput_set_vibration
        def _load_xinput_dll():
            if platform.system() != "Windows":
                return None
            for dll in ("XInput1_4.dll", "XInput1_3.dll", "XInput9_1_0.dll"):
                try:
                    return ctypes.WinDLL(dll)
                except OSError:
                    continue
            return None

        # Structures from XInput
        class XINPUT_GAMEPAD(ctypes.Structure):
            _fields_ = [
                ("wButtons", ctypes.c_ushort),
                ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short),
                ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short),
            ]

        class XINPUT_STATE(ctypes.Structure):
            _fields_ = [
                ("dwPacketNumber", ctypes.c_uint),
                ("Gamepad", XINPUT_GAMEPAD),
            ]

        class XINPUT_VIBRATION(ctypes.Structure):
            _fields_ = [
                ("wLeftMotorSpeed", ctypes.c_ushort),
                ("wRightMotorSpeed", ctypes.c_ushort),
            ]

        # Button bitmasks
        XINPUT_BUTTONS = {
            "DPAD_UP": 0x0001,
            "DPAD_DOWN": 0x0002,
            "DPAD_LEFT": 0x0004,
            "DPAD_RIGHT": 0x0008,
            "START": 0x0010,
            "BACK": 0x0020,
            "LEFT_THUMB": 0x0040,
            "RIGHT_THUMB": 0x0080,
            "LEFT_SHOULDER": 0x0100,
            "RIGHT_SHOULDER": 0x0200,
            "A": 0x1000,
            "B": 0x2000,
            "X": 0x4000,
            "Y": 0x8000,
        }

        def xinput_list() -> str:
            """List connected XInput (Xbox-compatible) controllers on Windows."""
            if platform.system() != "Windows":
                return "Unsupported platform (XInput is Windows/DirectX)"
            dll = _load_xinput_dll()
            if not dll:
                return "Error: XInput DLL not found"

            XInputGetState = dll.XInputGetState
            XInputGetState.argtypes = [ctypes.c_uint, ctypes.POINTER(XINPUT_STATE)]
            XInputGetState.restype = ctypes.c_uint  # 0 = success

            found = []
            for i in range(4):
                state = XINPUT_STATE()
                rc = XInputGetState(i, ctypes.byref(state))
                found.append({"id": i, "connected": (rc == 0)})

            return json.dumps(found)

        def xinput_get_state(controller_id: int = 0) -> str:
            """Get state for a specific controller id (0-3). Returns JSON with buttons/axes/triggers."""
            if platform.system() != "Windows":
                return "Unsupported platform"
            dll = _load_xinput_dll()
            if not dll:
                return "Error: XInput DLL not found"

            XInputGetState = dll.XInputGetState
            XInputGetState.argtypes = [ctypes.c_uint, ctypes.POINTER(XINPUT_STATE)]
            XInputGetState.restype = ctypes.c_uint

            state = XINPUT_STATE()
            rc = XInputGetState(int(controller_id), ctypes.byref(state))
            if rc != 0:
                return f"Error: controller {controller_id} not connected"

            gp = state.Gamepad
            buttons = gp.wButtons
            pressed = [name for name, mask in XINPUT_BUTTONS.items() if buttons & mask]

            data = {
                "id": int(controller_id),
                "packet": state.dwPacketNumber,
                "buttons": pressed,
                "left_trigger": int(gp.bLeftTrigger),
                "right_trigger": int(gp.bRightTrigger),
                "thumb_lx": int(gp.sThumbLX),
                "thumb_ly": int(gp.sThumbLY),
                "thumb_rx": int(gp.sThumbRX),
                "thumb_ry": int(gp.sThumbRY),
            }
            return json.dumps(data)

        def xinput_set_vibration(controller_id: int = 0,
                                 left_motor: int = 0,
                                 right_motor: int = 0) -> str:
            """Set rumble (0-65535 per motor)."""
            if platform.system() != "Windows":
                return "Unsupported platform"
            dll = _load_xinput_dll()
            if not dll:
                return "Error: XInput DLL not found"

            XInputSetState = dll.XInputSetState
            XInputSetState.argtypes = [ctypes.c_uint, ctypes.POINTER(XINPUT_VIBRATION)]
            XInputSetState.restype = ctypes.c_uint

            vib = XINPUT_VIBRATION(
                wLeftMotorSpeed=max(0, min(65535, int(left_motor))),
                wRightMotorSpeed=max(0, min(65535, int(right_motor)))
            )
            rc = XInputSetState(int(controller_id), ctypes.byref(vib))
            return "ok" if rc == 0 else f"Error: rc={rc}"

        # Register tools
        self.add_function("execute_python", execute_python, "Execute Python code and return results")
        self.add_function("web_search", web_search, "Search the web for information using DuckDuckGo")
        self.add_function("read_file", read_file, "Read contents of a file")
        self.add_function("write_file", write_file, "Write content to a file")
        self.add_function("list_files", list_files, "List files and directories")
        self.add_function("run_shell_command", run_shell_command, "Run shell commands")
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
        # Windows XInput (DirectX) gamepad tools
        self.add_function("xinput_list", xinput_list, "List XInput controllers (Windows/DirectX)")
        self.add_function("xinput_get_state", xinput_get_state, "Get XInput state for controller id (0-3)")
        self.add_function("xinput_set_vibration", xinput_set_vibration, "Set XInput vibration (rumble)")

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

    def send_dhal_message(
        self,
        conversation_id: str,
        message: str,
        stream: bool = False,
        on_token: Optional[Callable[[str, str], None]] = None,
        on_complete: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None
    ) -> None:
        """
        Public entry to send a message to the agent with optional streaming callbacks.
        Runs in a background thread to avoid blocking the UI thread.
        """
        # Add user message to history
        self.conversation_history.append(UserMessage(content=message))
        messages = self._prepare_messages()

        def _worker():
            import asyncio

            async def _run():
                try:
                    # Delegate to model client with callbacks
                    resp = await self.model_client.create_chat_completion(
                        messages=messages,
                        model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        tools=self.tools if self.tools else None,
                        stream=stream,
                        on_token=(lambda req_id, delta: on_token and on_token(conversation_id, delta)),
                        on_complete=(lambda req_id, full, meta: (
                            self.conversation_history.append(UserMessage(content=full)),
                            self._manage_memory(),
                            on_complete and on_complete(conversation_id, full, meta)
                        )),
                        on_error=(lambda req_id, err: on_error and on_error(conversation_id, err))
                    )
                    # If not streaming, we need to append and call on_complete here
                    if not stream and resp and hasattr(resp, "content"):
                        assistant_message = resp.content
                        self.conversation_history.append(UserMessage(content=assistant_message))
                        self._manage_memory()
                        if on_complete:
                            on_complete(conversation_id, assistant_message, {"finish_reason": "stop"})
                except Exception as e:
                    if on_error:
                        on_error(conversation_id, e)

            asyncio.run(_run())

        import threading as _threading
        t = _threading.Thread(target=_worker, daemon=True)
        t.start()

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
        try:
            model_client = HalModelClient(model_name=model)
        except Exception as e:
            print(f"[Dhal] Warning: Could not initialize model client: {e}")
            model_client = None

    return Dhal(config, model_client)
