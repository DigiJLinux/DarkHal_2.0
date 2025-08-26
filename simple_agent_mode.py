#!/usr/bin/env python3
"""
Simple Agent Mode - Direct system command execution through AI
"""

import os
import sys
import subprocess
import shutil
import platform
import ctypes
import json
import tempfile
import asyncio
import psutil
import time
from typing import Optional, Dict, Any, List

class SimpleAgentExecutor:
    """Simple agent that can execute system commands based on AI responses"""
    
    def __init__(self, log_callback=None):
        self.tools = self._register_tools()
        self.log_callback = log_callback or print
        self.active_processes = {}  # Store PIDs of opened processes
        
    def _register_tools(self) -> Dict[str, callable]:
        """Register available system tools"""
        tools = {}
        
        # PowerShell execution
        def powershell(command: str) -> str:
            """Execute PowerShell command"""
            try:
                exe = shutil.which("pwsh") or shutil.which("powershell")
                if not exe:
                    return "Error: PowerShell not found"
                result = subprocess.run(
                    [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr]: {result.stderr}"
                return output.strip() or "Command executed successfully"
            except Exception as e:
                return f"Error: {e}"
        
        # Bash execution
        def bash(command: str) -> str:
            """Execute Bash command"""
            try:
                exe = shutil.which("bash")
                if not exe:
                    return "Error: bash not found"
                result = subprocess.run(
                    [exe, "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr]: {result.stderr}"
                return output.strip() or "Command executed successfully"
            except Exception as e:
                return f"Error: {e}"
        
        # Generic shell command
        def shell(command: str) -> str:
            """Execute shell command"""
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr]: {result.stderr}"
                return output.strip() or "Command executed successfully"
            except Exception as e:
                return f"Error: {e}"
        
        # Open applications with PID tracking
        def open_app(app_name: str) -> str:
            """Open an application and track its PID"""
            try:
                self.log_callback(f"Opening application: {app_name}")
                
                if platform.system() == "Windows":
                    # Try common Windows apps
                    apps = {
                        "notepad": "notepad.exe",
                        "word": "winword.exe",
                        "excel": "excel.exe",
                        "powerpoint": "powerpnt.exe",
                        "calculator": "calc.exe",
                        "paint": "mspaint.exe",
                        "cmd": "cmd.exe",
                        "powershell": "powershell.exe",
                        "explorer": "explorer.exe",
                    }
                    
                    app_path = apps.get(app_name.lower(), app_name)
                    process = subprocess.Popen([app_path], shell=True)
                    
                    # Wait a bit for process to start
                    time.sleep(1)
                    
                    # Find the actual PID of the opened window
                    try:
                        for proc in psutil.process_iter(['pid', 'name']):
                            if proc.info['name'] and app_path.lower() in proc.info['name'].lower():
                                pid = proc.info['pid']
                                self.active_processes[app_name.lower()] = pid
                                self.log_callback(f"Tracked process {app_name} with PID: {pid}")
                                return f"Opened {app_name} (PID: {pid})"
                    except:
                        pass
                    
                    return f"Opened {app_name}"
                    
                elif platform.system() == "Darwin":  # macOS
                    process = subprocess.Popen(["open", "-a", app_name])
                    return f"Opened {app_name}"
                    
                else:  # Linux
                    process = subprocess.Popen([app_name], shell=True)
                    return f"Opened {app_name}"
                    
            except Exception as e:
                self.log_callback(f"Error opening {app_name}: {e}")
                return f"Error opening {app_name}: {e}"
        
        # File operations
        def read_file(filepath: str) -> str:
            """Read a file"""
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"File content:\n{content}"
            except Exception as e:
                return f"Error reading file: {e}"
        
        def write_file(filepath: str, content: str) -> str:
            """Write to a file"""
            try:
                os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Wrote {len(content)} bytes to {filepath}"
            except Exception as e:
                return f"Error writing file: {e}"
        
        def list_files(directory: str = ".") -> str:
            """List files in directory"""
            try:
                files = []
                for name in sorted(os.listdir(directory)):
                    path = os.path.join(directory, name)
                    if os.path.isdir(path):
                        files.append(f"[DIR] {name}/")
                    else:
                        files.append(f"[FILE] {name}")
                return "\n".join(files)
            except Exception as e:
                return f"Error listing files: {e}"
        
        # Python code execution
        def execute_python(code: str) -> str:
            """Execute Python code"""
            try:
                # Create temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                try:
                    result = subprocess.run(
                        [sys.executable, temp_file],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    output = result.stdout
                    if result.stderr:
                        output += f"\n[stderr]: {result.stderr}"
                    return output or "Code executed successfully"
                finally:
                    os.unlink(temp_file)
                    
            except Exception as e:
                return f"Error executing Python: {e}"
        
        # Send commands to specific processes
        def send_to_process(app_name: str, command: str) -> str:
            """Send command to a specific tracked process"""
            try:
                self.log_callback(f"Sending command to {app_name}: {command}")
                
                if app_name.lower() not in self.active_processes:
                    return f"Process {app_name} not tracked. Available: {list(self.active_processes.keys())}"
                
                pid = self.active_processes[app_name.lower()]
                
                # Check if process is still running
                try:
                    proc = psutil.Process(pid)
                    if not proc.is_running():
                        del self.active_processes[app_name.lower()]
                        return f"Process {app_name} (PID: {pid}) is no longer running"
                except psutil.NoSuchProcess:
                    del self.active_processes[app_name.lower()]
                    return f"Process {app_name} (PID: {pid}) not found"
                
                # For Windows, use multiple methods for better compatibility
                if platform.system() == "Windows":
                    import win32gui
                    import win32con
                    import win32api
                    import win32process
                    
                    # Find window by PID
                    def enum_windows_callback(hwnd, results):
                        try:
                            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                            if found_pid == pid and win32gui.IsWindowVisible(hwnd):
                                results.append(hwnd)
                        except:
                            pass
                        return True
                    
                    windows = []
                    win32gui.EnumWindows(enum_windows_callback, windows)
                    
                    if windows:
                        hwnd = windows[0]
                        window_title = win32gui.GetWindowText(hwnd)
                        self.log_callback(f"Found window: {window_title} (HWND: {hwnd})")
                        
                        # Bring window to foreground
                        try:
                            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                            win32gui.SetForegroundWindow(hwnd)
                            time.sleep(0.5)
                        except Exception as e:
                            self.log_callback(f"Warning: Could not bring window to foreground: {e}")
                        
                        # Method 1: Try using SendKeys via win32api (more reliable for terminal apps)
                        try:
                            # Ensure the window has focus
                            win32gui.SetActiveWindow(hwnd)
                            time.sleep(0.2)
                            
                            # For PowerShell/CMD, try different approaches
                            if "powershell" in app_name.lower() or "cmd" in app_name.lower():
                                # Method A: Use keybd_event for virtual key codes
                                import win32api
                                for char in command:
                                    if char.isalnum() or char in " .-_/\\:":
                                        vk_code = win32api.VkKeyScan(char) & 0xFF
                                        win32api.keybd_event(vk_code, 0, 0, 0)  # Key down
                                        time.sleep(0.01)
                                        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # Key up
                                        time.sleep(0.01)
                                
                                # Send Enter key
                                win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
                                time.sleep(0.01)
                                win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
                                
                                self.log_callback(f"Sent command using keyboard events to {app_name}")
                                return f"Command sent to {app_name} using keyboard simulation"
                            
                            else:
                                # Method B: Use WM_CHAR for other applications
                                for char in command:
                                    win32gui.SendMessage(hwnd, win32con.WM_CHAR, ord(char), 0)
                                    time.sleep(0.01)
                                
                                # Send Enter
                                win32gui.SendMessage(hwnd, win32con.WM_CHAR, 13, 0)
                                
                                self.log_callback(f"Sent command using WM_CHAR to {app_name}")
                                return f"Command sent to {app_name} using message posting"
                        
                        except Exception as e:
                            self.log_callback(f"Error in command sending methods: {e}")
                            
                            # Method 2: Fallback - try clipboard method for complex commands
                            try:
                                import win32clipboard
                                
                                # Copy command to clipboard
                                win32clipboard.OpenClipboard()
                                win32clipboard.EmptyClipboard()
                                win32clipboard.SetClipboardText(command)
                                win32clipboard.CloseClipboard()
                                
                                # Send Ctrl+V to paste
                                win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
                                win32api.keybd_event(ord('V'), 0, 0, 0)
                                time.sleep(0.05)
                                win32api.keybd_event(ord('V'), 0, win32con.KEYEVENTF_KEYUP, 0)
                                win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
                                
                                # Send Enter
                                time.sleep(0.1)
                                win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
                                time.sleep(0.01)
                                win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
                                
                                self.log_callback(f"Sent command using clipboard method to {app_name}")
                                return f"Command sent to {app_name} using clipboard paste"
                                
                            except Exception as e2:
                                self.log_callback(f"Clipboard method also failed: {e2}")
                                return f"Failed to send command to {app_name}: {e}"
                    
                    else:
                        # Try to get window processes for fallback selection
                        window_list = self.get_window_processes()
                        return f"No window found for {app_name} (PID: {pid}). Available windows:\n{window_list}"
                
                return f"Command sending not implemented for {platform.system()}"
                
            except Exception as e:
                self.log_callback(f"Error sending command to {app_name}: {e}")
                return f"Error sending command to {app_name}: {e}"
        
        def list_processes() -> str:
            """List available processes to send commands to"""
            try:
                self.log_callback("Listing available processes")
                
                if not self.active_processes:
                    return "No tracked processes. Open an application first."
                
                result = "Available tracked processes:\n"
                for app_name, pid in self.active_processes.items():
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            result += f"- {app_name} (PID: {pid}) - Running\n"
                        else:
                            result += f"- {app_name} (PID: {pid}) - Not running\n"
                    except psutil.NoSuchProcess:
                        result += f"- {app_name} (PID: {pid}) - Process not found\n"
                
                return result
                
            except Exception as e:
                self.log_callback(f"Error listing processes: {e}")
                return f"Error listing processes: {e}"
        
        def get_window_processes() -> str:
            """Get list of visible window processes for fallback selection"""
            try:
                self.log_callback("Getting visible window processes")
                
                if platform.system() == "Windows":
                    import win32gui
                    import win32process
                    
                    def enum_windows_callback(hwnd, results):
                        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                            _, pid = win32process.GetWindowThreadProcessId(hwnd)
                            window_title = win32gui.GetWindowText(hwnd)
                            try:
                                proc = psutil.Process(pid)
                                results.append(f"PID: {pid} | {proc.name()} | {window_title}")
                            except:
                                results.append(f"PID: {pid} | Unknown | {window_title}")
                        return True
                    
                    windows = []
                    win32gui.EnumWindows(enum_windows_callback, windows)
                    
                    if windows:
                        return "Visible window processes:\n" + "\n".join(windows[:20])  # Limit to 20
                    else:
                        return "No visible windows found"
                
                return f"Window enumeration not implemented for {platform.system()}"
                
            except Exception as e:
                self.log_callback(f"Error getting window processes: {e}")
                return f"Error getting window processes: {e}"
        
        def send_to_pid(pid: int, command: str) -> str:
            """Send command directly to a process by PID (fallback method)"""
            try:
                self.log_callback(f"Sending command to PID {pid}: {command}")
                
                # Check if process exists
                try:
                    proc = psutil.Process(pid)
                    if not proc.is_running():
                        return f"Process PID {pid} is not running"
                    app_name = proc.name()
                except psutil.NoSuchProcess:
                    return f"Process PID {pid} not found"
                
                # Use the same logic as send_to_process but with PID directly
                if platform.system() == "Windows":
                    import win32gui
                    import win32con
                    import win32api
                    import win32process
                    
                    # Find window by PID
                    def enum_windows_callback(hwnd, results):
                        try:
                            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                            if found_pid == pid and win32gui.IsWindowVisible(hwnd):
                                results.append(hwnd)
                        except:
                            pass
                        return True
                    
                    windows = []
                    win32gui.EnumWindows(enum_windows_callback, windows)
                    
                    if windows:
                        hwnd = windows[0]
                        window_title = win32gui.GetWindowText(hwnd)
                        self.log_callback(f"Found window: {window_title} (HWND: {hwnd})")
                        
                        # Bring window to foreground
                        try:
                            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                            win32gui.SetForegroundWindow(hwnd)
                            time.sleep(0.5)
                        except Exception as e:
                            self.log_callback(f"Warning: Could not bring window to foreground: {e}")
                        
                        # Try keyboard simulation for terminal apps
                        try:
                            win32gui.SetActiveWindow(hwnd)
                            time.sleep(0.2)
                            
                            # Use clipboard method for reliability
                            try:
                                import win32clipboard
                                
                                # Copy command to clipboard
                                win32clipboard.OpenClipboard()
                                win32clipboard.EmptyClipboard()
                                win32clipboard.SetClipboardText(command)
                                win32clipboard.CloseClipboard()
                                
                                # Send Ctrl+V to paste
                                win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
                                win32api.keybd_event(ord('V'), 0, 0, 0)
                                time.sleep(0.05)
                                win32api.keybd_event(ord('V'), 0, win32con.KEYEVENTF_KEYUP, 0)
                                win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
                                
                                # Send Enter
                                time.sleep(0.1)
                                win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
                                time.sleep(0.01)
                                win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
                                
                                self.log_callback(f"Sent command to PID {pid} ({app_name}) using clipboard")
                                
                                # Track this process for future use
                                self.active_processes[app_name.lower()] = pid
                                
                                return f"Command sent to PID {pid} ({app_name})"
                                
                            except Exception as e:
                                self.log_callback(f"Clipboard method failed: {e}")
                                return f"Failed to send command to PID {pid}: {e}"
                        
                        except Exception as e:
                            self.log_callback(f"Error in window interaction: {e}")
                            return f"Error sending command to PID {pid}: {e}"
                    
                    else:
                        return f"No visible window found for PID {pid}"
                
                return f"Command sending not implemented for {platform.system()}"
                
            except Exception as e:
                self.log_callback(f"Error sending command to PID {pid}: {e}")
                return f"Error sending command to PID {pid}: {e}"

        # Register all tools
        tools["powershell"] = powershell
        tools["bash"] = bash
        tools["shell"] = shell
        tools["open_app"] = open_app
        tools["read_file"] = read_file
        tools["write_file"] = write_file
        tools["list_files"] = list_files
        tools["execute_python"] = execute_python
        tools["send_to_process"] = send_to_process
        tools["send_to_pid"] = send_to_pid
        tools["list_processes"] = list_processes
        tools["get_window_processes"] = get_window_processes
        
        return tools
    
    def parse_and_execute(self, ai_response: str) -> str:
        """Parse AI response for commands and execute them"""
        output = []
        
        # Look for command patterns in the response
        lines = ai_response.split('\n')
        
        for line in lines:
            # Check for explicit command markers
            if line.strip().startswith("EXECUTE:"):
                command = line.replace("EXECUTE:", "").strip()
                result = self._execute_command(command)
                output.append(f"[Executed] {command}\n{result}")
                
            elif line.strip().startswith("OPEN:"):
                app = line.replace("OPEN:", "").strip()
                result = self.tools["open_app"](app)
                output.append(f"[Opened] {app}\n{result}")
                
            elif line.strip().startswith("PYTHON:"):
                code = line.replace("PYTHON:", "").strip()
                result = self.tools["execute_python"](code)
                output.append(f"[Python] {result}")
        
        if output:
            return "\n".join(output)
        else:
            # If no explicit commands, return the AI response as-is
            return ai_response
    
    def _execute_command(self, command: str) -> str:
        """Execute a command using appropriate shell"""
        if platform.system() == "Windows":
            # Detect if it's a PowerShell command
            ps_commands = ["Get-", "Set-", "New-", "Remove-", "Start-", "Stop-", "Test-"]
            if any(command.startswith(cmd) for cmd in ps_commands):
                return self.tools["powershell"](command)
            else:
                return self.tools["shell"](command)
        else:
            return self.tools["bash"](command)
    
    def process_request(self, user_request: str, model_response: str) -> str:
        """Process user request with model response"""
        # First, check if the model response contains tool calls
        if "EXECUTE:" in model_response or "OPEN:" in model_response or "PYTHON:" in model_response:
            return self.parse_and_execute(model_response)
        
        # Otherwise, try to infer intent from user request
        request_lower = user_request.lower()
        
        # Direct application opening requests
        if "open" in request_lower:
            if "word" in request_lower:
                return self.tools["open_app"]("word")
            elif "notepad" in request_lower:
                return self.tools["open_app"]("notepad")
            elif "powershell" in request_lower:
                return self.tools["open_app"]("powershell")
            elif "terminal" in request_lower or "cmd" in request_lower:
                return self.tools["open_app"]("cmd")
            elif "calculator" in request_lower:
                return self.tools["open_app"]("calculator")
        
        # File operations
        elif "list files" in request_lower or "show files" in request_lower:
            return self.tools["list_files"]()
        
        # If we can't determine intent, return the model response
        return model_response


def create_simple_agent(log_callback=None):
    """Create a simple agent executor"""
    return SimpleAgentExecutor(log_callback=log_callback)