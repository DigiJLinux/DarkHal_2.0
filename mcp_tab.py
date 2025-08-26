#!/usr/bin/env python3
"""
MCP Server Management Tab for DarkHal 2.0

Provides a comprehensive interface for managing and monitoring the MCP server,
including status monitoring, control, and configuration.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import threading
import json
import os
import sys
import time
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class MCPServerManager:
    """Manages the MCP server process and communication."""
    
    def __init__(self):
        self.process = None
        self.status = "stopped"
        self.start_time = None
        self.message_queue = queue.Queue()
        self.callbacks = {
            'on_start': [],
            'on_stop': [],
            'on_error': [],
            'on_message': []
        }
        
    def register_callback(self, event: str, callback):
        """Register a callback for server events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger all callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def start_server(self, config: Dict[str, Any] = None) -> bool:
        """Start the MCP server."""
        if self.process and self.process.poll() is None:
            return False  # Already running
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if config:
                env['MCP_CONFIG'] = json.dumps(config)
            
            # Start server process
            server_path = Path(__file__).parent / "mcp_server.py"
            self.process = subprocess.Popen(
                [sys.executable, str(server_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env
            )
            
            self.status = "running"
            self.start_time = time.time()
            
            # Start output monitoring threads
            threading.Thread(target=self._monitor_stdout, daemon=True).start()
            threading.Thread(target=self._monitor_stderr, daemon=True).start()
            
            self._trigger_callbacks('on_start')
            return True
            
        except Exception as e:
            self.status = "error"
            self._trigger_callbacks('on_error', str(e))
            return False
    
    def stop_server(self) -> bool:
        """Stop the MCP server."""
        if not self.process:
            return False
        
        try:
            # Send shutdown signal
            if self.process.poll() is None:
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            
            self.process = None
            self.status = "stopped"
            self.start_time = None
            self._trigger_callbacks('on_stop')
            return True
            
        except Exception as e:
            self._trigger_callbacks('on_error', str(e))
            return False
    
    def restart_server(self, config: Dict[str, Any] = None) -> bool:
        """Restart the MCP server."""
        self.stop_server()
        time.sleep(1)  # Brief pause
        return self.start_server(config)
    
    def send_command(self, command: Dict[str, Any]) -> bool:
        """Send a command to the server."""
        if not self.process or self.process.poll() is not None:
            return False
        
        try:
            command_json = json.dumps(command) + '\n'
            self.process.stdin.write(command_json)
            self.process.stdin.flush()
            return True
        except Exception:
            return False
    
    def _monitor_stdout(self):
        """Monitor stdout from the server."""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    self._trigger_callbacks('on_message', 'stdout', line.strip())
            except Exception:
                break
    
    def _monitor_stderr(self):
        """Monitor stderr from the server."""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stderr.readline()
                if line:
                    self._trigger_callbacks('on_message', 'stderr', line.strip())
            except Exception:
                break
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status information."""
        return {
            'status': self.status,
            'running': self.process and self.process.poll() is None,
            'pid': self.process.pid if self.process else None,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }


class MCPTab:
    """MCP Server management tab for DarkHal 2.0."""
    
    def __init__(self, parent: ttk.Frame, settings_manager):
        self.parent = parent
        self.settings = settings_manager
        self.server_manager = MCPServerManager()
        self.tools_info = []
        self.server_config = {}
        
        # Register callbacks
        self.server_manager.register_callback('on_start', self._on_server_start)
        self.server_manager.register_callback('on_stop', self._on_server_stop)
        self.server_manager.register_callback('on_error', self._on_server_error)
        self.server_manager.register_callback('on_message', self._on_server_message)
        
        self._build_ui()
        self._load_config()
        self._update_status()
    
    def _build_ui(self):
        """Build the MCP tab UI."""
        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Control and Status
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Server Status Frame
        status_frame = ttk.LabelFrame(left_panel, text="Server Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status indicators
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        # Status light
        self.status_canvas = tk.Canvas(status_grid, width=20, height=20)
        self.status_canvas.grid(row=0, column=0, padx=(0, 10))
        self.status_indicator = self.status_canvas.create_oval(2, 2, 18, 18, fill="red")
        
        ttk.Label(status_grid, text="Status:").grid(row=0, column=1, sticky=tk.W)
        self.status_label = ttk.Label(status_grid, text="Stopped", font=("TkDefaultFont", 10, "bold"))
        self.status_label.grid(row=0, column=2, sticky=tk.W, padx=(5, 20))
        
        ttk.Label(status_grid, text="PID:").grid(row=0, column=3, sticky=tk.W)
        self.pid_label = ttk.Label(status_grid, text="N/A")
        self.pid_label.grid(row=0, column=4, sticky=tk.W, padx=(5, 20))
        
        ttk.Label(status_grid, text="Uptime:").grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        self.uptime_label = ttk.Label(status_grid, text="00:00:00")
        self.uptime_label.grid(row=1, column=2, sticky=tk.W, padx=(5, 20), pady=(5, 0))
        
        ttk.Label(status_grid, text="Port:").grid(row=1, column=3, sticky=tk.W, pady=(5, 0))
        self.port_label = ttk.Label(status_grid, text="N/A")
        self.port_label.grid(row=1, column=4, sticky=tk.W, padx=(5, 20), pady=(5, 0))
        
        # Control Buttons Frame
        control_frame = ttk.LabelFrame(left_panel, text="Server Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(button_frame, text="Start Server", command=self._start_server)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Server", command=self._stop_server, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        self.restart_btn = ttk.Button(button_frame, text="Restart Server", command=self._restart_server, state=tk.DISABLED)
        self.restart_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="Configure", command=self._open_config).pack(side=tk.LEFT, padx=(20, 2))
        ttk.Button(button_frame, text="Test Connection", command=self._test_connection).pack(side=tk.LEFT, padx=2)
        
        # Configuration Frame
        config_frame = ttk.LabelFrame(left_panel, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        config_grid = ttk.Frame(config_frame)
        config_grid.pack(fill=tk.X)
        
        ttk.Label(config_grid, text="Model Cache:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cache_var = tk.StringVar(value="3 models")
        ttk.Label(config_grid, textvariable=self.cache_var).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(config_grid, text="Default Context:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ctx_var = tk.StringVar(value="4096 tokens")
        ttk.Label(config_grid, textvariable=self.ctx_var).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(config_grid, text="GPU Layers:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.gpu_var = tk.StringVar(value="Auto")
        ttk.Label(config_grid, textvariable=self.gpu_var).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(config_grid, text="Log Level:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.log_var = tk.StringVar(value="INFO")
        ttk.Label(config_grid, textvariable=self.log_var).grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        
        # Tools Frame
        tools_frame = ttk.LabelFrame(left_panel, text="Available Tools", padding=10)
        tools_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tools list
        columns = ("tool", "description", "status")
        self.tools_tree = ttk.Treeview(tools_frame, columns=columns, show="headings", height=8)
        
        self.tools_tree.heading("tool", text="Tool")
        self.tools_tree.heading("description", text="Description")
        self.tools_tree.heading("status", text="Status")
        
        self.tools_tree.column("tool", width=150)
        self.tools_tree.column("description", width=300)
        self.tools_tree.column("status", width=80)
        
        tools_scroll = ttk.Scrollbar(tools_frame, orient="vertical", command=self.tools_tree.yview)
        self.tools_tree.configure(yscrollcommand=tools_scroll.set)
        
        self.tools_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tools_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate default tools
        self._populate_tools()
        
        # Right panel - Logs
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Server Logs Frame
        log_frame = ttk.LabelFrame(right_panel, text="Server Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, wrap=tk.WORD, 
                                                  bg="#1a1a1a", fg="#00ff88",
                                                  font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure log tags
        self.log_text.tag_configure("error", foreground="#ff4444")
        self.log_text.tag_configure("warning", foreground="#ffaa00")
        self.log_text.tag_configure("info", foreground="#00aaff")
        self.log_text.tag_configure("success", foreground="#00ff88")
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_controls, text="Clear Logs", command=self._clear_logs).pack(side=tk.LEFT)
        ttk.Button(log_controls, text="Save Logs", command=self._save_logs).pack(side=tk.LEFT, padx=5)
        
        self.autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="Auto-scroll", variable=self.autoscroll_var).pack(side=tk.LEFT, padx=10)
        
        # Performance Metrics Frame
        metrics_frame = ttk.LabelFrame(right_panel, text="Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(10, 0))
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        ttk.Label(metrics_grid, text="Requests/sec:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.req_rate_label = ttk.Label(metrics_grid, text="0")
        self.req_rate_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 30))
        
        ttk.Label(metrics_grid, text="Avg Response:").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.response_time_label = ttk.Label(metrics_grid, text="0ms")
        self.response_time_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(metrics_grid, text="Total Requests:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.total_requests_label = ttk.Label(metrics_grid, text="0")
        self.total_requests_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 30))
        
        ttk.Label(metrics_grid, text="Errors:").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.errors_label = ttk.Label(metrics_grid, text="0")
        self.errors_label.grid(row=1, column=3, sticky=tk.W, padx=(10, 0))
        
        # Start status update timer
        self.parent.after(1000, self._update_status)
    
    def _populate_tools(self):
        """Populate the tools list with available MCP tools."""
        tools = [
            ("list_models", "List all available models in the library", "Ready"),
            ("load_model", "Load a model with specified parameters", "Ready"),
            ("unload_model", "Unload the current model from memory", "Ready"),
            ("generate_text", "Generate text using the loaded model", "Ready"),
            ("chat", "Interactive chat with context management", "Ready"),
            ("get_system_info", "Get system and GPU information", "Ready"),
            ("get_model_info", "Get information about loaded model", "Ready"),
            ("list_loras", "List available LoRA adapters", "Ready"),
            ("apply_lora", "Apply a LoRA adapter to the model", "Ready"),
            ("benchmark", "Run performance benchmarks", "Ready")
        ]
        
        for tool in tools:
            self.tools_tree.insert("", tk.END, values=tool)
    
    def _load_config(self):
        """Load MCP configuration."""
        try:
            config_file = Path("mcp_config.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.server_config = json.load(f)
                    
                # Update display
                cache = self.server_config.get('models', {}).get('cache_size', 3)
                self.cache_var.set(f"{cache} models")
                
                ctx = self.server_config.get('models', {}).get('default_context', 4096)
                self.ctx_var.set(f"{ctx} tokens")
                
                gpu = self.server_config.get('models', {}).get('default_gpu_layers', 0)
                self.gpu_var.set("Auto" if gpu == 0 else f"{gpu} layers")
                
                log_level = self.server_config.get('logging', {}).get('level', 'INFO')
                self.log_var.set(log_level)
                
        except Exception as e:
            self._log(f"Error loading config: {e}", "error")
    
    def _start_server(self):
        """Start the MCP server."""
        self._log("Starting MCP server...", "info")
        
        if self.server_manager.start_server(self.server_config):
            self._log("MCP server started successfully", "success")
        else:
            self._log("Failed to start MCP server", "error")
    
    def _stop_server(self):
        """Stop the MCP server."""
        self._log("Stopping MCP server...", "info")
        
        if self.server_manager.stop_server():
            self._log("MCP server stopped", "info")
        else:
            self._log("Failed to stop MCP server", "error")
    
    def _restart_server(self):
        """Restart the MCP server."""
        self._log("Restarting MCP server...", "info")
        
        if self.server_manager.restart_server(self.server_config):
            self._log("MCP server restarted successfully", "success")
        else:
            self._log("Failed to restart MCP server", "error")
    
    def _test_connection(self):
        """Test the server connection."""
        if not self.server_manager.get_status()['running']:
            messagebox.showinfo("Not Running", "Server is not running. Start it first.")
            return
        
        # Send a test command
        test_cmd = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1
        }
        
        if self.server_manager.send_command(test_cmd):
            self._log("Connection test sent", "info")
        else:
            self._log("Connection test failed", "error")
    
    def _open_config(self):
        """Open the configuration dialog."""
        from mcp_config import open_mcp_config
        open_mcp_config(self.parent.winfo_toplevel())
        self._load_config()  # Reload after config changes
    
    def _clear_logs(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)
    
    def _save_logs(self):
        """Save logs to file."""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self._log(f"Logs saved to {filename}", "success")
            except Exception as e:
                self._log(f"Error saving logs: {e}", "error")
    
    def _log(self, message: str, level: str = "info"):
        """Add a message to the log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Determine tag based on level
        tag = level.lower()
        if tag not in ["error", "warning", "info", "success"]:
            tag = "info"
        
        # Insert with tag
        self.log_text.insert(tk.END, log_entry, tag)
        
        # Auto-scroll if enabled
        if self.autoscroll_var.get():
            self.log_text.see(tk.END)
    
    def _on_server_start(self):
        """Handle server start event."""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.restart_btn.config(state=tk.NORMAL)
        self.status_canvas.itemconfig(self.status_indicator, fill="#00ff88")
        self.status_label.config(text="Running")
        
        # Update tool status
        for item in self.tools_tree.get_children():
            self.tools_tree.set(item, "status", "Active")
    
    def _on_server_stop(self):
        """Handle server stop event."""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.DISABLED)
        self.status_canvas.itemconfig(self.status_indicator, fill="red")
        self.status_label.config(text="Stopped")
        self.pid_label.config(text="N/A")
        self.uptime_label.config(text="00:00:00")
        
        # Update tool status
        for item in self.tools_tree.get_children():
            self.tools_tree.set(item, "status", "Ready")
    
    def _on_server_error(self, error: str):
        """Handle server error event."""
        self._log(f"Server error: {error}", "error")
        self.status_canvas.itemconfig(self.status_indicator, fill="#ff4444")
        self.status_label.config(text="Error")
    
    def _on_server_message(self, stream: str, message: str):
        """Handle server message event."""
        if stream == "stderr":
            if "error" in message.lower():
                self._log(message, "error")
            elif "warning" in message.lower():
                self._log(message, "warning")
            else:
                self._log(message, "info")
        else:
            # Try to parse as JSON for structured logs
            try:
                data = json.loads(message)
                if "level" in data:
                    self._log(data.get("message", message), data["level"].lower())
                else:
                    self._log(message, "info")
            except:
                self._log(message, "info")
    
    def _update_status(self):
        """Update status display periodically."""
        status = self.server_manager.get_status()
        
        if status['running']:
            # Update PID
            self.pid_label.config(text=str(status['pid']))
            
            # Update uptime
            uptime = int(status['uptime'])
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            seconds = uptime % 60
            self.uptime_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Update port (from config)
            port = self.server_config.get('server', {}).get('port', 'stdio')
            self.port_label.config(text=str(port))
        else:
            self.pid_label.config(text="N/A")
            self.port_label.config(text="N/A")
        
        # Schedule next update
        self.parent.after(1000, self._update_status)