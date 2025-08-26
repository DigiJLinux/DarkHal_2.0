#!/usr/bin/env python3
"""
LLM_Train Remote Control

A standalone GUI application for remotely controlling the LLM_Train MCP server.
Allows users to connect to the server, load models, configure settings, and 
perform inference operations remotely.
"""

import asyncio
import json
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from typing import Dict, Any, List, Optional, Callable
import subprocess
import os
from pathlib import Path
import queue
from datetime import datetime


class MCPClient:
    """Client for connecting to MCP server via subprocess."""
    
    def __init__(self):
        self.process = None
        self.connected = False
        self.request_id = 0
        self.callbacks: Dict[str, List[Callable]] = {
            'on_connect': [],
            'on_disconnect': [],
            'on_error': [],
            'on_response': []
        }
        self.pending_requests: Dict[int, Callable] = {}
        self.reader_thread = None
        self.writer_queue = queue.Queue()
        self.writer_thread = None
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for client events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")
    
    async def connect(self, server_path: str = "mcp_server.py"):
        """Connect to the MCP server."""
        try:
            # Start the MCP server process
            self.process = subprocess.Popen(
                [sys.executable, server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            self.connected = True
            
            # Start reader and writer threads
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            
            self.reader_thread.start()
            self.writer_thread.start()
            
            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {
                            "listChanged": True
                        },
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "LLM_Train Remote Control",
                        "version": "1.0.0"
                    }
                },
                "id": self._get_request_id()
            }
            
            await self._send_request(init_request)
            self._trigger_callback('on_connect')
            
            return True
            
        except Exception as e:
            self._trigger_callback('on_error', f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the MCP server."""
        self.connected = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
            self.process = None
        
        self._trigger_callback('on_disconnect')
    
    def _get_request_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: Dict[str, Any], callback: Optional[Callable] = None):
        """Send a request to the server."""
        if not self.connected or not self.process:
            return
        
        request_id = request.get('id')
        if request_id and callback:
            self.pending_requests[request_id] = callback
        
        # Queue the request for the writer thread
        self.writer_queue.put(json.dumps(request) + '\n')
    
    def _writer_loop(self):
        """Writer thread loop."""
        while self.connected and self.process:
            try:
                # Get request from queue
                request_str = self.writer_queue.get(timeout=1)
                
                if self.process and self.process.stdin:
                    self.process.stdin.write(request_str)
                    self.process.stdin.flush()
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.connected:
                    self._trigger_callback('on_error', f"Write error: {e}")
                break
    
    def _reader_loop(self):
        """Reader thread loop."""
        while self.connected and self.process:
            try:
                if self.process and self.process.stdout:
                    line = self.process.stdout.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    if line:
                        try:
                            response = json.loads(line)
                            self._handle_response(response)
                        except json.JSONDecodeError as e:
                            self._trigger_callback('on_error', f"JSON decode error: {e}")
                            
            except Exception as e:
                if self.connected:
                    self._trigger_callback('on_error', f"Read error: {e}")
                break
        
        # Connection lost
        if self.connected:
            self.connected = False
            self._trigger_callback('on_disconnect')
    
    def _handle_response(self, response: Dict[str, Any]):
        """Handle response from server."""
        request_id = response.get('id')
        
        if request_id and request_id in self.pending_requests:
            callback = self.pending_requests.pop(request_id)
            callback(response)
        
        self._trigger_callback('on_response', response)
    
    async def list_tools(self, callback: Optional[Callable] = None):
        """List available tools on the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": self._get_request_id()
        }
        await self._send_request(request, callback)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], callback: Optional[Callable] = None):
        """Call a tool on the server."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._get_request_id()
        }
        await self._send_request(request, callback)


class RemoteControlGUI:
    """Main GUI for the remote control application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LLM_Train Remote Control")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Initialize MCP client
        self.client = MCPClient()
        self.client.register_callback('on_connect', self._on_connect)
        self.client.register_callback('on_disconnect', self._on_disconnect)
        self.client.register_callback('on_error', self._on_error)
        
        # UI state
        self.connected = False
        self.available_tools = []
        self.available_models = []
        self.current_model = None
        self.system_info = {}
        
        # Setup UI
        self._setup_ui()
        
        # Setup asyncio loop for MCP client
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
    
    def _run_event_loop(self):
        """Run asyncio event loop in separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def _setup_ui(self):
        """Setup the main UI."""
        # Menu bar
        self._create_menu()
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Connection frame
        self._create_connection_frame(main_frame)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Model Management tab
        self._create_model_tab()
        
        # Inference tab
        self._create_inference_tab()
        
        # System Info tab
        self._create_system_tab()
        
        # Log tab
        self._create_log_tab()
        
        # Status bar
        self._create_status_bar(main_frame)
    
    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Connect to Server", command=self._connect_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Refresh Models", command=self._refresh_models)
        tools_menu.add_command(label="Get System Info", command=self._get_system_info)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_connection_frame(self, parent):
        """Create connection status frame."""
        conn_frame = ttk.LabelFrame(parent, text="Connection", padding=10)
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Connection status
        status_frame = ttk.Frame(conn_frame)
        status_frame.pack(fill=tk.X)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.connection_status = ttk.Label(status_frame, text="Disconnected", foreground="red")
        self.connection_status.pack(side=tk.LEFT, padx=(5, 20))
        
        # Server path
        ttk.Label(status_frame, text="Server:").pack(side=tk.LEFT)
        self.server_path_var = tk.StringVar(value="mcp_server.py")
        self.server_entry = ttk.Entry(status_frame, textvariable=self.server_path_var, width=30)
        self.server_entry.pack(side=tk.LEFT, padx=5)
        
        # Browse button
        ttk.Button(status_frame, text="Browse", command=self._browse_server).pack(side=tk.LEFT, padx=2)
        
        # Connect/Disconnect button
        self.connect_btn = ttk.Button(status_frame, text="Connect", command=self._toggle_connection)
        self.connect_btn.pack(side=tk.RIGHT, padx=5)
    
    def _create_model_tab(self):
        """Create model management tab."""
        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="Models")
        
        # Model list frame
        list_frame = ttk.LabelFrame(model_frame, text="Available Models", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model tree
        columns = ("name", "type", "size", "status")
        self.model_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        
        self.model_tree.heading("name", text="Model Name")
        self.model_tree.heading("type", text="Type")
        self.model_tree.heading("size", text="Size")
        self.model_tree.heading("status", text="Status")
        
        self.model_tree.column("name", width=300)
        self.model_tree.column("type", width=100)
        self.model_tree.column("size", width=100)
        self.model_tree.column("status", width=100)
        
        # Scrollbar
        model_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=model_scroll.set)
        
        self.model_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        model_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Model controls
        control_frame = ttk.Frame(model_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Refresh List", command=self._refresh_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Model", command=self._load_selected_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Unload Model", command=self._unload_model).pack(side=tk.LEFT, padx=5)
        
        # Current model info
        current_frame = ttk.LabelFrame(model_frame, text="Current Model", padding=10)
        current_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.current_model_label = ttk.Label(current_frame, text="No model loaded", font=("TkDefaultFont", 10, "bold"))
        self.current_model_label.pack(anchor=tk.W)
        
        # Model configuration
        config_frame = ttk.LabelFrame(model_frame, text="Model Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Context size
        ctx_frame = ttk.Frame(config_frame)
        ctx_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ctx_frame, text="Context Size:").pack(side=tk.LEFT)
        self.ctx_var = tk.IntVar(value=4096)
        ctx_spin = tk.Spinbox(ctx_frame, from_=512, to=32768, increment=512, textvariable=self.ctx_var, width=10)
        ctx_spin.pack(side=tk.LEFT, padx=5)
        
        # GPU layers
        gpu_frame = ttk.Frame(config_frame)
        gpu_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gpu_frame, text="GPU Layers:").pack(side=tk.LEFT)
        self.gpu_var = tk.IntVar(value=0)
        gpu_spin = tk.Spinbox(gpu_frame, from_=0, to=100, increment=1, textvariable=self.gpu_var, width=10)
        gpu_spin.pack(side=tk.LEFT, padx=5)
    
    def _create_inference_tab(self):
        """Create inference tab."""
        inference_frame = ttk.Frame(self.notebook)
        self.notebook.add(inference_frame, text="Inference")
        
        # Input frame
        input_frame = ttk.LabelFrame(inference_frame, text="Input", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Prompt input
        self.prompt_text = scrolledtext.ScrolledText(input_frame, height=6, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.X, pady=5)
        
        # Generation controls
        controls_frame = ttk.Frame(input_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Max tokens
        ttk.Label(controls_frame, text="Max Tokens:").pack(side=tk.LEFT)
        self.max_tokens_var = tk.IntVar(value=256)
        max_tokens_spin = tk.Spinbox(controls_frame, from_=1, to=8192, increment=16, textvariable=self.max_tokens_var, width=10)
        max_tokens_spin.pack(side=tk.LEFT, padx=5)
        
        # Temperature
        ttk.Label(controls_frame, text="Temperature:").pack(side=tk.LEFT, padx=(20, 0))
        self.temperature_var = tk.DoubleVar(value=0.7)
        temp_spin = tk.Spinbox(controls_frame, from_=0.0, to=2.0, increment=0.1, textvariable=self.temperature_var, width=10)
        temp_spin.pack(side=tk.LEFT, padx=5)
        
        # Generate button
        self.generate_btn = ttk.Button(controls_frame, text="Generate", command=self._generate_text)
        self.generate_btn.pack(side=tk.RIGHT, padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(inference_frame, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Chat mode
        chat_frame = ttk.LabelFrame(inference_frame, text="Chat Mode", padding=10)
        chat_frame.pack(fill=tk.X, padx=10, pady=10)
        
        chat_controls = ttk.Frame(chat_frame)
        chat_controls.pack(fill=tk.X)
        
        self.chat_mode_var = tk.BooleanVar()
        ttk.Checkbutton(chat_controls, text="Enable Chat Mode", variable=self.chat_mode_var).pack(side=tk.LEFT)
        
        ttk.Button(chat_controls, text="Clear History", command=self._clear_chat).pack(side=tk.RIGHT)
    
    def _create_system_tab(self):
        """Create system info tab."""
        system_frame = ttk.Frame(self.notebook)
        self.notebook.add(system_frame, text="System")
        
        # System info display
        info_frame = ttk.LabelFrame(system_frame, text="System Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.system_text = scrolledtext.ScrolledText(info_frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.system_text.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        ttk.Button(system_frame, text="Refresh System Info", command=self._get_system_info).pack(pady=10)
    
    def _create_log_tab(self):
        """Create log tab."""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log")
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(log_controls, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT)
        ttk.Button(log_controls, text="Save Log", command=self._save_log).pack(side=tk.LEFT, padx=5)
    
    def _create_status_bar(self, parent):
        """Create status bar."""
        self.status_bar = ttk.Label(parent, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _log(self, message: str, level: str = "INFO"):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _set_status(self, message: str):
        """Set status bar message."""
        self.status_bar.config(text=message)
    
    def _connect_dialog(self):
        """Show connection dialog."""
        if self.connected:
            self._disconnect()
        else:
            self._connect()
    
    def _browse_server(self):
        """Browse for server script."""
        file_path = filedialog.askopenfilename(
            title="Select MCP Server Script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if file_path:
            self.server_path_var.set(file_path)
    
    def _toggle_connection(self):
        """Toggle connection to server."""
        if self.connected:
            self._disconnect()
        else:
            self._connect()
    
    def _connect(self):
        """Connect to MCP server."""
        server_path = self.server_path_var.get().strip()
        if not server_path:
            messagebox.showerror("Error", "Please specify server path")
            return
        
        if not os.path.exists(server_path):
            messagebox.showerror("Error", f"Server file not found: {server_path}")
            return
        
        self._log(f"Connecting to server: {server_path}")
        self._set_status("Connecting...")
        
        # Connect in asyncio thread
        future = asyncio.run_coroutine_threadsafe(self.client.connect(server_path), self.loop)
        
        def check_result():
            if future.done():
                try:
                    success = future.result()
                    if success:
                        self._log("Connected successfully")
                    else:
                        self._log("Connection failed", "ERROR")
                except Exception as e:
                    self._log(f"Connection error: {e}", "ERROR")
            else:
                self.root.after(100, check_result)
        
        check_result()
    
    def _disconnect(self):
        """Disconnect from server."""
        self._log("Disconnecting from server")
        self.client.disconnect()
    
    def _on_connect(self):
        """Handle successful connection."""
        self.connected = True
        self.connection_status.config(text="Connected", foreground="green")
        self.connect_btn.config(text="Disconnect")
        self._set_status("Connected to server")
        
        # Refresh data
        self._refresh_models()
        self._get_system_info()
    
    def _on_disconnect(self):
        """Handle disconnection."""
        self.connected = False
        self.connection_status.config(text="Disconnected", foreground="red")
        self.connect_btn.config(text="Connect")
        self._set_status("Disconnected from server")
        
        # Clear data
        self.available_models = []
        self._update_model_list()
    
    def _on_error(self, error_message: str):
        """Handle client errors."""
        self._log(error_message, "ERROR")
        self._set_status(f"Error: {error_message}")
    
    def _refresh_models(self):
        """Refresh the list of available models."""
        if not self.connected:
            return
        
        self._log("Refreshing model list")
        
        def handle_response(response):
            if 'error' in response:
                self._log(f"Error listing models: {response['error']}", "ERROR")
                return
            
            try:
                result = response.get('result', [])
                if result and isinstance(result, list) and len(result) > 0:
                    content = result[0].get('text', '[]')
                    self.available_models = json.loads(content)
                else:
                    self.available_models = []
                
                self.root.after(0, self._update_model_list)
                self._log(f"Found {len(self.available_models)} models")
                
            except Exception as e:
                self._log(f"Error parsing model list: {e}", "ERROR")
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.call_tool("list_models", {}, handle_response), 
            self.loop
        )
    
    def _update_model_list(self):
        """Update the model list display."""
        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        
        # Add models
        for model in self.available_models:
            name = model.get('name', 'Unknown')
            model_type = model.get('type', 'Unknown')
            size_mb = model.get('size_mb', 0)
            size_text = f"{size_mb:.1f} MB" if size_mb > 0 else "Unknown"
            
            self.model_tree.insert("", tk.END, values=(name, model_type, size_text, "Available"))
    
    def _load_selected_model(self):
        """Load the selected model."""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a model to load")
            return
        
        item = self.model_tree.item(selection[0])
        model_name = item['values'][0]
        
        # Find the model path
        model_path = None
        for model in self.available_models:
            if model.get('name') == model_name:
                model_path = model.get('path')
                break
        
        if not model_path:
            messagebox.showerror("Error", "Model path not found")
            return
        
        self._log(f"Loading model: {model_name}")
        
        def handle_response(response):
            if 'error' in response:
                self._log(f"Error loading model: {response['error']}", "ERROR")
                return
            
            try:
                result = response.get('result', [])
                if result and isinstance(result, list) and len(result) > 0:
                    content = result[0].get('text', '')
                    if 'Successfully loaded' in content:
                        self.current_model = model_name
                        self.root.after(0, lambda: self.current_model_label.config(text=f"Loaded: {model_name}"))
                        self._log(f"Model loaded successfully: {model_name}")
                    else:
                        self._log(f"Failed to load model: {content}", "ERROR")
                
            except Exception as e:
                self._log(f"Error parsing load response: {e}", "ERROR")
        
        arguments = {
            "model_path": model_path,
            "n_ctx": self.ctx_var.get(),
            "n_gpu_layers": self.gpu_var.get()
        }
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.call_tool("load_model", arguments, handle_response), 
            self.loop
        )
    
    def _unload_model(self):
        """Unload the current model."""
        if not self.current_model:
            messagebox.showinfo("No Model", "No model is currently loaded")
            return
        
        self._log("Unloading current model")
        self.current_model = None
        self.current_model_label.config(text="No model loaded")
    
    def _generate_text(self):
        """Generate text using the current model."""
        if not self.connected:
            messagebox.showerror("Error", "Not connected to server")
            return
        
        if not self.current_model:
            messagebox.showerror("Error", "No model loaded")
            return
        
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt")
            return
        
        self._log(f"Generating text for prompt: {prompt[:50]}...")
        self._set_status("Generating...")
        
        def handle_response(response):
            if 'error' in response:
                self._log(f"Error generating text: {response['error']}", "ERROR")
                return
            
            try:
                result = response.get('result', [])
                if result and isinstance(result, list) and len(result) > 0:
                    content_str = result[0].get('text', '{}')
                    content = json.loads(content_str)
                    
                    if 'error' in content:
                        self._log(f"Generation error: {content['error']}", "ERROR")
                        return
                    
                    generated_text = content.get('text', '')
                    
                    # Update output
                    self.root.after(0, lambda: self._update_output(prompt, generated_text))
                    self._log("Text generation completed")
                    self.root.after(0, lambda: self._set_status("Generation completed"))
                
            except Exception as e:
                self._log(f"Error parsing generation response: {e}", "ERROR")
        
        arguments = {
            "prompt": prompt,
            "max_tokens": self.max_tokens_var.get(),
            "temperature": self.temperature_var.get()
        }
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.call_tool("generate_text", arguments, handle_response), 
            self.loop
        )
    
    def _update_output(self, prompt: str, generated_text: str):
        """Update the output text area."""
        self.output_text.config(state=tk.NORMAL)
        
        if self.chat_mode_var.get():
            # Chat mode - append to conversation
            self.output_text.insert(tk.END, f"User: {prompt}\n\n")
            self.output_text.insert(tk.END, f"Assistant: {generated_text}\n\n")
            self.output_text.insert(tk.END, "-" * 50 + "\n\n")
        else:
            # Replace mode - show only current generation
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Prompt: {prompt}\n\n")
            self.output_text.insert(tk.END, f"Response: {generated_text}")
        
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def _clear_chat(self):
        """Clear chat history."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def _get_system_info(self):
        """Get system information from server."""
        if not self.connected:
            return
        
        self._log("Getting system information")
        
        def handle_response(response):
            if 'error' in response:
                self._log(f"Error getting system info: {response['error']}", "ERROR")
                return
            
            try:
                result = response.get('result', [])
                if result and isinstance(result, list) and len(result) > 0:
                    content_str = result[0].get('text', '{}')
                    self.system_info = json.loads(content_str)
                    
                    self.root.after(0, self._update_system_display)
                    self._log("System information updated")
                
            except Exception as e:
                self._log(f"Error parsing system info: {e}", "ERROR")
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.call_tool("get_system_info", {}, handle_response), 
            self.loop
        )
    
    def _update_system_display(self):
        """Update system information display."""
        self.system_text.config(state=tk.NORMAL)
        self.system_text.delete(1.0, tk.END)
        
        # Format system info
        info_text = "System Information\n"
        info_text += "=" * 50 + "\n\n"
        
        info_text += f"Platform: {self.system_info.get('platform', 'Unknown')}\n"
        info_text += f"Architecture: {self.system_info.get('architecture', 'Unknown')}\n\n"
        
        # Acceleration info
        acceleration = self.system_info.get('acceleration', {})
        info_text += "Acceleration Support:\n"
        info_text += f"  CUDA Available: {acceleration.get('cuda_available', False)}\n"
        if acceleration.get('cuda_available'):
            info_text += f"  CUDA Version: {acceleration.get('cuda_version', 'Unknown')}\n"
            info_text += f"  CUDA Devices: {acceleration.get('cuda_devices', 0)}\n"
        
        info_text += f"  ROCm Available: {acceleration.get('rocm_available', False)}\n"
        info_text += f"  Metal Available: {acceleration.get('metal_available', False)}\n"
        info_text += f"  Intel GPU Available: {acceleration.get('intel_gpu_available', False)}\n"
        info_text += f"  Recommended GPU Layers: {acceleration.get('recommended_layers', 0)}\n\n"
        
        info_text += f"Current Model Acceleration: {self.system_info.get('current_model_acceleration', 'None')}\n"
        
        self.system_text.insert(tk.END, info_text)
        self.system_text.config(state=tk.DISABLED)
    
    def _clear_log(self):
        """Clear the log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _save_log(self):
        """Save log to file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self._log(f"Log saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log: {e}")
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """LLM_Train Remote Control v1.0.0

A remote control application for managing and 
interacting with LLM_Train MCP servers.

Features:
• Remote model loading and configuration
• Text generation and chat interface  
• System information monitoring
• Connection management

© 2024 LLM_Train Project"""
        
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run the application."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()
    
    def _on_closing(self):
        """Handle application closing."""
        if self.connected:
            self.client.disconnect()
        self._cleanup()
        self.root.destroy()
    
    def _cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


def main():
    """Main entry point."""
    try:
        app = RemoteControlGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())