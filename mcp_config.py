#!/usr/bin/env python3
"""
MCP Server Configuration and Management

This module provides utilities to configure and manage the MCP server
for LLM_Train models.
"""

import json
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Any, Optional
from pathlib import Path


class MCPServerConfig:
    """Configuration management for the MCP server."""
    
    def __init__(self, config_file: str = "mcp_config.json"):
        self.config_file = config_file
        self.default_config = {
            "server": {
                "name": "llm-train-models",
                "version": "1.0.0",
                "description": "Multi-model MCP server for LLM_Train",
                "host": "localhost",
                "port": 8000,
                "auto_start": False
            },
            "models": {
                "cache_size": 3,
                "default_context": 4096,
                "default_gpu_layers": 0
            },
            "logging": {
                "level": "INFO",
                "file": "mcp_server.log"
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults
                    return self._merge_config(self.default_config, loaded)
            return self.default_config.copy()
        except Exception as e:
            print(f"Error loading MCP config: {e}")
            return self.default_config.copy()
    
    def _merge_config(self, defaults: Dict, loaded: Dict) -> Dict:
        """Merge loaded config with defaults."""
        result = defaults.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving MCP config: {e}")
            return False
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, path: str, value: Any):
        """Set config value using dot notation."""
        keys = path.split('.')
        target = self.config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    
    def generate_claude_config(self) -> Dict[str, Any]:
        """Generate Claude Desktop MCP configuration."""
        script_path = os.path.abspath("mcp_server.py")
        python_path = sys.executable
        
        return {
            "mcpServers": {
                "llm-train-models": {
                    "command": python_path,
                    "args": [script_path],
                    "env": {
                        "PYTHONPATH": os.getcwd()
                    }
                }
            }
        }


class MCPConfigGUI:
    """GUI for configuring the MCP server."""
    
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.config = MCPServerConfig()
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("MCP Server Configuration")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._build_ui()
        self._load_values()
    
    def _build_ui(self):
        """Build the configuration UI."""
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Server Settings Tab
        server_frame = ttk.Frame(notebook)
        notebook.add(server_frame, text="Server")
        self._build_server_tab(server_frame)
        
        # Model Settings Tab
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Models")
        self._build_models_tab(model_frame)
        
        # Claude Integration Tab
        claude_frame = ttk.Frame(notebook)
        notebook.add(claude_frame, text="Claude Integration")
        self._build_claude_tab(claude_frame)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="Save", command=self._save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Test Server", command=self._test_server).pack(side=tk.LEFT)
    
    def _build_server_tab(self, parent: ttk.Frame):
        """Build server settings tab."""
        frame = ttk.LabelFrame(parent, text="Server Configuration", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Server name
        ttk.Label(frame, text="Server Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.server_name_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.server_name_var, width=30).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Auto start
        self.auto_start_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Auto-start server with application",
                       variable=self.auto_start_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        # Logging level
        ttk.Label(frame, text="Logging Level:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.log_level_var = tk.StringVar()
        ttk.Combobox(frame, textvariable=self.log_level_var, 
                    values=["DEBUG", "INFO", "WARNING", "ERROR"],
                    state="readonly", width=15).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Log file
        ttk.Label(frame, text="Log File:").grid(row=3, column=0, sticky=tk.W, pady=5)
        log_frame = ttk.Frame(frame)
        log_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        self.log_file_var = tk.StringVar()
        ttk.Entry(log_frame, textvariable=self.log_file_var, width=25).pack(side=tk.LEFT)
        ttk.Button(log_frame, text="Browse", 
                  command=self._browse_log_file).pack(side=tk.LEFT, padx=5)
    
    def _build_models_tab(self, parent: ttk.Frame):
        """Build model settings tab."""
        frame = ttk.LabelFrame(parent, text="Model Configuration", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Cache size
        ttk.Label(frame, text="Model Cache Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.cache_size_var = tk.IntVar()
        cache_spin = tk.Spinbox(frame, from_=1, to=10, textvariable=self.cache_size_var, width=10)
        cache_spin.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="models").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Default context
        ttk.Label(frame, text="Default Context Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.default_ctx_var = tk.IntVar()
        ctx_spin = tk.Spinbox(frame, from_=512, to=32768, increment=512, 
                             textvariable=self.default_ctx_var, width=10)
        ctx_spin.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="tokens").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Default GPU layers
        ttk.Label(frame, text="Default GPU Layers:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.default_gpu_var = tk.IntVar()
        gpu_spin = tk.Spinbox(frame, from_=0, to=100, textvariable=self.default_gpu_var, width=10)
        gpu_spin.grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="layers").grid(row=2, column=2, sticky=tk.W, padx=5)
        
        # Info
        info_text = ("Cache size determines how many models can be loaded simultaneously.\\n"
                    "Default settings are used when loading models via MCP.")
        ttk.Label(frame, text=info_text, foreground="gray").grid(row=3, column=0, columnspan=3, pady=10)
    
    def _build_claude_tab(self, parent: ttk.Frame):
        """Build Claude integration tab."""
        frame = ttk.LabelFrame(parent, text="Claude Desktop Integration", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Instructions - wrapped properly
        instructions_frame = ttk.Frame(frame)
        instructions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create a text widget for instructions with proper wrapping
        instructions_text = tk.Text(instructions_frame, height=8, wrap=tk.WORD, 
                                  background="#f0f0f0", relief=tk.FLAT, 
                                  borderwidth=0, state=tk.DISABLED)
        instructions_text.pack(fill=tk.X)
        
        # Add properly formatted instructions
        instructions_content = """To use this MCP server with Claude Desktop, follow these steps:

1. Locate your Claude Desktop configuration file:
   • Windows: %APPDATA%\\Claude\\claude_desktop_config.json
   • macOS: ~/Library/Application Support/Claude/claude_desktop_config.json  
   • Linux: ~/.config/Claude/claude_desktop_config.json

2. Open the configuration file in a text editor (create it if it doesn't exist)

3. Add the JSON configuration shown below to the file

4. Save the file and restart Claude Desktop

5. The LLM_Train models server will be available in Claude Desktop"""
        
        instructions_text.config(state=tk.NORMAL)
        instructions_text.insert(1.0, instructions_content)
        instructions_text.config(state=tk.DISABLED)
        
        # Config display
        ttk.Label(frame, text="Configuration JSON:", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(10,5))
        
        # Create text frame with proper layout
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Make the config text read-only and styled
        self.config_text = tk.Text(text_frame, height=12, width=70, wrap=tk.WORD,
                                 state=tk.DISABLED, background="#f8f8f8",
                                 font=("Courier", 9))
        config_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.config_text.yview)
        self.config_text.configure(yscrollcommand=config_scroll.set)
        
        self.config_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        config_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Copy to Clipboard", 
                  command=self._copy_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save to File", 
                  command=self._save_claude_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh", 
                  command=self._update_claude_config).pack(side=tk.LEFT, padx=5)
        
        self._update_claude_config()
    
    def _load_values(self):
        """Load configuration values into UI."""
        self.server_name_var.set(self.config.get('server.name', 'llm-train-models'))
        self.auto_start_var.set(self.config.get('server.auto_start', False))
        self.log_level_var.set(self.config.get('logging.level', 'INFO'))
        self.log_file_var.set(self.config.get('logging.file', 'mcp_server.log'))
        
        self.cache_size_var.set(self.config.get('models.cache_size', 3))
        self.default_ctx_var.set(self.config.get('models.default_context', 4096))
        self.default_gpu_var.set(self.config.get('models.default_gpu_layers', 0))
    
    def _save_config(self):
        """Save configuration."""
        self.config.set('server.name', self.server_name_var.get())
        self.config.set('server.auto_start', self.auto_start_var.get())
        self.config.set('logging.level', self.log_level_var.get())
        self.config.set('logging.file', self.log_file_var.get())
        
        self.config.set('models.cache_size', self.cache_size_var.get())
        self.config.set('models.default_context', self.default_ctx_var.get())
        self.config.set('models.default_gpu_layers', self.default_gpu_var.get())
        
        if self.config.save_config():
            messagebox.showinfo("Success", "Configuration saved successfully!")
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "Failed to save configuration")
    
    def _browse_log_file(self):
        """Browse for log file location."""
        filename = filedialog.asksaveasfilename(
            title="Select Log File",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if filename:
            self.log_file_var.set(filename)
    
    def _update_claude_config(self):
        """Update the Claude configuration display."""
        config = self.config.generate_claude_config()
        config_json = json.dumps(config, indent=2)
        
        self.config_text.delete(1.0, tk.END)
        self.config_text.insert(1.0, config_json)
    
    def _copy_config(self):
        """Copy configuration to clipboard."""
        config_text = self.config_text.get(1.0, tk.END)
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(config_text)
        messagebox.showinfo("Copied", "Configuration copied to clipboard!")
    
    def _save_claude_config(self):
        """Save Claude configuration to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Claude Config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialvalue="claude_desktop_config.json"
        )
        
        if filename:
            try:
                config = self.config.generate_claude_config()
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Saved", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def _test_server(self):
        """Test the MCP server."""
        try:
            # Try to run the server with --help to test if it's working
            result = subprocess.run([
                sys.executable, "mcp_server.py", "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                messagebox.showinfo("Test Successful", 
                                   "MCP server appears to be working correctly!")
            else:
                messagebox.showerror("Test Failed", 
                                   f"Server test failed:\\n{result.stderr}")
        except Exception as e:
            messagebox.showerror("Test Error", f"Failed to test server: {e}")


def open_mcp_config(parent: tk.Tk):
    """Open the MCP configuration dialog."""
    MCPConfigGUI(parent)


if __name__ == "__main__":
    # Test the configuration
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    app = MCPConfigGUI(root)
    root.mainloop()