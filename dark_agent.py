#!/usr/bin/env python3
"""
Dark Agent (DarkAgent) Tab Implementation
Handles the Dark Agent interface and controls for DarkHal 2.0
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

class DarkAgentTab:
    """Dark Agent tab with agent control and configuration."""
    
    def __init__(self, parent: ttk.Frame, settings_manager, main_app=None):
        self.parent = parent
        self.settings = settings_manager
        self.main_app = main_app
        
        # Initialize Dark Agent integration
        self._hal_integration = None
        
        # Create main frame
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create Dark Agent interface
        self._create_dark_agent_interface()
    
    def _create_dark_agent_interface(self):
        """Create Dark Agent control and configuration interface."""
        
        # Dark configuration frame
        config_frame = ttk.LabelFrame(self.main_frame, text="Dark Agent Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Configuration options
        options_frame = ttk.Frame(config_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        # Agent Name (fixed, not editable)
        ttk.Label(options_frame, text="Agent Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(options_frame, text="Dhal", font=("Arial", 9, "bold")).grid(row=0, column=1, sticky=tk.W, padx=10)
        
        # Model Configuration
        ttk.Label(options_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hal_model_var = tk.StringVar(value="local-llm")
        ttk.Entry(options_frame, textvariable=self.hal_model_var, width=20).grid(row=1, column=1, padx=10)
        
        # System Message
        ttk.Label(options_frame, text="System Message:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.hal_system_var = tk.StringVar(value="You are Dhal, an advanced AI assistant integrated into DarkHal 2.0.")
        ttk.Entry(options_frame, textvariable=self.hal_system_var, width=40).grid(row=0, column=3, padx=10)
        
        # Temperature
        ttk.Label(options_frame, text="Temperature:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0))
        self.hal_temp_var = tk.StringVar(value="0.7")
        ttk.Entry(options_frame, textvariable=self.hal_temp_var, width=10).grid(row=1, column=3, padx=10)
        
        # Max Tokens
        ttk.Label(options_frame, text="Max Tokens:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.hal_max_tokens_var = tk.StringVar(value="2048")
        ttk.Entry(options_frame, textvariable=self.hal_max_tokens_var, width=10).grid(row=2, column=1, padx=10)
        
        # Tool Configuration
        tools_frame = ttk.LabelFrame(config_frame, text="Available Tools", padding=5)
        tools_frame.pack(fill=tk.X, pady=10)
        
        self.hal_tools = {}
        tools = ["Web Search", "Code Execution", "File Operations", "System Commands"]
        for i, tool in enumerate(tools):
            var = tk.BooleanVar(value=True)
            self.hal_tools[tool.lower().replace(" ", "_")] = var
            ttk.Checkbutton(tools_frame, text=tool, variable=var).grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(config_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.hal_start_btn = ttk.Button(control_frame, text="Start Dhal Agent", 
                  command=self._start_hal)
        self.hal_start_btn.pack(side=tk.LEFT, padx=5)
        
        self.hal_stop_btn = ttk.Button(control_frame, text="Stop Agent", 
                 command=self._stop_hal, state="disabled")
        self.hal_stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.hal_reset_btn = ttk.Button(control_frame, text="Reset Conversation", 
                  command=self._reset_hal, state="disabled")
        self.hal_reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Configuration buttons
        config_btn_frame = ttk.Frame(control_frame)
        config_btn_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(config_btn_frame, text="Save Config", 
                  command=self._save_hal_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(config_btn_frame, text="Load Config", 
                  command=self._load_hal_config).pack(side=tk.LEFT, padx=2)
        
        # Chat interface
        chat_frame = ttk.LabelFrame(self.main_frame, text="Dark Agent Chat", padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Output area
        self.hal_output = tk.Text(chat_frame, height=15, wrap=tk.WORD)
        hal_scrollbar = ttk.Scrollbar(chat_frame, orient=tk.VERTICAL, command=self.hal_output.yview)
        self.hal_output.configure(yscrollcommand=hal_scrollbar.set)
        self.hal_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hal_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.hal_input_var = tk.StringVar()
        hal_input = ttk.Entry(input_frame, textvariable=self.hal_input_var)
        hal_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        hal_input.bind("<Return>", lambda e: self._send_hal_message())
        
        self.hal_send_btn = ttk.Button(input_frame, text="Send", 
                 command=self._send_hal_message, state="disabled")
        self.hal_send_btn.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.hal_status_var = tk.StringVar(value="Dhal Status: Not Started")
        ttk.Label(chat_frame, textvariable=self.hal_status_var).pack(anchor=tk.W, pady=(5, 0))
    
    # Dark Agent Methods
    def _start_hal(self):
        """Start Dark agent."""
        if not self._hal_integration:
            try:
                from agent_dhal_integration import DhalAgentIntegration
                self._hal_integration = DhalAgentIntegration(self)
            except ImportError:
                messagebox.showerror("Import Error", "Could not import DhalAgentIntegration")
                return
        self._hal_integration.start_agent()

    def _stop_hal(self):
        """Stop Dark agent."""
        if self._hal_integration:
            self._hal_integration.stop_agent()

    def _send_hal_message(self):
        """Send message to Dark agent."""
        if self._hal_integration:
            self._hal_integration.send_message()

    def _reset_hal(self):
        """Reset Dark agent conversation."""
        if self._hal_integration:
            self._hal_integration.reset_conversation()

    def _save_hal_config(self):
        """Save Dark agent configuration."""
        if not self._hal_integration:
            try:
                from agent_dhal_integration import DhalAgentIntegration
                self._hal_integration = DhalAgentIntegration(self)
            except ImportError:
                messagebox.showerror("Import Error", "Could not import DhalAgentIntegration")
                return
        self._hal_integration.save_config()

    def _load_hal_config(self):
        """Load Dark agent configuration."""
        if not self._hal_integration:
            try:
                from agent_dhal_integration import DhalAgentIntegration
                self._hal_integration = DhalAgentIntegration(self)
            except ImportError:
                messagebox.showerror("Import Error", "Could not import DhalAgentIntegration")
                return
        self._hal_integration.load_config()