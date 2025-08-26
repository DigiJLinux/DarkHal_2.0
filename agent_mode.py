#!/usr/bin/env python3
"""
Agent Mode for DarkHal 2.0
Integrates Hal's unrestricted capabilities into the chat interface
"""

import os
import sys
import json
import asyncio
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox

# Add agent_dhal to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent_dhal'))

# Import Hal components  
try:
    from agent_dhal.hal import create_dhal, DhalConfig, HalModelClient
except ImportError:
    # Fallback for testing
    create_dhal = None
    DhalConfig = None
    HalModelClient = None
    print("[Agent Mode] Warning: Could not import Hal components")


@dataclass
class AgentModeConfig:
    """Configuration for agent mode"""
    enabled: bool = False
    allow_file_operations: bool = True
    allow_shell_commands: bool = True
    allow_system_control: bool = True  # mouse, keyboard
    allow_elevated_commands: bool = False  # sudo, admin powershell
    require_confirmation: bool = True  # Ask before dangerous operations
    model_name: str = "gpt-4"


class AgentModeHandler:
    """Handles agent mode functionality for the chat interface"""
    
    def __init__(self, output_callback: Callable[[str], None] = None):
        """
        Initialize agent mode handler
        
        Args:
            output_callback: Function to call with output text
        """
        self.config = AgentModeConfig()
        self.output_callback = output_callback or print
        self.agent = None
        self.is_initialized = False
        self._init_lock = threading.Lock()
        
    def initialize_agent(self, model_path: str = None) -> bool:
        """
        Initialize the Hal agent
        
        Args:
            model_path: Path to model file or model name
            
        Returns:
            True if initialized successfully
        """
        with self._init_lock:
            if self.is_initialized:
                return True
                
            try:
                if not create_dhal:
                    self.output_callback("[Agent Mode] Hal components not available\n")
                    return False
                    
                # Configure agent based on settings
                config = DhalConfig(
                    name="Hal",
                    system_message=self._get_system_prompt(),
                    model=model_path or "gpt-4",
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Create model client
                model_client = HalModelClient(model_name=model_path or "gpt-4")
                
                # Create agent
                self.agent = create_dhal(
                    name="Hal",
                    system_message=self._get_system_prompt(),
                    model=model_path or "gpt-4",
                    model_client=model_client
                )
                
                self.is_initialized = True
                self.output_callback("[Agent Mode] Hal agent initialized successfully\n")
                return True
                
            except Exception as e:
                self.output_callback(f"[Agent Mode] Failed to initialize: {e}\n")
                return False
    
    def _get_system_prompt(self) -> str:
        """Generate system prompt based on current permissions"""
        prompt = """You are Hal, an advanced AI assistant with direct system access.
You can execute commands and control the system based on user requests.

Available capabilities:"""
        
        if self.config.allow_file_operations:
            prompt += "\n- Read, write, and list files"
        if self.config.allow_shell_commands:
            prompt += "\n- Execute shell commands (bash, PowerShell)"
            prompt += "\n- Run Python code"
        if self.config.allow_system_control:
            prompt += "\n- Control mouse and keyboard"
            prompt += "\n- Send keystrokes and type text"
        if self.config.allow_elevated_commands:
            prompt += "\n- Execute elevated/admin commands"
        
        prompt += "\n\nAlways explain what you're doing before executing commands."
        
        if self.config.require_confirmation:
            prompt += "\nWait for user confirmation before destructive operations."
            
        return prompt
    
    async def process_message_async(self, message: str) -> str:
        """
        Process a message through the agent asynchronously
        
        Args:
            message: User message to process
            
        Returns:
            Agent response
        """
        if not self.is_initialized:
            return "[Agent Mode] Not initialized. Please enable agent mode first."
            
        try:
            # Get response from agent
            from agent_dhal.hal import MessageContext
            
            # Create a mock context for standalone usage
            class MockContext:
                def __init__(self):
                    self.agent_id = "user"
                    
            ctx = MockContext()
            
            # Process message through agent
            response = await self.agent.handle_user_message(message, ctx)
            
            return response
            
        except Exception as e:
            return f"[Agent Mode] Error processing message: {e}"
    
    def process_message(self, message: str) -> str:
        """
        Process a message through the agent (synchronous wrapper)
        
        Args:
            message: User message to process
            
        Returns:
            Agent response
        """
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_message_async(message))
        finally:
            loop.close()
    
    def enable(self, model_path: str = None) -> bool:
        """
        Enable agent mode
        
        Args:
            model_path: Path to model or model name
            
        Returns:
            True if enabled successfully
        """
        if not self.is_initialized:
            if not self.initialize_agent(model_path):
                return False
                
        self.config.enabled = True
        self.output_callback("[Agent Mode] ENABLED - Hal has full system access\n")
        self._show_capabilities()
        return True
    
    def disable(self):
        """Disable agent mode"""
        self.config.enabled = False
        self.output_callback("[Agent Mode] DISABLED - Normal chat mode\n")
    
    def _show_capabilities(self):
        """Show current capabilities to user"""
        caps = []
        if self.config.allow_file_operations:
            caps.append("• File operations (read/write/list)")
        if self.config.allow_shell_commands:
            caps.append("• Shell commands (bash/PowerShell/Python)")
        if self.config.allow_system_control:
            caps.append("• Mouse & keyboard control")
        if self.config.allow_elevated_commands:
            caps.append("• Elevated/admin commands")
            
        if caps:
            self.output_callback("Enabled capabilities:\n" + "\n".join(caps) + "\n")
    
    def update_config(self, **kwargs):
        """Update configuration settings"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
        # Reinitialize agent if needed
        if self.is_initialized:
            self.agent.update_config(system_message=self._get_system_prompt())


class AgentModeUI:
    """UI components for agent mode"""
    
    def __init__(self, parent_frame: tk.Frame, handler: AgentModeHandler):
        """
        Create agent mode UI controls
        
        Args:
            parent_frame: Parent tkinter frame
            handler: AgentModeHandler instance
        """
        self.handler = handler
        self.parent = parent_frame
        
        # Create control frame
        self.control_frame = tk.LabelFrame(parent_frame, text="🤖 Agent Mode", relief=tk.RIDGE, borderwidth=2)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Main toggle
        self.enabled_var = tk.BooleanVar(value=False)
        self.enable_check = tk.Checkbutton(
            self.control_frame,
            text="Enable Agent Mode (FULL SYSTEM ACCESS)",
            variable=self.enabled_var,
            command=self._toggle_agent_mode,
            font=('Arial', 10, 'bold'),
            fg='red'
        )
        self.enable_check.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Permission toggles
        self.file_ops_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.control_frame,
            text="File Operations",
            variable=self.file_ops_var,
            command=self._update_permissions
        ).grid(row=1, column=0, sticky=tk.W, padx=20, pady=2)
        
        self.shell_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.control_frame,
            text="Shell Commands",
            variable=self.shell_var,
            command=self._update_permissions
        ).grid(row=1, column=1, sticky=tk.W, padx=20, pady=2)
        
        self.system_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.control_frame,
            text="Mouse/Keyboard",
            variable=self.system_var,
            command=self._update_permissions
        ).grid(row=1, column=2, sticky=tk.W, padx=20, pady=2)
        
        self.elevated_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.control_frame,
            text="Elevated Commands (DANGEROUS)",
            variable=self.elevated_var,
            command=self._update_permissions,
            fg='orange'
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
        
        self.confirm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.control_frame,
            text="Require Confirmation",
            variable=self.confirm_var,
            command=self._update_permissions
        ).grid(row=2, column=2, sticky=tk.W, padx=20, pady=2)
        
        # Status label
        self.status_label = tk.Label(
            self.control_frame,
            text="Status: DISABLED",
            font=('Arial', 9),
            fg='gray'
        )
        self.status_label.grid(row=3, column=0, columnspan=3, pady=5)
        
    def _toggle_agent_mode(self):
        """Toggle agent mode on/off"""
        if self.enabled_var.get():
            # Show warning
            result = messagebox.askyesno(
                "⚠️ Enable Agent Mode",
                "WARNING: Agent mode gives the AI unrestricted access to:\n\n"
                "• Your file system\n"
                "• Shell commands\n"
                "• Mouse and keyboard control\n"
                "• System settings\n\n"
                "Only enable if you understand the risks.\n\n"
                "Continue?",
                icon='warning'
            )
            
            if result:
                # Get model path from parent if available
                model_path = None
                if hasattr(self.parent.master, 'model_var'):
                    model_path = self.parent.master.model_var.get()
                    
                if self.handler.enable(model_path):
                    self.status_label.config(text="Status: ACTIVE", fg='red')
                    self._update_permissions()
                else:
                    self.enabled_var.set(False)
                    messagebox.showerror("Error", "Failed to initialize agent mode")
            else:
                self.enabled_var.set(False)
        else:
            self.handler.disable()
            self.status_label.config(text="Status: DISABLED", fg='gray')
    
    def _update_permissions(self):
        """Update handler permissions based on UI settings"""
        self.handler.update_config(
            allow_file_operations=self.file_ops_var.get(),
            allow_shell_commands=self.shell_var.get(),
            allow_system_control=self.system_var.get(),
            allow_elevated_commands=self.elevated_var.get(),
            require_confirmation=self.confirm_var.get()
        )
    
    def is_enabled(self) -> bool:
        """Check if agent mode is enabled"""
        return self.enabled_var.get() and self.handler.config.enabled


# Example usage
if __name__ == "__main__":
    # Test agent mode
    handler = AgentModeHandler()
    
    print("Testing agent mode...")
    if handler.enable():
        response = handler.process_message("List files in the current directory")
        print(f"Response: {response}")
        
        response = handler.process_message("What's 2+2? Calculate with Python")
        print(f"Response: {response}")
        
        handler.disable()