#!/usr/bin/env python3
"""
Agent Dhal Integration for DarkHal 2.0

Connects the DarkAgent to the main application's UI and model loading system.
Provides thread-safe communication between the agent and the Tkinter interface.
"""

import threading
import queue
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from tkinter import messagebox
from agent_debug_tracer import get_tracer, trace
import __spy as spy  # global announcer for current model

try:
    from agent_dhal.hal import Dhal as DarkAgent
except ImportError:
    try:
        from agent_dhal.hal import Hal as DarkAgent  # fallback class name
    except ImportError:
        DarkAgent = None  # type: ignore
AGENT_AVAILABLE = DarkAgent is not None

class ExistingRuntimeClient:
    """
    Thin adapter that wraps an already-loaded model object from the app and
    exposes a chat-completion-like interface for the agent.
    """
    def __init__(self, model):
        self._model = model

    def is_available(self) -> bool:
        return self._model is not None

    def create_chat_completion(
        self,
        messages,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools=None,
        stream: bool = False,
        on_token=None,
        on_complete=None,
        on_error=None,
        top_p: float = 0.9,
    ):
        """
        Minimal bridge:
        - Flattens chat messages into a single prompt.
        - Uses model.stream(...) if available when stream=True; otherwise model.generate(...).
        - Tries to construct a generation config if llm_runtime.GenerateConfig is available.
        """
        try:
            prompt = self._flatten_messages(messages)
            config = self._build_config(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

            if stream and hasattr(self._model, "stream"):
                full = []
                for chunk in self._model.stream(prompt, config) if config is not None else self._model.stream(prompt):
                    if on_token:
                        try:
                            on_token("request-1", chunk)
                        except Exception:
                            pass
                    full.append(chunk)
                final_text = "".join(full)
                if on_complete:
                    on_complete("request-1", final_text, {"finish_reason": "stop"})
                return {"text": final_text, "finish_reason": "stop"}

            # Non-streaming path
            if hasattr(self._model, "generate"):
                text = self._model.generate(prompt, config) if config is not None else self._model.generate(prompt)
                if on_complete:
                    on_complete("request-1", text, {"finish_reason": "stop"})
                return {"text": text, "finish_reason": "stop"}

            raise RuntimeError("Model does not support generate/stream")

        except Exception as e:
            if on_error:
                on_error("request-1", e)
            else:
                raise

    def _flatten_messages(self, messages) -> str:
        # Simple readable flattening
        lines = []
        for m in messages or []:
            role = (m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")).upper()
            content = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else ""

    def _build_config(self, max_tokens: int, temperature: float, top_p: float):
        # Let the underlying model decide all defaults (no explicit config).
        return None



class DhalAgentIntegration:
    """Integration layer between DarkAgent and DarkHal 2.0 UI."""
    
    def __init__(self, agents_tab):
        trace("INTEGRATION_INIT", "DhalAgentIntegration initializing")
        self.agents_tab = agents_tab
        self.agent: Optional[DarkAgent] = None
        self.client: Optional[ExistingRuntimeClient] = None
        self.conversation_id = "main_conversation"
        self.tracer = get_tracer()
        
        # UI update queue for thread-safe communication
        self.ui_queue = queue.Queue()
        self.is_running = False
        
        # Start UI update checker
        self._start_ui_updater()
    
    def _start_ui_updater(self):
        """Start the UI update thread."""
        def update_ui():
            while True:
                try:
                    action, data = self.ui_queue.get(timeout=0.1)
                    if action == "append_text":
                        self._append_text_safe(data)
                    elif action == "set_status":
                        self._set_status_safe(data)
                    elif action == "toggle_buttons":
                        self._toggle_buttons_safe(data)
                    self.ui_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"UI update error: {e}")
        
        ui_thread = threading.Thread(target=update_ui, daemon=True)
        ui_thread.start()
    
    def _append_text_safe(self, text: str):
        """Thread-safe text append to chat output."""
        try:
            self.agents_tab.hal_output.insert("end", text)
            self.agents_tab.hal_output.see("end")
        except Exception as e:
            print(f"Text append error: {e}")
    
    def _set_status_safe(self, status: str):
        """Thread-safe status update."""
        try:
            self.agents_tab.hal_status_var.set(status)
        except Exception as e:
            print(f"Status update error: {e}")
    
    def _toggle_buttons_safe(self, running: bool):
        """Thread-safe button state toggle."""
        try:
            if running:
                self.agents_tab.hal_start_btn.config(state="disabled")
                self.agents_tab.hal_stop_btn.config(state="normal")
                self.agents_tab.hal_send_btn.config(state="normal")
            else:
                self.agents_tab.hal_start_btn.config(state="normal")
                self.agents_tab.hal_stop_btn.config(state="disabled")
                self.agents_tab.hal_send_btn.config(state="disabled")
        except Exception as e:
            print(f"Button toggle error: {e}")
    
    def start_agent(self):
        """Start the Dark Agent."""
        if not AGENT_AVAILABLE:
            messagebox.showerror("Agent Error", "AgentDhal framework not available. Please check installation.")
            return
        
        try:
            # Find the main app object
            main_app = getattr(self.agents_tab, "main_app", None) or self._get_main_app()
            # Resolve the currently loaded model from common attribute names
            current_model = self._resolve_current_model(main_app)
            if current_model is None:
                messagebox.showwarning("No Model", "Please load a model first before starting the Dark Agent.")
                return

            # Create client adapter for the existing model
            self.client = ExistingRuntimeClient(current_model)
            
            # Get configuration from UI and spy
            agent_name = "Dhal"  # Fixed agent name
            system_prompt = self.agents_tab.hal_system_var.get() or f"You are {agent_name}, an advanced AI assistant integrated into DarkHal 2.0."
            model_name = (spy.get_model_name() or self.agents_tab.hal_model_var.get() or "local-llm")

            # Create agent object; prefer factory if available
            created = False
            try:
                from agent_dhal.hal import create_dhal  # factory supports (name, system_message, model, model_client)
                self.agent = create_dhal(name=agent_name, system_message=system_prompt, model=model_name, model_client=self.client)
                created = True
            except Exception:
                try:
                    from agent_dhal.hal import DhalConfig, Dhal as DarkAgentCtor
                    cfg = DhalConfig(name=agent_name, system_message=system_prompt, model=model_name)
                    self.agent = DarkAgentCtor(cfg, self.client)
                    created = True
                except Exception:
                    # Last fallback: legacy constructor (may not work on all versions)
                    try:
                        self.agent = DarkAgent(self.client)
                        created = True
                    except Exception:
                        created = False

            # Apply runtime configuration: let the model choose its own token/context defaults.
            try:
                tools_cfg = {name: var.get() for name, var in getattr(self.agents_tab, "hal_tools", {}).items()}
            except Exception:
                tools_cfg = {}

            if created and hasattr(self.agent, "update_config"):
                # Only set what is explicitly from UI that doesn't override model token defaults
                self.agent.update_config(system_message=system_prompt, tools=tools_cfg)

            # Start the agent runtime if start method exists
            if created and hasattr(self.agent, "start_dhal"):
                self.agent.start_dhal()
            
            self.is_running = True
            
            # Update UI
            self.ui_queue.put(("set_status", f"{agent_name} Status: Running"))
            self.ui_queue.put(("toggle_buttons", True))
            self.ui_queue.put(("append_text", f"\\n{agent_name} agent started successfully!\\n"))
            self.ui_queue.put(("append_text", f"System: {system_prompt}\\n\\n"))
            
        except Exception as e:
            messagebox.showerror("Start Error", f"Failed to start Dark Agent: {str(e)}")
            self.ui_queue.put(("set_status", "Dark Agent Status: Error"))
    
    def stop_agent(self):
        """Stop the Dark Agent."""
        try:
            if self.agent:
                self.agent.shutdown()
                self.agent = None
            
            if self.client:
                self.client = None
            
            self.is_running = False
            
            # Update UI
            agent_name = "Dhal"
            self.ui_queue.put(("set_status", f"{agent_name} Status: Stopped"))
            self.ui_queue.put(("toggle_buttons", False))
            self.ui_queue.put(("append_text", f"\\n{agent_name} agent stopped.\\n\\n"))
            
        except Exception as e:
            messagebox.showerror("Stop Error", f"Failed to stop Dark Agent: {str(e)}")
    
    def send_message(self):
        """Send message to Dark Agent."""
        if not self.is_running or not self.agent:
            messagebox.showwarning("Agent Not Running", "Please start the Dark Agent first.")
            return
        
        try:
            message = self.agents_tab.hal_input_var.get().strip()
            if not message:
                return
            
            # Clear input
            self.agents_tab.hal_input_var.set("")
            
            # Add user message to chat
            self.ui_queue.put(("append_text", f"User: {message}\\n"))
            
            # Define callback functions
            def on_token(request_id: str, delta: str):
                self.ui_queue.put(("append_text", delta))
            
            def on_complete(request_id: str, full_text: str, metadata: Dict[str, Any]):
                self.ui_queue.put(("append_text", "\\n\\n"))
            
            def on_error(request_id: str, error: Exception):
                self.ui_queue.put(("append_text", f"\\nError: {str(error)}\\n\\n"))
            
            # Add assistant prefix
            agent_name = "Dhal"
            self.ui_queue.put(("append_text", f"{agent_name}: "))
            
            # Send message to agent (non-blocking)
            self.agent.send_dhal_message(
                self.conversation_id,
                message,
                stream=True,
                on_token=on_token,
                on_complete=on_complete,
                on_error=on_error
            )
            
        except Exception as e:
            messagebox.showerror("Message Error", f"Failed to send message: {str(e)}")
            self.ui_queue.put(("append_text", f"\\nError: {str(e)}\\n\\n"))
    
    def reset_conversation(self):
        """Reset the agent conversation."""
        try:
            if self.agent:
                self.agent.reset_conversation(self.conversation_id)
            
            # Clear chat output
            self.agents_tab.hal_output.delete(1.0, "end")
            
            agent_name = "Dhal"
            self.ui_queue.put(("append_text", f"{agent_name} conversation reset.\\n\\n"))
            
        except Exception as e:
            messagebox.showerror("Reset Error", f"Failed to reset conversation: {str(e)}")
    
    def save_config(self):
        """Save agent configuration."""
        try:
            config = {
                "agent_name": "Dhal",
                "model": self.agents_tab.hal_model_var.get(),
                "system_message": self.agents_tab.hal_system_var.get(),
                "temperature": self.agents_tab.hal_temp_var.get(),
                "tools": {name: var.get() for name, var in self.agents_tab.hal_tools.items()}
            }
            
            config_file = Path("agent_dhal_config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.ui_queue.put(("append_text", f"Configuration saved to {config_file}\\n"))
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration: {str(e)}")
    
    def load_config(self):
        """Load agent configuration."""
        try:
            config_file = Path("agent_dhal_config.json")
            if not config_file.exists():
                messagebox.showinfo("No Config", "No saved configuration found.")
                return
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Apply configuration to UI
            if hasattr(self.agents_tab, "hal_name_var"):
                self.agents_tab.hal_name_var.set(config.get("agent_name", "Dhal"))
            self.agents_tab.hal_model_var.set(config.get("model", "local-llm"))
            self.agents_tab.hal_system_var.set(config.get("system_message", "You are Dhal, an advanced AI assistant."))
            self.agents_tab.hal_temp_var.set(config.get("temperature", "0.7"))
            
            # Apply tool settings
            tools_config = config.get("tools", {})
            for name, var in self.agents_tab.hal_tools.items():
                var.set(tools_config.get(name, True))
            
            self.ui_queue.put(("append_text", f"Configuration loaded from {config_file}\\n"))
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration: {str(e)}")
    
    def _resolve_current_model(self, main_app) -> Optional[Any]:
        """
        Try several common attribute names to find the loaded model on the main app/controller.
        Falls back to __spy if no attribute is found.
        Returns the model object or None.
        """
        # First, try the spy announcer if it has a model cached
        try:
            m = spy.get_model()
            if m is not None:
                return m
        except Exception:
            pass

        if main_app is None:
            return None
        candidate_attrs = [
            "current_model",
            "model",
            "llm_model",
            "loaded_model",
            "runtime_model",
        ]
        for attr in candidate_attrs:
            try:
                value = getattr(main_app, attr, None)
                if value is not None:
                    return value
            except Exception:
                continue
        return None

    def _get_main_app(self):
        """Get reference to main application."""
        try:
            # Navigate up the widget hierarchy to find the main app
            parent = self.agents_tab.parent
            while parent and not hasattr(parent, 'current_model'):
                parent = getattr(parent, 'master', None) or getattr(parent, 'parent', None)
                if hasattr(parent, 'winfo_toplevel'):
                    toplevel = parent.winfo_toplevel()
                    if hasattr(toplevel, 'current_model'):
                        return toplevel
            return parent
        except Exception:
            return None


# Legacy class name for compatibility
HALAgentIntegration = DhalAgentIntegration