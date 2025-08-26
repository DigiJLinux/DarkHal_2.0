"""
Chat Template Management System

This module provides chat template management for different model formats,
allowing users to apply proper conversation formatting for optimal model performance.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


@dataclass
class ChatTemplate:
    """Represents a chat template configuration"""
    name: str
    description: str
    system_prefix: str = ""
    system_suffix: str = ""
    user_prefix: str = ""
    user_suffix: str = ""
    assistant_prefix: str = ""
    assistant_suffix: str = ""
    turn_separator: str = ""
    eos_token: str = ""
    bos_token: str = ""
    stop_tokens: List[str] = None
    add_generation_prompt: bool = True
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = []


class ChatTemplateManager:
    """Manages chat templates with JSON persistence"""
    
    def __init__(self, templates_file: str = "chat_templates.json"):
        self.templates_file = templates_file
        self.templates: Dict[str, ChatTemplate] = {}
        self._load_templates()
        self._ensure_default_templates()
    
    def _load_templates(self):
        """Load templates from JSON file"""
        if os.path.exists(self.templates_file):
            try:
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, template_data in data.items():
                        self.templates[name] = ChatTemplate(**template_data)
            except Exception as e:
                print(f"Error loading chat templates: {e}")
    
    def _save_templates(self):
        """Save templates to JSON file"""
        try:
            data = {name: asdict(template) for name, template in self.templates.items()}
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat templates: {e}")
    
    def _ensure_default_templates(self):
        """Ensure default templates exist"""
        if "Llama-3.1-Instruct" not in self.templates:
            self.templates["Llama-3.1-Instruct"] = ChatTemplate(
                name="Llama-3.1-Instruct",
                description="Official Llama 3.1 Instruct chat template with proper headers and EOT tokens",
                bos_token="<|begin_of_text|>",
                system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
                system_suffix="<|eot_id|>",
                user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
                user_suffix="<|eot_id|>",
                assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
                assistant_suffix="",  # Model generates until <|eot_id|>
                eos_token="<|eot_id|>",
                stop_tokens=["<|eot_id|>"],
                add_generation_prompt=True
            )
            self._save_templates()
    
    def get_template_names(self) -> List[str]:
        """Get list of all template names"""
        return list(self.templates.keys())
    
    def get_template(self, name: str) -> Optional[ChatTemplate]:
        """Get a template by name"""
        return self.templates.get(name)
    
    def add_template(self, template: ChatTemplate) -> bool:
        """Add a new template"""
        if template.name in self.templates:
            return False  # Template already exists
        self.templates[template.name] = template
        self._save_templates()
        return True
    
    def update_template(self, template: ChatTemplate) -> None:
        """Update an existing template"""
        self.templates[template.name] = template
        self._save_templates()
    
    def delete_template(self, name: str) -> bool:
        """Delete a template by name"""
        if name in self.templates:
            del self.templates[name]
            self._save_templates()
            return True
        return False
    
    def format_conversation(self, template_name: str, messages: List[Dict[str, str]], 
                          add_generation_prompt: bool = True) -> str:
        """Format a conversation using the specified template"""
        template = self.get_template(template_name)
        if not template:
            # Fallback to simple User:/Assistant: format
            return self._format_simple(messages, add_generation_prompt)
        
        result = []
        
        # Add BOS token if specified
        if template.bos_token:
            result.append(template.bos_token)
        
        # System messages are handled by the calling code, so we don't need to add default ones here
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                if template.system_prefix or template.system_suffix:
                    result.append(f"{template.system_prefix}{content}{template.system_suffix}")
                else:
                    result.append(content)
            elif role == "user":
                result.append(f"{template.user_prefix}{content}{template.user_suffix}")
            elif role == "assistant":
                result.append(f"{template.assistant_prefix}{content}{template.assistant_suffix}")
            
            # Add turn separator if specified (but not after the last message if we're adding generation prompt)
            if template.turn_separator and not (add_generation_prompt and message == messages[-1]):
                result.append(template.turn_separator)
        
        # Add generation prompt for assistant response
        if add_generation_prompt and template.add_generation_prompt:
            result.append(template.assistant_prefix)
        
        return "".join(result)
    
    def _format_simple(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Simple fallback formatting with User:/Assistant: labels"""
        result = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                result.append(f"System: {content}")
            elif role == "user":
                result.append(f"User: {content}")
            elif role == "assistant":
                result.append(f"Assistant: {content}")
        
        if add_generation_prompt:
            result.append("Assistant:")
        
        return "\n".join(result)
    
    def get_stop_tokens(self, template_name: str) -> List[str]:
        """Get stop tokens for a template"""
        template = self.get_template(template_name)
        if template and template.stop_tokens:
            return template.stop_tokens
        return []


class ChatTemplateDialog:
    """Dialog for creating/editing chat templates"""
    
    def __init__(self, parent: tk.Tk, template: ChatTemplate = None):
        self.parent = parent
        self.template = template
        self.result = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Chat Template Editor")
        self.dialog.geometry("600x700")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._build_ui()
        
        if template:
            self._load_template_data()
        
        self._center_window()
    
    def _build_ui(self):
        """Build the template editor UI"""
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Basic Info
        info_frame = ttk.LabelFrame(main_frame, text="Template Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.name_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.name_var, width=40).grid(row=0, column=1, sticky=tk.EW, pady=2)
        
        ttk.Label(info_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.desc_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.desc_var, width=40).grid(row=1, column=1, sticky=tk.EW, pady=2)
        
        info_frame.grid_columnconfigure(1, weight=1)
        
        # Template Components
        components_frame = ttk.LabelFrame(main_frame, text="Template Components", padding=10)
        components_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create entry fields for all template components
        self.template_vars = {}
        components = [
            ("BOS Token", "bos_token"),
            ("System Prefix", "system_prefix"),
            ("System Suffix", "system_suffix"),
            ("User Prefix", "user_prefix"),
            ("User Suffix", "user_suffix"),
            ("Assistant Prefix", "assistant_prefix"),
            ("Assistant Suffix", "assistant_suffix"),
            ("Turn Separator", "turn_separator"),
            ("EOS Token", "eos_token"),
        ]
        
        for i, (label, key) in enumerate(components):
            ttk.Label(components_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(components_frame, textvariable=var, width=50)
            entry.grid(row=i, column=1, sticky=tk.EW, pady=2, padx=(5, 0))
            self.template_vars[key] = var
        
        # Stop tokens (text area)
        ttk.Label(components_frame, text="Stop Tokens (one per line):").grid(row=len(components), column=0, sticky=tk.W, pady=2)
        self.stop_tokens_text = tk.Text(components_frame, height=4, width=50)
        self.stop_tokens_text.grid(row=len(components), column=1, sticky=tk.EW, pady=2, padx=(5, 0))
        
        # Add generation prompt checkbox
        self.add_gen_prompt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(components_frame, text="Add generation prompt", 
                       variable=self.add_gen_prompt_var).grid(row=len(components)+1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        components_frame.grid_columnconfigure(1, weight=1)
        
        # Preview
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=10)
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(preview_frame, text="Generate Preview", command=self._generate_preview).pack(side=tk.LEFT)
        
        self.preview_text = tk.Text(preview_frame, height=6, wrap=tk.WORD)
        self.preview_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Save", command=self._save_template).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT)
    
    def _load_template_data(self):
        """Load existing template data into the form"""
        if not self.template:
            return
            
        self.name_var.set(self.template.name)
        self.desc_var.set(self.template.description)
        
        for key, var in self.template_vars.items():
            value = getattr(self.template, key, "")
            var.set(value)
        
        # Load stop tokens
        if self.template.stop_tokens:
            self.stop_tokens_text.insert('1.0', '\n'.join(self.template.stop_tokens))
        
        self.add_gen_prompt_var.set(self.template.add_generation_prompt)
    
    def _generate_preview(self):
        """Generate a preview of the template formatting"""
        try:
            # Create a temporary template from current form data
            template = self._create_template_from_form()
            
            # Sample conversation
            sample_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
                {"role": "user", "content": "Can you explain quantum physics?"}
            ]
            
            # Create temporary manager to format
            temp_manager = ChatTemplateManager()
            temp_manager.templates["preview"] = template
            
            formatted = temp_manager.format_conversation("preview", sample_messages, True)
            
            self.preview_text.delete('1.0', tk.END)
            self.preview_text.insert('1.0', formatted)
            
        except Exception as e:
            self.preview_text.delete('1.0', tk.END)
            self.preview_text.insert('1.0', f"Error generating preview: {e}")
    
    def _create_template_from_form(self) -> ChatTemplate:
        """Create a ChatTemplate object from form data"""
        # Get stop tokens
        stop_tokens_text = self.stop_tokens_text.get('1.0', tk.END).strip()
        stop_tokens = [token.strip() for token in stop_tokens_text.split('\n') if token.strip()]
        
        return ChatTemplate(
            name=self.name_var.get().strip(),
            description=self.desc_var.get().strip(),
            bos_token=self.template_vars["bos_token"].get(),
            system_prefix=self.template_vars["system_prefix"].get(),
            system_suffix=self.template_vars["system_suffix"].get(),
            user_prefix=self.template_vars["user_prefix"].get(),
            user_suffix=self.template_vars["user_suffix"].get(),
            assistant_prefix=self.template_vars["assistant_prefix"].get(),
            assistant_suffix=self.template_vars["assistant_suffix"].get(),
            turn_separator=self.template_vars["turn_separator"].get(),
            eos_token=self.template_vars["eos_token"].get(),
            stop_tokens=stop_tokens,
            add_generation_prompt=self.add_gen_prompt_var.get()
        )
    
    def _save_template(self):
        """Save the template"""
        try:
            template = self._create_template_from_form()
            
            # Validate required fields
            if not template.name:
                messagebox.showerror("Error", "Template name is required")
                return
            
            self.result = template
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving template: {e}")
    
    def _center_window(self):
        """Center the dialog on the parent window"""
        self.dialog.update_idletasks()
        
        # Get parent position
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")


# Global template manager instance
_template_manager = None

def get_template_manager() -> ChatTemplateManager:
    """Get the global chat template manager"""
    global _template_manager
    if _template_manager is None:
        _template_manager = ChatTemplateManager()
    return _template_manager