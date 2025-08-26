#!/usr/bin/env python3
"""
Fine Tune Tab for DarkHal 2.0

Model fine-tuning interface for training and customizing AI models.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class FineTuneTab:
    """Fine-tuning interface for training and customizing AI models."""
    
    def __init__(self, parent: ttk.Frame, settings_manager):
        self.parent = parent
        self.settings = settings_manager
        self.current_model = None
        self.training_in_progress = False
        
        # Create main frame
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create fine-tuning interface
        self._create_finetune_interface()
    
    def _create_finetune_interface(self):
        """Create the main fine-tuning interface."""
        
        # Model Selection Frame
        model_frame = ttk.LabelFrame(self.main_frame, text="Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Base model selection
        ttk.Label(model_frame, text="Base Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.base_model_var = tk.StringVar()
        self.base_model_entry = ttk.Entry(model_frame, textvariable=self.base_model_var, width=50)
        self.base_model_entry.grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="Browse", command=self._browse_base_model).grid(row=0, column=2, padx=5)
        
        # Output model name
        ttk.Label(model_frame, text="Output Model Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_model_var = tk.StringVar(value="my-finetuned-model")
        ttk.Entry(model_frame, textvariable=self.output_model_var, width=50).grid(row=1, column=1, padx=5)
        
        # Training Data Frame
        data_frame = ttk.LabelFrame(self.main_frame, text="Training Data", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Dataset file
        ttk.Label(data_frame, text="Dataset File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.dataset_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Browse", command=self._browse_dataset).grid(row=0, column=2, padx=5)
        
        # Dataset format
        ttk.Label(data_frame, text="Dataset Format:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.format_var = tk.StringVar(value="alpaca")
        format_combo = ttk.Combobox(data_frame, textvariable=self.format_var,
                                   values=["alpaca", "sharegpt", "completion", "chat", "custom"],
                                   state="readonly", width=20)
        format_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Training split
        ttk.Label(data_frame, text="Train/Val Split:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.split_var = tk.StringVar(value="90/10")
        ttk.Combobox(data_frame, textvariable=self.split_var,
                    values=["80/20", "90/10", "95/5", "100/0"],
                    state="readonly", width=20).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Training Parameters Frame
        params_frame = ttk.LabelFrame(self.main_frame, text="Training Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create two columns for parameters
        left_params = ttk.Frame(params_frame)
        left_params.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_params = ttk.Frame(params_frame)
        right_params.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left column parameters
        ttk.Label(left_params, text="Training Method:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar(value="LoRA")
        ttk.Combobox(left_params, textvariable=self.method_var,
                    values=["LoRA", "QLoRA", "Full Fine-tune", "PEFT"],
                    state="readonly", width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(left_params, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.StringVar(value="3")
        ttk.Spinbox(left_params, from_=1, to=100, textvariable=self.epochs_var, width=15).grid(row=1, column=1, padx=5)
        
        ttk.Label(left_params, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.batch_var = tk.StringVar(value="4")
        ttk.Spinbox(left_params, from_=1, to=64, textvariable=self.batch_var, width=15).grid(row=2, column=1, padx=5)
        
        ttk.Label(left_params, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.StringVar(value="2e-4")
        ttk.Entry(left_params, textvariable=self.lr_var, width=15).grid(row=3, column=1, padx=5)
        
        # Right column parameters
        ttk.Label(right_params, text="LoRA Rank:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.lora_rank_var = tk.StringVar(value="8")
        ttk.Spinbox(right_params, from_=4, to=128, textvariable=self.lora_rank_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(right_params, text="LoRA Alpha:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lora_alpha_var = tk.StringVar(value="16")
        ttk.Spinbox(right_params, from_=8, to=256, textvariable=self.lora_alpha_var, width=15).grid(row=1, column=1, padx=5)
        
        ttk.Label(right_params, text="Max Length:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_length_var = tk.StringVar(value="512")
        ttk.Spinbox(right_params, from_=128, to=4096, increment=128, textvariable=self.max_length_var, width=15).grid(row=2, column=1, padx=5)
        
        ttk.Label(right_params, text="Warmup Steps:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.warmup_var = tk.StringVar(value="100")
        ttk.Spinbox(right_params, from_=0, to=1000, textvariable=self.warmup_var, width=15).grid(row=3, column=1, padx=5)
        
        # Hardware Settings Frame
        hardware_frame = ttk.LabelFrame(self.main_frame, text="Hardware Settings", padding=10)
        hardware_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(hardware_frame, text="Device:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.device_var = tk.StringVar(value="cuda")
        ttk.Combobox(hardware_frame, textvariable=self.device_var,
                    values=["cuda", "cpu", "mps"],
                    state="readonly", width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(hardware_frame, text="Mixed Precision:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.mixed_precision_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hardware_frame, variable=self.mixed_precision_var).grid(row=0, column=3)
        
        ttk.Label(hardware_frame, text="Gradient Checkpointing:").grid(row=0, column=4, sticky=tk.W, padx=(20, 5))
        self.grad_checkpoint_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hardware_frame, variable=self.grad_checkpoint_var).grid(row=0, column=5)
        
        # Control Buttons Frame
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Start Training", command=self._start_training,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Training", command=self._stop_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Config", command=self._save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Config", command=self._load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Validate Dataset", command=self._validate_dataset).pack(side=tk.LEFT, padx=5)
        
        # Progress Frame
        progress_frame = ttk.LabelFrame(self.main_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(progress_frame, text="Ready to start training", foreground="green")
        self.status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Training log
        ttk.Label(progress_frame, text="Training Log:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=10, width=80, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def _browse_base_model(self):
        """Browse for base model file."""
        filename = filedialog.askopenfilename(
            title="Select Base Model",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if filename:
            self.base_model_var.set(filename)
            self._log(f"Selected base model: {filename}")
    
    def _browse_dataset(self):
        """Browse for dataset file."""
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[
                ("JSON files", "*.json"),
                ("JSONL files", "*.jsonl"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.dataset_var.set(filename)
            self._log(f"Selected dataset: {filename}")
    
    def _start_training(self):
        """Start the fine-tuning process."""
        if self.training_in_progress:
            messagebox.showwarning("Training in Progress", "Training is already in progress!")
            return
        
        # Validate inputs
        if not self.base_model_var.get():
            messagebox.showerror("Error", "Please select a base model")
            return
        
        if not self.dataset_var.get():
            messagebox.showerror("Error", "Please select a dataset")
            return
        
        self.training_in_progress = True
        self.status_label.config(text="Training in progress...", foreground="orange")
        self._log("=" * 50)
        self._log("Starting fine-tuning process...")
        self._log(f"Base Model: {self.base_model_var.get()}")
        self._log(f"Dataset: {self.dataset_var.get()}")
        self._log(f"Method: {self.method_var.get()}")
        self._log(f"Epochs: {self.epochs_var.get()}")
        self._log("=" * 50)
        
        # TODO: Implement actual training logic
        messagebox.showinfo("Coming Soon", "Fine-tuning functionality will be implemented soon!")
        
        self.training_in_progress = False
        self.status_label.config(text="Training complete (stub)", foreground="green")
    
    def _stop_training(self):
        """Stop the training process."""
        if not self.training_in_progress:
            messagebox.showinfo("No Training", "No training in progress")
            return
        
        self.training_in_progress = False
        self.status_label.config(text="Training stopped", foreground="red")
        self._log("Training stopped by user")
    
    def _save_config(self):
        """Save training configuration to file."""
        config = {
            "base_model": self.base_model_var.get(),
            "output_model": self.output_model_var.get(),
            "dataset": self.dataset_var.get(),
            "format": self.format_var.get(),
            "split": self.split_var.get(),
            "method": self.method_var.get(),
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_var.get(),
            "learning_rate": self.lr_var.get(),
            "lora_rank": self.lora_rank_var.get(),
            "lora_alpha": self.lora_alpha_var.get(),
            "max_length": self.max_length_var.get(),
            "warmup_steps": self.warmup_var.get(),
            "device": self.device_var.get(),
            "mixed_precision": self.mixed_precision_var.get(),
            "gradient_checkpointing": self.grad_checkpoint_var.get()
        }
        
        filename = filedialog.asksaveasfilename(
            title="Save Training Config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            self._log(f"Config saved to {filename}")
            messagebox.showinfo("Saved", f"Configuration saved to {filename}")
    
    def _load_config(self):
        """Load training configuration from file."""
        filename = filedialog.askopenfilename(
            title="Load Training Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Load values from config
                self.base_model_var.set(config.get("base_model", ""))
                self.output_model_var.set(config.get("output_model", "my-finetuned-model"))
                self.dataset_var.set(config.get("dataset", ""))
                self.format_var.set(config.get("format", "alpaca"))
                self.split_var.set(config.get("split", "90/10"))
                self.method_var.set(config.get("method", "LoRA"))
                self.epochs_var.set(config.get("epochs", "3"))
                self.batch_var.set(config.get("batch_size", "4"))
                self.lr_var.set(config.get("learning_rate", "2e-4"))
                self.lora_rank_var.set(config.get("lora_rank", "8"))
                self.lora_alpha_var.set(config.get("lora_alpha", "16"))
                self.max_length_var.set(config.get("max_length", "512"))
                self.warmup_var.set(config.get("warmup_steps", "100"))
                self.device_var.set(config.get("device", "cuda"))
                self.mixed_precision_var.set(config.get("mixed_precision", True))
                self.grad_checkpoint_var.set(config.get("gradient_checkpointing", True))
                
                self._log(f"Config loaded from {filename}")
                messagebox.showinfo("Loaded", f"Configuration loaded from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")
    
    def _validate_dataset(self):
        """Validate the selected dataset."""
        if not self.dataset_var.get():
            messagebox.showerror("Error", "Please select a dataset first")
            return
        
        dataset_path = self.dataset_var.get()
        if not os.path.exists(dataset_path):
            messagebox.showerror("Error", f"Dataset file not found: {dataset_path}")
            return
        
        # TODO: Implement actual dataset validation
        self._log(f"Validating dataset: {dataset_path}")
        self._log("Dataset validation (stub) - would check format, size, etc.")
        messagebox.showinfo("Validation", "Dataset validation complete (stub)")
    
    def _log(self, message: str):
        """Add message to training log."""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def set_model(self, model_path: Optional[str]):
        """Set the current model for fine-tuning."""
        self.current_model = model_path
        if model_path:
            self.base_model_var.set(model_path)
            self._log(f"Model selected: {Path(model_path).name}")