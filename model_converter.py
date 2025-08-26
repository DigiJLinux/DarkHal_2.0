#!/usr/bin/env python3
"""
Model Conversion and Editing Tools for DarkHal 2.0

Provides comprehensive model conversion between formats, quantization options,
and model editing capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import os
import sys
import json
import subprocess
import threading
import queue
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import tempfile


class ModelConverter:
    """Handles model conversion operations."""
    
    SUPPORTED_FORMATS = {
        'gguf': 'GGUF (llama.cpp)',
        'safetensors': 'SafeTensors (HuggingFace)',
        'bin': 'PyTorch Binary',
        'pt': 'PyTorch',
        'pth': 'PyTorch State Dict',
        'onnx': 'ONNX',
        'tflite': 'TensorFlow Lite',
        'h5': 'Keras/HDF5'
    }
    
    QUANTIZATION_TYPES = {
        'q4_0': 'Q4_0 - 4-bit (smallest, lower quality)',
        'q4_1': 'Q4_1 - 4-bit (small, better than q4_0)',
        'q4_k_m': 'Q4_K_M - 4-bit (medium, recommended)',
        'q4_k_s': 'Q4_K_S - 4-bit (small)',
        'q5_0': 'Q5_0 - 5-bit',
        'q5_1': 'Q5_1 - 5-bit (better than q5_0)',
        'q5_k_m': 'Q5_K_M - 5-bit (medium, recommended)',
        'q5_k_s': 'Q5_K_S - 5-bit (small)',
        'q6_k': 'Q6_K - 6-bit (good quality/size ratio)',
        'q8_0': 'Q8_0 - 8-bit (high quality)',
        'f16': 'FP16 - 16-bit float',
        'f32': 'FP32 - 32-bit float (original)'
    }
    
    def __init__(self):
        self.conversion_queue = queue.Queue()
        self.current_process = None
        
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a model file."""
        path = Path(model_path)
        if not path.exists():
            return None
        
        info = {
            'name': path.stem,
            'path': str(path),
            'format': path.suffix.lower().lstrip('.'),
            'size': path.stat().st_size,
            'size_mb': path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(path.stat().st_mtime)
        }
        
        # Try to extract additional metadata
        if info['format'] == 'gguf':
            info.update(self._get_gguf_info(path))
        elif info['format'] in ['safetensors', 'bin']:
            info.update(self._get_hf_info(path))
        
        return info
    
    def _get_gguf_info(self, path: Path) -> Dict[str, Any]:
        """Extract GGUF model information."""
        info = {}
        try:
            # Try to use llama-cpp-python if available
            from llama_cpp import Llama
            # This would require actually loading the model which is expensive
            # For now, extract from filename
            name = path.stem.lower()
            
            # Detect quantization
            for q_type in self.QUANTIZATION_TYPES.keys():
                if q_type in name:
                    info['quantization'] = q_type
                    break
            
            # Detect model size
            import re
            size_match = re.search(r'(\d+)b', name, re.IGNORECASE)
            if size_match:
                info['parameters'] = f"{size_match.group(1)}B"
            
        except Exception:
            pass
        
        return info
    
    def _get_hf_info(self, path: Path) -> Dict[str, Any]:
        """Extract HuggingFace model information."""
        info = {}
        try:
            # Look for config.json in the same directory
            config_path = path.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    info['model_type'] = config.get('model_type', 'unknown')
                    info['architectures'] = config.get('architectures', [])
                    info['vocab_size'] = config.get('vocab_size', 0)
                    
                    # Calculate parameters if possible
                    if 'hidden_size' in config and 'num_hidden_layers' in config:
                        hidden = config['hidden_size']
                        layers = config['num_hidden_layers']
                        vocab = config.get('vocab_size', 0)
                        # Rough parameter estimation
                        params = (hidden * hidden * 4 * layers + vocab * hidden) / 1e9
                        info['parameters'] = f"{params:.1f}B"
        except Exception:
            pass
        
        return info
    
    def convert_to_gguf(self, input_path: str, output_path: str, 
                       quantization: str = 'q4_k_m',
                       progress_callback: Optional[callable] = None) -> bool:
        """Convert a model to GGUF format."""
        try:
            input_format = Path(input_path).suffix.lower().lstrip('.')
            
            if input_format == 'gguf':
                # Already GGUF, just quantize if needed
                return self.quantize_gguf(input_path, output_path, quantization, progress_callback)
            
            # For HuggingFace models, use convert.py from llama.cpp
            if input_format in ['safetensors', 'bin']:
                return self._convert_hf_to_gguf(input_path, output_path, quantization, progress_callback)
            
            # For other formats, try generic conversion
            return self._generic_convert(input_path, output_path, 'gguf', progress_callback)
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}", 100, "error")
            return False
    
    def quantize_gguf(self, input_path: str, output_path: str,
                     quantization: str = 'q4_k_m',
                     progress_callback: Optional[callable] = None) -> bool:
        """Quantize a GGUF model."""
        try:
            # Look for quantize executable
            quantize_exe = self._find_quantize_executable()
            if not quantize_exe:
                if progress_callback:
                    progress_callback("Quantize executable not found", 100, "error")
                return False
            
            # Run quantization
            cmd = [str(quantize_exe), input_path, output_path, quantization]
            
            if progress_callback:
                progress_callback("Starting quantization...", 0, "info")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.current_process = process
            
            # Monitor progress
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                    
                if progress_callback:
                    # Parse progress from output
                    if "%" in line:
                        try:
                            import re
                            match = re.search(r'(\d+)%', line)
                            if match:
                                percent = int(match.group(1))
                                progress_callback(line.strip(), percent, "info")
                        except:
                            progress_callback(line.strip(), -1, "info")
                    else:
                        progress_callback(line.strip(), -1, "info")
            
            process.wait()
            
            if process.returncode == 0:
                if progress_callback:
                    progress_callback("Quantization complete!", 100, "success")
                return True
            else:
                if progress_callback:
                    progress_callback("Quantization failed", 100, "error")
                return False
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}", 100, "error")
            return False
        finally:
            self.current_process = None
    
    def _find_quantize_executable(self) -> Optional[Path]:
        """Find the quantize executable."""
        # Common locations
        locations = [
            Path("llama-cpp-python/vendor/llama.cpp/quantize"),
            Path("llama.cpp/quantize"),
            Path("bin/quantize"),
            Path("quantize"),
            Path("quantize.exe")
        ]
        
        for loc in locations:
            if loc.exists():
                return loc
            
            # Check in PATH
            import shutil
            exe = shutil.which("quantize")
            if exe:
                return Path(exe)
        
        return None
    
    def _convert_hf_to_gguf(self, input_path: str, output_path: str,
                           quantization: str,
                           progress_callback: Optional[callable] = None) -> bool:
        """Convert HuggingFace model to GGUF."""
        try:
            # Look for convert.py script
            convert_script = self._find_convert_script()
            if not convert_script:
                if progress_callback:
                    progress_callback("Convert script not found", 100, "error")
                return False
            
            # First convert to FP16 GGUF
            temp_gguf = output_path.replace('.gguf', '_fp16.gguf')
            
            cmd = [
                sys.executable,
                str(convert_script),
                str(Path(input_path).parent),  # Model directory
                "--outfile", temp_gguf,
                "--outtype", "f16"
            ]
            
            if progress_callback:
                progress_callback("Converting to GGUF format...", 0, "info")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                if progress_callback:
                    progress_callback(f"Conversion failed: {stderr}", 100, "error")
                return False
            
            if progress_callback:
                progress_callback("Conversion complete, quantizing...", 50, "info")
            
            # Then quantize if needed
            if quantization != 'f16':
                result = self.quantize_gguf(temp_gguf, output_path, quantization, progress_callback)
                # Clean up temp file
                try:
                    os.remove(temp_gguf)
                except:
                    pass
                return result
            else:
                # Just rename
                shutil.move(temp_gguf, output_path)
                if progress_callback:
                    progress_callback("Conversion complete!", 100, "success")
                return True
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}", 100, "error")
            return False
    
    def _find_convert_script(self) -> Optional[Path]:
        """Find the convert.py script."""
        locations = [
            Path("llama-cpp-python/vendor/llama.cpp/convert.py"),
            Path("llama.cpp/convert.py"),
            Path("scripts/convert.py"),
            Path("convert.py")
        ]
        
        for loc in locations:
            if loc.exists():
                return loc
        
        return None
    
    def _generic_convert(self, input_path: str, output_path: str,
                        target_format: str,
                        progress_callback: Optional[callable] = None) -> bool:
        """Generic conversion using available tools."""
        # This would use tools like ONNX converters, TensorFlow converters, etc.
        # For now, return False as not implemented
        if progress_callback:
            progress_callback(f"Conversion to {target_format} not yet implemented", 100, "error")
        return False
    
    def merge_lora(self, base_model: str, lora_path: str, output_path: str,
                  progress_callback: Optional[callable] = None) -> bool:
        """Merge a LoRA adapter into a base model."""
        try:
            # This would use a LoRA merging tool
            # For now, simplified implementation
            if progress_callback:
                progress_callback("LoRA merging not yet implemented", 100, "error")
            return False
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}", 100, "error")
            return False
    
    def split_model(self, input_path: str, output_dir: str, 
                   num_shards: int = 2,
                   progress_callback: Optional[callable] = None) -> bool:
        """Split a model into multiple shards."""
        try:
            # This would split large models for easier distribution
            if progress_callback:
                progress_callback("Model splitting not yet implemented", 100, "error")
            return False
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}", 100, "error")
            return False


class ModelConverterTab:
    """Model conversion and editing tab for DarkHal 2.0."""
    
    def __init__(self, parent: ttk.Frame, settings_manager):
        self.parent = parent
        self.settings = settings_manager
        self.converter = ModelConverter()
        self.current_model = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the converter tab UI."""
        # Create paned window for split view
        paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Model Selection and Info
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Model Selection Frame
        select_frame = ttk.LabelFrame(left_frame, text="Model Selection", padding=10)
        select_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input model
        ttk.Label(select_frame, text="Input Model:").pack(anchor=tk.W)
        
        input_frame = ttk.Frame(select_frame)
        input_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.input_path_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_path_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(input_frame, text="Browse", command=self._browse_input).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(input_frame, text="Analyze", command=self._analyze_model).pack(side=tk.LEFT, padx=(5, 0))
        
        # Model Information Frame
        info_frame = ttk.LabelFrame(left_frame, text="Model Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Conversion Options Frame
        options_frame = ttk.LabelFrame(left_frame, text="Conversion Options", padding=10)
        options_frame.pack(fill=tk.X)
        
        # Output format
        ttk.Label(options_frame, text="Output Format:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.output_format_var = tk.StringVar(value="gguf")
        format_combo = ttk.Combobox(options_frame, textvariable=self.output_format_var,
                                   values=list(ModelConverter.SUPPORTED_FORMATS.keys()),
                                   state="readonly", width=20)
        format_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        format_combo.bind('<<ComboboxSelected>>', self._on_format_change)
        
        # Quantization (for GGUF)
        ttk.Label(options_frame, text="Quantization:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.quantization_var = tk.StringVar(value="q4_k_m")
        self.quant_combo = ttk.Combobox(options_frame, textvariable=self.quantization_var,
                                       values=list(ModelConverter.QUANTIZATION_TYPES.keys()),
                                       state="readonly", width=20)
        self.quant_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Output path
        ttk.Label(options_frame, text="Output Path:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        output_frame = ttk.Frame(options_frame)
        output_frame.grid(row=2, column=1, sticky=tk.W+tk.E, pady=5)
        
        self.output_path_var = tk.StringVar()
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var, width=30)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(output_frame, text="Browse", command=self._browse_output).pack(side=tk.LEFT, padx=(5, 0))
        
        # Conversion button
        self.convert_btn = ttk.Button(options_frame, text="Start Conversion",
                                     command=self._start_conversion)
        self.convert_btn.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        # Right panel - Advanced Tools and Progress
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Advanced Tools Frame
        tools_frame = ttk.LabelFrame(right_frame, text="Advanced Tools", padding=10)
        tools_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tool buttons
        tool_grid = ttk.Frame(tools_frame)
        tool_grid.pack(fill=tk.X)
        
        ttk.Button(tool_grid, text="Merge LoRA", command=self._open_lora_merger).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(tool_grid, text="Split Model", command=self._open_model_splitter).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(tool_grid, text="Optimize", command=self._open_optimizer).grid(row=0, column=2, padx=2, pady=2)
        
        ttk.Button(tool_grid, text="Batch Convert", command=self._open_batch_converter).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(tool_grid, text="Compare Models", command=self._open_model_compare).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(tool_grid, text="Edit Metadata", command=self._open_metadata_editor).grid(row=1, column=2, padx=2, pady=2)
        
        # Quantization Comparison
        compare_frame = ttk.LabelFrame(right_frame, text="Quantization Comparison", padding=10)
        compare_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Comparison table
        columns = ("Type", "Size", "Quality", "Speed")
        self.compare_tree = ttk.Treeview(compare_frame, columns=columns, show="headings", height=6)
        
        self.compare_tree.heading("Type", text="Type")
        self.compare_tree.heading("Size", text="Size")
        self.compare_tree.heading("Quality", text="Quality")
        self.compare_tree.heading("Speed", text="Speed")
        
        self.compare_tree.column("Type", width=80)
        self.compare_tree.column("Size", width=80)
        self.compare_tree.column("Quality", width=80)
        self.compare_tree.column("Speed", width=80)
        
        self.compare_tree.pack(fill=tk.X)
        
        # Populate comparison table
        self._populate_comparison()
        
        # Progress Frame
        progress_frame = ttk.LabelFrame(right_frame, text="Conversion Progress", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Progress log
        self.progress_text = scrolledtext.ScrolledText(progress_frame, height=10, wrap=tk.WORD,
                                                      bg="#1a1a1a", fg="#00ff88",
                                                      font=("Consolas", 9))
        self.progress_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(progress_frame)
        control_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.cancel_btn = ttk.Button(control_frame, text="Cancel", command=self._cancel_conversion,
                                    state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=5)
    
    def _populate_comparison(self):
        """Populate the quantization comparison table."""
        comparisons = [
            ("Q4_0", "~3.5GB", "★★☆☆☆", "★★★★★"),
            ("Q4_K_M", "~3.8GB", "★★★☆☆", "★★★★★"),
            ("Q5_K_M", "~4.5GB", "★★★★☆", "★★★★☆"),
            ("Q6_K", "~5.5GB", "★★★★☆", "★★★☆☆"),
            ("Q8_0", "~7GB", "★★★★★", "★★☆☆☆"),
            ("FP16", "~14GB", "★★★★★", "★☆☆☆☆"),
        ]
        
        for comp in comparisons:
            self.compare_tree.insert("", tk.END, values=comp)
    
    def _browse_input(self):
        """Browse for input model."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("All Models", "*.gguf;*.safetensors;*.bin;*.pt;*.pth;*.onnx"),
                ("GGUF", "*.gguf"),
                ("SafeTensors", "*.safetensors"),
                ("PyTorch", "*.bin;*.pt;*.pth"),
                ("ONNX", "*.onnx"),
                ("All Files", "*.*")
            ]
        )
        
        if filename:
            self.input_path_var.set(filename)
            self._analyze_model()
            
            # Auto-generate output path
            input_path = Path(filename)
            output_format = self.output_format_var.get()
            quantization = self.quantization_var.get()
            
            output_name = f"{input_path.stem}_{quantization}.{output_format}"
            output_path = input_path.parent / output_name
            self.output_path_var.set(str(output_path))
    
    def _browse_output(self):
        """Browse for output location."""
        filename = filedialog.asksaveasfilename(
            title="Save Converted Model As",
            defaultextension=f".{self.output_format_var.get()}",
            filetypes=[
                (f"{self.output_format_var.get().upper()}", f"*.{self.output_format_var.get()}"),
                ("All Files", "*.*")
            ]
        )
        
        if filename:
            self.output_path_var.set(filename)
    
    def _analyze_model(self):
        """Analyze the selected model."""
        model_path = self.input_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "Analyzing model...\n\n")
        
        # Get model info
        info = self.converter.get_model_info(model_path)
        if info:
            self.current_model = info
            
            # Display info
            self.info_text.insert(tk.END, f"Name: {info['name']}\n")
            self.info_text.insert(tk.END, f"Format: {info['format'].upper()}\n")
            self.info_text.insert(tk.END, f"Size: {info['size_mb']:.1f} MB\n")
            self.info_text.insert(tk.END, f"Modified: {info['modified'].strftime('%Y-%m-%d %H:%M')}\n")
            
            if 'quantization' in info:
                self.info_text.insert(tk.END, f"Quantization: {info['quantization']}\n")
            if 'parameters' in info:
                self.info_text.insert(tk.END, f"Parameters: {info['parameters']}\n")
            if 'model_type' in info:
                self.info_text.insert(tk.END, f"Model Type: {info['model_type']}\n")
            if 'architectures' in info:
                self.info_text.insert(tk.END, f"Architecture: {', '.join(info['architectures'])}\n")
            if 'vocab_size' in info:
                self.info_text.insert(tk.END, f"Vocab Size: {info['vocab_size']:,}\n")
    
    def _on_format_change(self, event=None):
        """Handle output format change."""
        format_type = self.output_format_var.get()
        
        # Enable/disable quantization based on format
        if format_type == 'gguf':
            self.quant_combo.config(state="readonly")
        else:
            self.quant_combo.config(state="disabled")
        
        # Update output path
        if self.input_path_var.get():
            input_path = Path(self.input_path_var.get())
            quantization = self.quantization_var.get() if format_type == 'gguf' else ''
            
            if quantization:
                output_name = f"{input_path.stem}_{quantization}.{format_type}"
            else:
                output_name = f"{input_path.stem}.{format_type}"
            
            output_path = input_path.parent / output_name
            self.output_path_var.set(str(output_path))
    
    def _start_conversion(self):
        """Start the conversion process."""
        input_path = self.input_path_var.get()
        output_path = self.output_path_var.get()
        
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid input model")
            return
        
        if not output_path:
            messagebox.showerror("Error", "Please specify an output path")
            return
        
        # Confirm overwrite if exists
        if os.path.exists(output_path):
            if not messagebox.askyesno("Confirm", f"Output file exists. Overwrite?\n{output_path}"):
                return
        
        # Disable UI
        self.convert_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        
        # Clear progress
        self.progress_var.set(0)
        self.progress_text.delete(1.0, tk.END)
        
        # Start conversion in thread
        threading.Thread(target=self._conversion_thread,
                        args=(input_path, output_path),
                        daemon=True).start()
    
    def _conversion_thread(self, input_path: str, output_path: str):
        """Run conversion in background thread."""
        def progress_callback(message: str, percent: int, level: str = "info"):
            # Update UI in main thread
            self.parent.after(0, self._update_progress, message, percent, level)
        
        try:
            output_format = self.output_format_var.get()
            
            if output_format == 'gguf':
                quantization = self.quantization_var.get()
                success = self.converter.convert_to_gguf(
                    input_path, output_path, quantization, progress_callback
                )
            else:
                # Other format conversions
                success = False
                progress_callback(f"Conversion to {output_format} not yet implemented", 100, "error")
            
            if success:
                self.parent.after(0, self._conversion_complete, True)
            else:
                self.parent.after(0, self._conversion_complete, False)
                
        except Exception as e:
            progress_callback(f"Error: {e}", 100, "error")
            self.parent.after(0, self._conversion_complete, False)
    
    def _update_progress(self, message: str, percent: int, level: str):
        """Update progress display."""
        # Update progress bar
        if percent >= 0:
            self.progress_var.set(percent)
        
        # Add to log
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color based on level
        if level == "error":
            self.progress_text.insert(tk.END, f"[{timestamp}] ERROR: {message}\n", "error")
        elif level == "success":
            self.progress_text.insert(tk.END, f"[{timestamp}] SUCCESS: {message}\n", "success")
        else:
            self.progress_text.insert(tk.END, f"[{timestamp}] {message}\n")
        
        self.progress_text.see(tk.END)
    
    def _conversion_complete(self, success: bool):
        """Handle conversion completion."""
        self.convert_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        
        if success:
            messagebox.showinfo("Success", "Model conversion completed successfully!")
        else:
            messagebox.showerror("Error", "Model conversion failed. Check the log for details.")
    
    def _cancel_conversion(self):
        """Cancel the current conversion."""
        if self.converter.current_process:
            self.converter.current_process.terminate()
            self._update_progress("Conversion cancelled", 0, "error")
            self.convert_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
    
    def _clear_log(self):
        """Clear the progress log."""
        self.progress_text.delete(1.0, tk.END)
        self.progress_var.set(0)
    
    def _open_lora_merger(self):
        """Open LoRA merger dialog."""
        messagebox.showinfo("LoRA Merger", "LoRA merging tool coming soon!")
    
    def _open_model_splitter(self):
        """Open model splitter dialog."""
        messagebox.showinfo("Model Splitter", "Model splitting tool coming soon!")
    
    def _open_optimizer(self):
        """Open model optimizer dialog."""
        messagebox.showinfo("Model Optimizer", "Model optimization tool coming soon!")
    
    def _open_batch_converter(self):
        """Open batch converter dialog."""
        messagebox.showinfo("Batch Converter", "Batch conversion tool coming soon!")
    
    def _open_model_compare(self):
        """Open model comparison dialog."""
        messagebox.showinfo("Model Compare", "Model comparison tool coming soon!")
    
    def _open_metadata_editor(self):
        """Open metadata editor dialog."""
        messagebox.showinfo("Metadata Editor", "Metadata editing tool coming soon!")