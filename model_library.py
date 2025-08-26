import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelInfo:
    """Information about a model file."""
    name: str
    path: str
    size: int
    modified_time: float
    file_type: str
    hash: str = ""
    metadata: Dict[str, Any] = None
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
    
    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)
    
    @property
    def modified_date(self) -> str:
        return datetime.fromtimestamp(self.modified_time).strftime("%Y-%m-%d %H:%M")


class ModelLibrary:
    """Manages a library of model files with scanning and indexing."""
    
    def __init__(self, library_root: str, max_depth: int = 3):
        self.library_root = Path(library_root)
        self.max_depth = max_depth
        self.models: Dict[str, ModelInfo] = {}
        self.index_file = self.library_root / ".model_index.json"
        self.supported_extensions = {'.gguf', '.bin', '.safetensors', '.pt', '.pth', '.onnx'}
        self.scan_in_progress = False
        
    def scan_library(self, progress_callback: Optional[callable] = None) -> List[ModelInfo]:
        """Scan the library directory for model files."""
        if self.scan_in_progress:
            return list(self.models.values())
            
        self.scan_in_progress = True
        new_models = {}
        total_files = 0
        processed_files = 0
        
        try:
            # Count total files first for progress
            if progress_callback:
                for root, dirs, files in os.walk(self.library_root):
                    depth = len(Path(root).relative_to(self.library_root).parts)
                    if depth >= self.max_depth:
                        dirs.clear()
                    total_files += len([f for f in files if Path(f).suffix.lower() in self.supported_extensions])
                
                progress_callback(0, total_files, "Counting files...")
            
            # Scan for model files
            for root, dirs, files in os.walk(self.library_root):
                # Limit scan depth
                depth = len(Path(root).relative_to(self.library_root).parts)
                if depth >= self.max_depth:
                    dirs.clear()
                    continue
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # Check if it's a supported model file
                    if file_path.suffix.lower() not in self.supported_extensions:
                        continue
                    
                    try:
                        stat = file_path.stat()
                        file_hash = self._calculate_file_hash(str(file_path))
                        
                        # Check if file already exists in index
                        existing_model = self.models.get(str(file_path))
                        if (existing_model and 
                            existing_model.modified_time == stat.st_mtime and 
                            existing_model.hash == file_hash):
                            # File unchanged, use existing data
                            new_models[str(file_path)] = existing_model
                        else:
                            # New or changed file
                            model_info = ModelInfo(
                                name=file_path.stem,
                                path=str(file_path),
                                size=stat.st_size,
                                modified_time=stat.st_mtime,
                                file_type=file_path.suffix.lower(),
                                hash=file_hash
                            )
                            # Try to extract metadata
                            self._extract_metadata(model_info)
                            new_models[str(file_path)] = model_info
                        
                        processed_files += 1
                        if progress_callback:
                            progress_callback(processed_files, total_files, f"Processing {file}")
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
            
            self.models = new_models
            self._save_index()
            
        finally:
            self.scan_in_progress = False
            
        return list(self.models.values())
    
    def _calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of first and last chunks for quick identification."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # Hash first chunk
                chunk = f.read(chunk_size)
                if chunk:
                    hash_md5.update(chunk)
                
                # Hash last chunk if file is large enough
                file_size = os.path.getsize(file_path)
                if file_size > chunk_size * 2:
                    f.seek(-chunk_size, 2)
                    chunk = f.read(chunk_size)
                    hash_md5.update(chunk)
                    
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _extract_metadata(self, model_info: ModelInfo):
        """Extract metadata from model files."""
        try:
            file_path = Path(model_info.path)
            
            if model_info.file_type == '.gguf':
                # Try to extract GGUF metadata
                model_info.metadata = self._extract_gguf_metadata(file_path)
            elif model_info.file_type in ['.bin', '.safetensors']:
                # Try to extract HuggingFace metadata
                model_info.metadata = self._extract_hf_metadata(file_path)
            
            # Extract tags from path
            path_parts = file_path.parts
            model_info.tags = [part.lower() for part in path_parts 
                             if any(keyword in part.lower() 
                                  for keyword in ['q4', 'q5', 'q8', 'fp16', 'fp32', 'instruct', 'chat'])]
                             
        except Exception as e:
            print(f"Error extracting metadata from {model_info.path}: {e}")
    
    def _extract_gguf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from GGUF files."""
        metadata = {}
        try:
            # This is a simplified version - you'd need proper GGUF parsing
            # For now, extract info from filename
            name = file_path.stem.lower()
            
            if 'q4' in name:
                metadata['quantization'] = 'Q4'
            elif 'q5' in name:
                metadata['quantization'] = 'Q5'
            elif 'q8' in name:
                metadata['quantization'] = 'Q8'
            elif 'fp16' in name:
                metadata['quantization'] = 'FP16'
                
            if 'instruct' in name:
                metadata['type'] = 'instruct'
            elif 'chat' in name:
                metadata['type'] = 'chat'
            else:
                metadata['type'] = 'base'
                
            # Extract model size if present
            for part in name.split('-'):
                if part.endswith('b') and part[:-1].replace('.', '').isdigit():
                    metadata['parameters'] = part
                    break
                    
        except Exception as e:
            print(f"Error extracting GGUF metadata: {e}")
            
        return metadata
    
    def _extract_hf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from HuggingFace model files."""
        metadata = {}
        try:
            # Look for config.json in the same directory
            config_path = file_path.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    metadata.update({
                        'architecture': config.get('architectures', []),
                        'model_type': config.get('model_type', ''),
                        'vocab_size': config.get('vocab_size', 0)
                    })
        except Exception as e:
            print(f"Error extracting HF metadata: {e}")
            
        return metadata
    
    def _save_index(self):
        """Save the model index to disk."""
        try:
            index_data = {
                'last_scan': time.time(),
                'library_root': str(self.library_root),
                'max_depth': self.max_depth,
                'models': {path: asdict(model) for path, model in self.models.items()}
            }
            
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def _load_index(self):
        """Load the model index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                
                # Validate index is current
                if (index_data.get('library_root') == str(self.library_root) and
                    index_data.get('max_depth') == self.max_depth):
                    
                    models_data = index_data.get('models', {})
                    self.models = {
                        path: ModelInfo(**data) 
                        for path, data in models_data.items()
                    }
                    
        except Exception as e:
            print(f"Error loading index: {e}")
    
    def search_models(self, query: str, file_type: str = "", tags: List[str] = None) -> List[ModelInfo]:
        """Search models by name, type, or tags."""
        if tags is None:
            tags = []
            
        results = []
        query_lower = query.lower()
        
        for model in self.models.values():
            # Check name match
            name_match = query_lower in model.name.lower()
            
            # Check file type match
            type_match = not file_type or model.file_type == file_type
            
            # Check tags match
            tags_match = not tags or any(tag in model.tags for tag in tags)
            
            if name_match and type_match and tags_match:
                results.append(model)
        
        return sorted(results, key=lambda x: x.modified_time, reverse=True)
    
    def get_models_by_type(self) -> Dict[str, List[ModelInfo]]:
        """Get models grouped by file type."""
        grouped = {}
        for model in self.models.values():
            if model.file_type not in grouped:
                grouped[model.file_type] = []
            grouped[model.file_type].append(model)
        
        return grouped
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        if not self.models:
            return {}
            
        total_size = sum(model.size for model in self.models.values())
        file_types = {}
        
        for model in self.models.values():
            if model.file_type not in file_types:
                file_types[model.file_type] = {'count': 0, 'size': 0}
            file_types[model.file_type]['count'] += 1
            file_types[model.file_type]['size'] += model.size
        
        return {
            'total_models': len(self.models),
            'total_size': total_size,
            'file_types': file_types,
            'last_scan': getattr(self, 'last_scan_time', 0)
        }


class ModelLibraryTab:
    """GUI tab for the Model Library."""
    
    def __init__(self, parent: ttk.Frame, settings_manager):
        self.parent = parent
        self.settings = settings_manager
        self.library = None
        self.current_models = []
        
        self._build_ui()
        self._load_library()
    
    def _build_ui(self):
        """Build the model library UI."""
        # Top toolbar
        toolbar = ttk.Frame(self.parent)
        toolbar.pack(fill=tk.X, padx=10, pady=5)
        
        # Left side controls
        left_controls = ttk.Frame(toolbar)
        left_controls.pack(side=tk.LEFT)
        
        ttk.Button(left_controls, text="Scan Library", 
                  command=self._scan_library).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Refresh", 
                  command=self._refresh).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Settings", 
                  command=self._open_library_settings).pack(side=tk.LEFT, padx=2)
        
        # Search controls
        search_frame = ttk.Frame(toolbar)
        search_frame.pack(side=tk.RIGHT)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=2)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        self.search_entry.pack(side=tk.LEFT, padx=2)
        self.search_entry.bind('<KeyRelease>', self._on_search)
        
        ttk.Label(search_frame, text="Type:").pack(side=tk.LEFT, padx=(10,2))
        self.type_var = tk.StringVar(value="All")
        self.type_combo = ttk.Combobox(search_frame, textvariable=self.type_var, 
                                      values=["All", ".gguf", ".bin", ".safetensors", ".pt", ".onnx"],
                                      state="readonly", width=10)
        self.type_combo.pack(side=tk.LEFT, padx=2)
        self.type_combo.bind('<<ComboboxSelected>>', self._on_filter_change)
        
        # Main content area
        content_frame = ttk.Frame(self.parent)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Model list
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Model treeview
        columns = ("name", "type", "size", "modified", "tags")
        self.tree = ttk.Treeview(left_panel, columns=columns, show="headings", height=20)
        
        self.tree.heading("name", text="Model Name")
        self.tree.heading("type", text="Type")
        self.tree.heading("size", text="Size")
        self.tree.heading("modified", text="Modified")
        self.tree.heading("tags", text="Tags")
        
        self.tree.column("name", width=300)
        self.tree.column("type", width=80)
        self.tree.column("size", width=100)
        self.tree.column("modified", width=120)
        self.tree.column("tags", width=150)
        
        # Scrollbars for treeview
        v_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=self.tree.yview)
        h_scroll = ttk.Scrollbar(left_panel, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Right panel - Details and actions
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Model details
        details_frame = ttk.LabelFrame(right_panel, text="Model Details", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True)
        
        self.details_text = tk.Text(details_frame, width=40, height=15, wrap=tk.WORD)
        details_scroll = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scroll.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        actions_frame = ttk.LabelFrame(right_panel, text="Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(actions_frame, text="Load Model", 
                  command=self._load_selected_model).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Open Folder", 
                  command=self._open_model_folder).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Copy Path", 
                  command=self._copy_model_path).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Add Tags", 
                  command=self._add_tags).pack(fill=tk.X, pady=2)
        
        # Status bar
        self.status_label = ttk.Label(self.parent, text="Ready")
        self.status_label.pack(fill=tk.X, padx=10, pady=2)
        
        # Bind events
        self.tree.bind('<<TreeviewSelect>>', self._on_model_select)
        self.tree.bind('<Double-Button-1>', self._load_selected_model)
    
    def _load_library(self):
        """Load the model library based on settings."""
        library_root = self.settings.get('library.root_folder', '')
        if library_root and os.path.exists(library_root):
            max_depth = self.settings.get('library.max_depth', 3)
            self.library = ModelLibrary(library_root, max_depth)
            self.library._load_index()
            self._update_model_list()
        else:
            self.status_label.config(text="No library folder configured. Click Settings to configure.")
    
    def _scan_library(self):
        """Scan the library for models."""
        if not self.library:
            self._open_library_settings()
            return
        
        # Show progress dialog
        progress_dialog = tk.Toplevel(self.parent)
        progress_dialog.title("Scanning Library")
        progress_dialog.geometry("400x150")
        progress_dialog.transient(self.parent.winfo_toplevel())
        progress_dialog.grab_set()
        
        ttk.Label(progress_dialog, text="Scanning model library...").pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, 
                                       maximum=100, length=300)
        progress_bar.pack(pady=10)
        
        status_var = tk.StringVar(value="Starting scan...")
        status_label = ttk.Label(progress_dialog, textvariable=status_var)
        status_label.pack(pady=5)
        
        def progress_callback(current, total, message):
            if total > 0:
                progress_var.set((current / total) * 100)
            status_var.set(message)
            progress_dialog.update()
        
        def scan_thread():
            try:
                models = self.library.scan_library(progress_callback)
                progress_dialog.after(0, lambda: progress_dialog.destroy())
                self.parent.after(0, self._update_model_list)
                self.parent.after(0, lambda: self.status_label.config(
                    text=f"Scan complete. Found {len(models)} models."))
            except Exception as e:
                progress_dialog.after(0, lambda: progress_dialog.destroy())
                self.parent.after(0, lambda: messagebox.showerror("Scan Error", str(e)))
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def _refresh(self):
        """Refresh the model list."""
        if self.library:
            self._update_model_list()
    
    def _update_model_list(self):
        """Update the model list display."""
        if not self.library:
            return
            
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Apply filters
        query = self.search_var.get()
        file_type = self.type_var.get() if self.type_var.get() != "All" else ""
        
        if query or file_type:
            models = self.library.search_models(query, file_type)
        else:
            models = list(self.library.models.values())
        
        self.current_models = models
        
        # Populate treeview
        for model in models:
            tags_str = ", ".join(model.tags[:3])  # Show first 3 tags
            self.tree.insert("", tk.END, values=(
                model.name,
                model.file_type,
                f"{model.size_mb:.1f} MB",
                model.modified_date,
                tags_str
            ))
        
        # Update status
        stats = self.library.get_library_stats()
        total_size_gb = stats.get('total_size', 0) / (1024**3)
        self.status_label.config(
            text=f"Showing {len(models)} of {stats.get('total_models', 0)} models "
                 f"({total_size_gb:.1f} GB total)"
        )
    
    def _on_search(self, event=None):
        """Handle search input."""
        self._update_model_list()
    
    def _on_filter_change(self, event=None):
        """Handle filter change."""
        self._update_model_list()
    
    def _on_model_select(self, event):
        """Handle model selection."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        model_name = item['values'][0]
        
        # Find the selected model
        selected_model = None
        for model in self.current_models:
            if model.name == model_name:
                selected_model = model
                break
        
        if selected_model:
            self._show_model_details(selected_model)
    
    def _show_model_details(self, model: ModelInfo):
        """Show detailed information about a model."""
        details = f"Name: {model.name}\\n"
        details += f"Path: {model.path}\\n"
        details += f"Type: {model.file_type}\\n"
        details += f"Size: {model.size_mb:.1f} MB\\n"
        details += f"Modified: {model.modified_date}\\n"
        
        if model.tags:
            details += f"Tags: {', '.join(model.tags)}\\n"
        
        if model.metadata:
            details += "\\nMetadata:\\n"
            for key, value in model.metadata.items():
                details += f"  {key}: {value}\\n"
        
        if model.description:
            details += f"\\nDescription:\\n{model.description}"
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details)
    
    def _load_selected_model(self, event=None):
        """Load the selected model in the main application."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a model to load")
            return
        
        item = self.tree.item(selection[0])
        model_name = item['values'][0]
        
        # Find the selected model
        selected_model = None
        for model in self.current_models:
            if model.name == model_name:
                selected_model = model
                break
        
        if selected_model:
            # Set the model path in the main application
            parent_window = self.parent.winfo_toplevel()
            if hasattr(parent_window, 'model_var'):
                parent_window.model_var.set(selected_model.path)
                messagebox.showinfo("Model Loaded", f"Loaded model: {selected_model.name}")
    
    def _open_model_folder(self):
        """Open the folder containing the selected model."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        model_name = item['values'][0]
        
        selected_model = None
        for model in self.current_models:
            if model.name == model_name:
                selected_model = model
                break
        
        if selected_model:
            folder = os.path.dirname(selected_model.path)
            import subprocess
            import platform
            
            try:
                if platform.system() == "Windows":
                    subprocess.run(["explorer", folder])
                elif platform.system() == "Darwin":
                    subprocess.run(["open", folder])
                else:
                    subprocess.run(["xdg-open", folder])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder: {e}")
    
    def _copy_model_path(self):
        """Copy the selected model's path to clipboard."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        model_name = item['values'][0]
        
        selected_model = None
        for model in self.current_models:
            if model.name == model_name:
                selected_model = model
                break
        
        if selected_model:
            self.parent.clipboard_clear()
            self.parent.clipboard_append(selected_model.path)
            self.status_label.config(text="Model path copied to clipboard")
    
    def _add_tags(self):
        """Add tags to the selected model."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        model_name = item['values'][0]
        
        selected_model = None
        for model in self.current_models:
            if model.name == model_name:
                selected_model = model
                break
        
        if selected_model:
            import tkinter.simpledialog as simpledialog
            
            current_tags = ", ".join(selected_model.tags)
            new_tags = simpledialog.askstring(
                "Add Tags",
                f"Enter tags (comma-separated):\\nCurrent: {current_tags}",
                initialvalue=current_tags
            )
            
            if new_tags is not None:
                selected_model.tags = [tag.strip() for tag in new_tags.split(",") if tag.strip()]
                self.library._save_index()
                self._update_model_list()
                self._show_model_details(selected_model)
    
    def _open_library_settings(self):
        """Open library settings dialog."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Library Settings")
        dialog.geometry("500x300")
        dialog.transient(self.parent.winfo_toplevel())
        dialog.grab_set()
        
        # Root folder setting
        ttk.Label(dialog, text="Library Root Folder:").pack(anchor=tk.W, padx=10, pady=5)
        
        folder_frame = ttk.Frame(dialog)
        folder_frame.pack(fill=tk.X, padx=10, pady=5)
        
        folder_var = tk.StringVar(value=self.settings.get('library.root_folder', ''))
        folder_entry = ttk.Entry(folder_frame, textvariable=folder_var, width=50)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        def browse_folder():
            folder = filedialog.askdirectory(initialdir=folder_var.get())
            if folder:
                folder_var.set(folder)
        
        ttk.Button(folder_frame, text="Browse", command=browse_folder).pack(side=tk.LEFT, padx=5)
        
        # Max depth setting
        ttk.Label(dialog, text="Maximum Scan Depth:").pack(anchor=tk.W, padx=10, pady=(10,5))
        
        depth_var = tk.IntVar(value=self.settings.get('library.max_depth', 3))
        depth_frame = ttk.Frame(dialog)
        depth_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Scale(depth_frame, from_=1, to=10, variable=depth_var, 
                 orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(depth_frame, textvariable=depth_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(depth_frame, text="levels").pack(side=tk.LEFT)
        
        # Info label
        info_text = ("Scan depth determines how many subdirectory levels to search.\\n"
                    "Higher values find more models but take longer to scan.")
        ttk.Label(dialog, text=info_text, foreground="gray").pack(padx=10, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_settings():
            self.settings.set('library.root_folder', folder_var.get())
            self.settings.set('library.max_depth', depth_var.get())
            self.settings.save_settings()
            
            # Reload library
            self._load_library()
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)