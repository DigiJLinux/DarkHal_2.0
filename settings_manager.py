import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class SettingsManager:
    """Manages application settings with JSON persistence."""
    
    def __init__(self, settings_file: str = "settings.json"):
        self.settings_file = settings_file
        self.default_settings = {
            "api": {
                "huggingface_token": "",
                "use_env_token": True,
                "use_organization": False,
                "organization": ""
            },
            "paths": {
                "models_directory": "./models",
                "downloads_directory": "./downloads",
                "last_model_path": "",
                "last_lora_path": ""
            },
            "model_settings": {
                "default_n_ctx": 4096,
                "default_n_gpu_layers": 0,
                "default_max_tokens": 256,
                "stream_by_default": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 0,
                "min_p": 0.0,
                "typical_p": 1.0
            },
            "search_preferences": {
                "default_search_type": "Models",
                "default_sort": "downloads",
                "search_limit": 50,
                "auto_filter_gguf": True
            },
            "ui_preferences": {
                "window_width": 1200,
                "window_height": 700,
                "theme": "default",
                "show_tooltips": True
            },
            "library": {
                "root_folder": "",
                "max_depth": 3,
                "auto_scan_on_startup": False,
                "watch_for_changes": False
            },
            "download_settings": {
                "max_concurrent_downloads": 3,
                "max_download_speed": 0,
                "min_download_speed": 0,
                "retry_attempts": 3,
                "timeout_seconds": 30
            }
        }
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or create default."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults to handle new keys
                    return self._merge_settings(self.default_settings, loaded)
            except Exception as e:
                print(f"Error loading settings: {e}")
                return self.default_settings.copy()
        return self.default_settings.copy()
    
    def _merge_settings(self, defaults: Dict, loaded: Dict) -> Dict:
        """Merge loaded settings with defaults, preserving user values."""
        result = defaults.copy()
        for key, value in loaded.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_settings(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def save_settings(self):
        """Save current settings to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a setting value using dot notation (e.g., 'api.huggingface_token')."""
        keys = path.split('.')
        value = self.settings
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, path: str, value: Any):
        """Set a setting value using dot notation."""
        keys = path.split('.')
        target = self.settings
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = self.default_settings.copy()
        self.save_settings()


class SettingsDialog:
    """Settings dialog window for the application."""
    
    def __init__(self, parent: tk.Tk, settings_manager: SettingsManager):
        self.parent = parent
        self.settings = settings_manager
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("700x500")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Variables for settings
        self.vars = {}
        self._create_variables()
        
        # Build UI
        self._build_ui()
        
        # Load current settings into UI
        self._load_current_settings()
        
        # Center the dialog
        self._center_window()
    
    def _create_variables(self):
        """Create tkinter variables for settings."""
        self.vars = {
            # API Settings
            'hf_token': tk.StringVar(),
            'use_env_token': tk.BooleanVar(),
            'use_organization': tk.BooleanVar(),
            'organization': tk.StringVar(),
            
            # Path Settings
            'models_dir': tk.StringVar(),
            'downloads_dir': tk.StringVar(),
            
            # Model Settings
            'default_n_ctx': tk.IntVar(),
            'default_n_gpu': tk.IntVar(),
            'default_max_tokens': tk.IntVar(),
            'stream_default': tk.BooleanVar(),
            'temperature': tk.DoubleVar(),
            'top_p': tk.DoubleVar(),
            'repetition_penalty': tk.DoubleVar(),
            'no_repeat_ngram_size': tk.IntVar(),
            'min_p': tk.DoubleVar(),
            'typical_p': tk.DoubleVar(),
            
            # Search Settings
            'search_type': tk.StringVar(),
            'default_sort': tk.StringVar(),
            'search_limit': tk.IntVar(),
            'auto_filter_gguf': tk.BooleanVar(),
            
            # UI Settings
            'show_tooltips': tk.BooleanVar(),
            'theme': tk.StringVar(),
            
            # Library Settings
            'library_root': tk.StringVar(),
            'library_depth': tk.IntVar(),
            'auto_scan': tk.BooleanVar(),
            'watch_changes': tk.BooleanVar(),
            
            # Download Settings
            'max_downloads': tk.IntVar(),
            'max_speed': tk.IntVar(),
            'min_speed': tk.IntVar(),
            'retry_attempts': tk.IntVar(),
            'timeout_seconds': tk.IntVar()
        }
    
    def _build_ui(self):
        """Build the settings dialog UI."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # API Settings Tab
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="API")
        self._build_api_tab(api_frame)
        
        # Paths Tab
        paths_frame = ttk.Frame(notebook)
        notebook.add(paths_frame, text="Paths")
        self._build_paths_tab(paths_frame)
        
        # Model Settings Tab
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Model Defaults")
        self._build_model_tab(model_frame)
        
        # Search Settings Tab
        search_frame = ttk.Frame(notebook)
        notebook.add(search_frame, text="Search")
        self._build_search_tab(search_frame)
        
        # UI Preferences Tab
        ui_frame = ttk.Frame(notebook)
        notebook.add(ui_frame, text="Interface")
        self._build_ui_tab(ui_frame)
        
        # Library Settings Tab
        library_frame = ttk.Frame(notebook)
        notebook.add(library_frame, text="Library")
        self._build_library_tab(library_frame)
        
        # Download Settings Tab
        download_frame = ttk.Frame(notebook)
        notebook.add(download_frame, text="Downloads")
        self._build_download_tab(download_frame)
        
        # Buttons at bottom
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="Save", command=self._save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Reset to Defaults", command=self._reset_defaults).pack(side=tk.LEFT)
    
    def _build_api_tab(self, parent: ttk.Frame):
        """Build API settings tab."""
        # Main container with scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # API Token Frame
        token_frame = ttk.LabelFrame(scrollable_frame, text="API Token Management", padding=10)
        token_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Use environment token checkbox
        ttk.Checkbutton(token_frame, text="Use token from HUGGINGFACE.env file",
                       variable=self.vars['use_env_token'],
                       command=self._toggle_token_entry).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # API Token entry
        ttk.Label(token_frame, text="API Token:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.token_entry = ttk.Entry(token_frame, textvariable=self.vars['hf_token'], width=40, show="*")
        self.token_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Token action buttons
        token_buttons = ttk.Frame(token_frame)
        token_buttons.grid(row=1, column=2, padx=5)
        
        self.show_token_btn = ttk.Button(token_buttons, text="View", width=8,
                                         command=self._toggle_token_visibility)
        self.show_token_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(token_buttons, text="Change", width=8,
                  command=self._change_token).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(token_buttons, text="Test", width=8,
                  command=self._test_api_key).pack(side=tk.LEFT, padx=2)
        
        # Organization Frame
        org_frame = ttk.LabelFrame(scrollable_frame, text="Organization Settings", padding=10)
        org_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Use organization checkbox
        ttk.Checkbutton(org_frame, text="Use as HuggingFace organization member",
                       variable=self.vars['use_organization'],
                       command=self._toggle_organization).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Organization dropdown
        ttk.Label(org_frame, text="Organization:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.org_combo = ttk.Combobox(org_frame, textvariable=self.vars['organization'],
                                      state="disabled", width=30)
        self.org_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Button(org_frame, text="Fetch Organizations", 
                  command=self._fetch_organizations).grid(row=1, column=2, padx=5)
        
        # Organizations list (for display)
        self.org_listbox = tk.Listbox(org_frame, height=5, width=50)
        self.org_listbox.grid(row=2, column=0, columnspan=3, pady=10)
        self.org_listbox.bind('<<ListboxSelect>>', self._on_org_select)
        
        # Info labels
        info_frame = ttk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        info_text = ("• API tokens can be obtained from: https://huggingface.co/settings/tokens\n"
                    "• Organizations allow you to access private repos and team resources\n"
                    "• Test your API key to verify it's working correctly")
        ttk.Label(info_frame, text=info_text, foreground="gray").pack(anchor=tk.W)
        
        # API Status label
        self.api_status_label = ttk.Label(info_frame, text="", foreground="green")
        self.api_status_label.pack(anchor=tk.W, pady=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _build_paths_tab(self, parent: ttk.Frame):
        """Build paths settings tab."""
        frame = ttk.LabelFrame(parent, text="Default Directories", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Models directory
        ttk.Label(frame, text="Models Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.vars['models_dir'], width=40).grid(row=0, column=1, pady=5)
        ttk.Button(frame, text="Browse", 
                  command=lambda: self._browse_directory('models_dir')).grid(row=0, column=2, padx=5)
        
        # Downloads directory
        ttk.Label(frame, text="Downloads Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.vars['downloads_dir'], width=40).grid(row=1, column=1, pady=5)
        ttk.Button(frame, text="Browse",
                  command=lambda: self._browse_directory('downloads_dir')).grid(row=1, column=2, padx=5)
        
        # Create directories checkbox
        self.create_dirs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Create directories if they don't exist",
                       variable=self.create_dirs_var).grid(row=2, column=0, columnspan=3, pady=10)
    
    def _build_model_tab(self, parent: ttk.Frame):
        """Build model defaults tab."""
        frame = ttk.LabelFrame(parent, text="Default Model Settings", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Context size
        ttk.Label(frame, text="Default Context Size (n_ctx):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ctx_spin = tk.Spinbox(frame, from_=512, to=32768, increment=512,
                             textvariable=self.vars['default_n_ctx'], width=15)
        ctx_spin.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # GPU layers
        ttk.Label(frame, text="Default GPU Layers:").grid(row=1, column=0, sticky=tk.W, pady=5)
        gpu_spin = tk.Spinbox(frame, from_=0, to=100, increment=1,
                             textvariable=self.vars['default_n_gpu'], width=15)
        gpu_spin.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Max tokens
        ttk.Label(frame, text="Default Max Tokens:").grid(row=2, column=0, sticky=tk.W, pady=5)
        tokens_spin = tk.Spinbox(frame, from_=16, to=8192, increment=16,
                                textvariable=self.vars['default_max_tokens'], width=15)
        tokens_spin.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Stream by default
        ttk.Checkbutton(frame, text="Stream output by default",
                       variable=self.vars['stream_default']).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Temperature
        ttk.Label(frame, text="Temperature:").grid(row=4, column=0, sticky=tk.W, pady=5)
        temp_spin = tk.Spinbox(frame, from_=0.0, to=2.0, increment=0.1,
                              textvariable=self.vars['temperature'], width=15, format="%.1f")
        temp_spin.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Top P
        ttk.Label(frame, text="Top P:").grid(row=5, column=0, sticky=tk.W, pady=5)
        top_p_spin = tk.Spinbox(frame, from_=0.0, to=1.0, increment=0.1,
                               textvariable=self.vars['top_p'], width=15, format="%.1f")
        top_p_spin.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # Repetition Penalty
        ttk.Label(frame, text="Repetition Penalty:").grid(row=6, column=0, sticky=tk.W, pady=5)
        rep_pen_spin = tk.Spinbox(frame, from_=0.5, to=2.0, increment=0.1,
                                 textvariable=self.vars['repetition_penalty'], width=15, format="%.1f")
        rep_pen_spin.grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # No Repeat N-gram Size
        ttk.Label(frame, text="No Repeat N-gram Size:").grid(row=7, column=0, sticky=tk.W, pady=5)
        ngram_spin = tk.Spinbox(frame, from_=0, to=10, increment=1,
                               textvariable=self.vars['no_repeat_ngram_size'], width=15)
        ngram_spin.grid(row=7, column=1, sticky=tk.W, pady=5)
        
        # Min P
        ttk.Label(frame, text="Min P:").grid(row=8, column=0, sticky=tk.W, pady=5)
        min_p_spin = tk.Spinbox(frame, from_=0.0, to=1.0, increment=0.01,
                               textvariable=self.vars['min_p'], width=15, format="%.2f")
        min_p_spin.grid(row=8, column=1, sticky=tk.W, pady=5)
        
        # Typical P
        ttk.Label(frame, text="Typical P:").grid(row=9, column=0, sticky=tk.W, pady=5)
        typical_p_spin = tk.Spinbox(frame, from_=0.0, to=1.0, increment=0.1,
                                   textvariable=self.vars['typical_p'], width=15, format="%.1f")
        typical_p_spin.grid(row=9, column=1, sticky=tk.W, pady=5)
    
    def _build_search_tab(self, parent: ttk.Frame):
        """Build search settings tab."""
        frame = ttk.LabelFrame(parent, text="Search Preferences", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Default search type
        ttk.Label(frame, text="Default Search Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(frame, textvariable=self.vars['search_type'],
                    values=["Models", "Datasets"], state="readonly", width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Default sort
        ttk.Label(frame, text="Default Sort By:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(frame, textvariable=self.vars['default_sort'],
                    values=["downloads", "likes", "lastModified"], state="readonly", width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Search limit
        ttk.Label(frame, text="Results Limit:").grid(row=2, column=0, sticky=tk.W, pady=5)
        limit_spin = tk.Spinbox(frame, from_=10, to=200, increment=10,
                               textvariable=self.vars['search_limit'], width=20)
        limit_spin.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Auto filter GGUF
        ttk.Checkbutton(frame, text="Automatically filter for GGUF files when downloading models",
                       variable=self.vars['auto_filter_gguf']).grid(row=3, column=0, columnspan=2, pady=10)
    
    def _build_ui_tab(self, parent: ttk.Frame):
        """Build UI preferences tab."""
        frame = ttk.LabelFrame(parent, text="Interface Settings", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Theme selection
        ttk.Label(frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(frame, textvariable=self.vars['theme'],
                    values=["default", "dark", "light"], state="readonly", width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Show tooltips
        ttk.Checkbutton(frame, text="Show tooltips",
                       variable=self.vars['show_tooltips']).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Note about themes
        ttk.Label(frame, text="Note: Theme changes will take effect after restart",
                 foreground="gray").grid(row=2, column=0, columnspan=2, pady=5)
    
    def _build_library_tab(self, parent: ttk.Frame):
        """Build library settings tab."""
        frame = ttk.LabelFrame(parent, text="Model Library Settings", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Library root folder
        ttk.Label(frame, text="Library Root Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        root_frame = ttk.Frame(frame)
        root_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Entry(root_frame, textvariable=self.vars['library_root'], width=40).pack(side=tk.LEFT)
        ttk.Button(root_frame, text="Browse", 
                  command=lambda: self._browse_directory('library_root')).pack(side=tk.LEFT, padx=5)
        
        # Scan depth
        ttk.Label(frame, text="Maximum Scan Depth:").grid(row=1, column=0, sticky=tk.W, pady=5)
        depth_frame = ttk.Frame(frame)
        depth_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        depth_scale = ttk.Scale(depth_frame, from_=1, to=10, variable=self.vars['library_depth'],
                               orient=tk.HORIZONTAL, length=200)
        depth_scale.pack(side=tk.LEFT)
        
        depth_label = ttk.Label(depth_frame, textvariable=self.vars['library_depth'])
        depth_label.pack(side=tk.LEFT, padx=10)
        ttk.Label(depth_frame, text="levels").pack(side=tk.LEFT)
        
        # Auto scan options
        ttk.Checkbutton(frame, text="Auto-scan library on startup",
                       variable=self.vars['auto_scan']).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        ttk.Checkbutton(frame, text="Watch for file system changes (experimental)",
                       variable=self.vars['watch_changes']).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Info text
        info_text = ("The library scanner searches for model files in the specified folder.\n"
                    "Scan depth controls how many subdirectory levels to search.\n"
                    "Supported formats: .gguf, .bin, .safetensors, .pt, .pth, .onnx")
        ttk.Label(frame, text=info_text, foreground="gray").grid(row=4, column=0, columnspan=3, pady=10)
    
    def _build_download_tab(self, parent: ttk.Frame):
        """Build download settings tab."""
        # Download limits frame
        limits_frame = ttk.LabelFrame(parent, text="Download Limits", padding=10)
        limits_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Max concurrent downloads
        ttk.Label(limits_frame, text="Maximum Concurrent Downloads:").grid(row=0, column=0, sticky=tk.W, pady=5)
        concurrent_spin = tk.Spinbox(limits_frame, from_=1, to=10, increment=1,
                                   textvariable=self.vars['max_downloads'], width=15)
        concurrent_spin.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(limits_frame, text="(1-10 downloads)").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Speed limits frame
        speed_frame = ttk.LabelFrame(parent, text="Speed Limits (KB/s)", padding=10)
        speed_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Max download speed
        ttk.Label(speed_frame, text="Maximum Download Speed:").grid(row=0, column=0, sticky=tk.W, pady=5)
        max_speed_spin = tk.Spinbox(speed_frame, from_=0, to=100000, increment=100,
                                  textvariable=self.vars['max_speed'], width=15)
        max_speed_spin.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(speed_frame, text="(0 = unlimited)").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Min download speed
        ttk.Label(speed_frame, text="Minimum Download Speed:").grid(row=1, column=0, sticky=tk.W, pady=5)
        min_speed_spin = tk.Spinbox(speed_frame, from_=0, to=10000, increment=10,
                                  textvariable=self.vars['min_speed'], width=15)
        min_speed_spin.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(speed_frame, text="(0 = no minimum)").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Connection settings frame
        conn_frame = ttk.LabelFrame(parent, text="Connection Settings", padding=10)
        conn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Retry attempts
        ttk.Label(conn_frame, text="Retry Attempts:").grid(row=0, column=0, sticky=tk.W, pady=5)
        retry_spin = tk.Spinbox(conn_frame, from_=0, to=10, increment=1,
                              textvariable=self.vars['retry_attempts'], width=15)
        retry_spin.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(conn_frame, text="(number of retries on failure)").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Timeout
        ttk.Label(conn_frame, text="Connection Timeout:").grid(row=1, column=0, sticky=tk.W, pady=5)
        timeout_spin = tk.Spinbox(conn_frame, from_=5, to=300, increment=5,
                                textvariable=self.vars['timeout_seconds'], width=15)
        timeout_spin.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(conn_frame, text="(seconds)").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Info text
        info_text = ("• Speed limits help manage bandwidth usage\n"
                    "• Concurrent downloads should be balanced with your internet connection\n"
                    "• Higher timeout values help with slow connections")
        ttk.Label(parent, text=info_text, foreground="gray").pack(anchor=tk.W, padx=10, pady=10)
    
    def _toggle_token_entry(self):
        """Enable/disable token entry based on checkbox."""
        if self.vars['use_env_token'].get():
            self.token_entry.config(state="disabled")
        else:
            self.token_entry.config(state="normal")
    
    def _toggle_token_visibility(self):
        """Toggle token visibility."""
        if self.token_entry['show'] == "*":
            self.token_entry.config(show="")
            self.show_token_btn.config(text="Hide")
        else:
            self.token_entry.config(show="*")
            self.show_token_btn.config(text="View")
    
    def _change_token(self):
        """Open dialog to change API token."""
        import tkinter.simpledialog as simpledialog
        
        new_token = simpledialog.askstring(
            "Change API Token",
            "Enter new HuggingFace API token:",
            parent=self.dialog,
            show='*'
        )
        
        if new_token:
            self.vars['hf_token'].set(new_token)
            self.vars['use_env_token'].set(False)
            self._toggle_token_entry()
            self.api_status_label.config(text="Token updated. Click Test to verify.", foreground="blue")
    
    def _test_api_key(self):
        """Test the API key."""
        import requests
        
        # Get the token to test
        if self.vars['use_env_token'].get():
            import os
            from dotenv import load_dotenv
            load_dotenv("HUGGINGFACE.env")
            token = os.getenv("HF_API_KEY")
        else:
            token = self.vars['hf_token'].get()
        
        if not token:
            self.api_status_label.config(text="No API token configured", foreground="red")
            return
        
        try:
            # Test API by fetching user info
            # Ensure token is properly stripped of whitespace and newlines
            clean_token = token.strip().replace('\n', '').replace('\r', '')
            headers = {"Authorization": f"Bearer {clean_token}"}
            response = requests.get("https://huggingface.co/api/whoami", headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                username = user_data.get('name', 'Unknown')
                self.api_status_label.config(
                    text=f"✓ API key valid. Logged in as: {username}", 
                    foreground="green"
                )
                
                # Update organizations if found
                orgs = user_data.get('orgs', [])
                if orgs:
                    org_names = [org.get('name', '') for org in orgs]
                    self.org_listbox.delete(0, tk.END)
                    for org in org_names:
                        self.org_listbox.insert(tk.END, org)
                    self.org_combo['values'] = org_names
                    
            elif response.status_code == 401:
                self.api_status_label.config(text="✗ Invalid API token", foreground="red")
            else:
                self.api_status_label.config(
                    text=f"✗ API test failed: {response.status_code}", 
                    foreground="red"
                )
                
        except Exception as e:
            self.api_status_label.config(text=f"✗ Connection error: {str(e)[:50]}", foreground="red")
    
    def _toggle_organization(self):
        """Enable/disable organization controls."""
        if self.vars['use_organization'].get():
            self.org_combo.config(state="readonly")
            if not self.org_combo['values']:
                self._fetch_organizations()
        else:
            self.org_combo.config(state="disabled")
    
    def _fetch_organizations(self):
        """Fetch organizations for the current API key."""
        import requests
        
        # Get the token
        if self.vars['use_env_token'].get():
            import os
            from dotenv import load_dotenv
            load_dotenv("HUGGINGFACE.env")
            token = os.getenv("HF_API_KEY")
        else:
            token = self.vars['hf_token'].get()
        
        if not token:
            messagebox.showwarning("No Token", "Please configure an API token first")
            return
        
        try:
            # Ensure token is properly stripped of whitespace and newlines
            clean_token = token.strip().replace('\n', '').replace('\r', '')
            headers = {"Authorization": f"Bearer {clean_token}"}
            response = requests.get("https://huggingface.co/api/whoami", headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                orgs = user_data.get('orgs', [])
                
                if orgs:
                    org_names = [org.get('name', '') for org in orgs]
                    self.org_listbox.delete(0, tk.END)
                    for org in org_names:
                        self.org_listbox.insert(tk.END, org)
                    self.org_combo['values'] = org_names
                    
                    if org_names:
                        self.org_combo.set(org_names[0])
                        self.api_status_label.config(
                            text=f"Found {len(org_names)} organization(s)", 
                            foreground="green"
                        )
                else:
                    self.api_status_label.config(
                        text="No organizations found for this account", 
                        foreground="blue"
                    )
            else:
                self.api_status_label.config(
                    text=f"Failed to fetch organizations: {response.status_code}", 
                    foreground="red"
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch organizations: {str(e)}")
    
    def _on_org_select(self, event):
        """Handle organization selection from listbox."""
        selection = self.org_listbox.curselection()
        if selection:
            org_name = self.org_listbox.get(selection[0])
            self.vars['organization'].set(org_name)
    
    def _browse_directory(self, var_name: str):
        """Browse for directory."""
        directory = filedialog.askdirectory(
            parent=self.dialog,
            initialdir=self.vars[var_name].get() or "."
        )
        if directory:
            self.vars[var_name].set(directory)
    
    def _load_current_settings(self):
        """Load current settings into UI variables."""
        self.vars['hf_token'].set(self.settings.get('api.huggingface_token', ''))
        self.vars['use_env_token'].set(self.settings.get('api.use_env_token', True))
        self.vars['use_organization'].set(self.settings.get('api.use_organization', False))
        self.vars['organization'].set(self.settings.get('api.organization', ''))
        
        self.vars['models_dir'].set(self.settings.get('paths.models_directory', './models'))
        self.vars['downloads_dir'].set(self.settings.get('paths.downloads_directory', './downloads'))
        
        self.vars['default_n_ctx'].set(self.settings.get('model_settings.default_n_ctx', 4096))
        self.vars['default_n_gpu'].set(self.settings.get('model_settings.default_n_gpu_layers', 0))
        self.vars['default_max_tokens'].set(self.settings.get('model_settings.default_max_tokens', 256))
        self.vars['stream_default'].set(self.settings.get('model_settings.stream_by_default', True))
        self.vars['temperature'].set(self.settings.get('model_settings.temperature', 0.7))
        self.vars['top_p'].set(self.settings.get('model_settings.top_p', 0.9))
        self.vars['repetition_penalty'].set(self.settings.get('model_settings.repetition_penalty', 1.1))
        self.vars['no_repeat_ngram_size'].set(self.settings.get('model_settings.no_repeat_ngram_size', 0))
        self.vars['min_p'].set(self.settings.get('model_settings.min_p', 0.0))
        self.vars['typical_p'].set(self.settings.get('model_settings.typical_p', 1.0))
        
        self.vars['search_type'].set(self.settings.get('search_preferences.default_search_type', 'Models'))
        self.vars['default_sort'].set(self.settings.get('search_preferences.default_sort', 'downloads'))
        self.vars['search_limit'].set(self.settings.get('search_preferences.search_limit', 50))
        self.vars['auto_filter_gguf'].set(self.settings.get('search_preferences.auto_filter_gguf', True))
        
        self.vars['show_tooltips'].set(self.settings.get('ui_preferences.show_tooltips', True))
        self.vars['theme'].set(self.settings.get('ui_preferences.theme', 'default'))
        
        # Library settings
        self.vars['library_root'].set(self.settings.get('library.root_folder', ''))
        self.vars['library_depth'].set(self.settings.get('library.max_depth', 3))
        self.vars['auto_scan'].set(self.settings.get('library.auto_scan_on_startup', False))
        self.vars['watch_changes'].set(self.settings.get('library.watch_for_changes', False))
        
        # Download settings
        self.vars['max_downloads'].set(self.settings.get('download_settings.max_concurrent_downloads', 3))
        self.vars['max_speed'].set(self.settings.get('download_settings.max_download_speed', 0))
        self.vars['min_speed'].set(self.settings.get('download_settings.min_download_speed', 0))
        self.vars['retry_attempts'].set(self.settings.get('download_settings.retry_attempts', 3))
        self.vars['timeout_seconds'].set(self.settings.get('download_settings.timeout_seconds', 30))
        
        # Update token entry state
        self._toggle_token_entry()
    
    def _save_settings(self):
        """Save settings from UI to settings manager."""
        # API settings (strip whitespace from strings)
        self.settings.set('api.huggingface_token', self.vars['hf_token'].get().strip())
        self.settings.set('api.use_env_token', self.vars['use_env_token'].get())
        self.settings.set('api.use_organization', self.vars['use_organization'].get())
        self.settings.set('api.organization', self.vars['organization'].get().strip())
        
        # Path settings
        models_dir = self.vars['models_dir'].get()
        downloads_dir = self.vars['downloads_dir'].get()
        
        # Create directories if requested
        if self.create_dirs_var.get():
            for directory in [models_dir, downloads_dir]:
                if directory and not os.path.exists(directory):
                    try:
                        os.makedirs(directory, exist_ok=True)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to create directory {directory}: {e}")
        
        self.settings.set('paths.models_directory', models_dir)
        self.settings.set('paths.downloads_directory', downloads_dir)
        
        # Model settings
        self.settings.set('model_settings.default_n_ctx', self.vars['default_n_ctx'].get())
        self.settings.set('model_settings.default_n_gpu_layers', self.vars['default_n_gpu'].get())
        self.settings.set('model_settings.default_max_tokens', self.vars['default_max_tokens'].get())
        self.settings.set('model_settings.stream_by_default', self.vars['stream_default'].get())
        self.settings.set('model_settings.temperature', self.vars['temperature'].get())
        self.settings.set('model_settings.top_p', self.vars['top_p'].get())
        self.settings.set('model_settings.repetition_penalty', self.vars['repetition_penalty'].get())
        self.settings.set('model_settings.no_repeat_ngram_size', self.vars['no_repeat_ngram_size'].get())
        self.settings.set('model_settings.min_p', self.vars['min_p'].get())
        self.settings.set('model_settings.typical_p', self.vars['typical_p'].get())
        
        # Search settings
        self.settings.set('search_preferences.default_search_type', self.vars['search_type'].get())
        self.settings.set('search_preferences.default_sort', self.vars['default_sort'].get())
        self.settings.set('search_preferences.search_limit', self.vars['search_limit'].get())
        self.settings.set('search_preferences.auto_filter_gguf', self.vars['auto_filter_gguf'].get())
        
        # UI settings
        self.settings.set('ui_preferences.show_tooltips', self.vars['show_tooltips'].get())
        self.settings.set('ui_preferences.theme', self.vars['theme'].get())
        
        # Library settings
        self.settings.set('library.root_folder', self.vars['library_root'].get().strip())
        self.settings.set('library.max_depth', self.vars['library_depth'].get())
        self.settings.set('library.auto_scan_on_startup', self.vars['auto_scan'].get())
        self.settings.set('library.watch_for_changes', self.vars['watch_changes'].get())
        
        # Download settings
        self.settings.set('download_settings.max_concurrent_downloads', self.vars['max_downloads'].get())
        self.settings.set('download_settings.max_download_speed', self.vars['max_speed'].get())
        self.settings.set('download_settings.min_download_speed', self.vars['min_speed'].get())
        self.settings.set('download_settings.retry_attempts', self.vars['retry_attempts'].get())
        self.settings.set('download_settings.timeout_seconds', self.vars['timeout_seconds'].get())
        
        # Save to file
        if self.settings.save_settings():
            messagebox.showinfo("Settings", "Settings saved successfully!")
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "Failed to save settings")
    
    def _reset_defaults(self):
        """Reset settings to defaults."""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            self.settings.reset_to_defaults()
            self._load_current_settings()
            messagebox.showinfo("Settings", "Settings reset to defaults")
    
    def _center_window(self):
        """Center the dialog on the parent window."""
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


def open_settings_dialog(parent: tk.Tk, settings_manager: SettingsManager):
    """Convenience function to open the settings dialog."""
    dialog = SettingsDialog(parent, settings_manager)
    return dialog