import os
import requests
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import queue
import uuid


class DownloadStatus(Enum):
    """Download status enumeration."""
    QUEUED = "Queued"
    DOWNLOADING = "Downloading"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    AUTH_REQUIRED = "Auth Required"


@dataclass
class DownloadItem:
    """Represents a single file download."""
    id: str
    group_id: str
    repo_id: str
    filename: str
    url: str
    save_path: str
    total_size: int = 0
    downloaded_size: int = 0
    status: DownloadStatus = DownloadStatus.QUEUED
    error_message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speed: float = 0.0
    eta: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    resume_position: int = 0
    selected: bool = True  # Whether this file should be downloaded
    
    @property
    def progress(self) -> float:
        """Calculate download progress percentage."""
        if self.total_size > 0:
            return (self.downloaded_size / self.total_size) * 100
        return 0.0
    
    @property
    def is_resumable(self) -> bool:
        """Check if download can be resumed."""
        return self.status in [DownloadStatus.PAUSED, DownloadStatus.FAILED]


@dataclass
class DownloadGroup:
    """Represents a group of related downloads (e.g., all files from one model)."""
    id: str
    repo_id: str
    name: str
    description: str = ""
    created_time: float = field(default_factory=time.time)
    files: Dict[str, DownloadItem] = field(default_factory=dict)
    expanded: bool = False
    
    @property
    def total_size(self) -> int:
        """Total size of all selected files in group."""
        return sum(item.total_size for item in self.files.values() if item.selected)
    
    @property
    def downloaded_size(self) -> int:
        """Total downloaded size of all selected files."""
        return sum(item.downloaded_size for item in self.files.values() if item.selected)
    
    @property
    def progress(self) -> float:
        """Overall progress percentage for the group."""
        if self.total_size > 0:
            return (self.downloaded_size / self.total_size) * 100
        return 0.0
    
    @property
    def status(self) -> DownloadStatus:
        """Overall status of the group."""
        selected_files = [item for item in self.files.values() if item.selected]
        if not selected_files:
            return DownloadStatus.COMPLETED
        
        statuses = [item.status for item in selected_files]
        
        # If any are downloading, group is downloading
        if DownloadStatus.DOWNLOADING in statuses:
            return DownloadStatus.DOWNLOADING
        
        # If all completed, group is completed
        if all(s == DownloadStatus.COMPLETED for s in statuses):
            return DownloadStatus.COMPLETED
        
        # If any failed, group shows failed
        if DownloadStatus.FAILED in statuses or DownloadStatus.AUTH_REQUIRED in statuses:
            return DownloadStatus.FAILED
        
        # If any cancelled, show cancelled
        if DownloadStatus.CANCELLED in statuses:
            return DownloadStatus.CANCELLED
        
        # If any paused, show paused
        if DownloadStatus.PAUSED in statuses:
            return DownloadStatus.PAUSED
        
        # Otherwise queued
        return DownloadStatus.QUEUED
    
    @property
    def active_speed(self) -> float:
        """Combined download speed of active files."""
        return sum(item.speed for item in self.files.values() 
                  if item.status == DownloadStatus.DOWNLOADING and item.selected)
    
    @property
    def eta(self) -> int:
        """Estimated time to completion for the group."""
        if self.active_speed > 0:
            remaining = self.total_size - self.downloaded_size
            return int(remaining / self.active_speed)
        return 0


class GroupedDownloadManager:
    """Manages grouped downloads with pause/resume support."""
    
    def __init__(self, max_concurrent: int = 3):
        self.groups: Dict[str, DownloadGroup] = {}
        self.download_queue: queue.Queue = queue.Queue()
        self.active_downloads: Dict[str, threading.Thread] = {}
        self.max_concurrent = max_concurrent
        self.callbacks: Dict[str, List[Callable]] = {
            'on_progress': [],
            'on_status_change': [],
            'on_complete': [],
            'on_error': [],
            'on_remove': [],
            'on_group_change': []
        }
        self._stop_flags: Dict[str, threading.Event] = {}
        self._pause_flags: Dict[str, threading.Event] = {}
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
    
    def create_download_group(self, repo_id: str, name: str, description: str = "") -> str:
        """Create a new download group."""
        group_id = str(uuid.uuid4())
        group = DownloadGroup(
            id=group_id,
            repo_id=repo_id,
            name=name,
            description=description
        )
        self.groups[group_id] = group
        self._trigger_callback('on_group_change', group)
        return group_id
    
    def add_file_to_group(self, group_id: str, filename: str, url: str, save_path: str,
                         headers: Optional[Dict[str, str]] = None, selected: bool = True) -> str:
        """Add a file to an existing download group."""
        if group_id not in self.groups:
            raise ValueError(f"Group {group_id} not found")
        
        group = self.groups[group_id]
        download_id = f"{group_id}_{filename}_{int(time.time())}"
        
        item = DownloadItem(
            id=download_id,
            group_id=group_id,
            repo_id=group.repo_id,
            filename=filename,
            url=url,
            save_path=save_path,
            headers=headers or {},
            selected=selected
        )
        
        group.files[download_id] = item
        
        # Queue for download if selected
        if selected:
            self.download_queue.put(download_id)
        
        self._trigger_callback('on_status_change', item)
        self._trigger_callback('on_group_change', group)
        
        return download_id
    
    def toggle_file_selection(self, download_id: str):
        """Toggle whether a file should be downloaded."""
        for group in self.groups.values():
            if download_id in group.files:
                item = group.files[download_id]
                item.selected = not item.selected
                
                if item.selected and item.status == DownloadStatus.QUEUED:
                    # Add to queue if now selected
                    self.download_queue.put(download_id)
                elif not item.selected and item.status in [DownloadStatus.QUEUED, DownloadStatus.PAUSED]:
                    # Cancel if unselected and not yet completed
                    self.cancel_download(download_id)
                
                self._trigger_callback('on_status_change', item)
                self._trigger_callback('on_group_change', group)
                break
    
    def pause_download(self, download_id: str):
        """Pause a download."""
        if download_id in self._pause_flags:
            self._pause_flags[download_id].set()
            
        for group in self.groups.values():
            if download_id in group.files:
                item = group.files[download_id]
                item.status = DownloadStatus.PAUSED
                self._trigger_callback('on_status_change', item)
                self._trigger_callback('on_group_change', group)
                break
    
    def pause_group(self, group_id: str):
        """Pause all downloads in a group."""
        if group_id in self.groups:
            group = self.groups[group_id]
            for download_id in group.files:
                if group.files[download_id].selected:
                    self.pause_download(download_id)
    
    def resume_download(self, download_id: str):
        """Resume a paused download."""
        for group in self.groups.values():
            if download_id in group.files:
                item = group.files[download_id]
                if item.is_resumable and item.selected:
                    item.status = DownloadStatus.QUEUED
                    item.resume_position = item.downloaded_size
                    self.download_queue.put(download_id)
                    self._trigger_callback('on_status_change', item)
                    self._trigger_callback('on_group_change', group)
                break
    
    def resume_group(self, group_id: str):
        """Resume all paused downloads in a group."""
        if group_id in self.groups:
            group = self.groups[group_id]
            for download_id in group.files:
                item = group.files[download_id]
                if item.is_resumable and item.selected:
                    self.resume_download(download_id)
    
    def cancel_download(self, download_id: str):
        """Cancel a download."""
        if download_id in self._stop_flags:
            self._stop_flags[download_id].set()
        
        for group in self.groups.values():
            if download_id in group.files:
                item = group.files[download_id]
                item.status = DownloadStatus.CANCELLED
                
                # Remove partial file
                if os.path.exists(item.save_path):
                    try:
                        os.remove(item.save_path)
                    except Exception:
                        pass
                
                self._trigger_callback('on_status_change', item)
                self._trigger_callback('on_group_change', group)
                break
    
    def cancel_group(self, group_id: str):
        """Cancel all downloads in a group."""
        if group_id in self.groups:
            group = self.groups[group_id]
            for download_id in group.files:
                self.cancel_download(download_id)
    
    def remove_group(self, group_id: str):
        """Remove an entire download group."""
        if group_id in self.groups:
            group = self.groups[group_id]
            
            # Cancel all active downloads first
            for download_id in group.files:
                if download_id in self._stop_flags:
                    self._stop_flags[download_id].set()
            
            # Remove the group
            del self.groups[group_id]
            self._trigger_callback('on_remove', group)
    
    def _process_queue(self):
        """Process download queue."""
        while True:
            # Check if we can start more downloads
            if len(self.active_downloads) < self.max_concurrent:
                try:
                    download_id = self.download_queue.get(timeout=1)
                    
                    # Find the download item
                    item = None
                    for group in self.groups.values():
                        if download_id in group.files:
                            item = group.files[download_id]
                            break
                    
                    if item and item.selected:
                        thread = threading.Thread(
                            target=self._download_file,
                            args=(download_id,),
                            daemon=True
                        )
                        self.active_downloads[download_id] = thread
                        thread.start()
                        
                except queue.Empty:
                    pass
            
            # Clean up finished downloads
            finished = []
            for download_id, thread in self.active_downloads.items():
                if not thread.is_alive():
                    finished.append(download_id)
            
            for download_id in finished:
                del self.active_downloads[download_id]
            
            time.sleep(0.5)
    
    def _download_file(self, download_id: str):
        """Download a file with resume support."""
        # Find the item
        item = None
        group = None
        for g in self.groups.values():
            if download_id in g.files:
                item = g.files[download_id]
                group = g
                break
        
        if not item or not item.selected:
            return
        
        # Create flags for this download
        self._stop_flags[download_id] = threading.Event()
        self._pause_flags[download_id] = threading.Event()
        
        try:
            # Update status
            item.status = DownloadStatus.DOWNLOADING
            item.start_time = time.time()
            self._trigger_callback('on_status_change', item)
            self._trigger_callback('on_group_change', group)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(item.save_path), exist_ok=True)
            
            # Setup headers for resume
            headers = item.headers.copy()
            if item.resume_position > 0:
                headers['Range'] = f'bytes={item.resume_position}-'
            
            # Make request
            response = requests.get(item.url, headers=headers, stream=True, timeout=30)
            
            # Check for authentication issues
            if response.status_code == 401 or response.status_code == 403:
                item.status = DownloadStatus.AUTH_REQUIRED
                item.error_message = f"Authentication failed: {response.status_code}"
                self._trigger_callback('on_error', item)
                self._trigger_callback('on_group_change', group)
                return
            
            response.raise_for_status()
            
            # Get total size
            if item.resume_position == 0:
                item.total_size = int(response.headers.get('content-length', 0))
            
            # Open file for writing (append if resuming)
            mode = 'ab' if item.resume_position > 0 else 'wb'
            
            # Optimize chunk size based on file size
            base_chunk_size = 1024 * 1024  # 1MB base chunk
            if item.total_size > 100 * 1024 * 1024:  # Files > 100MB
                chunk_size = base_chunk_size * 4  # 4MB chunks
            elif item.total_size > 10 * 1024 * 1024:  # Files > 10MB
                chunk_size = base_chunk_size * 2  # 2MB chunks
            else:
                chunk_size = base_chunk_size  # 1MB chunks
            
            # Use buffered writing for better performance
            buffer_size = chunk_size * 8
            
            with open(item.save_path, mode, buffering=buffer_size) as f:
                last_update = time.time()
                bytes_since_update = 0
                update_interval = 0.5  # Update UI every 0.5 seconds
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    # Check stop flag
                    if self._stop_flags[download_id].is_set():
                        return
                    
                    # Check pause flag
                    if self._pause_flags[download_id].is_set():
                        item.status = DownloadStatus.PAUSED
                        self._trigger_callback('on_status_change', item)
                        self._trigger_callback('on_group_change', group)
                        return
                    
                    if chunk:
                        f.write(chunk)
                        item.downloaded_size += len(chunk)
                        bytes_since_update += len(chunk)
                        
                        # Calculate speed and ETA
                        current_time = time.time()
                        time_diff = current_time - last_update
                        
                        if time_diff >= update_interval:
                            item.speed = bytes_since_update / time_diff
                            
                            if item.speed > 0 and item.total_size > item.downloaded_size:
                                remaining = item.total_size - item.downloaded_size
                                item.eta = int(remaining / item.speed)
                            
                            self._trigger_callback('on_progress', item)
                            self._trigger_callback('on_group_change', group)
                            last_update = current_time
                            bytes_since_update = 0
                            
                            # Force flush for USB drives
                            f.flush()
                            os.fsync(f.fileno())
            
            # Download completed
            item.status = DownloadStatus.COMPLETED
            item.end_time = time.time()
            self._trigger_callback('on_complete', item)
            self._trigger_callback('on_group_change', group)
            
        except requests.exceptions.RequestException as e:
            item.status = DownloadStatus.FAILED
            item.error_message = str(e)
            self._trigger_callback('on_error', item)
            self._trigger_callback('on_group_change', group)
            
        except Exception as e:
            item.status = DownloadStatus.FAILED
            item.error_message = f"Unexpected error: {e}"
            self._trigger_callback('on_error', item)
            self._trigger_callback('on_group_change', group)
        
        finally:
            # Clean up flags
            if download_id in self._stop_flags:
                del self._stop_flags[download_id]
            if download_id in self._pause_flags:
                del self._pause_flags[download_id]
            
            self._trigger_callback('on_status_change', item)
            self._trigger_callback('on_group_change', group)
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for download events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, item):
        """Trigger callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(item)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def get_all_groups(self) -> List[DownloadGroup]:
        """Get all download groups."""
        return list(self.groups.values())
    
    def clear_completed_groups(self):
        """Clear completed download groups."""
        to_remove = []
        for group_id, group in self.groups.items():
            if group.status == DownloadStatus.COMPLETED:
                to_remove.append(group_id)
        
        for group_id in to_remove:
            self.remove_group(group_id)


class FileSelectionDialog:
    """Dialog for selecting which files to download from a model."""
    
    def __init__(self, parent: tk.Tk, repo_id: str, files: List[Dict[str, Any]], 
                 title: str = "Select Files to Download"):
        self.parent = parent
        self.repo_id = repo_id
        self.files = files
        self.selected_files = []
        self.result = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)
        
        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.build_ui()
        self.center_dialog()
    
    def build_ui(self):
        """Build the file selection dialog UI."""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and info
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text=f"Select files to download from:", 
                 font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(title_frame, text=self.repo_id, 
                 font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W)
        
        # Selection controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="Select All", 
                  command=self.select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Deselect All", 
                  command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Select GGUF Only", 
                  command=self.select_gguf_only).pack(side=tk.LEFT, padx=5)
        
        # File list
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview with file info
        columns = ("size", "type", "modified")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="tree headings", height=15)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure columns
        self.file_tree.heading("#0", text="Filename")
        self.file_tree.heading("size", text="Size")
        self.file_tree.heading("type", text="Type")
        self.file_tree.heading("modified", text="Modified")
        
        self.file_tree.column("#0", width=300)
        self.file_tree.column("size", width=100)
        self.file_tree.column("type", width=80)
        self.file_tree.column("modified", width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        # Populate file list
        self.file_vars = {}
        self.populate_file_list()
        
        # Bind click events
        self.file_tree.bind("<Button-1>", self.on_tree_click)
        
        # Summary frame
        summary_frame = ttk.LabelFrame(main_frame, text="Download Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.summary_label = ttk.Label(summary_frame, text="")
        self.summary_label.pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Download Selected", 
                  command=self.download_selected).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", 
                  command=self.cancel).pack(side=tk.RIGHT)
        
        # Update summary
        self.update_summary()
    
    def populate_file_list(self):
        """Populate the file list with checkboxes."""
        for i, file_info in enumerate(self.files):
            filename = file_info.get('rfilename', file_info.get('path', 'Unknown'))
            size = self.format_size(file_info.get('size', 0))
            file_type = self.get_file_type(filename)
            modified = file_info.get('lastModified', 'Unknown')
            
            # Create selection variable
            var = tk.BooleanVar(value=self.should_auto_select(filename))
            self.file_vars[filename] = var
            
            # Insert into tree
            item_id = self.file_tree.insert(
                "", "end", 
                text=f"☐ {filename}",
                values=(size, file_type, modified),
                tags=(filename,)
            )
    
    def should_auto_select(self, filename: str) -> bool:
        """Determine if a file should be auto-selected."""
        # Auto-select GGUF files and small files like README
        if filename.lower().endswith('.gguf'):
            return True
        if filename.lower() in ['readme.md', 'readme', 'license', 'license.txt']:
            return True
        if any(filename.lower().endswith(ext) for ext in ['.json', '.txt']) and 'config' in filename.lower():
            return True
        return False
    
    def get_file_type(self, filename: str) -> str:
        """Get file type from extension."""
        ext = os.path.splitext(filename)[1].lower()
        type_map = {
            '.gguf': 'GGUF',
            '.safetensors': 'SafeTensors',
            '.bin': 'Binary',
            '.json': 'Config',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.py': 'Python',
            '.yaml': 'YAML',
            '.yml': 'YAML'
        }
        return type_map.get(ext, 'Other')
    
    def format_size(self, bytes_size: int) -> str:
        """Format bytes to human readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
    
    def on_tree_click(self, event):
        """Handle tree item clicks."""
        item = self.file_tree.identify_row(event.y)
        if item:
            tags = self.file_tree.item(item, "tags")
            if tags:
                filename = tags[0]
                if filename in self.file_vars:
                    # Toggle selection
                    current = self.file_vars[filename].get()
                    self.file_vars[filename].set(not current)
                    self.update_file_display(item, filename)
                    self.update_summary()
    
    def update_file_display(self, item_id: str, filename: str):
        """Update file display based on selection."""
        selected = self.file_vars[filename].get()
        checkbox = "☑" if selected else "☐"
        text = f"{checkbox} {filename}"
        self.file_tree.item(item_id, text=text)
    
    def select_all(self):
        """Select all files."""
        for var in self.file_vars.values():
            var.set(True)
        self.refresh_display()
    
    def deselect_all(self):
        """Deselect all files."""
        for var in self.file_vars.values():
            var.set(False)
        self.refresh_display()
    
    def select_gguf_only(self):
        """Select only GGUF files and essential files."""
        for filename, var in self.file_vars.items():
            should_select = (
                filename.lower().endswith('.gguf') or
                filename.lower() in ['readme.md', 'readme', 'license'] or
                'config' in filename.lower()
            )
            var.set(should_select)
        self.refresh_display()
    
    def refresh_display(self):
        """Refresh the display of all items."""
        for item in self.file_tree.get_children():
            tags = self.file_tree.item(item, "tags")
            if tags:
                filename = tags[0]
                self.update_file_display(item, filename)
        self.update_summary()
    
    def update_summary(self):
        """Update the download summary."""
        selected_count = sum(1 for var in self.file_vars.values() if var.get())
        total_count = len(self.file_vars)
        
        selected_size = sum(
            file_info.get('size', 0) 
            for i, file_info in enumerate(self.files)
            if self.file_vars.get(file_info.get('rfilename', file_info.get('path', '')), tk.BooleanVar()).get()
        )
        
        size_text = self.format_size(selected_size)
        self.summary_label.config(
            text=f"Selected: {selected_count}/{total_count} files, Total size: {size_text}"
        )
    
    def download_selected(self):
        """Start download of selected files."""
        self.selected_files = [
            (filename, file_info) 
            for filename, var in self.file_vars.items() 
            if var.get()
            for file_info in self.files
            if file_info.get('rfilename', file_info.get('path', '')) == filename
        ]
        
        if not self.selected_files:
            messagebox.showwarning("No Selection", "Please select at least one file to download.")
            return
        
        self.result = 'download'
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel the dialog."""
        self.result = 'cancel'
        self.dialog.destroy()
    
    def center_dialog(self):
        """Center dialog on parent."""
        self.dialog.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def show(self):
        """Show the dialog and return the result."""
        self.dialog.wait_window()
        return self.result, self.selected_files