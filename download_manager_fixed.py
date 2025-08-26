import os
import requests
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import queue


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
    """Represents a download item."""
    id: str
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
    headers: Dict[str, str] = None
    resume_position: int = 0
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
    
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


class DownloadManager:
    """Manages multiple downloads with pause/resume support."""
    
    def __init__(self, max_concurrent: int = 3):
        self.downloads: Dict[str, DownloadItem] = {}
        self.download_queue: queue.Queue = queue.Queue()
        self.active_downloads: Dict[str, threading.Thread] = {}
        self.max_concurrent = max_concurrent
        self.callbacks: Dict[str, List[Callable]] = {
            'on_progress': [],
            'on_status_change': [],
            'on_complete': [],
            'on_error': [],
            'on_remove': []
        }
        self._stop_flags: Dict[str, threading.Event] = {}
        self._pause_flags: Dict[str, threading.Event] = {}
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
    
    def add_download(self, repo_id: str, filename: str, url: str, save_path: str, 
                     headers: Optional[Dict[str, str]] = None) -> str:
        """Add a new download to the queue."""
        download_id = f"{repo_id}_{filename}_{int(time.time())}"
        
        # Create download item
        item = DownloadItem(
            id=download_id,
            repo_id=repo_id,
            filename=filename,
            url=url,
            save_path=save_path,
            headers=headers or {}
        )
        
        self.downloads[download_id] = item
        self.download_queue.put(download_id)
        self._trigger_callback('on_status_change', item)
        
        return download_id
    
    def pause_download(self, download_id: str):
        """Pause a download."""
        if download_id in self._pause_flags:
            self._pause_flags[download_id].set()
            if download_id in self.downloads:
                self.downloads[download_id].status = DownloadStatus.PAUSED
                self._trigger_callback('on_status_change', self.downloads[download_id])
    
    def resume_download(self, download_id: str):
        """Resume a paused download."""
        item = self.downloads.get(download_id)
        if item and item.is_resumable:
            item.status = DownloadStatus.QUEUED
            item.resume_position = item.downloaded_size
            self.download_queue.put(download_id)
            self._trigger_callback('on_status_change', item)
    
    def cancel_download(self, download_id: str):
        """Cancel a download."""
        if download_id in self._stop_flags:
            self._stop_flags[download_id].set()
        
        if download_id in self.downloads:
            item = self.downloads[download_id]
            item.status = DownloadStatus.CANCELLED
            self._trigger_callback('on_status_change', item)
            
            # Remove partial file
            if os.path.exists(item.save_path):
                try:
                    os.remove(item.save_path)
                except Exception:
                    pass
            
            # Auto-remove cancelled download after a short delay
            def auto_remove():
                time.sleep(2)  # Wait 2 seconds
                self.remove_download(download_id)
            
            threading.Thread(target=auto_remove, daemon=True).start()
    
    def remove_download(self, download_id: str):
        """Remove a specific download from the list."""
        if download_id in self.downloads:
            # Cancel if still active
            if download_id in self._stop_flags:
                self._stop_flags[download_id].set()
            
            # Remove from downloads
            item = self.downloads[download_id]
            del self.downloads[download_id]
            self._trigger_callback('on_remove', item)
    
    def retry_download(self, download_id: str):
        """Retry a failed download."""
        item = self.downloads.get(download_id)
        if item and item.status in [DownloadStatus.FAILED, DownloadStatus.AUTH_REQUIRED]:
            item.status = DownloadStatus.QUEUED
            item.downloaded_size = 0
            item.resume_position = 0
            item.error_message = ""
            self.download_queue.put(download_id)
            self._trigger_callback('on_status_change', item)
    
    def _process_queue(self):
        """Process download queue."""
        while True:
            # Check if we can start more downloads
            if len(self.active_downloads) < self.max_concurrent:
                try:
                    download_id = self.download_queue.get(timeout=1)
                    if download_id in self.downloads:
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
        item = self.downloads[download_id]
        
        # Create flags for this download
        self._stop_flags[download_id] = threading.Event()
        self._pause_flags[download_id] = threading.Event()
        
        try:
            # Update status
            item.status = DownloadStatus.DOWNLOADING
            item.start_time = time.time()
            self._trigger_callback('on_status_change', item)
            
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
                return
            
            response.raise_for_status()
            
            # Get total size
            if item.resume_position == 0:
                item.total_size = int(response.headers.get('content-length', 0))
            
            # Open file for writing (append if resuming)
            mode = 'ab' if item.resume_position > 0 else 'wb'
            
            # Optimize chunk size based on file size and storage type
            base_chunk_size = 1024 * 1024  # 1MB base chunk
            if item.total_size > 100 * 1024 * 1024:  # Files > 100MB
                chunk_size = base_chunk_size * 4  # 4MB chunks
            elif item.total_size > 10 * 1024 * 1024:  # Files > 10MB
                chunk_size = base_chunk_size * 2  # 2MB chunks
            else:
                chunk_size = base_chunk_size  # 1MB chunks
            
            # Use buffered writing for better performance
            buffer_size = chunk_size * 8  # 8x chunk size buffer
            
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
                        return
                    
                    if chunk:
                        f.write(chunk)
                        item.downloaded_size += len(chunk)
                        bytes_since_update += len(chunk)
                        
                        # Calculate speed and ETA (less frequent updates for performance)
                        current_time = time.time()
                        time_diff = current_time - last_update
                        
                        if time_diff >= update_interval:
                            item.speed = bytes_since_update / time_diff
                            
                            if item.speed > 0 and item.total_size > item.downloaded_size:
                                remaining = item.total_size - item.downloaded_size
                                item.eta = int(remaining / item.speed)
                            
                            self._trigger_callback('on_progress', item)
                            last_update = current_time
                            bytes_since_update = 0
                            
                            # Force flush for USB drives
                            f.flush()
                            os.fsync(f.fileno())
            
            # Download completed
            item.status = DownloadStatus.COMPLETED
            item.end_time = time.time()
            self._trigger_callback('on_complete', item)
            
        except requests.exceptions.RequestException as e:
            item.status = DownloadStatus.FAILED
            item.error_message = str(e)
            self._trigger_callback('on_error', item)
            
        except Exception as e:
            item.status = DownloadStatus.FAILED
            item.error_message = f"Unexpected error: {e}"
            self._trigger_callback('on_error', item)
        
        finally:
            # Clean up flags
            if download_id in self._stop_flags:
                del self._stop_flags[download_id]
            if download_id in self._pause_flags:
                del self._pause_flags[download_id]
            
            self._trigger_callback('on_status_change', item)
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for download events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, item: DownloadItem):
        """Trigger callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(item)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def get_all_downloads(self) -> List[DownloadItem]:
        """Get all download items."""
        return list(self.downloads.values())
    
    def clear_completed(self):
        """Clear completed downloads from the list."""
        to_remove = []
        for download_id, item in self.downloads.items():
            if item.status in [DownloadStatus.COMPLETED, DownloadStatus.CANCELLED]:
                to_remove.append(download_id)
        
        for download_id in to_remove:
            self.remove_download(download_id)


class DownloadManagerTab:
    """Download Manager GUI tab with improved real-time updates."""
    
    def __init__(self, parent: ttk.Frame, download_manager: DownloadManager):
        self.parent = parent
        self.manager = download_manager
        self.item_widgets: Dict[str, Dict[str, Any]] = {}
        
        # Register callbacks
        self.manager.register_callback('on_progress', self._on_progress)
        self.manager.register_callback('on_status_change', self._on_status_change)
        self.manager.register_callback('on_complete', self._on_complete)
        self.manager.register_callback('on_error', self._on_error)
        self.manager.register_callback('on_remove', self._on_remove)
        
        self._build_ui()
        
        # Start update timer with shorter interval for real-time updates
        self._update_display()
    
    def _build_ui(self):
        """Build the download manager UI."""
        # Top controls with better layout
        controls_frame = ttk.Frame(self.parent)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left side controls
        left_controls = ttk.Frame(controls_frame)
        left_controls.pack(side=tk.LEFT)
        
        ttk.Button(left_controls, text="Clear Completed", 
                  command=self._clear_completed).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Pause All", 
                  command=self._pause_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Resume All", 
                  command=self._resume_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Refresh", 
                  command=self._refresh_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Remove Selected", 
                  command=self._remove_selected).pack(side=tk.LEFT, padx=2)
        
        # Right side status
        right_controls = ttk.Frame(controls_frame)
        right_controls.pack(side=tk.RIGHT)
        
        self.status_label = ttk.Label(right_controls, text="Downloads: 0 active, 0 queued")
        self.status_label.pack(side=tk.RIGHT)
        
        # Main downloads area using Treeview for better layout
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for downloads
        columns = ("repo", "file", "status", "progress", "size", "speed", "eta")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        self.tree.heading("repo", text="Repository")
        self.tree.heading("file", text="File")
        self.tree.heading("status", text="Status")
        self.tree.heading("progress", text="Progress")
        self.tree.heading("size", text="Size")
        self.tree.heading("speed", text="Speed")
        self.tree.heading("eta", text="ETA")
        
        self.tree.column("repo", width=200)
        self.tree.column("file", width=250)
        self.tree.column("status", width=100)
        self.tree.column("progress", width=100)
        self.tree.column("size", width=100)
        self.tree.column("speed", width=100)
        self.tree.column("eta", width=80)
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Context menu for downloads
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Pause", command=self._context_pause)
        self.context_menu.add_command(label="Resume", command=self._context_resume)
        self.context_menu.add_command(label="Cancel", command=self._context_cancel)
        self.context_menu.add_command(label="Retry", command=self._context_retry)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Remove", command=self._context_remove)
        self.context_menu.add_command(label="Open Folder", command=self._context_open_folder)
        
        self.tree.bind("<Button-3>", self._show_context_menu)
        
        # Details frame for selected download
        details_frame = ttk.LabelFrame(self.parent, text="Download Details", padding=5)
        details_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.details_text = tk.Text(details_frame, height=3, wrap=tk.WORD)
        self.details_text.pack(fill=tk.X)
        
        self.tree.bind("<<TreeviewSelect>>", self._on_selection_change)
    
    def add_download_item(self, item: DownloadItem):
        """Add a download item to the treeview."""
        if item.id in self.item_widgets:
            return
        
        # Insert into treeview
        tree_id = self.tree.insert("", tk.END, values=(
            item.repo_id,
            item.filename,
            item.status.value,
            f"{item.progress:.1f}%",
            self._format_size(item.total_size) if item.total_size > 0 else "-",
            "-",
            "-"
        ))
        
        # Store mapping
        self.item_widgets[item.id] = {
            'tree_id': tree_id,
            'item': item
        }
        
        # Initial update
        self._update_download_item(item)
    
    def _update_download_item(self, item: DownloadItem):
        """Update a download item in the treeview."""
        if item.id not in self.item_widgets:
            self.add_download_item(item)
            return
        
        tree_id = self.item_widgets[item.id]['tree_id']
        
        # Check if tree item still exists
        if not self.tree.exists(tree_id):
            # Tree item was deleted, remove from our tracking
            del self.item_widgets[item.id]
            return
        
        # Update size display
        if item.total_size > 0:
            size_text = f"{self._format_size(item.downloaded_size)} / {self._format_size(item.total_size)}"
        else:
            size_text = self._format_size(item.downloaded_size)
        
        # Update speed display
        speed_text = f"{self._format_size(item.speed)}/s" if item.speed > 0 else "-"
        
        # Update ETA display
        eta_text = self._format_time(item.eta) if item.eta > 0 else "-"
        
        try:
            # Update treeview row
            self.tree.item(tree_id, values=(
                item.repo_id,
                item.filename,
                item.status.value,
                f"{item.progress:.1f}%",
                size_text,
                speed_text,
                eta_text
            ))
            
            # Update row color based on status
            if item.status == DownloadStatus.COMPLETED:
                self.tree.item(tree_id, tags=("completed",))
            elif item.status == DownloadStatus.FAILED:
                self.tree.item(tree_id, tags=("failed",))
            elif item.status == DownloadStatus.DOWNLOADING:
                self.tree.item(tree_id, tags=("downloading",))
            else:
                self.tree.item(tree_id, tags=())
            
            # Configure tag colors
            self.tree.tag_configure("completed", background="#d4edda")
            self.tree.tag_configure("failed", background="#f8d7da")
            self.tree.tag_configure("downloading", background="#d1ecf1")
            
        except tk.TclError:
            # Tree item no longer exists
            if item.id in self.item_widgets:
                del self.item_widgets[item.id]
    
    def _format_size(self, bytes_size: float) -> str:
        """Format bytes to human readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
    
    def _format_time(self, seconds: int) -> str:
        """Format seconds to human readable time."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def _show_context_menu(self, event):
        """Show context menu for downloads."""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            self.context_menu.post(event.x_root, event.y_root)
    
    def _get_selected_download_id(self):
        """Get the download ID of the selected item."""
        selection = self.tree.selection()
        if not selection:
            return None
        
        tree_id = selection[0]
        for download_id, data in self.item_widgets.items():
            if data['tree_id'] == tree_id:
                return download_id
        return None
    
    def _context_pause(self):
        download_id = self._get_selected_download_id()
        if download_id:
            self.manager.pause_download(download_id)
    
    def _context_resume(self):
        download_id = self._get_selected_download_id()
        if download_id:
            self.manager.resume_download(download_id)
    
    def _context_cancel(self):
        download_id = self._get_selected_download_id()
        if download_id:
            self.manager.cancel_download(download_id)
    
    def _context_retry(self):
        download_id = self._get_selected_download_id()
        if download_id:
            self.manager.retry_download(download_id)
    
    def _context_remove(self):
        download_id = self._get_selected_download_id()
        if download_id:
            self.manager.remove_download(download_id)
    
    def _context_open_folder(self):
        download_id = self._get_selected_download_id()
        if download_id and download_id in self.manager.downloads:
            item = self.manager.downloads[download_id]
            folder = os.path.dirname(item.save_path)
            if os.path.exists(folder):
                import subprocess
                import platform
                if platform.system() == "Windows":
                    subprocess.run(["explorer", folder])
                elif platform.system() == "Darwin":
                    subprocess.run(["open", folder])
                else:
                    subprocess.run(["xdg-open", folder])
    
    def _refresh_list(self):
        """Refresh the download list display."""
        # Clear all items
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        
        # Clear widget mapping
        self.item_widgets.clear()
        
        # Re-add all downloads
        for item in self.manager.get_all_downloads():
            self.add_download_item(item)
    
    def _remove_selected(self):
        """Remove the selected download."""
        download_id = self._get_selected_download_id()
        if download_id:
            self.manager.remove_download(download_id)
    
    def _on_selection_change(self, event):
        """Handle selection change in treeview."""
        download_id = self._get_selected_download_id()
        if download_id and download_id in self.manager.downloads:
            item = self.manager.downloads[download_id]
            details = f"File: {item.filename}\\n"
            details += f"Repository: {item.repo_id}\\n"
            details += f"Save Path: {item.save_path}"
            
            if item.error_message:
                details += f"\\nError: {item.error_message}"
            
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, details)
    
    def _on_progress(self, item: DownloadItem):
        """Handle progress update."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_download_item(item))
    
    def _on_status_change(self, item: DownloadItem):
        """Handle status change."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_download_item(item))
    
    def _on_complete(self, item: DownloadItem):
        """Handle download completion."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_download_item(item))
    
    def _on_error(self, item: DownloadItem):
        """Handle download error."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_download_item(item))
        
        # Show error notification for auth issues
        if item.status == DownloadStatus.AUTH_REQUIRED:
            if hasattr(self.parent, 'after'):
                self.parent.after(0, lambda: messagebox.showerror(
                    "Authentication Required",
                    f"Authentication failed for {item.filename}.\\n"
                    f"Please check your API key in Settings."
                ))
    
    def _on_remove(self, item: DownloadItem):
        """Handle download removal."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._remove_download_from_tree(item.id))
    
    def _remove_download_from_tree(self, download_id: str):
        """Remove download from treeview."""
        if download_id in self.item_widgets:
            try:
                tree_id = self.item_widgets[download_id]['tree_id']
                self.tree.delete(tree_id)
                del self.item_widgets[download_id]
            except (tk.TclError, KeyError):
                # Item already removed or doesn't exist
                if download_id in self.item_widgets:
                    del self.item_widgets[download_id]
    
    def _clear_completed(self):
        """Clear completed downloads."""
        self.manager.clear_completed()
    
    def _pause_all(self):
        """Pause all active downloads."""
        for item in self.manager.get_all_downloads():
            if item.status == DownloadStatus.DOWNLOADING:
                self.manager.pause_download(item.id)
    
    def _resume_all(self):
        """Resume all paused downloads."""
        for item in self.manager.get_all_downloads():
            if item.status == DownloadStatus.PAUSED:
                self.manager.resume_download(item.id)
    
    def _update_display(self):
        """Update the display periodically with real-time updates."""
        try:
            # Update status summary
            all_downloads = self.manager.get_all_downloads()
            active = sum(1 for d in all_downloads if d.status == DownloadStatus.DOWNLOADING)
            queued = sum(1 for d in all_downloads if d.status == DownloadStatus.QUEUED)
            completed = sum(1 for d in all_downloads if d.status == DownloadStatus.COMPLETED)
            failed = sum(1 for d in all_downloads if d.status in [DownloadStatus.FAILED, DownloadStatus.CANCELLED])
            
            status_text = f"Downloads: {active} active, {queued} queued, {completed} completed, {failed} failed"
            self.status_label.config(text=status_text)
            
            # Add any new downloads
            for item in all_downloads:
                if item.id not in self.item_widgets:
                    self.add_download_item(item)
            
            # Force update all items for real-time progress
            for item in all_downloads:
                if item.id in self.item_widgets:
                    self._update_download_item(item)
            
        except Exception as e:
            print(f"Error updating download display: {e}")
        finally:
            # Schedule next update with shorter interval for real-time feel
            if hasattr(self.parent, 'after'):
                self.parent.after(200, self._update_display)  # Update every 200ms for smooth progress