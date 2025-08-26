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
            'on_error': []
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
            self.downloads[download_id].status = DownloadStatus.CANCELLED
            self._trigger_callback('on_status_change', self.downloads[download_id])
            
            # Remove partial file
            item = self.downloads[download_id]
            if os.path.exists(item.save_path):
                try:
                    os.remove(item.save_path)
                except Exception:
                    pass
    
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
            
            with open(item.save_path, mode) as f:
                chunk_size = 8192
                last_update = time.time()
                bytes_since_update = 0
                
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
                        
                        # Calculate speed and ETA
                        current_time = time.time()
                        time_diff = current_time - last_update
                        
                        if time_diff >= 1.0:  # Update every second
                            item.speed = bytes_since_update / time_diff
                            
                            if item.speed > 0:
                                remaining = item.total_size - item.downloaded_size
                                item.eta = int(remaining / item.speed)
                            
                            self._trigger_callback('on_progress', item)
                            last_update = current_time
                            bytes_since_update = 0
            
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
            del self.downloads[download_id]


class DownloadManagerTab:
    """Download Manager GUI tab."""
    
    def __init__(self, parent: ttk.Frame, download_manager: DownloadManager):
        self.parent = parent
        self.manager = download_manager
        self.item_widgets: Dict[str, Dict[str, Any]] = {}
        
        # Register callbacks
        self.manager.register_callback('on_progress', self._on_progress)
        self.manager.register_callback('on_status_change', self._on_status_change)
        self.manager.register_callback('on_complete', self._on_complete)
        self.manager.register_callback('on_error', self._on_error)
        
        self._build_ui()
        
        # Start update timer
        self._update_display()
    
    def _build_ui(self):
        """Build the download manager UI."""
        # Top controls
        controls_frame = ttk.Frame(self.parent)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Clear Completed", 
                  command=self._clear_completed).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Pause All", 
                  command=self._pause_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Resume All", 
                  command=self._resume_all).pack(side=tk.LEFT, padx=5)
        
        # Status summary
        self.status_label = ttk.Label(controls_frame, text="Downloads: 0 active, 0 queued")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Downloads list with scrollbar
        list_frame = ttk.Frame(self.parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for scrollable content
        self.canvas = tk.Canvas(list_frame, bg='white')
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def add_download_widget(self, item: DownloadItem):
        """Add a download widget to the display."""
        if item.id in self.item_widgets:
            return
        
        # Create frame for this download
        frame = ttk.LabelFrame(self.scrollable_frame, text=f"{item.repo_id} / {item.filename}")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Info row
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        status_label = ttk.Label(info_frame, text=f"Status: {item.status.value}")
        status_label.pack(side=tk.LEFT, padx=5)
        
        size_label = ttk.Label(info_frame, text="Size: -")
        size_label.pack(side=tk.LEFT, padx=5)
        
        speed_label = ttk.Label(info_frame, text="Speed: -")
        speed_label.pack(side=tk.LEFT, padx=5)
        
        eta_label = ttk.Label(info_frame, text="ETA: -")
        eta_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_bar = ttk.Progressbar(frame, length=400, mode='determinate')
        progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Error label (hidden by default)
        error_label = ttk.Label(frame, text="", foreground="red")
        
        # Control buttons
        controls_frame = ttk.Frame(frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        pause_btn = ttk.Button(controls_frame, text="Pause", width=10,
                               command=lambda: self.manager.pause_download(item.id))
        pause_btn.pack(side=tk.LEFT, padx=2)
        
        resume_btn = ttk.Button(controls_frame, text="Resume", width=10,
                               command=lambda: self.manager.resume_download(item.id))
        resume_btn.pack(side=tk.LEFT, padx=2)
        
        cancel_btn = ttk.Button(controls_frame, text="Cancel", width=10,
                               command=lambda: self.manager.cancel_download(item.id))
        cancel_btn.pack(side=tk.LEFT, padx=2)
        
        retry_btn = ttk.Button(controls_frame, text="Retry", width=10,
                              command=lambda: self.manager.retry_download(item.id))
        retry_btn.pack(side=tk.LEFT, padx=2)
        
        # Store widgets
        self.item_widgets[item.id] = {
            'frame': frame,
            'status_label': status_label,
            'size_label': size_label,
            'speed_label': speed_label,
            'eta_label': eta_label,
            'progress_bar': progress_bar,
            'error_label': error_label,
            'pause_btn': pause_btn,
            'resume_btn': resume_btn,
            'cancel_btn': cancel_btn,
            'retry_btn': retry_btn
        }
        
        # Initial update
        self._update_download_widget(item)
    
    def _update_download_widget(self, item: DownloadItem):
        """Update a download widget."""
        if item.id not in self.item_widgets:
            self.add_download_widget(item)
            return
        
        widgets = self.item_widgets[item.id]
        
        # Update status
        widgets['status_label'].config(text=f"Status: {item.status.value}")
        
        # Update size
        if item.total_size > 0:
            size_text = f"Size: {self._format_size(item.downloaded_size)} / {self._format_size(item.total_size)}"
        else:
            size_text = f"Size: {self._format_size(item.downloaded_size)}"
        widgets['size_label'].config(text=size_text)
        
        # Update speed
        if item.speed > 0:
            widgets['speed_label'].config(text=f"Speed: {self._format_size(item.speed)}/s")
        else:
            widgets['speed_label'].config(text="Speed: -")
        
        # Update ETA
        if item.eta > 0:
            eta_text = self._format_time(item.eta)
            widgets['eta_label'].config(text=f"ETA: {eta_text}")
        else:
            widgets['eta_label'].config(text="ETA: -")
        
        # Update progress bar
        widgets['progress_bar']['value'] = item.progress
        
        # Update error message
        if item.error_message:
            widgets['error_label'].config(text=f"Error: {item.error_message}")
            widgets['error_label'].pack(fill=tk.X, padx=5, pady=2)
        else:
            widgets['error_label'].pack_forget()
        
        # Update button states
        if item.status == DownloadStatus.DOWNLOADING:
            widgets['pause_btn'].config(state="normal")
            widgets['resume_btn'].config(state="disabled")
            widgets['cancel_btn'].config(state="normal")
            widgets['retry_btn'].config(state="disabled")
        elif item.status == DownloadStatus.PAUSED:
            widgets['pause_btn'].config(state="disabled")
            widgets['resume_btn'].config(state="normal")
            widgets['cancel_btn'].config(state="normal")
            widgets['retry_btn'].config(state="disabled")
        elif item.status in [DownloadStatus.FAILED, DownloadStatus.AUTH_REQUIRED]:
            widgets['pause_btn'].config(state="disabled")
            widgets['resume_btn'].config(state="disabled")
            widgets['cancel_btn'].config(state="disabled")
            widgets['retry_btn'].config(state="normal")
        elif item.status == DownloadStatus.COMPLETED:
            widgets['pause_btn'].config(state="disabled")
            widgets['resume_btn'].config(state="disabled")
            widgets['cancel_btn'].config(state="disabled")
            widgets['retry_btn'].config(state="disabled")
        else:
            widgets['pause_btn'].config(state="disabled")
            widgets['resume_btn'].config(state="disabled")
            widgets['cancel_btn'].config(state="normal")
            widgets['retry_btn'].config(state="disabled")
    
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
    
    def _on_progress(self, item: DownloadItem):
        """Handle progress update."""
        self.parent.after(0, lambda: self._update_download_widget(item))
    
    def _on_status_change(self, item: DownloadItem):
        """Handle status change."""
        self.parent.after(0, lambda: self._update_download_widget(item))
    
    def _on_complete(self, item: DownloadItem):
        """Handle download completion."""
        self.parent.after(0, lambda: self._update_download_widget(item))
    
    def _on_error(self, item: DownloadItem):
        """Handle download error."""
        self.parent.after(0, lambda: self._update_download_widget(item))
        
        # Show error notification for auth issues
        if item.status == DownloadStatus.AUTH_REQUIRED:
            self.parent.after(0, lambda: messagebox.showerror(
                "Authentication Required",
                f"Authentication failed for {item.filename}.\n"
                f"Please check your API key in Settings."
            ))
    
    def _clear_completed(self):
        """Clear completed downloads."""
        # Remove widgets for completed downloads
        for download_id in list(self.item_widgets.keys()):
            item = self.manager.downloads.get(download_id)
            if item and item.status in [DownloadStatus.COMPLETED, DownloadStatus.CANCELLED]:
                self.item_widgets[download_id]['frame'].destroy()
                del self.item_widgets[download_id]
        
        # Clear from manager
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
        """Update the display periodically."""
        # Update status summary
        all_downloads = self.manager.get_all_downloads()
        active = sum(1 for d in all_downloads if d.status == DownloadStatus.DOWNLOADING)
        queued = sum(1 for d in all_downloads if d.status == DownloadStatus.QUEUED)
        self.status_label.config(text=f"Downloads: {active} active, {queued} queued")
        
        # Add any new downloads
        for item in all_downloads:
            if item.id not in self.item_widgets:
                self.add_download_item(item)
        
        # Schedule next update
        self.parent.after(500, self._update_display)