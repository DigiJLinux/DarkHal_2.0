import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional
import os
import subprocess
import platform
from grouped_download_manager import (
    GroupedDownloadManager, DownloadGroup, DownloadItem, DownloadStatus, FileSelectionDialog
)


class CollapsibleDownloadWidget:
    """A collapsible widget for displaying a download group and its files."""
    
    def __init__(self, parent: ttk.Frame, group: DownloadGroup, manager: GroupedDownloadManager):
        self.parent = parent
        self.group = group
        self.manager = manager
        self.expanded = group.expanded
        
        # Main container frame
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Group header frame (always visible)
        self.header_frame = ttk.Frame(self.main_frame, relief=tk.RAISED, borderwidth=1)
        self.header_frame.pack(fill=tk.X)
        
        # Group details frame (collapsible)
        self.details_frame = ttk.Frame(self.main_frame)
        
        self.build_header()
        self.build_details()
        self.update_display()
        
        # Set initial expansion state
        if self.expanded:
            self.show_details()
        else:
            self.hide_details()
    
    def build_header(self):
        """Build the group header with overall progress and controls."""
        # Left side - expand/collapse button and info
        left_frame = ttk.Frame(self.header_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Expand/collapse button
        self.expand_btn = ttk.Button(
            left_frame, 
            text="▼" if self.expanded else "▶", 
            width=3,
            command=self.toggle_expansion
        )
        self.expand_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Group info
        info_frame = ttk.Frame(left_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Title and status
        title_frame = ttk.Frame(info_frame)
        title_frame.pack(fill=tk.X)
        
        self.title_label = ttk.Label(
            title_frame, 
            text=self.group.name, 
            font=("TkDefaultFont", 10, "bold")
        )
        self.title_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(
            title_frame, 
            text=self.group.status.value,
            foreground=self.get_status_color(self.group.status)
        )
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress info
        progress_frame = ttk.Frame(info_frame)
        progress_frame.pack(fill=tk.X, pady=(2, 0))
        
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(side=tk.LEFT)
        
        self.speed_label = ttk.Label(progress_frame, text="")
        self.speed_label.pack(side=tk.LEFT, padx=(10, 0))
        
        self.eta_label = ttk.Label(progress_frame, text="")
        self.eta_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            info_frame, 
            length=300, 
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(2, 0))
        
        # Right side - control buttons
        control_frame = ttk.Frame(self.header_frame)
        control_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.pause_btn = ttk.Button(
            control_frame, 
            text="Pause", 
            width=8,
            command=self.pause_group
        )
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.resume_btn = ttk.Button(
            control_frame, 
            text="Resume", 
            width=8,
            command=self.resume_group
        )
        self.resume_btn.pack(side=tk.LEFT, padx=2)
        
        self.cancel_btn = ttk.Button(
            control_frame, 
            text="Cancel", 
            width=8,
            command=self.cancel_group
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=2)
        
        self.remove_btn = ttk.Button(
            control_frame, 
            text="Remove", 
            width=8,
            command=self.remove_group
        )
        self.remove_btn.pack(side=tk.LEFT, padx=2)
    
    def build_details(self):
        """Build the collapsible details section with individual files."""
        # File list frame
        files_frame = ttk.LabelFrame(self.details_frame, text="Files", padding=5)
        files_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for files
        columns = ("status", "progress", "size", "speed")
        self.file_tree = ttk.Treeview(
            files_frame, 
            columns=columns, 
            show="tree headings", 
            height=8
        )
        self.file_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure columns
        self.file_tree.heading("#0", text="Filename")
        self.file_tree.heading("status", text="Status")
        self.file_tree.heading("progress", text="Progress")
        self.file_tree.heading("size", text="Size")
        self.file_tree.heading("speed", text="Speed")
        
        self.file_tree.column("#0", width=200)
        self.file_tree.column("status", width=100)
        self.file_tree.column("progress", width=100)
        self.file_tree.column("size", width=100)
        self.file_tree.column("speed", width=100)
        
        # File controls
        file_controls = ttk.Frame(files_frame)
        file_controls.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(
            file_controls, 
            text="Select All", 
            command=self.select_all_files
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            file_controls, 
            text="Deselect All", 
            command=self.deselect_all_files
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            file_controls, 
            text="Open Folder", 
            command=self.open_folder
        ).pack(side=tk.RIGHT, padx=2)
        
        # Bind file selection
        self.file_tree.bind("<Button-1>", self.on_file_click)
        self.file_tree.bind("<Double-1>", self.on_file_double_click)
        
        # Populate files
        self.populate_files()
    
    def populate_files(self):
        """Populate the file tree with download items."""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Add files
        for download_id, item in self.group.files.items():
            checkbox = "☑" if item.selected else "☐"
            filename = f"{checkbox} {item.filename}"
            
            size_text = self.format_size(item.total_size) if item.total_size > 0 else "-"
            if item.downloaded_size > 0 and item.total_size > 0:
                size_text = f"{self.format_size(item.downloaded_size)} / {size_text}"
            
            speed_text = f"{self.format_size(item.speed)}/s" if item.speed > 0 else "-"
            
            tree_id = self.file_tree.insert(
                "", "end",
                text=filename,
                values=(
                    item.status.value,
                    f"{item.progress:.1f}%",
                    size_text,
                    speed_text
                ),
                tags=(download_id,)
            )
            
            # Color code by status
            if item.status == DownloadStatus.COMPLETED:
                self.file_tree.item(tree_id, tags=(download_id, "completed"))
            elif item.status == DownloadStatus.FAILED:
                self.file_tree.item(tree_id, tags=(download_id, "failed"))
            elif item.status == DownloadStatus.DOWNLOADING:
                self.file_tree.item(tree_id, tags=(download_id, "downloading"))
        
        # Configure tag colors
        self.file_tree.tag_configure("completed", background="#d4edda")
        self.file_tree.tag_configure("failed", background="#f8d7da")
        self.file_tree.tag_configure("downloading", background="#d1ecf1")
    
    def toggle_expansion(self):
        """Toggle the expansion state of the widget."""
        self.expanded = not self.expanded
        self.group.expanded = self.expanded  # Update group state
        
        if self.expanded:
            self.show_details()
        else:
            self.hide_details()
        
        self.expand_btn.config(text="▼" if self.expanded else "▶")
    
    def show_details(self):
        """Show the details frame."""
        self.details_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
    
    def hide_details(self):
        """Hide the details frame."""
        self.details_frame.pack_forget()
    
    def update_display(self):
        """Update the display with current group status."""
        # Update status
        self.status_label.config(
            text=self.group.status.value,
            foreground=self.get_status_color(self.group.status)
        )
        
        # Update progress
        self.progress_bar['value'] = self.group.progress
        
        # Update progress text
        if self.group.total_size > 0:
            progress_text = (
                f"{self.group.progress:.1f}% - "
                f"{self.format_size(self.group.downloaded_size)} / "
                f"{self.format_size(self.group.total_size)}"
            )
        else:
            progress_text = f"{self.group.progress:.1f}%"
        
        self.progress_label.config(text=progress_text)
        
        # Update speed and ETA
        if self.group.active_speed > 0:
            self.speed_label.config(text=f"Speed: {self.format_size(self.group.active_speed)}/s")
        else:
            self.speed_label.config(text="")
        
        if self.group.eta > 0:
            self.eta_label.config(text=f"ETA: {self.format_time(self.group.eta)}")
        else:
            self.eta_label.config(text="")
        
        # Update file list if expanded
        if self.expanded:
            self.populate_files()
        
        # Update button states
        self.update_button_states()
    
    def update_button_states(self):
        """Update button states based on group status."""
        status = self.group.status
        
        if status == DownloadStatus.DOWNLOADING:
            self.pause_btn.config(state="normal")
            self.resume_btn.config(state="disabled")
            self.cancel_btn.config(state="normal")
        elif status == DownloadStatus.PAUSED:
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="normal")
            self.cancel_btn.config(state="normal")
        elif status in [DownloadStatus.FAILED, DownloadStatus.AUTH_REQUIRED]:
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="normal")
            self.cancel_btn.config(state="disabled")
        elif status == DownloadStatus.COMPLETED:
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
            self.cancel_btn.config(state="disabled")
        else:  # QUEUED, CANCELLED
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="normal")
            self.cancel_btn.config(state="normal")
    
    def get_status_color(self, status: DownloadStatus) -> str:
        """Get color for status display."""
        color_map = {
            DownloadStatus.QUEUED: "blue",
            DownloadStatus.DOWNLOADING: "green",
            DownloadStatus.PAUSED: "orange",
            DownloadStatus.COMPLETED: "green",
            DownloadStatus.FAILED: "red",
            DownloadStatus.CANCELLED: "gray",
            DownloadStatus.AUTH_REQUIRED: "red"
        }
        return color_map.get(status, "black")
    
    def format_size(self, bytes_size: float) -> str:
        """Format bytes to human readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
    
    def format_time(self, seconds: int) -> str:
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
    
    def on_file_click(self, event):
        """Handle file clicks for selection toggle."""
        item_id = self.file_tree.identify_row(event.y)
        if item_id:
            tags = self.file_tree.item(item_id, "tags")
            if tags:
                download_id = tags[0]
                self.manager.toggle_file_selection(download_id)
    
    def on_file_double_click(self, event):
        """Handle file double-click for individual file controls."""
        item_id = self.file_tree.identify_row(event.y)
        if item_id:
            tags = self.file_tree.item(item_id, "tags")
            if tags:
                download_id = tags[0]
                # Show context menu or individual file controls
                self.show_file_context_menu(event, download_id)
    
    def show_file_context_menu(self, event, download_id: str):
        """Show context menu for individual file."""
        # Find the download item
        item = self.group.files.get(download_id)
        if not item:
            return
        
        context_menu = tk.Menu(self.file_tree, tearoff=0)
        
        if item.selected:
            context_menu.add_command(label="Deselect", command=lambda: self.manager.toggle_file_selection(download_id))
        else:
            context_menu.add_command(label="Select", command=lambda: self.manager.toggle_file_selection(download_id))
        
        context_menu.add_separator()
        
        if item.status == DownloadStatus.DOWNLOADING:
            context_menu.add_command(label="Pause", command=lambda: self.manager.pause_download(download_id))
        elif item.is_resumable:
            context_menu.add_command(label="Resume", command=lambda: self.manager.resume_download(download_id))
        
        if item.status not in [DownloadStatus.COMPLETED]:
            context_menu.add_command(label="Cancel", command=lambda: self.manager.cancel_download(download_id))
        
        context_menu.add_separator()
        context_menu.add_command(label="Open Folder", command=lambda: self.open_file_folder(item))
        
        try:
            context_menu.post(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def select_all_files(self):
        """Select all files in the group."""
        for download_id, item in self.group.files.items():
            if not item.selected:
                self.manager.toggle_file_selection(download_id)
    
    def deselect_all_files(self):
        """Deselect all files in the group."""
        for download_id, item in self.group.files.items():
            if item.selected:
                self.manager.toggle_file_selection(download_id)
    
    def pause_group(self):
        """Pause all downloads in the group."""
        self.manager.pause_group(self.group.id)
    
    def resume_group(self):
        """Resume all downloads in the group."""
        self.manager.resume_group(self.group.id)
    
    def cancel_group(self):
        """Cancel all downloads in the group."""
        if messagebox.askyesno("Cancel Downloads", 
                              f"Cancel all downloads in '{self.group.name}'?"):
            self.manager.cancel_group(self.group.id)
    
    def remove_group(self):
        """Remove the entire group."""
        if messagebox.askyesno("Remove Group", 
                              f"Remove download group '{self.group.name}'?\\nThis will cancel any active downloads."):
            self.manager.remove_group(self.group.id)
    
    def open_folder(self):
        """Open the download folder."""
        if self.group.files:
            # Get folder from first file
            first_file = next(iter(self.group.files.values()))
            folder = os.path.dirname(first_file.save_path)
            self.open_directory(folder)
    
    def open_file_folder(self, item: DownloadItem):
        """Open folder for specific file."""
        folder = os.path.dirname(item.save_path)
        self.open_directory(folder)
    
    def open_directory(self, path: str):
        """Open directory in file explorer."""
        if os.path.exists(path):
            try:
                if platform.system() == "Windows":
                    subprocess.run(["explorer", path])
                elif platform.system() == "Darwin":
                    subprocess.run(["open", path])
                else:
                    subprocess.run(["xdg-open", path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder: {e}")
    
    def destroy(self):
        """Clean up the widget."""
        self.main_frame.destroy()


class GroupedDownloadManagerTab:
    """Main tab for the grouped download manager."""
    
    def __init__(self, parent: ttk.Frame, download_manager: GroupedDownloadManager):
        self.parent = parent
        self.manager = download_manager
        self.group_widgets: Dict[str, CollapsibleDownloadWidget] = {}
        
        # Register callbacks
        self.manager.register_callback('on_progress', self._on_progress)
        self.manager.register_callback('on_status_change', self._on_status_change)
        self.manager.register_callback('on_complete', self._on_complete)
        self.manager.register_callback('on_error', self._on_error)
        self.manager.register_callback('on_remove', self._on_remove)
        self.manager.register_callback('on_group_change', self._on_group_change)
        
        self._build_ui()
        self._update_display()
    
    def _build_ui(self):
        """Build the main UI."""
        # Top controls
        controls_frame = ttk.Frame(self.parent)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left controls
        left_controls = ttk.Frame(controls_frame)
        left_controls.pack(side=tk.LEFT)
        
        ttk.Button(left_controls, text="Clear Completed", 
                  command=self._clear_completed).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Pause All", 
                  command=self._pause_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Resume All", 
                  command=self._resume_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Expand All", 
                  command=self._expand_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_controls, text="Collapse All", 
                  command=self._collapse_all).pack(side=tk.LEFT, padx=2)
        
        # Right status
        right_controls = ttk.Frame(controls_frame)
        right_controls.pack(side=tk.RIGHT)
        
        self.status_label = ttk.Label(right_controls, text="Groups: 0, Downloads: 0")
        self.status_label.pack(side=tk.RIGHT)
        
        # Scrollable frame for download groups
        canvas_frame = ttk.Frame(self.parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        self.scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse wheel
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def add_download_group(self, group: DownloadGroup):
        """Add a new download group widget."""
        if group.id not in self.group_widgets:
            widget = CollapsibleDownloadWidget(self.scrollable_frame, group, self.manager)
            self.group_widgets[group.id] = widget
    
    def _on_progress(self, item):
        """Handle progress updates."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_group_display(item.group_id))
    
    def _on_status_change(self, item):
        """Handle status changes."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_group_display(item.group_id))
    
    def _on_complete(self, item):
        """Handle completion."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_group_display(item.group_id))
    
    def _on_error(self, item):
        """Handle errors."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_group_display(item.group_id))
    
    def _on_remove(self, group):
        """Handle group removal."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._remove_group_widget(group.id))
    
    def _on_group_change(self, group):
        """Handle group changes."""
        if hasattr(self.parent, 'after'):
            self.parent.after(0, lambda: self._update_group_display(group.id))
    
    def _update_group_display(self, group_id: str):
        """Update display for a specific group."""
        if group_id in self.group_widgets and group_id in self.manager.groups:
            widget = self.group_widgets[group_id]
            widget.update_display()
    
    def _remove_group_widget(self, group_id: str):
        """Remove a group widget."""
        if group_id in self.group_widgets:
            widget = self.group_widgets[group_id]
            widget.destroy()
            del self.group_widgets[group_id]
    
    def _clear_completed(self):
        """Clear completed groups."""
        self.manager.clear_completed_groups()
    
    def _pause_all(self):
        """Pause all active downloads."""
        for group in self.manager.get_all_groups():
            self.manager.pause_group(group.id)
    
    def _resume_all(self):
        """Resume all paused downloads."""
        for group in self.manager.get_all_groups():
            self.manager.resume_group(group.id)
    
    def _expand_all(self):
        """Expand all groups."""
        for widget in self.group_widgets.values():
            if not widget.expanded:
                widget.toggle_expansion()
    
    def _collapse_all(self):
        """Collapse all groups."""
        for widget in self.group_widgets.values():
            if widget.expanded:
                widget.toggle_expansion()
    
    def _update_display(self):
        """Update the display periodically."""
        try:
            # Update status summary
            all_groups = self.manager.get_all_groups()
            total_files = sum(len(group.files) for group in all_groups)
            active_downloads = sum(
                1 for group in all_groups 
                for item in group.files.values()
                if item.status == DownloadStatus.DOWNLOADING
            )
            
            self.status_label.config(text=f"Groups: {len(all_groups)}, Downloads: {active_downloads} active / {total_files} total")
            
            # Add any new groups
            for group in all_groups:
                if group.id not in self.group_widgets:
                    self.add_download_group(group)
            
            # Remove widgets for deleted groups
            to_remove = []
            for group_id in self.group_widgets:
                if group_id not in self.manager.groups:
                    to_remove.append(group_id)
            
            for group_id in to_remove:
                self._remove_group_widget(group_id)
            
        except Exception as e:
            print(f"Error updating grouped download display: {e}")
        finally:
            # Schedule next update
            if hasattr(self.parent, 'after'):
                self.parent.after(200, self._update_display)  # Update every 200ms