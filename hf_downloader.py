import os
import requests
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, List, Dict, Any
import threading
import json
from pathlib import Path
from dotenv import load_dotenv

# Load the HuggingFace API key from HUGGINGFACE.env
load_dotenv("HUGGINGFACE.env")


class HuggingFaceAPI:
    """Web API-based HuggingFace interface using direct HTTP requests."""
    
    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        # Ensure API key is properly cleaned of whitespace and newlines
        raw_key = api_key or os.getenv("HF_API_KEY", "")
        self.api_key = raw_key.strip().replace('\n', '').replace('\r', '')
        if not self.api_key:
            raise ValueError("HuggingFace API key not found. Please set HF_API_KEY in HUGGINGFACE.env")
        
        self.organization = organization.strip() if organization else None
        self.base_url = "https://huggingface.co"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Add organization header if specified
        if self.organization:
            self.headers["X-Organization"] = self.organization
    
    def search_models(self, query: str = "", limit: int = 50, sort: str = "downloads") -> List[Dict[str, Any]]:
        """Search for models using the web API."""
        url = f"{self.base_url}/api/models"
        params = {
            "limit": limit,
            "sort": sort,
            "direction": -1,
            "full": True
        }
        if query:
            params["search"] = query
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching models: {e}")
            return []
    
    def search_datasets(self, query: str = "", limit: int = 50, sort: str = "downloads") -> List[Dict[str, Any]]:
        """Search for datasets using the web API."""
        url = f"{self.base_url}/api/datasets"
        params = {
            "limit": limit,
            "sort": sort,
            "direction": -1,
            "full": True
        }
        if query:
            params["search"] = query
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching datasets: {e}")
            return []
    
    def get_model_files(self, repo_id: str) -> List[Dict[str, Any]]:
        """Get list of files in a model repository."""
        url = f"{self.base_url}/api/models/{repo_id}"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("siblings", [])
        except Exception as e:
            print(f"Error getting model files: {e}")
            return []
    
    def download_file(self, repo_id: str, filename: str, save_path: str, 
                     progress_callback=None) -> bool:
        """Download a file from HuggingFace."""
        url = f"{self.base_url}/{repo_id}/resolve/main/{filename}"
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            response = requests.get(url, headers=self.headers, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(progress, downloaded, total_size)
            
            return True
            
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False


class HuggingFaceDownloaderGUI:
    """GUI for HuggingFace model and dataset search and download."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("HuggingFace Downloader")
        self.root.geometry("1200x700")
        
        # Initialize API
        try:
            self.api = HuggingFaceAPI()
        except ValueError as e:
            messagebox.showerror("API Key Error", str(e))
            self.api = None
        
        # Search variables
        self.search_query = tk.StringVar()
        self.search_type = tk.StringVar(value="Models")
        self.filter_most_downloaded = tk.BooleanVar(value=True)
        self.filter_most_liked = tk.BooleanVar(value=False)
        self.filter_size = tk.BooleanVar(value=False)
        
        # Current results storage
        self.current_results = []
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the main UI."""
        # Search bar frame
        search_frame = ttk.Frame(self.root)
        search_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Search entry
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_query, width=60)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_entry.bind("<Return>", lambda e: self._perform_search())
        
        # Search type dropdown
        self.type_dropdown = ttk.Combobox(search_frame, textvariable=self.search_type, 
                                          values=["Models", "Datasets"], 
                                          state="readonly", width=15)
        self.type_dropdown.pack(side=tk.LEFT, padx=(10, 0))
        
        # Search button
        self.search_button = ttk.Button(search_frame, text="Search", command=self._perform_search)
        self.search_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Results frame with treeview
        results_frame = ttk.Frame(self.root)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create treeview with columns
        columns = ("creator", "name", "description", "keywords", "size", "metadata")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=20)
        
        # Define column headings and widths
        self.results_tree.heading("creator", text="Creator")
        self.results_tree.heading("name", text="Name")
        self.results_tree.heading("description", text="Description")
        self.results_tree.heading("keywords", text="Keywords")
        self.results_tree.heading("size", text="Size")
        self.results_tree.heading("metadata", text="Metadata")
        
        self.results_tree.column("creator", width=150)
        self.results_tree.column("name", width=200)
        self.results_tree.column("description", width=300)
        self.results_tree.column("keywords", width=150)
        self.results_tree.column("size", width=100)
        self.results_tree.column("metadata", width=200)
        
        # Scrollbars
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Bind double-click to download
        self.results_tree.bind("<Double-Button-1>", self._on_item_double_click)
        
        # Filter footer frame
        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        
        ttk.Checkbutton(filter_frame, text="Most Downloaded", 
                       variable=self.filter_most_downloaded,
                       command=self._update_filters).pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Checkbutton(filter_frame, text="Most Liked", 
                       variable=self.filter_most_liked,
                       command=self._update_filters).pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Checkbutton(filter_frame, text="Size", 
                       variable=self.filter_size,
                       command=self._update_filters).pack(side=tk.LEFT, padx=(10, 0))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Download button
        download_frame = ttk.Frame(self.root)
        download_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.download_button = ttk.Button(download_frame, text="Download Selected", 
                                         command=self._download_selected)
        self.download_button.pack(side=tk.RIGHT)
    
    def _update_filters(self):
        """Update filter settings and re-sort results if needed."""
        # Ensure at least one filter is selected
        if not any([self.filter_most_downloaded.get(), 
                   self.filter_most_liked.get(), 
                   self.filter_size.get()]):
            self.filter_most_downloaded.set(True)
    
    def _perform_search(self):
        """Perform the search based on current settings."""
        if not self.api:
            messagebox.showerror("Error", "API not initialized")
            return
        
        query = self.search_query.get().strip()
        search_type = self.search_type.get()
        
        # Determine sort parameter
        sort = "downloads"
        if self.filter_most_liked.get() and not self.filter_most_downloaded.get():
            sort = "likes"
        elif self.filter_size.get() and not self.filter_most_downloaded.get() and not self.filter_most_liked.get():
            sort = "lastModified"
        
        self.status_var.set(f"Searching {search_type.lower()}...")
        self.search_button.config(state="disabled")
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Perform search in thread
        thread = threading.Thread(target=self._search_thread, 
                                 args=(query, search_type, sort))
        thread.daemon = True
        thread.start()
    
    def _search_thread(self, query: str, search_type: str, sort: str):
        """Thread function for performing search."""
        try:
            if search_type == "Models":
                results = self.api.search_models(query, limit=50, sort=sort)
            else:
                results = self.api.search_datasets(query, limit=50, sort=sort)
            
            self.current_results = results
            
            # Update UI in main thread
            self.root.after(0, self._populate_results, results, search_type)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Search Error", str(e)))
        finally:
            self.root.after(0, lambda: self.search_button.config(state="normal"))
    
    def _populate_results(self, results: List[Dict], search_type: str):
        """Populate the treeview with search results."""
        count = 0
        
        for item in results:
            try:
                # Extract common fields
                if search_type == "Models":
                    repo_id = item.get("modelId", item.get("id", ""))
                    pipeline_tag = item.get("pipeline_tag", "")
                    tags = item.get("tags", [])
                    keywords = ", ".join(tags[:3]) if tags else pipeline_tag
                else:
                    repo_id = item.get("id", "")
                    task_ids = item.get("cardData", {}).get("task_ids", [])
                    keywords = ", ".join(task_ids[:3]) if task_ids else "dataset"
                
                creator = repo_id.split("/")[0] if "/" in repo_id else ""
                name = repo_id.split("/")[1] if "/" in repo_id else repo_id
                
                # Get description
                description = ""
                if search_type == "Models":
                    description = item.get("description", "")
                else:
                    card_data = item.get("cardData", {})
                    description = card_data.get("description", card_data.get("summary", ""))
                
                # Truncate description
                if len(description) > 100:
                    description = description[:97] + "..."
                
                # Calculate size
                size_bytes = 0
                siblings = item.get("siblings", [])
                for sibling in siblings:
                    if isinstance(sibling, dict):
                        size = sibling.get("size", 0)
                        if isinstance(size, (int, float)):
                            size_bytes += size
                
                size_str = self._format_size(size_bytes) if size_bytes > 0 else "-"
                
                # Get metadata
                metadata_parts = []
                downloads = item.get("downloads", 0)
                likes = item.get("likes", 0)
                
                if downloads > 0:
                    metadata_parts.append(f"↓{self._format_number(downloads)}")
                if likes > 0:
                    metadata_parts.append(f"♥{self._format_number(likes)}")
                
                if search_type == "Models":
                    library = item.get("library_name", "")
                    if library:
                        metadata_parts.append(library)
                
                metadata = " | ".join(metadata_parts)
                
                # Insert into treeview
                self.results_tree.insert("", tk.END, values=(
                    creator, name, description, keywords, size_str, metadata
                ))
                
                count += 1
                
            except Exception as e:
                print(f"Error processing result: {e}")
                continue
        
        self.status_var.set(f"Found {count} {search_type.lower()}")
    
    def _format_size(self, bytes_size: int) -> str:
        """Format bytes to human readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
    
    def _format_number(self, num: int) -> str:
        """Format large numbers with K, M suffixes."""
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        return str(num)
    
    def _on_item_double_click(self, event):
        """Handle double-click on a result item."""
        self._download_selected()
    
    def _download_selected(self):
        """Download the selected model or dataset."""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select an item to download")
            return
        
        item = self.results_tree.item(selection[0])
        values = item['values']
        
        if len(values) < 2:
            return
        
        creator = values[0]
        name = values[1]
        repo_id = f"{creator}/{name}" if creator else name
        
        # Ask for download location
        download_dir = filedialog.askdirectory(title="Select Download Directory")
        if not download_dir:
            return
        
        # Create download window
        self._show_download_window(repo_id, download_dir)
    
    def _show_download_window(self, repo_id: str, download_dir: str):
        """Show a window for selecting files to download."""
        download_window = tk.Toplevel(self.root)
        download_window.title(f"Download: {repo_id}")
        download_window.geometry("800x500")
        
        # Get files list
        ttk.Label(download_window, text="Fetching file list...").pack(pady=10)
        
        def fetch_files():
            files = self.api.get_model_files(repo_id)
            download_window.after(0, lambda: self._populate_download_window(
                download_window, repo_id, download_dir, files))
        
        thread = threading.Thread(target=fetch_files)
        thread.daemon = True
        thread.start()
    
    def _populate_download_window(self, window: tk.Toplevel, repo_id: str, 
                                  download_dir: str, files: List[Dict]):
        """Populate the download window with file list."""
        # Clear window
        for widget in window.winfo_children():
            widget.destroy()
        
        ttk.Label(window, text=f"Select files to download from {repo_id}:").pack(pady=5)
        
        # File list frame
        list_frame = ttk.Frame(window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for files
        columns = ("filename", "size")
        file_tree = ttk.Treeview(list_frame, columns=columns, show="tree headings", height=15)
        file_tree.heading("#0", text="Select")
        file_tree.heading("filename", text="File")
        file_tree.heading("size", text="Size")
        
        file_tree.column("#0", width=50)
        file_tree.column("filename", width=500)
        file_tree.column("size", width=100)
        
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=file_tree.yview)
        file_tree.configure(yscrollcommand=vsb.set)
        
        file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add files to tree with checkboxes
        file_vars = {}
        for file_info in files:
            filename = file_info.get("rfilename", "")
            size = file_info.get("size", 0)
            size_str = self._format_size(size) if size > 0 else "-"
            
            item_id = file_tree.insert("", tk.END, text="☐", 
                                       values=(filename, size_str))
            file_vars[item_id] = {"filename": filename, "selected": False}
        
        # Toggle selection on click
        def toggle_selection(event):
            item = file_tree.identify("item", event.x, event.y)
            if item in file_vars:
                file_vars[item]["selected"] = not file_vars[item]["selected"]
                check = "☑" if file_vars[item]["selected"] else "☐"
                file_tree.item(item, text=check)
        
        file_tree.bind("<Button-1>", toggle_selection)
        
        # Button frame
        button_frame = ttk.Frame(window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def select_all():
            for item_id in file_vars:
                file_vars[item_id]["selected"] = True
                file_tree.item(item_id, text="☑")
        
        def select_none():
            for item_id in file_vars:
                file_vars[item_id]["selected"] = False
                file_tree.item(item_id, text="☐")
        
        def select_gguf():
            for item_id in file_vars:
                filename = file_vars[item_id]["filename"]
                is_gguf = filename.lower().endswith(".gguf")
                file_vars[item_id]["selected"] = is_gguf
                file_tree.item(item_id, text="☑" if is_gguf else "☐")
        
        ttk.Button(button_frame, text="Select All", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Select None", command=select_none).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Select GGUF Only", command=select_gguf).pack(side=tk.LEFT, padx=5)
        
        def start_download():
            selected_files = [info["filename"] for info in file_vars.values() if info["selected"]]
            if not selected_files:
                messagebox.showinfo("No Selection", "Please select at least one file to download")
                return
            
            window.destroy()
            self._download_files(repo_id, selected_files, download_dir)
        
        ttk.Button(button_frame, text="Download Selected", 
                  command=start_download).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _download_files(self, repo_id: str, files: List[str], download_dir: str):
        """Download selected files."""
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Downloading...")
        progress_window.geometry("500x200")
        
        ttk.Label(progress_window, text=f"Downloading from {repo_id}").pack(pady=10)
        
        current_file_var = tk.StringVar(value="Preparing...")
        ttk.Label(progress_window, textvariable=current_file_var).pack(pady=5)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                       maximum=100, length=400)
        progress_bar.pack(pady=10)
        
        status_var = tk.StringVar(value="Starting download...")
        ttk.Label(progress_window, textvariable=status_var).pack(pady=5)
        
        cancel_flag = {"cancelled": False}
        
        def cancel_download():
            cancel_flag["cancelled"] = True
            progress_window.destroy()
        
        ttk.Button(progress_window, text="Cancel", command=cancel_download).pack(pady=10)
        
        def download_thread():
            total_files = len(files)
            completed = 0
            
            for filename in files:
                if cancel_flag["cancelled"]:
                    break
                
                current_file_var.set(f"Downloading: {filename}")
                save_path = os.path.join(download_dir, repo_id.replace("/", "_"), filename)
                
                def update_progress(percent, downloaded, total):
                    progress_var.set(percent)
                    size_str = f"{self._format_size(downloaded)} / {self._format_size(total)}"
                    status_var.set(f"File {completed + 1}/{total_files}: {size_str}")
                
                success = self.api.download_file(repo_id, filename, save_path, update_progress)
                
                if success:
                    completed += 1
                
                if cancel_flag["cancelled"]:
                    break
            
            if not cancel_flag["cancelled"]:
                progress_window.after(0, lambda: messagebox.showinfo(
                    "Download Complete", 
                    f"Downloaded {completed}/{total_files} files to {download_dir}"))
            
            progress_window.after(0, progress_window.destroy)
        
        thread = threading.Thread(target=download_thread)
        thread.daemon = True
        thread.start()


def main():
    """Main entry point for the HuggingFace Downloader GUI."""
    root = tk.Tk()
    app = HuggingFaceDownloaderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()