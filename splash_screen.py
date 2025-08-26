#!/usr/bin/env python3
"""
DarkHal 2.0 Splash Screen

A professional splash screen with logo, disclaimers, and loading animation.
"""

import tkinter as tk
from tkinter import ttk
import os
import sys
import time
import threading
from pathlib import Path

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SplashScreen:
    """Professional splash screen for DarkHal 2.0"""
    
    def __init__(self, callback=None):
        """
        Initialize splash screen.
        
        Args:
            callback (callable): Function to call when splash closes with selected hardware acceleration
        """
        self.callback = callback
        self.selected_acceleration = None
        self.root = tk.Tk()
        self.showing_logo = True
        # References for dynamic logo sizing
        self.logo_original = None   # PIL.Image (original)
        self.logo_photo = None      # ImageTk.PhotoImage or tk.PhotoImage (current)
        self.logo_label = None      # tk.Label showing the logo
        # References for dynamic logo sizing
        self.logo_original = None   # PIL.Image (original)
        self.logo_photo = None      # ImageTk.PhotoImage or tk.PhotoImage (current)
        self.logo_label = None      # tk.Label that displays the logo
        # References for dynamic logo sizing
        self.logo_original = None   # PIL.Image (original)
        self.logo_photo = None      # ImageTk.PhotoImage or tk.PhotoImage (current)
        self.logo_label = None      # tk.Label showing the logo
        self.setup_window()
        self.create_logo_screen()
        
        # Switch to main screen after 2.5 seconds
        self.root.after(2500, self.switch_to_main_screen)
        
    def setup_window(self):
        """Configure the splash window"""
        # Remove window decorations
        self.root.overrideredirect(True)
        
        # Set window size - larger to accommodate all elements
        width = 700
        height = 600
        
        # Center on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.configure(bg='#1a1a1a')  # Dark background
        
        # Make window topmost
        self.root.attributes('-topmost', True)
        
        # Set window icon if available
        try:
            icon_path = Path("assets/Halico.ico")
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except:
            pass
    
    def load_logo(self):
        """Load and prepare the logo image"""
        logo_path = Path("assets/logo.png")
        
        print(f"Looking for logo at: {logo_path.absolute()}")
        print(f"Logo exists: {logo_path.exists()}")
        print(f"PIL available: {PIL_AVAILABLE}")
        
        if not logo_path.exists():
            print("Logo file not found")
            return None
            
        try:
            if PIL_AVAILABLE:
                # Use Pillow for better image handling
                image = Image.open(logo_path)
                print(f"Original image size: {image.size}")
                # Resize to fit in the fixed 75x75 box
                image = image.resize((65, 65), Image.Resampling.LANCZOS)
                print(f"Resized image to: {image.size}")
                return ImageTk.PhotoImage(image)
            else:
                # Fallback to tkinter's basic image support
                return tk.PhotoImage(file=str(logo_path))
        except Exception as e:
            print(f"Error loading logo: {e}")
            return None
    
    def create_logo_screen(self):
        """Create the initial logo-only screen"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Center container for logo
        center_frame = tk.Frame(self.root, bg='#1a1a1a')
        center_frame.pack(expand=True, fill=tk.BOTH)
        
        logo_path = Path("assets/logo.png")
        # Dynamic path if PIL available, else static fallback
        if PIL_AVAILABLE:
            self.logo_original = self._load_logo_original()
            if self.logo_original is not None:
                # Create label and center it; image set during update
                self.logo_label = tk.Label(center_frame, bg='#1a1a1a', borderwidth=0, highlightthickness=0)
                self.logo_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                # Bind window resize to keep logo responsive
                self.root.bind("<Configure>", self.update_logo_size)
                # Initial sizing
                self.update_logo_size()
                return
            elif logo_path.exists():
                # PIL present but image failed -> fallback to static display using tk.PhotoImage
                try:
                    self.logo_photo = tk.PhotoImage(file=str(logo_path))
                    self.logo_label = tk.Label(center_frame, image=self.logo_photo, bg='#1a1a1a')
                    self.logo_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                    return
                except Exception:
                    pass
        else:
            # No PIL -> static display if possible
            if logo_path.exists():
                try:
                    self.logo_photo = tk.PhotoImage(file=str(logo_path))
                    self.logo_label = tk.Label(center_frame, image=self.logo_photo, bg='#1a1a1a')
                    self.logo_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                    return
                except Exception:
                    pass

        # Fallback - large app name
        logo_text = tk.Label(center_frame, text="DarkHal 2.0", font=("Arial", 48, "bold"), fg='#00ff88', bg='#1a1a1a')
        logo_text.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def load_logo_large(self):
        """Load logo at original size for logo screen"""
        logo_path = Path("assets/logo.png")
        
        if not logo_path.exists():
            return None
            
        try:
            if PIL_AVAILABLE:
                image = Image.open(logo_path)
                # Keep original size or resize to reasonable splash size
                max_size = min(300, min(image.size))
                image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(image)
            else:
                return tk.PhotoImage(file=str(logo_path))
        except Exception as e:
            print(f"Error loading large logo: {e}")
            return None
    
    def switch_to_main_screen(self):
        """Switch from logo screen to main text screen"""
        self.showing_logo = False
        # Stop handling resize events once the logo screen ends
        try:
            self.root.unbind("<Configure>")
        except Exception:
            pass
        # Stop handling resize events once the logo screen ends
        try:
            self.root.unbind("<Configure>")
        except Exception:
            pass
        try:
            self.root.unbind("<Configure>")
        except Exception:
            pass
        self.create_widgets()
    
    def create_widgets(self):
        """Create the main text-only screen with buttons"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Application name
        title_label = tk.Label(
            main_frame,
            text="DarkHal 2.0",
            font=("Arial", 24, "bold"),
            fg='#00ff88',  # Green accent color
            bg='#1a1a1a'
        )
        title_label.pack(pady=(0, 5))
        
        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="AI Model Management & Training Platform",
            font=("Arial", 11),
            fg='#cccccc',
            bg='#1a1a1a'
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Warning/Disclaimer section
        warning_frame = tk.Frame(main_frame, bg='#2a2a2a', relief=tk.RAISED, bd=1)
        warning_frame.pack(fill=tk.X, pady=(0, 15))
        
        warning_title = tk.Label(
            warning_frame,
            text="⚠️ IMPORTANT WARNING",
            font=("Arial", 10, "bold"),
            fg='#ff4444',
            bg='#2a2a2a'
        )
        warning_title.pack(pady=(8, 5))
        
        warning_text = """This software is provided "as is" without any warranties or guarantees.
The user assumes all responsibility for the use of this software and any
consequences that may arise from its use. The developers are not liable
for any damages, data loss, or other issues that may occur."""
        
        warning_label = tk.Label(
            warning_frame,
            text=warning_text,
            font=("Arial", 9),
            fg='#cccccc',
            bg='#2a2a2a',
            wraplength=600,
            justify=tk.CENTER
        )
        warning_label.pack(pady=(0, 8))
        
        # Terms agreement section
        terms_frame = tk.Frame(main_frame, bg='#1a1a1a')
        terms_frame.pack(pady=(10, 15))
        
        terms_label = tk.Label(
            terms_frame,
            text="By continuing you agree to our terms",
            font=("Arial", 10),
            fg='#ffaa00',
            bg='#1a1a1a'
        )
        terms_label.pack()
        
        # Agreement section
        agreement_frame = tk.Frame(main_frame, bg='#1a1a1a')
        agreement_frame.pack(pady=(10, 20))
        
        agreement_label = tk.Label(
            agreement_frame,
            text="I agree",
            font=("Arial", 12, "bold"),
            fg='#00ff88',
            bg='#1a1a1a'
        )
        agreement_label.pack(pady=(0, 15))
        
        # Hardware acceleration buttons
        buttons_frame = tk.Frame(agreement_frame, bg='#1a1a1a')
        buttons_frame.pack()
        
        # CUDA button
        cuda_btn = tk.Button(
            buttons_frame,
            text="Start with CUDA",
            font=("Arial", 11, "bold"),
            bg='#00aa55',
            fg='white',
            activebackground='#00ff88',
            activeforeground='white',
            relief=tk.RAISED,
            bd=2,
            padx=20,
            pady=8,
            command=lambda: self.start_application('cuda')
        )
        cuda_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Intel button
        intel_btn = tk.Button(
            buttons_frame,
            text="Start with Intel",
            font=("Arial", 11, "bold"),
            bg='#0078d4',
            fg='white',
            activebackground='#106ebe',
            activeforeground='white',
            relief=tk.RAISED,
            bd=2,
            padx=20,
            pady=8,
            command=lambda: self.start_application('intel')
        )
        intel_btn.pack(side=tk.LEFT, padx=5)
        
        # CPU button
        cpu_btn = tk.Button(
            buttons_frame,
            text="Start with CPU",
            font=("Arial", 11, "bold"),
            bg='#666666',
            fg='white',
            activebackground='#888888',
            activeforeground='white',
            relief=tk.RAISED,
            bd=2,
            padx=20,
            pady=8,
            command=lambda: self.start_application('cpu')
        )
        cpu_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Copyright section at bottom
        copyright_frame = tk.Frame(main_frame, bg='#1a1a1a')
        copyright_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))
        
        copyright_label = tk.Label(
            copyright_frame,
            text="© 2025 Setec Labs",
            font=("Arial", 9, "bold"),
            fg='#888888',
            bg='#1a1a1a'
        )
        copyright_label.pack(side=tk.LEFT)
        
        author_label = tk.Label(
            copyright_frame,
            text="by ssSnake",
            font=("Arial", 9, "italic"),
            fg='#888888',
            bg='#1a1a1a'
        )
        author_label.pack(side=tk.RIGHT)
    
    def start_application(self, acceleration_type):
        """Start the application with selected hardware acceleration"""
        self.selected_acceleration = acceleration_type
        print(f"Starting DarkHal 2.0 with {acceleration_type.upper()} acceleration...")
        self.close_splash()
    
    def close_splash(self):
        """Close the splash screen and call callback"""
        self.root.destroy()
        
        if self.callback:
            self.callback(self.selected_acceleration)
    
    def show(self):
        """Show the splash screen"""
        self.root.mainloop()

    # --- Dynamic logo helpers ---
    def _load_logo_original(self):
        """Load the original logo as a PIL Image (RGBA) without resizing."""
        logo_path = Path("assets/logo.png")
        if not logo_path.exists():
            return None
        try:
            image = Image.open(logo_path)
            return image.convert("RGBA")
        except Exception as e:
            print(f"Error loading original logo: {e}")
            return None

    def update_logo_size(self, event=None):
        """Resize the logo image dynamically to match the window size."""
        if not self.showing_logo:
            return
        if self.logo_label is None or self.logo_original is None:
            return
        try:
            # Current window size
            win_w = max(1, self.root.winfo_width())
            win_h = max(1, self.root.winfo_height())

            # Target size: fraction of shortest side
            target_box = int(min(win_w, win_h) * 0.45)
            # Clamp to reasonable bounds
            target_box = max(96, min(512, target_box))

            ow, oh = self.logo_original.size
            scale = min(target_box / ow, target_box / oh)
            new_w = max(1, int(ow * scale))
            new_h = max(1, int(oh * scale))

            resized = self.logo_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(resized)
            self.logo_label.configure(image=self.logo_photo)
        except Exception as e:
            print(f"Error dynamically resizing logo: {e}")
        """Resize the logo to fit dynamically within the current window size."""
        if not self.showing_logo:
            return  # Only active on the logo screen
        if self.logo_label is None or self.logo_original is None:
            return
        try:
            # Current window size
            win_w = max(1, self.root.winfo_width())
            win_h = max(1, self.root.winfo_height())

            # Target box as a fraction of the shortest side
            target_box = int(min(win_w, win_h) * 0.45)
            # Clamp to reasonable bounds
            target_box = max(96, min(512, target_box))

            ow, oh = self.logo_original.size
            scale = min(target_box / ow, target_box / oh)
            new_w = max(1, int(ow * scale))
            new_h = max(1, int(oh * scale))

            # Resize with high-quality resampling
            resized = self.logo_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(resized)
            self.logo_label.configure(image=self.logo_photo)
        except Exception as e:
            print(f"Error dynamically resizing logo: {e}")

class SplashManager:
    """Manager for showing splash screen and launching main application"""
    
    def __init__(self, main_app_callback=None):
        """
        Initialize splash manager.
        
        Args:
            main_app_callback (callable): Function to call when splash closes with acceleration type
        """
        self.main_app_callback = main_app_callback
    
    def show_splash_and_launch(self):
        """Show splash screen then launch main application"""
        splash = SplashScreen(callback=self.launch_main_app)
        splash.show()
    
    def launch_main_app(self, acceleration_type):
        """Launch the main application after splash"""
        if self.main_app_callback:
            self.main_app_callback(acceleration_type)
        else:
            print(f"DarkHal 2.0 ready with {acceleration_type.upper() if acceleration_type else 'default'} acceleration!")


def demo_main_app(acceleration_type):
    """Demo main application for testing"""
    root = tk.Tk()
    root.title(f"DarkHal 2.0 - Main Application ({acceleration_type.upper()})")
    root.geometry("800x600")
    
    label = tk.Label(root, text=f"Welcome to DarkHal 2.0!\nRunning with {acceleration_type.upper()} acceleration", 
                     font=("Arial", 16), justify=tk.CENTER)
    label.pack(expand=True)
    
    root.mainloop()


def main():
    """Main entry point for testing splash screen"""
    print("Starting DarkHal 2.0 with splash screen...")
    
    # Create splash manager
    splash_manager = SplashManager(main_app_callback=demo_main_app)
    
    # Show splash and launch app
    splash_manager.show_splash_and_launch()


if __name__ == "__main__":
    main()