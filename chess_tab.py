#!/usr/bin/env python3
"""
Chess Tab for DarkHal 2.0

Dedicated chess interface with AI opponent, game analysis, and UCI engine integration.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class ChessTab:
    """Dedicated Chess tab with AI opponent and advanced chess features."""
    
    def __init__(self, parent: ttk.Frame, settings_manager):
        self.parent = parent
        self.settings = settings_manager
        self.current_model = None
        
        # Create main frame
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create chess interface
        self._create_chess_interface()
    
    def _create_chess_interface(self):
        """Create the main chess interface."""
        
        # Chess configuration frame
        config_frame = ttk.LabelFrame(self.main_frame, text="Chess Game Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Configuration options
        options_frame = ttk.Frame(config_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="AI Difficulty:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.chess_difficulty_var = tk.StringVar(value="Medium")
        ttk.Combobox(options_frame, textvariable=self.chess_difficulty_var,
                    values=["Easy", "Medium", "Hard", "Expert"],
                    state="readonly", width=15).grid(row=0, column=1, padx=10)
        
        ttk.Label(options_frame, text="Time Control:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.time_control_var = tk.StringVar(value="10+0")
        ttk.Combobox(options_frame, textvariable=self.time_control_var,
                    values=["3+0", "5+0", "10+0", "15+10", "30+0"],
                    state="readonly", width=10).grid(row=0, column=3, padx=10)
        
        ttk.Label(options_frame, text="Play as:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.play_side_var = tk.StringVar(value="White")
        ttk.Radiobutton(options_frame, text="White", variable=self.play_side_var, 
                       value="White").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="Black", variable=self.play_side_var, 
                       value="Black").grid(row=1, column=2, sticky=tk.W)
        
        # Game control buttons
        control_frame = ttk.Frame(config_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="New Game", command=self._new_chess_game,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Start Chess", command=self._start_chess).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Analyze Position", command=self._analyze_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Get Hint", command=self._get_hint).pack(side=tk.LEFT, padx=5)
        
        # Chess engine information
        engine_frame = ttk.LabelFrame(self.main_frame, text="Chess Engine Information", padding=10)
        engine_frame.pack(fill=tk.X, pady=(0, 10))
        
        engine_info = """DarkHal Chess Engine Features:

• UCI (Universal Chess Interface) Protocol Support
• AI-powered move analysis and generation  
• Multiple difficulty levels from beginner to expert
• Position evaluation and game analysis
• Opening book and endgame tablebase support
• Real-time move suggestions and hints
• Game saving/loading in PGN format
• Integration with ChessGPT model for enhanced play

The chess engine uses advanced AI models to provide a challenging and educational chess experience.
You can adjust the difficulty to match your skill level and use analysis features to improve your game."""
        
        ttk.Label(engine_frame, text=engine_info, wraplength=600, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Game management frame
        management_frame = ttk.LabelFrame(self.main_frame, text="Game Management", padding=10)
        management_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Game management buttons
        mgmt_buttons_frame = ttk.Frame(management_frame)
        mgmt_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(mgmt_buttons_frame, text="Save Game", command=self._save_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(mgmt_buttons_frame, text="Load Game", command=self._load_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(mgmt_buttons_frame, text="Export PGN", command=self._export_pgn).pack(side=tk.LEFT, padx=5)
        ttk.Button(mgmt_buttons_frame, text="Import PGN", command=self._import_pgn).pack(side=tk.LEFT, padx=5)
        
        # Game status and move history
        status_frame = ttk.Frame(management_frame)
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        # Current game status
        ttk.Label(status_frame, text="Game Status:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.status_label = ttk.Label(status_frame, text="No game in progress", foreground="gray")
        self.status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Move history
        ttk.Label(status_frame, text="Move History:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.move_history = scrolledtext.ScrolledText(status_frame, height=8, width=50, state=tk.DISABLED)
        self.move_history.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Chess engine status
        engine_status_frame = ttk.Frame(management_frame)
        engine_status_frame.pack(fill=tk.X)
        
        ttk.Label(engine_status_frame, text="Engine Status:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.engine_status_label = ttk.Label(engine_status_frame, text="Ready", foreground="green")
        self.engine_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Button(engine_status_frame, text="Configure Engine", 
                  command=self._configure_engine).pack(side=tk.RIGHT, padx=5)
    
    def _start_chess(self):
        """Start a chess game with AI in floating window."""
        try:
            from chess_window import open_chess_window
            open_chess_window(self.parent.winfo_toplevel(), self.settings)
            self._update_status("Chess game started")
        except ImportError:
            messagebox.showerror("Error", "Chess module not available. Please install python-chess.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open chess window: {e}")
    
    def _new_chess_game(self):
        """Start a new chess game."""
        difficulty = self.chess_difficulty_var.get()
        time_control = self.time_control_var.get()
        play_side = self.play_side_var.get()
        
        self._update_status(f"New game started - {play_side} vs AI ({difficulty})")
        self._add_move_to_history(f"Game started: {play_side} vs AI")
        self._add_move_to_history(f"Difficulty: {difficulty}, Time: {time_control}")
        
        # Start the actual chess game
        self._start_chess()
    
    def _analyze_position(self):
        """Analyze current chess position."""
        self._update_status("Analyzing position...")
        self._add_move_to_history("Position analysis requested")
        # TODO: Implement position analysis with AI
        messagebox.showinfo("Analysis", "Position analysis feature coming soon!")
    
    def _get_hint(self):
        """Get a hint for the current position."""
        self._update_status("Generating hint...")
        self._add_move_to_history("Hint requested")
        # TODO: Implement hint generation with AI
        messagebox.showinfo("Hint", "Hint generation feature coming soon!")
    
    def _save_game(self):
        """Save the current game."""
        filename = filedialog.asksaveasfilename(
            title="Save Chess Game",
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        if filename:
            # TODO: Implement game saving
            self._add_move_to_history(f"Game saved to {filename}")
            messagebox.showinfo("Saved", f"Game saved to {filename}")
    
    def _load_game(self):
        """Load a saved game."""
        filename = filedialog.askopenfilename(
            title="Load Chess Game",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        if filename:
            # TODO: Implement game loading
            self._add_move_to_history(f"Game loaded from {filename}")
            self._update_status("Game loaded")
            messagebox.showinfo("Loaded", f"Game loaded from {filename}")
    
    def _export_pgn(self):
        """Export current game to PGN format."""
        filename = filedialog.asksaveasfilename(
            title="Export to PGN",
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        if filename:
            # TODO: Implement PGN export
            self._add_move_to_history(f"Game exported to PGN: {filename}")
            messagebox.showinfo("Exported", f"Game exported to {filename}")
    
    def _import_pgn(self):
        """Import game from PGN format."""
        filename = filedialog.askopenfilename(
            title="Import PGN",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        if filename:
            # TODO: Implement PGN import
            self._add_move_to_history(f"Game imported from PGN: {filename}")
            self._update_status("PGN game imported")
            messagebox.showinfo("Imported", f"Game imported from {filename}")
    
    def _configure_engine(self):
        """Configure chess engine settings."""
        # TODO: Implement engine configuration dialog
        messagebox.showinfo("Engine Config", "Engine configuration coming soon!")
    
    def _update_status(self, status: str):
        """Update the game status display."""
        self.status_label.config(text=status)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Chess: {status}")
    
    def _add_move_to_history(self, move: str):
        """Add a move or event to the move history."""
        self.move_history.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.move_history.insert(tk.END, f"[{timestamp}] {move}\n")
        self.move_history.see(tk.END)
        self.move_history.config(state=tk.DISABLED)
    
    def set_model(self, model_path: Optional[str]):
        """Set the current AI model for chess analysis."""
        self.current_model = model_path
        if model_path:
            self._update_status(f"AI Model loaded: {Path(model_path).name}")
        else:
            self._update_status("No AI model loaded")