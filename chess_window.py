#!/usr/bin/env python3
"""
Floating Chess Window for DarkHal 2.0

A separate window for playing chess against the AI with a visual board.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import sys
import os
import threading
from pathlib import Path
import json
import datetime

try:
    import chess
    import chess.svg
except ImportError:
    chess = None


class ChessWindow:
    """Floating chess game window."""
    
    def __init__(self, parent, settings_manager):
        self.parent = parent
        self.settings = settings_manager
        self.window = None
        self.board = None
        self.engine_process = None
        self.selected_square = None
        self.move_history = []
        self.game_saved = True
        self.llm_cache = None  # Cache for LLM instance
        self.ai_thinking = False
        self.flipped_board = False
        
        if not chess:
            messagebox.showerror("Missing Dependency", 
                               "Please install python-chess: pip install python-chess")
            return
        
        self._create_window()
    
    def _create_window(self):
        """Create the floating chess window."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("DarkHal Chess - Human vs AI")
        self.window.geometry("800x900")
        self.window.resizable(False, False)
        
        # Set icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "assets", "Halico.ico")
            if os.path.exists(icon_path):
                self.window.iconbitmap(icon_path)
        except Exception:
            pass
        
        # Initialize chess board
        self.board = chess.Board()
        
        # Create UI
        self._create_ui()
        
        # Handle window closing
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_ui(self):
        """Create the chess UI."""
        # Title frame
        title_frame = ttk.Frame(self.window)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        title_label = ttk.Label(title_frame, text="DarkHal Chess Engine", 
                               font=("Arial", 16, "bold"))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Human vs AI Chess Match", 
                                  font=("Arial", 10))
        subtitle_label.pack()
        
        # Game controls frame
        controls_frame = ttk.LabelFrame(self.window, text="Game Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Game settings
        settings_frame = ttk.Frame(controls_frame)
        settings_frame.pack(fill=tk.X)
        
        ttk.Label(settings_frame, text="Play as:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.play_side_var = tk.StringVar(value="White")
        ttk.Radiobutton(settings_frame, text="White", variable=self.play_side_var, 
                       value="White", command=self._side_changed).grid(row=0, column=1)
        ttk.Radiobutton(settings_frame, text="Black", variable=self.play_side_var, 
                       value="Black", command=self._side_changed).grid(row=0, column=2)
        
        ttk.Label(settings_frame, text="AI Difficulty:").grid(row=0, column=3, sticky=tk.W, padx=(20, 5))
        self.difficulty_var = tk.StringVar(value="Medium")
        ttk.Combobox(settings_frame, textvariable=self.difficulty_var,
                    values=["Easy", "Medium", "Hard", "Expert"],
                    state="readonly", width=10).grid(row=0, column=4)
        
        # Control buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="New Game", command=self._new_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Flip Board", command=self._flip_board).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Hint", command=self._get_hint).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Undo", command=self._undo_move).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Analyze", command=self._analyze_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Save Game", command=self._save_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Load Game", command=self._load_game).pack(side=tk.LEFT, padx=5)
        
        # Game status
        status_frame = ttk.Frame(controls_frame)
        status_frame.pack(fill=tk.X)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Ready to play")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(status_frame, text="Turn:").pack(side=tk.LEFT, padx=(20, 0))
        self.turn_var = tk.StringVar(value="White")
        self.turn_label = ttk.Label(status_frame, textvariable=self.turn_var, 
                                   font=("Arial", 10, "bold"))
        self.turn_label.pack(side=tk.LEFT, padx=5)
        
        # Chess board frame
        board_frame = ttk.LabelFrame(self.window, text="Chess Board", padding=10)
        board_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create chess board canvas
        self.board_canvas = tk.Canvas(board_frame, width=640, height=640, bg="white")
        self.board_canvas.pack()
        
        # Bind mouse events
        self.board_canvas.bind("<Button-1>", self._on_square_click)
        self.board_canvas.bind("<Motion>", self._on_mouse_motion)
        
        # Move history frame
        history_frame = ttk.LabelFrame(self.window, text="Move History", padding=10)
        history_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Move history text
        self.history_text = tk.Text(history_frame, height=4, wrap=tk.WORD, 
                                   font=("Consolas", 9), state=tk.DISABLED)
        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scroll.set)
        
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Draw initial board
        self._draw_board()
        self._update_status()
        
        # Start AI if playing as black
        if self.play_side_var.get() == "Black":
            self._ai_move()
    
    def _draw_board(self):
        """Draw the chess board and pieces."""
        self.board_canvas.delete("all")
        
        # Board dimensions
        square_size = 80
        board_size = square_size * 8
        
        # Colors
        light_color = "#F0D9B5"
        dark_color = "#B58863"
        highlight_color = "#FFFF00"
        
        # Draw squares
        for rank in range(8):
            for file in range(8):
                x1 = file * square_size
                y1 = rank * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Determine square color
                if (rank + file) % 2 == 0:
                    color = light_color
                else:
                    color = dark_color
                
                # Highlight selected square
                square = chess.square(file, 7 - rank)
                if square == self.selected_square:
                    color = highlight_color
                
                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
                # Draw coordinates
                if file == 0:  # Left edge - rank labels
                    self.board_canvas.create_text(x1 + 5, y1 + 10, text=str(8 - rank), 
                                                 font=("Arial", 8), anchor="nw")
                if rank == 7:  # Bottom edge - file labels
                    self.board_canvas.create_text(x2 - 10, y2 - 5, text=chr(ord('a') + file), 
                                                 font=("Arial", 8), anchor="se")
        
        # Draw pieces
        self._draw_pieces()
    
    def _draw_pieces(self):
        """Draw chess pieces on the board."""
        square_size = 80
        
        # Unicode chess pieces
        piece_symbols = {
            chess.PAWN: "♟♙", chess.ROOK: "♜♖", chess.KNIGHT: "♞♘",
            chess.BISHOP: "♝♗", chess.QUEEN: "♛♕", chess.KING: "♚♔"
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                
                x = file * square_size + square_size // 2
                y = (7 - rank) * square_size + square_size // 2
                
                # Get piece symbol
                symbol = piece_symbols[piece.piece_type][0 if piece.color == chess.BLACK else 1]
                
                self.board_canvas.create_text(x, y, text=symbol, font=("Arial", 48), 
                                            fill="black" if piece.color == chess.BLACK else "white",
                                            anchor="center")
    
    def _on_square_click(self, event):
        """Handle square click events."""
        if self.board.is_game_over():
            return
        
        # Convert canvas coordinates to square
        square_size = 80
        file = event.x // square_size
        rank = 7 - (event.y // square_size)
        
        if 0 <= file <= 7 and 0 <= rank <= 7:
            square = chess.square(file, rank)
            
            if self.selected_square is None:
                # Select piece
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    # Only allow selecting our pieces on our turn
                    human_color = chess.WHITE if self.play_side_var.get() == "White" else chess.BLACK
                    if self.board.turn == human_color:
                        self.selected_square = square
                        self._draw_board()
            else:
                # Try to make move
                try:
                    move = chess.Move(self.selected_square, square)
                    
                    # Check for promotion
                    piece = self.board.piece_at(self.selected_square)
                    if (piece and piece.piece_type == chess.PAWN and 
                        ((piece.color == chess.WHITE and rank == 7) or 
                         (piece.color == chess.BLACK and rank == 0))):
                        # Auto-promote to queen for simplicity
                        move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                    
                    if move in self.board.legal_moves:
                        self._make_human_move(move)
                    else:
                        messagebox.showwarning("Illegal Move", "That move is not legal!")
                
                except Exception as e:
                    messagebox.showerror("Move Error", f"Error making move: {e}")
                
                self.selected_square = None
                self._draw_board()
    
    def _on_mouse_motion(self, event):
        """Handle mouse motion for hover effects."""
        # Could add hover highlighting here
        pass
    
    def _make_human_move(self, move):
        """Make a human move and trigger AI response."""
        print(f"[CHESS DEBUG] Making human move: {move.uci()}")
        
        # Get SAN notation BEFORE applying the move
        san_notation = self.board.san(move)
        print(f"[CHESS DEBUG] Human move SAN notation: {san_notation}")
        
        # Add move to board
        self.board.push(move)
        print(f"[CHESS DEBUG] Move applied to board. New position: {self.board.fen()}")
        
        self._add_move_to_history_with_san(move, san_notation)
        self._draw_board()
        self._update_status()
        
        # Check if game is over
        if self.board.is_game_over():
            print(f"[CHESS DEBUG] Game is over after human move")
            self._game_over()
            return
        
        print(f"[CHESS DEBUG] Starting AI move in background thread")
        # AI's turn
        threading.Thread(target=self._ai_move, daemon=True).start()
    
    def _ai_move(self):
        """Let AI make a move."""
        print(f"[CHESS DEBUG] AI move started")
        
        if self.board.is_game_over():
            print(f"[CHESS DEBUG] Game is over, AI move cancelled")
            return
        
        # Check whose turn it is
        ai_color = chess.BLACK if self.play_side_var.get() == "White" else chess.WHITE
        ai_color_str = "Black" if ai_color == chess.BLACK else "White"
        board_turn_str = "Black" if self.board.turn == chess.BLACK else "White"
        
        if self.board.turn != ai_color:
            print(f"[CHESS DEBUG] Not AI's turn! Board turn: {board_turn_str}, AI color: {ai_color_str}")
            return
        
        print(f"[CHESS DEBUG] AI's turn confirmed. Board turn: {board_turn_str}, AI color: {ai_color_str}")
        
        # Update status
        self.window.after(0, lambda: self.status_var.set("AI is thinking..."))
        
        try:
            # Get AI move using simple logic for now
            # In a full implementation, this would call the UCI engine
            ai_move = self._get_ai_move()
            
            if ai_move:
                print(f"[CHESS DEBUG] AI found move: {ai_move.uci()}")
                self.window.after(0, lambda: self._apply_ai_move(ai_move))
            else:
                print(f"[CHESS DEBUG] AI couldn't find a move")
                self.window.after(0, lambda: self.status_var.set("AI couldn't find a move"))
        
        except Exception as e:
            print(f"[CHESS DEBUG] AI move exception: {e}")
            error_msg = str(e)
            self.window.after(0, lambda: messagebox.showerror("AI Error", f"AI move failed: {error_msg}"))
    
    def _get_ai_move(self):
        """Get AI move using LLM integration with repetition avoidance."""
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        
        # Filter out moves that would repeat recent positions
        filtered_moves = self._filter_repetitive_moves(legal_moves)
        if not filtered_moves:
            filtered_moves = legal_moves  # Fall back to all legal moves if filtering removes everything
        
        # Try LLM first
        llm_move = self._query_llm_for_move()
        if llm_move and llm_move in filtered_moves:
            return llm_move
        elif llm_move and llm_move in legal_moves:
            print(f"[CHESS DEBUG] LLM suggested repetitive move: {llm_move.uci()}, using fallback")
        
        # Fallback to strategic heuristics with filtered moves
        return self._get_strategic_move(filtered_moves)
    
    def _filter_repetitive_moves(self, legal_moves):
        """Filter out moves that would create repetitive positions."""
        if len(self.board.move_stack) < 4:
            return legal_moves  # Not enough moves to check repetition
        
        current_fen = self.board.fen().split()[0]  # Just board position
        filtered_moves = []
        
        for move in legal_moves:
            # Test if this move would create a repetition
            self.board.push(move)
            new_fen = self.board.fen().split()[0]
            self.board.pop()
            
            # Check if this position appeared recently
            is_repetitive = False
            temp_board = chess.Board()
            recent_positions = []
            
            # Build recent position history
            for historical_move in self.board.move_stack[-6:]:
                temp_board.push(historical_move)
                recent_positions.append(temp_board.fen().split()[0])
            
            # Check if new position would repeat a recent one
            if new_fen in recent_positions[-4:]:  # Last 2 moves
                is_repetitive = True
                print(f"[CHESS DEBUG] Filtering repetitive move: {move.uci()}")
            
            if not is_repetitive:
                filtered_moves.append(move)
        
        return filtered_moves if filtered_moves else legal_moves
    
    def _query_llm_for_move(self):
        """Query the LLM for a chess move using DarkHal's main chat system."""
        try:
            print(f"[CHESS DEBUG] Starting LLM query for move")
            
            # Check if chess mode is enabled
            chess_mode = self.settings.get('model_settings.chess_mode', False)
            print(f"[CHESS DEBUG] Chess mode enabled: {chess_mode}")
            
            # Import DarkHal's main chat function
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from main import run_prompt
            
            # Get model path and settings
            model_path = self.settings.get('paths.last_model_path', '')
            if not model_path or not os.path.exists(model_path):
                print(f"[CHESS DEBUG] No valid model path: {model_path}")
                return None
            
            # Prepare the chess prompt
            prompt = self._create_chess_prompt()
            print(f"[CHESS DEBUG] Created chess prompt: {len(prompt)} characters")
            
            # Get settings for LLM
            n_ctx = self.settings.get('model_settings.default_n_ctx', 4096)
            n_gpu_layers = self.settings.get('model_settings.default_n_gpu_layers', 0)
            
            # Use multiple attempts with different temperatures
            max_attempts = 3
            temperatures = [0.1, 0.3, 0.5]
            
            for attempt in range(max_attempts):
                try:
                    print(f"[CHESS DEBUG] Attempt {attempt + 1} with temperature {temperatures[attempt]}")
                    
                    # Use DarkHal's run_prompt function (which returns a string directly)
                    response_text = run_prompt(
                        model_path=model_path,
                        prompt=prompt,
                        stream=False,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        max_tokens=20,
                        chess_mode=chess_mode
                    )
                    
                    print(f"[CHESS DEBUG] LLM response: '{response_text}'")
                    
                    # Parse the response text directly
                    move = self._parse_move_from_response(response_text)
                    
                    if move and move in self.board.legal_moves:
                        print(f"[CHESS DEBUG] Valid move found: {move.uci()} (attempt {attempt + 1})")
                        return move
                    elif move:
                        print(f"[CHESS DEBUG] Invalid move suggested: {move.uci()} not in legal moves")
                    else:
                        print(f"[CHESS DEBUG] Could not parse move from response")
                        
                except Exception as e:
                    print(f"[CHESS DEBUG] LLM attempt {attempt + 1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"[CHESS DEBUG] All LLM attempts failed")
            return None
            
        except Exception as e:
            print(f"[CHESS DEBUG] LLM query failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_move_from_response(self, text):
        """Parse chess move from LLM response using multiple methods."""
        # Clean the text
        text = text.strip().lower()
        legal_moves = list(self.board.legal_moves)
        legal_uci = [move.uci() for move in legal_moves]
        
        # Method 1: Direct UCI format match
        for move_uci in legal_uci:
            if move_uci in text:
                return chess.Move.from_uci(move_uci)
        
        # Method 2: Look for 4-5 character sequences that could be UCI
        import re
        uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
        matches = re.findall(uci_pattern, text)
        for match in matches:
            if match in legal_uci:
                return chess.Move.from_uci(match)
        
        # Method 3: Try to find SAN notation and convert
        # Use a safe approach that doesn't call .san() on invalid moves
        for move in legal_moves:
            try:
                # Calculate SAN for this legal move
                san = self.board.san(move).lower()
                san_clean = san.replace('+', '').replace('#', '').replace('x', '')
                if san_clean in text or san in text:
                    return move
            except:
                # Skip if SAN calculation fails
                continue
        
        # Method 4: Extract first plausible move-like string
        tokens = text.replace(',', ' ').replace('.', ' ').split()
        for token in tokens:
            token = token.strip('.,()[]{}')
            if len(token) >= 4 and len(token) <= 5:
                try:
                    # Try as UCI
                    if token in legal_uci:
                        return chess.Move.from_uci(token)
                except:
                    continue
        
        return None
    
    def _get_llm_instance(self):
        """Get cached LLM instance."""
        if self.llm_cache:
            return self.llm_cache
        
        try:
            # Import from main project
            sys.path.append('..')
            from llama_cpp import Llama
            
            # Get model path from settings
            model_path = self.settings.get('paths.last_model_path', '')
            if not model_path or not os.path.exists(model_path):
                return None
            
            # Create LLM instance
            self.llm_cache = Llama(
                model_path=model_path,
                n_ctx=self.settings.get('model_settings.default_n_ctx', 4096),
                n_gpu_layers=self.settings.get('model_settings.default_n_gpu_layers', 0),
                verbose=False
            )
            
            return self.llm_cache
            
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            return None
    
    def _create_chess_prompt(self):
        """Create a chess-specific prompt for the LLM with anti-repetition measures."""
        # Get game context
        fen = self.board.fen()
        legal_moves = [move.uci() for move in self.board.legal_moves]
        # Get move history safely - use stored SAN notation instead of converting
        move_history = []
        for item in self.move_history[-10:]:
            if ' ' in item:
                move_history.append(item.split(' ')[0])  # Get just the SAN part
        
        # Determine player color
        ai_color = "Black" if self.play_side_var.get() == "White" else "White"
        current_turn = "White" if self.board.turn == chess.WHITE else "Black"
        
        # Get recent position repetitions
        recent_positions = []
        temp_board = chess.Board()
        for move in self.board.move_stack[-6:]:  # Last 3 moves (6 half-moves)
            temp_board.push(move)
            recent_positions.append(temp_board.fen().split()[0])  # Just board position, not full FEN
        
        # Check for repetitive patterns
        repetition_warning = ""
        if len(recent_positions) >= 4:
            if recent_positions[-1] == recent_positions[-3] and recent_positions[-2] == recent_positions[-4]:
                repetition_warning = "WARNING: Position is repeating! Choose a different strategy to avoid draws."
        
        # Analyze game phase
        material_count = len([p for p in str(self.board) if p.isalpha()])
        if material_count > 28:
            game_phase = "opening"
            phase_advice = "Focus on piece development, center control, and king safety."
        elif material_count > 16:
            game_phase = "middlegame"  
            phase_advice = "Look for tactical combinations and improve piece coordination."
        else:
            game_phase = "endgame"
            phase_advice = "Activate your king, push passed pawns, and simplify advantageous positions."
        
        # Create enhanced strategic prompt
        prompt = f"""You are an expert chess player playing as {ai_color}. It's {current_turn}'s turn.

Position (FEN): {fen}
Game phase: {game_phase}
Recent moves: {' '.join(move_history) if move_history else 'Game start'}

{repetition_warning}

Available moves (UCI format): {', '.join(legal_moves[:20])}

Strategic priorities for {game_phase}:
{phase_advice}

Choose the BEST move considering:
1. Avoid repeating recent moves or positions
2. King safety and piece protection
3. {phase_advice.split('.')[0].lower()}
4. Tactical opportunities (captures, forks, pins, skewers)
5. Long-term strategic advantages

IMPORTANT: Do not repeat the same move or create position repetitions.

Respond with ONLY the move in UCI format (e.g., e2e4, g1f3, e7e8q):"""

        return prompt
    
    def _get_strategic_move(self, legal_moves):
        """Get strategic move using chess heuristics."""
        import random
        
        scored_moves = []
        
        for move in legal_moves:
            score = 0
            
            # Make move temporarily to evaluate
            self.board.push(move)
            
            # Prefer captures with good piece values
            if self.board.move_stack and self.board.is_capture(self.board.move_stack[-1]):
                captured_piece = self.board.piece_at(move.to_square)
                if captured_piece:
                    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                                  chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                    score += piece_values.get(captured_piece.piece_type, 0) * 10
            
            # Prefer central squares
            to_square = move.to_square
            file = chess.square_file(to_square)
            rank = chess.square_rank(to_square)
            center_distance = abs(3.5 - file) + abs(3.5 - rank)
            score += (7 - center_distance) * 2
            
            # Prefer piece development
            piece = self.board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                if chess.square_rank(move.from_square) in [0, 7]:  # From back rank
                    score += 15
            
            # Avoid moving king early (unless castling)
            if piece and piece.piece_type == chess.KING and not self.board.is_castling(move):
                if len(self.board.move_stack) < 10:  # Early game
                    score -= 10
            
            # Prefer checks
            if self.board.is_check():
                score += 5
            
            # Avoid leaving pieces hanging
            if self.board.is_attacked_by(not self.board.turn, move.to_square):
                score -= piece_values.get(piece.piece_type if piece else chess.PAWN, 0) * 5
            
            self.board.pop()  # Undo temporary move
            
            scored_moves.append((move, score))
        
        # Sort by score and add randomness for variety
        scored_moves.sort(key=lambda x: x[1] + random.random() * 2, reverse=True)
        return scored_moves[0][0]
    
    def _apply_ai_move(self, move):
        """Apply AI move to the board with validation."""
        try:
            print(f"[CHESS DEBUG] Applying AI move: {move.uci()}")
            
            # Validate move is legal
            if move not in self.board.legal_moves:
                print(f"[CHESS DEBUG] Illegal AI move attempted: {move.uci()}")
                return False
            
            # Get SAN notation BEFORE applying the move
            san_notation = self.board.san(move)
            print(f"[CHESS DEBUG] Move SAN notation: {san_notation}")
            
            # Apply move
            self.board.push(move)
            print(f"[CHESS DEBUG] Move applied to board")
            
            # Add to history using the pre-calculated SAN
            self._add_move_to_history_with_san(move, san_notation)
            self._draw_board()
            self._update_status()
            
            print(f"[CHESS DEBUG] AI played: {san_notation} ({move.uci()})")
            
            if self.board.is_game_over():
                print(f"[CHESS DEBUG] Game over after AI move")
                self._game_over()
                
            return True
            
        except Exception as e:
            print(f"[CHESS DEBUG] Error applying AI move: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_move_to_history(self, move):
        """Add move to history display (deprecated - use _add_move_to_history_with_san)."""
        # This method is kept for compatibility but should not be used
        # as it can cause the SAN notation error
        print(f"[CHESS DEBUG] WARNING: Using deprecated _add_move_to_history method")
        try:
            san_notation = self.board.san(move)
            self._add_move_to_history_with_san(move, san_notation)
        except Exception as e:
            print(f"[CHESS DEBUG] Error in deprecated history method: {e}")
            # Fallback to just UCI notation
            self._add_move_to_history_with_san(move, move.uci())
    
    def _add_move_to_history_with_san(self, move, san_notation):
        """Add move to history display using pre-calculated SAN notation."""
        print(f"[CHESS DEBUG] Adding move to history: {san_notation} ({move.uci()})")
        
        self.history_text.config(state=tk.NORMAL)
        
        move_num = len(self.board.move_stack) // 2 + 1
        if len(self.board.move_stack) % 2 == 1:  # White move (odd number in stack)
            self.history_text.insert(tk.END, f"{move_num}. {san_notation} ")
        else:  # Black move (even number in stack)
            self.history_text.insert(tk.END, f"{san_notation}\n")
        
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)
    
    def _update_status(self):
        """Update game status display."""
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "White" if self.board.turn == chess.BLACK else "Black"
                self.status_var.set(f"Checkmate! {winner} wins!")
            elif self.board.is_stalemate():
                self.status_var.set("Stalemate - Draw!")
            elif self.board.is_insufficient_material():
                self.status_var.set("Draw - Insufficient material")
            else:
                self.status_var.set("Game Over")
        elif self.board.is_check():
            self.status_var.set("Check!")
        else:
            self.status_var.set("Game in progress")
        
        # Update turn
        self.turn_var.set("White" if self.board.turn == chess.WHITE else "Black")
    
    def _game_over(self):
        """Handle game over."""
        result = "Unknown"
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            result = f"{winner} wins by checkmate!"
        elif self.board.is_stalemate():
            result = "Draw by stalemate!"
        elif self.board.is_insufficient_material():
            result = "Draw by insufficient material!"
        
        messagebox.showinfo("Game Over", result)
    
    def _new_game(self):
        """Start a new game."""
        if not self.game_saved:
            result = messagebox.askyesnocancel("Unsaved Game", 
                                              "Current game is not saved. Save before starting new game?")
            if result is True:  # Yes - save first
                if not self._save_game():
                    return  # Save cancelled
            elif result is None:  # Cancel
                return
        
        self.board = chess.Board()
        self.selected_square = None
        self.move_history = []
        self.game_saved = True
        
        # Clear history display
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        self.history_text.config(state=tk.DISABLED)
        
        self._draw_board()
        self._update_status()
        
        # AI goes first if human plays black
        if self.play_side_var.get() == "Black":
            threading.Thread(target=self._ai_move, daemon=True).start()
    
    def _flip_board(self):
        """Flip the board view."""
        self.flipped_board = not self.flipped_board
        self._draw_board()
        messagebox.showinfo("Board Flipped", f"Board view {'flipped' if self.flipped_board else 'normal'}")
    
    def _get_hint(self):
        """Get a hint for the current position using AI analysis."""
        if self.board.is_game_over():
            messagebox.showinfo("Hint", "Game is over!")
            return
        
        if self.ai_thinking:
            messagebox.showinfo("Hint", "AI is currently thinking. Please wait.")
            return
        
        # Check if it's human's turn
        human_color = chess.WHITE if self.play_side_var.get() == "White" else chess.BLACK
        if self.board.turn != human_color:
            messagebox.showinfo("Hint", "It's not your turn!")
            return
        
        # Get AI suggestion
        self.status_var.set("Analyzing position for hint...")
        
        def get_hint_thread():
            try:
                # Use the same AI logic to get best move
                hint_move = self._get_ai_move()
                
                if hint_move:
                    from_square = chess.square_name(hint_move.from_square)
                    to_square = chess.square_name(hint_move.to_square)
                    
                    # Get piece name
                    piece = self.board.piece_at(hint_move.from_square)
                    piece_name = piece.symbol().upper() if piece else "Piece"
                    
                    # Check if it's a special move
                    move_type = ""
                    if self.board.is_capture(hint_move):
                        move_type += " (Capture)"
                    if self.board.is_castling(hint_move):
                        move_type += " (Castling)"
                    if hint_move.promotion:
                        move_type += f" (Promote to {chess.piece_name(hint_move.promotion)})"
                    
                    hint_text = f"Suggested move: {piece_name} from {from_square} to {to_square}{move_type}\n\n"
                    hint_text += f"Move notation: {self.board.san(hint_move)}\n"
                    hint_text += f"UCI format: {hint_move.uci()}"
                    
                    self.window.after(0, lambda: messagebox.showinfo("Chess Hint", hint_text))
                else:
                    self.window.after(0, lambda: messagebox.showinfo("Hint", "No good moves found!"))
                
                self.window.after(0, lambda: self.status_var.set("Ready"))
                
            except Exception as e:
                self.window.after(0, lambda: messagebox.showerror("Hint Error", f"Failed to get hint: {e}"))
                self.window.after(0, lambda: self.status_var.set("Ready"))
        
        threading.Thread(target=get_hint_thread, daemon=True).start()
    
    def _undo_move(self):
        """Undo the last move(s)."""
        if self.ai_thinking:
            messagebox.showinfo("Undo", "Cannot undo while AI is thinking.")
            return
        
        if len(self.board.move_stack) >= 2:
            # Undo both AI and human moves
            self.board.pop()
            self.board.pop()
            self.move_history = self.move_history[:-2]
            self.game_saved = False
            
            # Update history display
            self.history_text.config(state=tk.NORMAL)
            self.history_text.delete(1.0, tk.END)
            
            # Rebuild history display
            for i, move in enumerate(self.move_history):
                move_num = (i + 2) // 2
                if i % 2 == 0:  # White move
                    self.history_text.insert(tk.END, f"{move_num}. {move} ")
                else:  # Black move
                    self.history_text.insert(tk.END, f"{move}\n")
            
            self.history_text.config(state=tk.DISABLED)
            
            self._draw_board()
            self._update_status()
            
        elif len(self.board.move_stack) == 1:
            # Undo only one move
            self.board.pop()
            self.move_history = self.move_history[:-1]
            self.game_saved = False
            
            self.history_text.config(state=tk.NORMAL)
            self.history_text.delete(1.0, tk.END)
            self.history_text.config(state=tk.DISABLED)
            
            self._draw_board()
            self._update_status()
        else:
            messagebox.showinfo("Undo", "No moves to undo!")
    
    def _analyze_position(self):
        """Analyze current position with detailed information."""
        if self.ai_thinking:
            messagebox.showinfo("Analysis", "AI is currently thinking. Please wait.")
            return
        
        # Create analysis window
        analysis_window = tk.Toplevel(self.window)
        analysis_window.title("Position Analysis")
        analysis_window.geometry("600x500")
        analysis_window.transient(self.window)
        
        # Create notebook for different analysis types
        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Basic analysis tab
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Info")
        
        basic_text = tk.Text(basic_frame, wrap=tk.WORD, font=("Consolas", 10))
        basic_scroll = ttk.Scrollbar(basic_frame, orient="vertical", command=basic_text.yview)
        basic_text.configure(yscrollcommand=basic_scroll.set)
        
        basic_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        basic_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate basic analysis
        analysis = f"Position Analysis\n{'='*50}\n\n"
        analysis += f"FEN: {self.board.fen()}\n\n"
        analysis += f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}\n"
        analysis += f"Move number: {self.board.fullmove_number}\n"
        analysis += f"Half-move clock: {self.board.halfmove_clock}\n\n"
        
        analysis += f"Legal moves: {len(list(self.board.legal_moves))}\n"
        analysis += f"In check: {'Yes' if self.board.is_check() else 'No'}\n"
        analysis += f"Can castle kingside: {self.board.has_kingside_castling_rights(self.board.turn)}\n"
        analysis += f"Can castle queenside: {self.board.has_queenside_castling_rights(self.board.turn)}\n\n"
        
        # Material count
        analysis += "Material Count:\n"
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            analysis += f"\n{color_name}:\n"
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                count = len(self.board.pieces(piece_type, color))
                if count > 0:
                    analysis += f"  {chess.piece_name(piece_type).title()}s: {count}\n"
        
        basic_text.insert(1.0, analysis)
        basic_text.config(state=tk.DISABLED)
        
        # AI Analysis tab
        ai_frame = ttk.Frame(notebook)
        notebook.add(ai_frame, text="AI Analysis")
        
        ai_text = scrolledtext.ScrolledText(ai_frame, wrap=tk.WORD, font=("Consolas", 10))
        ai_text.pack(fill=tk.BOTH, expand=True)
        
        # Get AI analysis
        def get_ai_analysis():
            ai_text.insert(tk.END, "Getting AI analysis...\n\n")
            try:
                llm = self._get_llm_instance()
                if llm:
                    analysis_prompt = f"""Analyze this chess position as an expert player:

Position (FEN): {self.board.fen()}
Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}
Recent moves: {' '.join([self.board.san(move) for move in self.board.move_stack[-5:]])}

Provide analysis covering:
1. Position evaluation (who's better and why)
2. Key tactical and positional themes
3. Best moves for the current player
4. Strategic plans for both sides
5. Critical weaknesses to address

Analysis:"""
                    
                    # Use consistent temperature based on analysis depth
                    response = llm(analysis_prompt, max_tokens=500, temperature=0.4)
                    ai_analysis = response['choices'][0]['text'].strip()
                    
                    ai_text.delete(1.0, tk.END)
                    ai_text.insert(1.0, ai_analysis)
                else:
                    ai_text.delete(1.0, tk.END)
                    ai_text.insert(1.0, "AI analysis not available - no model loaded")
                    
            except Exception as e:
                ai_text.delete(1.0, tk.END)
                ai_text.insert(1.0, f"AI analysis failed: {e}")
        
        threading.Thread(target=get_ai_analysis, daemon=True).start()
    
    def _save_game(self):
        """Save the current game to a PGN file."""
        if len(self.board.move_stack) == 0:
            messagebox.showinfo("Save Game", "No moves to save!")
            return False
        
        try:
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                title="Save Chess Game",
                defaultextension=".pgn",
                filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")],
                initialname=f"darkhal_chess_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
            )
            
            if not filename:
                return False
            
            # Create PGN content
            pgn_content = self._create_pgn()
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(pgn_content)
            
            self.game_saved = True
            messagebox.showinfo("Game Saved", f"Game saved to {os.path.basename(filename)}")
            return True
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save game: {e}")
            return False
    
    def _load_game(self):
        """Load a game from a PGN file."""
        if not self.game_saved:
            result = messagebox.askyesnocancel("Unsaved Game", 
                                              "Current game is not saved. Save before loading?")
            if result is True:  # Yes - save first
                if not self._save_game():
                    return  # Save cancelled
            elif result is None:  # Cancel
                return
        
        try:
            # Ask for file to load
            filename = filedialog.askopenfilename(
                title="Load Chess Game",
                filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
            # Read PGN file
            with open(filename, 'r', encoding='utf-8') as f:
                pgn_content = f.read()
            
            # Parse and load game
            if self._parse_pgn(pgn_content):
                self.game_saved = True
                messagebox.showinfo("Game Loaded", f"Game loaded from {os.path.basename(filename)}")
            else:
                messagebox.showerror("Load Error", "Invalid PGN format or unsupported game")
                
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load game: {e}")
    
    def _create_pgn(self):
        """Create PGN content from current game."""
        # PGN headers
        headers = [
            '[Event "DarkHal Chess Game"]',
            f'[Date "{datetime.datetime.now().strftime("%Y.%m.%d")}"]',
            '[White "Human"]' if self.play_side_var.get() == "White" else '[White "DarkHal AI"]',
            '[Black "DarkHal AI"]' if self.play_side_var.get() == "White" else '[Black "Human"]',
            f'[Site "DarkHal 2.0"]',
            '[Round "1"]'
        ]
        
        # Game result
        if self.board.is_game_over():
            if self.board.is_checkmate():
                result = "1-0" if self.board.turn == chess.BLACK else "0-1"
            else:
                result = "1/2-1/2"
        else:
            result = "*"
        
        headers.append(f'[Result "{result}"]')
        
        # Create moves section
        moves = []
        temp_board = chess.Board()
        
        for i, move in enumerate(self.board.move_stack):
            if i % 2 == 0:  # White move
                move_num = (i // 2) + 1
                moves.append(f"{move_num}. {temp_board.san(move)}")
            else:  # Black move
                moves.append(temp_board.san(move))
            temp_board.push(move)
        
        moves_text = " ".join(moves)
        if result != "*":
            moves_text += f" {result}"
        
        # Combine headers and moves
        pgn = "\n".join(headers) + "\n\n" + moves_text + "\n"
        return pgn
    
    def _parse_pgn(self, pgn_content):
        """Parse PGN content and load the game."""
        try:
            # Simple PGN parser - extract moves section
            lines = pgn_content.strip().split('\n')
            moves_text = ""
            
            # Find the moves section (after headers)
            in_moves = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('['):
                    continue
                else:
                    moves_text += line + " "
                    in_moves = True
            
            if not moves_text:
                return False
            
            # Clean up moves text
            moves_text = moves_text.replace('\n', ' ').strip()
            
            # Remove result markers
            for result in ['1-0', '0-1', '1/2-1/2', '*']:
                moves_text = moves_text.replace(result, '').strip()
            
            # Parse moves
            self.board = chess.Board()
            self.history_text.config(state=tk.NORMAL)
            self.history_text.delete(1.0, tk.END)
            self.history_text.config(state=tk.DISABLED)
            
            # Split into tokens and process
            tokens = moves_text.split()
            for token in tokens:
                token = token.strip('.')
                if not token or token.isdigit():
                    continue
                
                try:
                    # Try to parse as SAN (Standard Algebraic Notation)
                    move = self.board.parse_san(token)
                    # Get SAN notation before applying move
                    san_notation = self.board.san(move)
                    self.board.push(move)
                    self._add_move_to_history_with_san(move, san_notation)
                except:
                    # Skip invalid moves
                    continue
            
            self._draw_board()
            self._update_status()
            return True
            
        except Exception as e:
            print(f"PGN parsing error: {e}")
            return False
    
    def _side_changed(self):
        """Handle play side change."""
        if hasattr(self, 'board') and self.board:
            # Reset game when side changes
            self._new_game()
    
    def _on_closing(self):
        """Handle window closing."""
        if self.engine_process:
            try:
                self.engine_process.terminate()
            except:
                pass
        
        self.window.destroy()
    
    def _on_difficulty_changed(self, event=None):
        """Handle difficulty setting change."""
        difficulty = self.difficulty_var.get()
        self.status_var.set(f"AI difficulty set to {difficulty}")
        
        # Clear LLM cache to ensure new difficulty settings take effect
        self.llm_cache = None


def open_chess_window(parent, settings_manager):
    """Open the floating chess window."""
    ChessWindow(parent, settings_manager)