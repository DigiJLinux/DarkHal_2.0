#!/usr/bin/env python3
"""
Minimal UCI chess engine that delegates move selection to an LLM (optional).
Works with GUIs like Arena/Cute Chess. Acts as "player 2" (black) when the GUI pairs it that way.

This is integrated with DarkHal 2.0's LLM system.
"""

import os
import sys
import time
import json
import random
import requests
import threading
from pathlib import Path

try:
    import chess
except ImportError:
    print("Please install python-chess: pip install python-chess==1.999")
    sys.exit(1)

ENGINE_NAME = "DarkHal-Chess-Engine"
ENGINE_AUTHOR = "Setec Labs"

class DarkHalChessEngine:
    """UCI Chess Engine integrated with DarkHal 2.0"""
    
    def __init__(self):
        self.board = chess.Board()
        self.history_san = []
        self.settings_file = Path("../settings.json")
        self.llm_settings = self._load_llm_settings()
    
    def _load_llm_settings(self):
        """Load LLM settings from DarkHal 2.0 settings file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                return {
                    'model_path': settings.get('paths', {}).get('last_model_path', ''),
                    'temperature': 0.3,
                    'max_tokens': 50,
                    'n_ctx': settings.get('model_settings', {}).get('default_n_ctx', 4096),
                    'n_gpu_layers': settings.get('model_settings', {}).get('default_n_gpu_layers', 0)
                }
        except Exception:
            pass
        
        return {
            'model_path': '',
            'temperature': 0.3,
            'max_tokens': 50,
            'n_ctx': 4096,
            'n_gpu_layers': 0
        }
    
    def _query_darkhal_llm(self, prompt: str) -> str:
        """Query DarkHal's loaded LLM model for a chess move."""
        try:
            # Try to import llama_cpp from the main project
            sys.path.append('..')
            from llama_cpp import Llama
            
            # Check if we have a model path
            model_path = self.llm_settings.get('model_path', '')
            if not model_path or not os.path.exists(model_path):
                return None
            
            # Create Llama instance (simplified - in real implementation this would be cached)
            llm = Llama(
                model_path=model_path,
                n_ctx=self.llm_settings['n_ctx'],
                n_gpu_layers=self.llm_settings['n_gpu_layers'],
                verbose=False
            )
            
            # Generate response
            response = llm(
                prompt,
                max_tokens=self.llm_settings['max_tokens'],
                temperature=self.llm_settings['temperature'],
                stop=["\n", ".", ",", " "],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            # If LLM fails, return None to fall back to random moves
            return None
    
    def _query_ollama(self, prompt: str) -> str:
        """Query an Ollama server for a chess move."""
        model = os.getenv("OLLAMA_MODEL")
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not model:
            return None
        try:
            r = requests.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")
        except Exception:
            return None

    def _query_llamacpp_server(self, prompt: str) -> str:
        """Query a llama.cpp-compatible server."""
        url = os.getenv("LLAMACPP_URL")  # e.g., "http://localhost:8080"
        if not url:
            return None
        try:
            r = requests.post(
                f"{url}/completion",
                json={"prompt": prompt, "n_predict": 64, "temperature": 0.3, "stop": ["\n"]},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return (data.get("content") or data.get("result") or "").strip()
        except Exception:
            return None

    def _ask_llm_for_move(self, fen: str, legal_uci: list, history_san: list) -> str:
        """Ask the LLM to choose a move from legal_uci."""
        legal_str = ", ".join(legal_uci[:20])  # Limit to first 20 moves to avoid token limit
        hist_str = " ".join(history_san[-10:]) if history_san else "game start"
        
        # Create a chess-focused prompt
        prompt = f"""You are a chess engine playing as {'Black' if self.board.turn == chess.BLACK else 'White'}.
        
Current position (FEN): {fen}
Recent moves: {hist_str}
Legal moves available: {legal_str}

Choose the BEST move from the legal moves list. Consider:
- Piece safety and development
- Control of center squares
- King safety
- Tactical opportunities

Respond with ONLY the move in UCI format (e.g., "e2e4" or "g1f3"):"""

        # Try different LLM sources
        response = None
        
        # 1. Try DarkHal's internal LLM first
        response = self._query_darkhal_llm(prompt)
        
        # 2. Fall back to Ollama
        if not response:
            response = self._query_ollama(prompt)
        
        # 3. Fall back to llama.cpp server
        if not response:
            response = self._query_llamacpp_server(prompt)
        
        if not response:
            return None

        # Parse the response to extract UCI move
        response = response.strip().lower()
        
        # Look for exact match first
        for move in legal_uci:
            if move in response:
                return move
        
        # Try to extract move-like patterns
        import re
        move_pattern = r'[a-h][1-8][a-h][1-8][qrnb]?'
        matches = re.findall(move_pattern, response)
        
        for match in matches:
            if match in legal_uci:
                return match
        
        return None

    def loop(self):
        """Main UCI communication loop."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                
                if line == "uci":
                    self.cmd_uci()
                elif line == "isready":
                    self.cmd_isready()
                elif line == "ucinewgame":
                    self.cmd_ucinewgame()
                elif line.startswith("position"):
                    self.cmd_position(line)
                elif line.startswith("go"):
                    self.cmd_go(line)
                elif line == "quit":
                    break
                # Ignore other commands: "stop", "ponderhit", "setoption", etc.
                
            except (EOFError, KeyboardInterrupt):
                break

    def cmd_uci(self):
        """Handle UCI identification command."""
        print(f"id name {ENGINE_NAME}")
        print(f"id author {ENGINE_AUTHOR}")
        # Engine options
        print("option name Skill Level type spin default 5 min 0 max 10")
        print("option name Use LLM type check default true")
        print("uciok")
        sys.stdout.flush()

    def cmd_isready(self):
        """Handle UCI ready check."""
        print("readyok")
        sys.stdout.flush()

    def cmd_ucinewgame(self):
        """Handle new game command."""
        self.board = chess.Board()
        self.history_san.clear()

    def cmd_position(self, line: str):
        """Handle position setup command."""
        parts = line.split()
        
        if "startpos" in parts:
            self.board = chess.Board()
            moves_index = parts.index("startpos") + 1
        elif "fen" in parts:
            fen_index = parts.index("fen") + 1
            fen = " ".join(parts[fen_index:fen_index + 6])
            self.board = chess.Board(fen)
            moves_index = fen_index + 6
        else:
            return

        # Apply moves if present
        if moves_index < len(parts) and parts[moves_index] == "moves":
            self.history_san.clear()
            for mv in parts[moves_index + 1:]:
                try:
                    move = self.board.parse_uci(mv)
                    self.history_san.append(self.board.san(move))
                    self.board.push(move)
                except Exception:
                    # Ignore illegal moves from GUI (shouldn't happen)
                    pass

    def cmd_go(self, line: str):
        """Handle go (search for best move) command."""
        # Get legal moves
        legal_moves = list(self.board.legal_moves)
        legal_uci = [move.uci() for move in legal_moves]
        
        if not legal_uci:
            print("bestmove 0000")
            sys.stdout.flush()
            return

        # Ask LLM for move
        fen = self.board.fen()
        chosen_move = self._ask_llm_for_move(fen, legal_uci, self.history_san)
        
        # Fall back to strategic random if LLM fails
        if chosen_move not in legal_uci:
            chosen_move = self._choose_fallback_move(legal_moves)

        # Validate and make move
        try:
            move = self.board.parse_uci(chosen_move)
            if move in legal_moves:
                # Optional: Print thinking info
                print(f"info depth 1 score cp 0 pv {chosen_move}")
                print(f"bestmove {chosen_move}")
            else:
                # Safety fallback
                print(f"bestmove {random.choice(legal_uci)}")
        except Exception:
            print(f"bestmove {random.choice(legal_uci)}")
        
        sys.stdout.flush()

    def _choose_fallback_move(self, legal_moves):
        """Choose a strategic move when LLM fails."""
        # Simple heuristics for move selection
        scored_moves = []
        
        for move in legal_moves:
            score = 0
            
            # Prefer captures
            if self.board.is_capture(move):
                score += 10
            
            # Prefer central squares
            to_square = move.to_square
            file = chess.square_file(to_square)
            rank = chess.square_rank(to_square)
            if 2 <= file <= 5 and 2 <= rank <= 5:  # Central squares
                score += 3
            
            # Prefer piece development
            piece = self.board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                if chess.square_rank(move.from_square) in [0, 7]:  # From back rank
                    score += 5
            
            # Avoid moving king early
            if piece and piece.piece_type == chess.KING:
                score -= 5
            
            scored_moves.append((move, score))
        
        # Sort by score and add some randomness
        scored_moves.sort(key=lambda x: x[1] + random.random(), reverse=True)
        return scored_moves[0][0].uci()


def main():
    """Main entry point."""
    # Ensure unbuffered output
    engine = DarkHalChessEngine()
    engine.loop()


if __name__ == "__main__":
    main()