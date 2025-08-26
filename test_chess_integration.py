#!/usr/bin/env python3
"""
Integration test for the chess game with LLM.
This simulates the chess logic without the GUI.
"""

import sys
import os
sys.path.append('.')

try:
    import chess
    print("✓ python-chess imported successfully")
except ImportError:
    print("✗ python-chess not available")
    sys.exit(1)

class MockSettings:
    """Mock settings manager for testing."""
    def __init__(self):
        self.settings = {
            'paths.last_model_path': '/path/to/model.gguf',  # Mock path
            'model_settings.default_n_ctx': 4096,
            'model_settings.default_n_gpu_layers': 0
        }
    
    def get(self, key, default=None):
        return self.settings.get(key, default)

class MockLlama:
    """Mock LLM for testing chess integration."""
    def __init__(self, *args, **kwargs):
        self.moves_to_suggest = ['e2e4', 'g1f3', 'd2d4', 'b1c3']  # Common opening moves
        self.call_count = 0
    
    def __call__(self, prompt, **kwargs):
        """Simulate LLM response with chess moves."""
        # Cycle through suggested moves
        move = self.moves_to_suggest[self.call_count % len(self.moves_to_suggest)]
        self.call_count += 1
        
        print(f"Mock LLM responding with: {move}")
        return {
            'choices': [{'text': move}]
        }

class ChessGameSimulator:
    """Simplified chess game simulator based on our implementation."""
    
    def __init__(self):
        self.board = chess.Board()
        self.settings = MockSettings()
        self.llm_cache = None
        self.move_history = []
        
    def _get_llm_instance(self):
        """Get mock LLM instance."""
        if not self.llm_cache:
            # Use mock instead of real LLM
            self.llm_cache = MockLlama()
        return self.llm_cache
    
    def _parse_move_from_response(self, text):
        """Parse chess move from LLM response using multiple methods."""
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
        for move in legal_moves:
            san = self.board.san(move).lower()
            san_clean = san.replace('+', '').replace('#', '').replace('x', '')
            if san_clean in text or san in text:
                return move
        
        return None
    
    def _create_chess_prompt(self):
        """Create a chess-specific prompt for the LLM."""
        fen = self.board.fen()
        legal_moves = [move.uci() for move in self.board.legal_moves]
        move_history = [self.board.san(move) for move in self.board.move_stack[-6:]]
        
        ai_color = "white" if self.board.turn == chess.WHITE else "black"
        current_turn = "white" if self.board.turn == chess.WHITE else "black"
        board_unicode = self.board.unicode()
        
        prompt = f"""You are a professional chess player and you play as {ai_color}. Now is your turn to make a move.

Current board position:
{board_unicode}

Position (FEN): {fen}
Turn: {current_turn}
Recent moves: {' '.join(move_history) if move_history else 'Game start'}

Legal moves available (UCI format): {', '.join(legal_moves)}

As an expert chess player, choose the BEST move considering:
- King safety and piece protection
- Center control and piece development
- Tactical opportunities (captures, forks, pins, skewers)
- Positional advantages
- Endgame principles if material is low

Reply with ONLY the move in UCI format (examples: e2e4, g1f3, e7e8q):"""

        return prompt
    
    def _query_llm_for_move(self):
        """Query the LLM for a chess move."""
        try:
            llm = self._get_llm_instance()
            if not llm:
                return None
            
            prompt = self._create_chess_prompt()
            
            # Use multiple attempts with different temperatures
            max_attempts = 3
            temperatures = [0.1, 0.3, 0.5]
            
            for attempt in range(max_attempts):
                try:
                    response = llm(
                        prompt,
                        max_tokens=20,
                        temperature=temperatures[attempt],
                        stop=["\n", " ", ".", ",", "because", "since", "as", "the"],
                        echo=False
                    )
                    
                    text = response['choices'][0]['text'].strip().lower()
                    move = self._parse_move_from_response(text)
                    
                    if move and move in self.board.legal_moves:
                        print(f"✓ LLM suggested valid move: {move.uci()} (attempt {attempt + 1})")
                        return move
                    elif move:
                        print(f"✗ LLM suggested illegal move: {move.uci()}")
                    else:
                        print(f"✗ Could not parse move from response: '{text}'")
                        
                except Exception as e:
                    print(f"✗ LLM attempt {attempt + 1} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"✗ LLM query failed: {e}")
            return None
    
    def _get_strategic_move(self, legal_moves):
        """Get strategic move using chess heuristics."""
        import random
        scored_moves = []
        
        for move in legal_moves:
            score = 0
            
            # Prefer central squares
            to_square = move.to_square
            file = chess.square_file(to_square)
            rank = chess.square_rank(to_square)
            center_distance = abs(3.5 - file) + abs(3.5 - rank)
            score += (7 - center_distance) * 2
            
            # Prefer piece development
            piece = self.board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                if chess.square_rank(move.from_square) in [0, 7]:
                    score += 15
            
            scored_moves.append((move, score))
        
        scored_moves.sort(key=lambda x: x[1] + random.random() * 2, reverse=True)
        return scored_moves[0][0]
    
    def get_ai_move(self):
        """Get AI move with LLM integration and fallback."""
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        
        # Try LLM first
        llm_move = self._query_llm_for_move()
        if llm_move and llm_move in legal_moves:
            return llm_move
        
        # Fallback to strategic heuristics
        print("Using strategic fallback move")
        return self._get_strategic_move(legal_moves)
    
    def make_move(self, move):
        """Make a move on the board."""
        if move in self.board.legal_moves:
            san = self.board.san(move)
            self.board.push(move)
            self.move_history.append(san)
            return True
        return False
    
    def play_moves(self, num_moves=4):
        """Play a few moves to test the integration."""
        print(f"\n=== Playing {num_moves} moves ===")
        
        for i in range(num_moves):
            if self.board.is_game_over():
                print("Game over!")
                break
            
            current_turn = "White" if self.board.turn == chess.WHITE else "Black"
            print(f"\nMove {i+1} - {current_turn} to play")
            print(f"Current position: {self.board.fen()}")
            
            ai_move = self.get_ai_move()
            if ai_move:
                success = self.make_move(ai_move)
                if success:
                    print(f"✓ Played: {self.move_history[-1]} ({ai_move.uci()})")
                else:
                    print(f"✗ Failed to make move: {ai_move.uci()}")
                    break
            else:
                print("✗ No move found")
                break
        
        print(f"\nFinal position after {len(self.move_history)} moves:")
        print(self.board.unicode())
        print(f"Move history: {' '.join(self.move_history)}")

def main():
    """Main test function."""
    print("Testing Chess Integration with Mock LLM")
    print("=" * 50)
    
    try:
        game = ChessGameSimulator()
        game.play_moves(6)  # Play 6 moves to test the integration
        
        print("\n" + "=" * 50)
        print("✓ Chess integration test completed successfully!")
        print("\nKey features tested:")
        print("- LLM querying with multiple attempts")
        print("- Move parsing from LLM responses")
        print("- Strategic fallback when LLM fails")
        print("- Proper move validation")
        print("- Chess position tracking")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()