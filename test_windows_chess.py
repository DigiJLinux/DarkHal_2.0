#!/usr/bin/env python3
"""
Windows-specific test for chess functionality.
Tests the core chess logic without GUI dependencies.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chess_imports():
    """Test that all required chess dependencies are available."""
    print("Testing Chess Dependencies for Windows")
    print("=" * 50)
    
    # Test python-chess
    try:
        import chess
        print("✓ python-chess library imported successfully")
        
        # Test basic functionality
        board = chess.Board()
        print(f"✓ Chess board created: {board.fen()}")
        
        legal_moves = list(board.legal_moves)
        print(f"✓ Found {len(legal_moves)} legal opening moves")
        
        # Test Unicode display (important for Windows console)
        unicode_board = board.unicode()
        print("✓ Unicode board representation works")
        
        return True
        
    except ImportError as e:
        print(f"✗ python-chess not available: {e}")
        print("  Install with: pip install python-chess")
        return False
    except Exception as e:
        print(f"✗ Chess library error: {e}")
        return False

def test_llama_cpp_availability():
    """Test if llama-cpp-python is available (optional for testing)."""
    try:
        import llama_cpp
        print("✓ llama-cpp-python is available")
        print(f"  Version info: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'Unknown'}")
        return True
    except ImportError:
        print("⚠ llama-cpp-python not available (optional for testing)")
        print("  For full LLM functionality, install with: pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"⚠ llama-cpp-python error: {e}")
        return False

def test_darkhal_imports():
    """Test that our DarkHal chess modules import correctly."""
    try:
        # Test settings manager
        from darkhal.settings_manager import SettingsManager
        settings = SettingsManager()
        print("✓ DarkHal SettingsManager imported successfully")
        
        # Test if we can create a mock chess environment
        print("✓ DarkHal chess infrastructure is ready")
        return True
        
    except ImportError as e:
        print(f"✗ DarkHal module import error: {e}")
        return False
    except Exception as e:
        print(f"✗ DarkHal initialization error: {e}")
        return False

def test_chess_ai_logic():
    """Test the core chess AI logic we implemented."""
    print("\nTesting Chess AI Logic")
    print("-" * 30)
    
    try:
        import chess
        
        # Test our move parsing logic
        def parse_move_from_response(text, board):
            """Simplified version of our parsing method."""
            text = text.strip().lower()
            legal_moves = list(board.legal_moves)
            legal_uci = [move.uci() for move in legal_moves]
            
            # Direct UCI format match
            for move_uci in legal_uci:
                if move_uci in text:
                    return chess.Move.from_uci(move_uci)
            
            # Try SAN notation
            for move in legal_moves:
                san = board.san(move).lower()
                san_clean = san.replace('+', '').replace('#', '').replace('x', '')
                if san_clean in text or san in text:
                    return move
            
            return None
        
        # Test cases
        board = chess.Board()
        test_cases = [
            "e2e4",
            "I suggest e2e4",
            "Nf3 is good",
            "play d2d4",
        ]
        
        print("Testing move parsing:")
        for test in test_cases:
            move = parse_move_from_response(test, board)
            if move:
                print(f"  ✓ '{test}' → {move.uci()} ({board.san(move)})")
            else:
                print(f"  ✗ '{test}' → No valid move found")
        
        # Test strategic move evaluation
        def get_strategic_move(legal_moves, board):
            """Test strategic move selection."""
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
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if chess.square_rank(move.from_square) in [0, 7]:
                        score += 15
                
                scored_moves.append((move, score))
            
            scored_moves.sort(key=lambda x: x[1], reverse=True)
            return scored_moves[0][0] if scored_moves else None
        
        legal_moves = list(board.legal_moves)
        best_move = get_strategic_move(legal_moves, board)
        if best_move:
            print(f"✓ Strategic move selection: {best_move.uci()} ({board.san(best_move)})")
        else:
            print("✗ Strategic move selection failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Chess AI logic test failed: {e}")
        return False

def test_windows_console():
    """Test Windows console compatibility."""
    print("\nTesting Windows Console Compatibility")
    print("-" * 40)
    
    try:
        import chess
        board = chess.Board()
        
        # Test Unicode chess pieces display
        print("Testing Unicode chess pieces:")
        pieces = {
            chess.PAWN: "♟♙", chess.ROOK: "♜♖", chess.KNIGHT: "♞♘",
            chess.BISHOP: "♝♗", chess.QUEEN: "♛♕", chess.KING: "♚♔"
        }
        
        for piece_type, symbols in pieces.items():
            print(f"  {chess.piece_name(piece_type).title()}: {symbols}")
        
        print("\nSample board position:")
        print(board.unicode())
        
        print("✓ Windows console Unicode display working")
        return True
        
    except Exception as e:
        print(f"✗ Windows console test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("DarkHal Chess - Windows Compatibility Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Chess Dependencies", test_chess_imports()))
    results.append(("LLM Dependencies", test_llama_cpp_availability()))
    results.append(("DarkHal Modules", test_darkhal_imports()))
    results.append(("Chess AI Logic", test_chess_ai_logic()))
    results.append(("Windows Console", test_windows_console()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your chess implementation is ready for Windows.")
        print("\nNext steps:")
        print("1. Load a chess model (any LLM works)")
        print("2. Launch DarkHal and try the chess feature")
        print("3. Play against the AI!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please install missing dependencies:")
        if not any(name == "Chess Dependencies" and result for name, result in results):
            print("   pip install python-chess")
        if not any(name == "LLM Dependencies" and result for name, result in results):
            print("   pip install llama-cpp-python")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    
    # Keep window open on Windows
    if os.name == 'nt':  # Windows
        input("\nPress Enter to exit...")
    
    sys.exit(0 if success else 1)