#!/usr/bin/env python3
"""
Chess AI Testing Tool
Tests your chess AI against various opponents with parallel processing
"""

import chess
import chess.engine
import time
import random
import csv
import sys
import os
from multiprocessing import Pool, cpu_count, Manager
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class OpponentType(Enum):
    RANDOM = "random"
    MATERIAL = "material"
    BASIC = "basic"
    STOCKFISH_0 = "stockfish_level_0"
    STOCKFISH_1 = "stockfish_level_1"
    STOCKFISH_3 = "stockfish_level_3"
    STOCKFISH_5 = "stockfish_level_5"
    STOCKFISH_10 = "stockfish_level_10"
    STOCKFISH_15 = "stockfish_level_15"
    STOCKFISH_20 = "stockfish_level_20"
    STOCKFISH_ELO_1320 = "stockfish_elo_1320"
    STOCKFISH_ELO_1500 = "stockfish_elo_1500"
    STOCKFISH_ELO_1800 = "stockfish_elo_1800"
    STOCKFISH_ELO_2000 = "stockfish_elo_2000"
    STOCKFISH_ELO_2200 = "stockfish_elo_2200"
    STOCKFISH_ELO_2500 = "stockfish_elo_2500"


@dataclass
class GameResult:
    game_number: int
    result: str  # 'win', 'loss', 'draw'
    your_color: str  # 'white' or 'black'
    opponent_type: str
    moves: List[str]
    pgn: str
    move_count: int
    opponent_elo: Optional[int] = None  # Stockfish ELO if applicable


# Piece values for evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-square tables (from white's perspective)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE
}


def get_piece_value(piece: chess.Piece, square: chess.Square) -> int:
    """Get the value of a piece at a specific square."""
    piece_type = piece.piece_type
    base_value = PIECE_VALUES[piece_type]
    
    # Get position value from piece-square table
    table = PIECE_TABLES[piece_type]
    
    # Flip square index for black pieces
    if piece.color == chess.BLACK:
        square = chess.square_mirror(square)
    
    position_value = table[square]
    
    return base_value + position_value


def evaluate_board(board: chess.Board) -> int:
    """Evaluate the board position. Positive is good for white, negative for black."""
    if board.is_checkmate():
        return -20000 if board.turn == chess.WHITE else 20000
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    evaluation = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = get_piece_value(piece, square)
            evaluation += value if piece.color == chess.WHITE else -value
    
    return evaluation


def evaluate_material(board: chess.Board) -> int:
    """Evaluate board based only on material count."""
    if board.is_checkmate():
        return -20000 if board.turn == chess.WHITE else 20000
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    evaluation = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            evaluation += value if piece.color == chess.WHITE else -value
    
    return evaluation


def minimax(board: chess.Board, depth: int, alpha: int, beta: int, 
            maximizing_player: bool, use_positional: bool = True) -> int:
    """Minimax algorithm with alpha-beta pruning."""
    if depth == 0 or board.is_game_over():
        return evaluate_board(board) if use_positional else evaluate_material(board)
    
    legal_moves = list(board.legal_moves)
    
    # Move ordering: prioritize captures
    legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False, use_positional)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True, use_positional)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval


def get_best_move(board: chess.Board, depth: int = 3, 
                  use_positional: bool = True) -> Optional[chess.Move]:
    """Find the best move using minimax algorithm."""
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None
    
    # Simple opening book for speed
    if board.fullmove_number <= 2:
        opening_moves = [m for m in legal_moves if m.uci() in 
                        ['e2e4', 'e7e5', 'd2d4', 'd7d5', 'g1f3', 'b1c3', 'g8f6', 'b8c6']]
        if opening_moves:
            return random.choice(opening_moves)
    
    best_move = None
    best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
    
    # Move ordering
    legal_moves.sort(key=lambda move: board.is_capture(move), reverse=True)
    
    for move in legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, float('-inf'), float('inf'), 
                             board.turn == chess.WHITE, use_positional)
        board.pop()
        
        if board.turn == chess.WHITE:
            if board_value > best_value:
                best_value = board_value
                best_move = move
        else:
            if board_value < best_value:
                best_value = board_value
                best_move = move
    
    return best_move or legal_moves[0]


def get_random_move(board: chess.Board) -> Optional[chess.Move]:
    """Get a random legal move."""
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves) if legal_moves else None


def find_stockfish() -> Optional[str]:
    """
    Try to find Stockfish executable.
    Returns path to Stockfish or None if not found.
    """
    # Common Stockfish locations on macOS
    possible_paths = [
        "/opt/homebrew/bin/stockfish",  # Homebrew on Apple Silicon
        "/usr/local/bin/stockfish",     # Homebrew on Intel
        "/usr/bin/stockfish",            # System installation
        "./stockfish",                   # Current directory
        "stockfish",                     # In PATH
    ]
    
    for path in possible_paths:
        if os.path.exists(path) or os.access(path, os.X_OK):
            try:
                # Try to verify it's actually Stockfish
                engine = chess.engine.SimpleEngine.popen_uci(path)
                engine.quit()
                return path
            except:
                continue
    
    return None


def get_stockfish_move(board: chess.Board, opponent_type: OpponentType, 
                      stockfish_path: str, time_limit: float = 0.1) -> Optional[chess.Move]:
    """Get a move from Stockfish at specified strength."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        # Configure Stockfish based on opponent type
        if opponent_type.value.startswith("stockfish_level"):
            # Extract skill level (0-20)
            level = int(opponent_type.value.split('_')[-1])
            engine.configure({"Skill Level": level})
            result = engine.play(board, chess.engine.Limit(time=time_limit))
        elif "elo" in opponent_type.value:
            # Extract ELO from opponent type (minimum 1320)
            elo = int(opponent_type.value.split('_')[-1])
            if elo < 1320:
                print(f"\nWarning: Stockfish minimum ELO is 1320, using level {elo//200} instead")
                engine.configure({"Skill Level": min(20, elo // 200)})
            else:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            result = engine.play(board, chess.engine.Limit(time=time_limit))
        else:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
        
        engine.quit()
        return result.move
    except Exception as e:
        print(f"\nError using Stockfish: {e}")
        return None


def get_opponent_move(board: chess.Board, opponent_type: OpponentType, 
                     depth: int = 3, stockfish_path: Optional[str] = None) -> Optional[chess.Move]:
    """Get move based on opponent type."""
    # Check if this is a Stockfish opponent
    if opponent_type.value.startswith("stockfish"):
        if stockfish_path:
            return get_stockfish_move(board, opponent_type, stockfish_path)
        else:
            print("\nWarning: Stockfish opponent requested but Stockfish not found. Using random moves.")
            return get_random_move(board)
    
    # Non-Stockfish opponents
    if opponent_type == OpponentType.RANDOM:
        return get_random_move(board)
    elif opponent_type == OpponentType.MATERIAL:
        return get_best_move(board, depth=1, use_positional=False)
    elif opponent_type == OpponentType.BASIC:
        return get_best_move(board, depth=depth, use_positional=True)
    return get_random_move(board)


def play_single_game(args: Tuple) -> GameResult:
    """Play a single game. Designed to work with multiprocessing."""
    game_number, your_color, opponent_type, depth, stockfish_path, progress_dict, lock = args
    
    board = chess.Board()
    moves = []
    move_stack = []  # Store moves separately for PGN
    
    your_ai_color = chess.WHITE if your_color == 'white' else chess.BLACK
    
    while not board.is_game_over():
        if board.turn == your_ai_color:
            move = get_best_move(board, depth=depth, use_positional=True)
        else:
            move = get_opponent_move(board, opponent_type, depth=depth, stockfish_path=stockfish_path)
        
        if move:
            moves.append(board.san(move))
            move_stack.append(move)
            board.push(move)
        else:
            break
    
    # Determine result
    if board.is_checkmate():
        winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
        result = 'win' if winner == your_ai_color else 'loss'
    else:
        result = 'draw'
    
    # Generate PGN from the move stack
    pgn_board = chess.Board()
    pgn_moves = []
    for move in move_stack:
        pgn_moves.append(pgn_board.san(move))
        pgn_board.push(move)
    
    # Extract ELO if Stockfish opponent
    opponent_elo = None
    if "elo" in opponent_type.value:
        opponent_elo = int(opponent_type.value.split('_')[-1])
    
    # Update progress counter
    with lock:
        progress_dict['completed'] += 1
    
    return GameResult(
        game_number=game_number,
        result=result,
        your_color=your_color,
        opponent_type=opponent_type.value,
        moves=moves,
        pgn=' '.join(pgn_moves),
        move_count=len(moves),
        opponent_elo=opponent_elo
    )


def print_progress(completed: int, total: int, start_time: float, 
                  wins: int = 0, losses: int = 0, draws: int = 0):
    """Print progress bar with statistics."""
    elapsed = time.time() - start_time
    progress = completed / total if total > 0 else 0
    
    # Calculate ETA
    if completed > 0:
        avg_time_per_game = elapsed / completed
        remaining = total - completed
        eta = avg_time_per_game * remaining
        eta_str = f"{int(eta//60)}m {int(eta%60)}s" if eta >= 60 else f"{int(eta)}s"
    else:
        eta_str = "calculating..."
    
    # Progress bar
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Win rate
    win_rate = (wins / completed * 100) if completed > 0 else 0
    
    # Build progress line
    progress_line = (
        f"\r  [{bar}] {completed}/{total} games "
        f"({progress*100:.1f}%) | "
        f"W:{wins} L:{losses} D:{draws} ({win_rate:.1f}% wins) | "
        f"Elapsed: {int(elapsed)}s | ETA: {eta_str}"
    )
    
    # Print with carriage return to overwrite
    sys.stdout.write(progress_line)
    sys.stdout.flush()


def run_test_suite(num_games: int, opponent_type: OpponentType, 
                  your_color: str = 'both', depth: int = 3, 
                  num_workers: Optional[int] = None,
                  stockfish_path: Optional[str] = None) -> List[GameResult]:
    """
    Run a test suite of games.
    
    Args:
        num_games: Number of games to play
        opponent_type: Type of opponent (RANDOM, MATERIAL, BASIC, or Stockfish variants)
        your_color: 'white', 'black', or 'both' (alternating)
        depth: Search depth for minimax (1-5, higher is stronger but slower)
        num_workers: Number of parallel workers (default: CPU count)
        stockfish_path: Path to Stockfish executable (auto-detected if None)
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    # Auto-detect Stockfish if using Stockfish opponent
    if opponent_type.value.startswith("stockfish") and stockfish_path is None:
        stockfish_path = find_stockfish()
        if stockfish_path:
            print(f"Found Stockfish at: {stockfish_path}")
        else:
            print("WARNING: Stockfish not found! Install with: brew install stockfish")
            print("Continuing without Stockfish (will use fallback)...\n")
    
    # Display opponent info
    opponent_display = opponent_type.value
    if opponent_type.value.startswith("stockfish"):
        if "elo" in opponent_type.value:
            elo = opponent_type.value.split('_')[-1]
            opponent_display = f"Stockfish (ELO {elo})"
        else:
            level = opponent_type.value.split('_')[-1]
            opponent_display = f"Stockfish (Level {level})"
    
    print(f"\n{'='*70}")
    print(f"Starting test suite:")
    print(f"  Games: {num_games}")
    print(f"  Opponent: {opponent_display}")
    print(f"  Your color: {your_color}")
    print(f"  Search depth: {depth}")
    print(f"  CPU cores: {num_workers}")
    print(f"{'='*70}\n")
    
    # Prepare game configurations
    game_configs = []
    for i in range(num_games):
        if your_color == 'both':
            color = 'white' if i % 2 == 0 else 'black'
        else:
            color = your_color
        game_configs.append((i + 1, color, opponent_type, depth, stockfish_path))
    
    start_time = time.time()
    
    # Use Manager for shared progress tracking
    with Manager() as manager:
        progress_dict = manager.dict()
        progress_dict['completed'] = 0
        lock = manager.Lock()
        
        # Add progress tracking to game configs
        game_configs_with_progress = [
            (*config, progress_dict, lock) for config in game_configs
        ]
        
        # Determine update frequency (update every 2% or minimum 1 game)
        update_interval = max(1, num_games // 50)
        
        results = []
        last_update = 0
        
        print("  Progress:")
        
        # Run games in parallel with progress tracking
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for better progress tracking
            for result in pool.imap_unordered(play_single_game, game_configs_with_progress):
                results.append(result)
                completed = len(results)
                
                # Update progress bar at intervals
                if completed - last_update >= update_interval or completed == num_games:
                    wins = sum(1 for r in results if r.result == 'win')
                    losses = sum(1 for r in results if r.result == 'loss')
                    draws = sum(1 for r in results if r.result == 'draw')
                    
                    print_progress(completed, num_games, start_time, wins, losses, draws)
                    last_update = completed
    
    # Final newline after progress bar
    print("\n")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    wins = sum(1 for r in results if r.result == 'win')
    losses = sum(1 for r in results if r.result == 'loss')
    draws = sum(1 for r in results if r.result == 'draw')
    win_rate = (wins / num_games * 100) if num_games > 0 else 0
    
    print(f"{'='*70}")
    print(f"Testing Complete!")
    print(f"{'='*70}")
    print(f"Results:")
    print(f"  Wins:      {wins:4d} ({wins/num_games*100:5.1f}%)")
    print(f"  Losses:    {losses:4d} ({losses/num_games*100:5.1f}%)")
    print(f"  Draws:     {draws:4d} ({draws/num_games*100:5.1f}%)")
    print(f"  Total:     {num_games:4d}")
    print(f"\nPerformance:")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"  Games/sec:    {num_games/total_time:.2f}")
    print(f"  Avg game:     {total_time/num_games:.2f}s")
    
    # ELO estimation if playing against Stockfish
    if opponent_type.value.startswith("stockfish") and "elo" in opponent_type.value:
        opponent_elo = int(opponent_type.value.split('_')[-1])
        estimated_elo = estimate_elo_from_results(win_rate, opponent_elo, num_games)
        print(f"\nELO Estimation:")
        print(f"  Opponent ELO: {opponent_elo}")
        print(f"  Your estimated ELO: {estimated_elo[0]} - {estimated_elo[1]}")
        print(f"  Confidence: {estimated_elo[2]}")
    
    print(f"{'='*70}\n")
    
    return results


def estimate_elo_from_results(win_rate: float, opponent_elo: int, num_games: int) -> Tuple[int, int, str]:
    """
    Estimate your ELO based on win rate against a known opponent.
    Returns (min_elo, max_elo, confidence_level)
    """
    # Using the ELO expected score formula: E = 1 / (1 + 10^((opponent_elo - your_elo)/400))
    # Solving for your_elo: your_elo = opponent_elo - 400 * log10((1/E) - 1)
    
    # Clamp win rate to avoid division by zero
    win_rate = max(1, min(99, win_rate))
    expected_score = win_rate / 100
    
    import math
    
    # Calculate estimated ELO
    if expected_score >= 0.99:
        your_elo = opponent_elo + 400
    elif expected_score <= 0.01:
        your_elo = opponent_elo - 400
    else:
        elo_diff = -400 * math.log10((1 / expected_score) - 1)
        your_elo = int(opponent_elo + elo_diff)
    
    # Calculate confidence interval based on number of games
    if num_games < 10:
        confidence = "Very Low"
        margin = 300
    elif num_games < 30:
        confidence = "Low"
        margin = 200
    elif num_games < 50:
        confidence = "Medium"
        margin = 150
    elif num_games < 100:
        confidence = "Good"
        margin = 100
    else:
        confidence = "High"
        margin = 75
    
    return (your_elo - margin, your_elo + margin, confidence)


def export_to_csv(results: List[GameResult], filename: str = "test_results.csv"):
    """Export results to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['game_number', 'result', 'your_color', 'opponent_type', 
                     'opponent_elo', 'move_count', 'moves']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'game_number': result.game_number,
                'result': result.result,
                'your_color': result.your_color,
                'opponent_type': result.opponent_type,
                'opponent_elo': result.opponent_elo or 'N/A',
                'move_count': result.move_count,
                'moves': ' '.join(result.moves)
            })
    
    print(f"Results exported to {filename}")


def export_to_pgn(results: List[GameResult], filename: str = "test_games.pgn"):
    """Export games to PGN format."""
    with open(filename, 'w') as pgnfile:
        for result in results:
            pgnfile.write(f'[Event "AI Testing Match"]\n')
            pgnfile.write(f'[Round "{result.game_number}"]\n')
            pgnfile.write(f'[White "{"Your AI" if result.your_color == "white" else result.opponent_type}"]\n')
            pgnfile.write(f'[Black "{"Your AI" if result.your_color == "black" else result.opponent_type}"]\n')
            
            if result.result == 'win':
                pgn_result = '1-0' if result.your_color == 'white' else '0-1'
            elif result.result == 'loss':
                pgn_result = '0-1' if result.your_color == 'white' else '1-0'
            else:
                pgn_result = '1/2-1/2'
            
            pgnfile.write(f'[Result "{pgn_result}"]\n\n')
            pgnfile.write(' '.join(result.moves) + f' {pgn_result}\n\n')
    
    print(f"Games exported to {filename}")


if __name__ == "__main__":
    # Example usage
    print("Chess AI Testing Tool")
    print("=====================\n")
    
    # Check if Stockfish is available
    stockfish_path = find_stockfish()
    if stockfish_path:
        print(f"✓ Stockfish found at: {stockfish_path}\n")
    else:
        print("✗ Stockfish not found. Install with: brew install stockfish")
        print("  (You can still test against non-Stockfish opponents)\n")
    
    # Quick test: 10 games vs random opponent
    print("Running quick test: 10 games vs random opponent...")
    results = run_test_suite(
        num_games=10,
        opponent_type=OpponentType.RANDOM,
        your_color='both',
        depth=2,
        num_workers=4
    )
    
    # Export results
    export_to_csv(results, "quick_test_results.csv")
    export_to_pgn(results, "quick_test_games.pgn")
    
    # Uncomment for Stockfish testing (if available):
    
    if stockfish_path:
        print("\n" + "="*70)
        print("STOCKFISH TESTING")
        print("="*70)
        print("\nRunning calibration: 50 games vs Stockfish Level 1 (~800-1000 ELO)...")
        results = run_test_suite(
            num_games=50,
            opponent_type=OpponentType.STOCKFISH_1,
            your_color='both',
            depth=3,
            num_workers=8,
            stockfish_path=stockfish_path
        )
        export_to_csv(results, "stockfish_level_1_results.csv")
    
    # More comprehensive tests (uncomment as needed):
    
    # # Test vs Stockfish Skill Levels (for sub-1320 ELO estimation)
    # print("\nTesting against Stockfish Level 0 (weakest)...")
    # results = run_test_suite(
    #     num_games=50,
    #     opponent_type=OpponentType.STOCKFISH_0,
    #     your_color='both',
    #     depth=3
    # )
    # export_to_csv(results, "stockfish_level_0_results.csv")
    
    # print("\nTesting against Stockfish Level 3...")
    # results = run_test_suite(
    #     num_games=50,
    #     opponent_type=OpponentType.STOCKFISH_3,
    #     your_color='both',
    #     depth=3
    # )
    # export_to_csv(results, "stockfish_level_3_results.csv")
    
    # # Test vs Stockfish ELO (1320+)
    # print("\nTesting against Stockfish ELO 1320...")
    # results = run_test_suite(
    #     num_games=50,
    #     opponent_type=OpponentType.STOCKFISH_ELO_1320,
    #     your_color='both',
    #     depth=3
    # )
    # export_to_csv(results, "stockfish_elo_1320_results.csv")
    
    # # Test vs higher ELOs
    # print("\nTesting against Stockfish ELO 1500...")
    # results = run_test_suite(
    #     num_games=50,
    #     opponent_type=OpponentType.STOCKFISH_ELO_1500,
    #     your_color='both',
    #     depth=3
    # )
    # export_to_csv(results, "stockfish_1500_results.csv")
    
    # # Comprehensive ladder test
    # print("\n" + "="*70)
    # print("COMPREHENSIVE ELO LADDER TEST")
    # print("="*70)
    # print("\nTesting against multiple skill levels to find your ELO...")
    # 
    # # Test skill levels 0, 1, 3, 5, 10 (roughly 600-1200 ELO)
    # for level in [0, 1, 3, 5, 10]:
    #     opponent = OpponentType[f"STOCKFISH_{level}"]
    #     print(f"\nTesting against Skill Level {level}...")
    #     results = run_test_suite(
    #         num_games=30,
    #         opponent_type=opponent,
    #         your_color='both',
    #         depth=3
    #     )
    #     export_to_csv(results, f"skill_level_{level}_test.csv")
    # 
    # # If doing well, test actual ELO ratings
    # for elo in [1320, 1500, 1800]:
    #     opponent = OpponentType[f"STOCKFISH_ELO_{elo}"]
    #     print(f"\nTesting against ELO {elo}...")
    #     results = run_test_suite(
    #         num_games=30,
    #         opponent_type=opponent,
    #         your_color='both',
    #         depth=3
    #     )
    #     export_to_csv(results, f"elo_{elo}_test.csv")