#!/usr/bin/env python3
"""
Chess AI Testing Tool
Tests your chess AI against various opponents with parallel processing
"""

import chess
import time
import random
import csv
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class OpponentType(Enum):
    RANDOM = "random"
    MATERIAL = "material"
    BASIC = "basic"


@dataclass
class GameResult:
    game_number: int
    result: str  # 'win', 'loss', 'draw'
    your_color: str  # 'white' or 'black'
    opponent_type: str
    moves: List[str]
    pgn: str
    move_count: int


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


def get_opponent_move(board: chess.Board, opponent_type: OpponentType, 
                     depth: int = 3) -> Optional[chess.Move]:
    """Get move based on opponent type."""
    if opponent_type == OpponentType.RANDOM:
        return get_random_move(board)
    elif opponent_type == OpponentType.MATERIAL:
        return get_best_move(board, depth=1, use_positional=False)
    elif opponent_type == OpponentType.BASIC:
        return get_best_move(board, depth=depth, use_positional=True)
    return get_random_move(board)


def play_single_game(args: Tuple) -> GameResult:
    """Play a single game. Designed to work with multiprocessing."""
    game_number, your_color, opponent_type, depth = args
    
    board = chess.Board()
    moves = []
    move_stack = []  # Store moves separately for PGN
    
    your_ai_color = chess.WHITE if your_color == 'white' else chess.BLACK
    
    while not board.is_game_over():
        if board.turn == your_ai_color:
            move = get_best_move(board, depth=depth, use_positional=True)
        else:
            move = get_opponent_move(board, opponent_type, depth=depth)
        
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
    
    return GameResult(
        game_number=game_number,
        result=result,
        your_color=your_color,
        opponent_type=opponent_type.value,
        moves=moves,
        pgn=' '.join(pgn_moves),
        move_count=len(moves)
    )


def run_test_suite(num_games: int, opponent_type: OpponentType, 
                  your_color: str = 'both', depth: int = 3, 
                  num_workers: Optional[int] = None) -> List[GameResult]:
    """
    Run a test suite of games.
    
    Args:
        num_games: Number of games to play
        opponent_type: Type of opponent (RANDOM, MATERIAL, BASIC)
        your_color: 'white', 'black', or 'both' (alternating)
        depth: Search depth for minimax (1-5, higher is stronger but slower)
        num_workers: Number of parallel workers (default: CPU count)
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n{'='*60}")
    print(f"Starting test suite:")
    print(f"  Games: {num_games}")
    print(f"  Opponent: {opponent_type.value}")
    print(f"  Your color: {your_color}")
    print(f"  Search depth: {depth}")
    print(f"  CPU cores: {num_workers}")
    print(f"{'='*60}\n")
    
    # Prepare game configurations
    game_configs = []
    for i in range(num_games):
        if your_color == 'both':
            color = 'white' if i % 2 == 0 else 'black'
        else:
            color = your_color
        game_configs.append((i + 1, color, opponent_type, depth))
    
    start_time = time.time()
    
    # Run games in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(play_single_game, game_configs)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    wins = sum(1 for r in results if r.result == 'win')
    losses = sum(1 for r in results if r.result == 'loss')
    draws = sum(1 for r in results if r.result == 'draw')
    win_rate = (wins / num_games * 100) if num_games > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Testing Complete!")
    print(f"{'='*60}")
    print(f"Results:")
    print(f"  Wins:      {wins:4d} ({wins/num_games*100:5.1f}%)")
    print(f"  Losses:    {losses:4d} ({losses/num_games*100:5.1f}%)")
    print(f"  Draws:     {draws:4d} ({draws/num_games*100:5.1f}%)")
    print(f"  Total:     {num_games:4d}")
    print(f"\nPerformance:")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"  Games/sec:    {num_games/total_time:.2f}")
    print(f"  Avg game:     {total_time/num_games:.2f}s")
    print(f"{'='*60}\n")
    
    return results


def export_to_csv(results: List[GameResult], filename: str = "test_results.csv"):
    """Export results to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['game_number', 'result', 'your_color', 'opponent_type', 
                     'move_count', 'moves']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'game_number': result.game_number,
                'result': result.result,
                'your_color': result.your_color,
                'opponent_type': result.opponent_type,
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
    
    # Uncomment for more comprehensive testing:
    
    # # Test vs Material-only AI
    # print("\n\nRunning comprehensive test: 50 games vs material AI...")
    # results = run_test_suite(
    #     num_games=50,
    #     opponent_type=OpponentType.MATERIAL,
    #     your_color='both',
    #     depth=3
    # )
    # export_to_csv(results, "material_test_results.csv")
    
    # # Test vs Basic AI
    # print("\n\nRunning advanced test: 50 games vs basic AI...")
    # results = run_test_suite(
    #     num_games=50,
    #     opponent_type=OpponentType.BASIC,
    #     your_color='both',
    #     depth=3
    # )
    # export_to_csv(results, "basic_test_results.csv")