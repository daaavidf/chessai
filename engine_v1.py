#!/usr/bin/env python3
"""
Chess Engine v1
A basic chess engine using minimax with alpha-beta pruning and piece-square tables
"""

import chess
import random
from typing import Optional


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