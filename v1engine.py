"""Chess engine implementing basic minimax with alpha-beta pruning."""
import chess

class V1Engine:
    # Piece values for evaluation
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    # Piece-square tables (copied from original)
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
    
    # ... (copy remaining piece-square tables)

    def __init__(self, depth=3):
        self.depth = depth

    def get_best_move(self, board):
        """Find best move in current position."""
        legal_moves = list(board.legal_moves)
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in legal_moves:
            board.push(move)
            value = -self.minimax(self.depth - 1, -beta, -alpha, board)
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            
        return best_move

    def minimax(self, depth, alpha, beta, board):
        """Minimax search with alpha-beta pruning."""
        if depth == 0:
            return self.evaluate_board(board)
            
        legal_moves = list(board.legal_moves)
        
        if len(legal_moves) == 0:
            if board.is_checkmate():
                return -20000
            return 0
            
        max_value = float('-inf')
        for move in legal_moves:
            board.push(move)
            value = -self.minimax(depth - 1, -beta, -alpha, board)
            board.pop()
            
            max_value = max(max_value, value)
            alpha = max(alpha, value)
            
            if alpha >= beta:
                break
                
        return max_value

    def evaluate_board(self, board):
        """Static evaluation of board position."""
        if board.is_checkmate():
            if board.turn:
                return -20000
            return 20000
            
        total = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
                
            value = self.PIECE_VALUES[piece.piece_type]
            
            # Add position bonus based on piece-square tables
            if piece.piece_type == chess.PAWN:
                value += self.PAWN_TABLE[square if piece.color else 63 - square]
            elif piece.piece_type == chess.KNIGHT:
                value += self.KNIGHT_TABLE[square if piece.color else 63 - square]
            # ... add remaining piece type position values
                
            if not piece.color:
                value = -value
                
            total += value
            
        return total if board.turn else -total