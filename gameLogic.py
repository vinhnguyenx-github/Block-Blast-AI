import random
from piece import Piece
from solver import solve_any_order

# Hard diagonal/awkward shapes
HARD_SHAPES = [
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],       # diagonal ↘

    [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]],       # diagonal ↙

    [[0, 1],
     [1, 0]],  
    
    [[1, 0],
     [0, 1]],
    
    [[1, 1],
     [1, 0]],

    [[0, 1],
     [1, 1],
     [0, 1]],          # vertical + cross
]

def _make_piece_from_shape(shape):
    """Build a Piece from a raw 2D list shape."""
    p = Piece()
    p.grid.fill(0)
    for r in range(len(shape)):
        for c in range(len(shape[0])):
            p.grid[r, c] = shape[r][c]
    return p

def generate_solvable_pieces(board, max_tries=1000, hard_prob=0.01):
    """
    Generate 3 pieces that are solvable on the given board.
    Each piece has 'hard_prob' chance of being from HARD_SHAPES,
    otherwise uses Piece.random_piece().
    """
    B = (board.grid > 0).astype("uint8")

    for _ in range(max_tries):
        pieces = []
        piece_arrays = []
        for _ in range(3):
            if random.random() < hard_prob:
                shape = random.choice(HARD_SHAPES)
                p = _make_piece_from_shape(shape)
            else:
                p = Piece()
                p.random_piece()
            pieces.append(p)
            piece_arrays.append(p.grid)

        # Check if the 3 pieces can be solved
        if solve_any_order(B, piece_arrays):
            return pieces

    # fallback if solver fails repeatedly
    return [Piece() for _ in range(3)]

def clear_lines(board):
    """Clear full rows and columns and return number of squares cleared."""
    full_rows = [r for r in range(board.rows) if all(board.grid[r, :] == 1)]
    full_cols = [c for c in range(board.cols) if all(board.grid[:, c] == 1)]

    squares_cleared = 0

    # Count squares cleared
    squares_cleared += len(full_rows) * board.cols
    squares_cleared += len(full_cols) * board.rows

    # Clear them
    for r in full_rows:
        board.grid[r, :] = 0
    for c in full_cols:
        board.grid[:, c] = 0

    return squares_cleared

def can_place_piece(board, piece, row, col):
    """Check if piece can be placed at (row, col) without overlap/out of bounds."""
    for r in range(piece.rows):
        for c in range(piece.cols):
            if piece.grid[r, c] == 1:
                br, bc = row + r, col + c
                if not (0 <= br < board.rows and 0 <= bc < board.cols):
                    return False
                if board.grid[br, bc] == 1:
                    return False
    return True

def can_place_any(board, pieces):
    """Check if at least one piece can be placed anywhere on the board."""
    for piece in pieces:
        if piece is None:
            continue
        for row in range(board.rows):
            for col in range(board.cols):
                if can_place_piece(board, piece, row, col):
                    return True
    return False