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