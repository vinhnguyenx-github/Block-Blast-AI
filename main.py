import pygame
from grid import Grid
from piece import Piece
from config import *
from button import Button
from gameLogic import clear_lines, can_place_piece, can_place_any, generate_solvable_pieces

pygame.init()

# Window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Block Blast")

# Board
board = Grid(BOARD_ROWS, BOARD_COLS, BOARD_CELL_SIZE)

# Fonts (hacker theme)
font = pygame.font.SysFont("consolas", 32, bold=True)
game_over_font = pygame.font.SysFont("consolas", 48, bold=True)

# Score
score = 0

# Game state
game_over = False
reset_button = Button(WIDTH//2 - 80, 400, 160, 40, "RESET", font)
def draw_game_over(surface, font, game_over_font, reset_button, score, board, pieces):
    """Draw Game Over screen and handle reset button."""
    global game_over  # use the outer variable

    # Draw green bordered box
    box_width, box_height = 400, 120
    box_x = WIDTH // 2 - box_width // 2
    box_y = HEIGHT // 2 - box_height // 2
    box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
    pygame.draw.rect(surface, (0, 0, 0), box_rect)      # background
    pygame.draw.rect(surface, (0, 255, 0), box_rect, 3) # border

    # Title text
    text_surface = game_over_font.render("GAME OVER", True, (0, 255, 0))
    surface.blit(text_surface, (
        box_x + box_width // 2 - text_surface.get_width() // 2,
        box_y + 30
    ))

    # Draw reset button
    reset_button.draw(surface)

    # Handle click
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, board, pieces, score, True
        if reset_button.is_clicked(event):
            board.clear()
            pieces = generate_pieces()
            score = 0
            game_over = False
            return True, board, pieces, score, False

    return True, board, pieces, score, game_over

# ---------------- Helpers ----------------
def generate_pieces():
    new_pieces = []
    for _ in range(3):
        p = Piece()
        p.random_piece()
        new_pieces.append(p)
    return new_pieces

pieces = generate_solvable_pieces(board)
dragging = None 

# ---------------- Main Loop ----------------
running = True
while running:
    screen.fill((0, 0, 0))

    # ---------------- Events ----------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if game_over:
            if reset_button.is_clicked(event):
                board.clear()
                pieces = generate_pieces()
                score = 0
                game_over = False

        # Pick up piece
        elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            mx, my = pygame.mouse.get_pos()
            for i, piece in enumerate(pieces):
                if piece is None:
                    continue
                px = PIECES_OFFSET_X
                py = PIECES_OFFSET_Y + i * (piece.rows * piece.cell_size + 20)
                rect = pygame.Rect(px, py,
                                   piece.cols * piece.cell_size,
                                   piece.rows * piece.cell_size)
                if rect.collidepoint(mx, my):
                    dragging = {"piece": piece, "index": i}
                    break

        # Drop piece
        elif event.type == pygame.MOUSEBUTTONUP and dragging and not game_over:
            mx, my = pygame.mouse.get_pos()
            row = (my - BOARD_OFFSET_Y) // BOARD_CELL_SIZE
            col = (mx - BOARD_OFFSET_X) // BOARD_CELL_SIZE

            piece = dragging["piece"]
            if can_place_piece(board, piece, row, col):
                # Place piece
                for r in range(piece.rows):
                    for c in range(piece.cols):
                        if piece.grid[r, c] == 1:
                            board.set_cell(row + r, col + c, 1)

                # Clear lines & update score
                cleared = clear_lines(board)
                score += cleared

                pieces[dragging["index"]] = None
                if all(p is None for p in pieces):
                    pieces = generate_pieces()

                # Check game over
                if not can_place_any(board, pieces):
                    game_over = True

            dragging = None

    # ---------------- Draw ----------------
    # Draw board
    board.draw(screen, offset_x=BOARD_OFFSET_X, offset_y=BOARD_OFFSET_Y)

    # Draw available pieces
    for i, piece in enumerate(pieces):
        if piece is None:
            continue
        if dragging and dragging["index"] == i:
            continue
        piece.draw(screen,
                   offset_x=PIECES_OFFSET_X,
                   offset_y=PIECES_OFFSET_Y + i * (piece.rows * piece.cell_size + 20))

    # Draw dragging preview
    if dragging:
        mx, my = pygame.mouse.get_pos()
        piece = dragging["piece"]
        row = (my - BOARD_OFFSET_Y) // BOARD_CELL_SIZE
        col = (mx - BOARD_OFFSET_X) // BOARD_CELL_SIZE
        valid = can_place_piece(board, piece, row, col)

        for r in range(piece.rows):
            for c in range(piece.cols):
                if piece.grid[r, c] == 1:
                    x = BOARD_OFFSET_X + (col + c) * BOARD_CELL_SIZE
                    y = BOARD_OFFSET_Y + (row + r) * BOARD_CELL_SIZE
                    rect = pygame.Rect(x, y, BOARD_CELL_SIZE, BOARD_CELL_SIZE)

                    color = (0, 255, 0) if valid else (255, 0, 0)
                    fill = (0, 180, 0) if valid else (180, 0, 0)

                    pygame.draw.rect(screen, color, rect, 2)
                    inner_rect = rect.inflate(-10, -10)
                    pygame.draw.rect(screen, fill, inner_rect)

    # Draw score at top-left
    score_text = font.render(f"SCORE: {score}", True, (0, 255, 0))
    screen.blit(score_text, (40, 20))

    # Game Over state
    if game_over:
        # Draw green bordered box
        running, board, pieces, score, game_over = draw_game_over(
        screen, font, game_over_font, reset_button, score, board, pieces
        )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()