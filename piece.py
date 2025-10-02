import numpy as np
import pygame
from grid import Grid
from config import *

class Piece(Grid):
    def __init__(self, matrix=None, cell_size=PIECE_CELL_SIZE):
        super().__init__(5, 5, cell_size=cell_size)
        if matrix is not None:
            self.grid = np.array(matrix, dtype=int)

    def random_piece(self):
        """Generate a random piece shape inside a 5x5 grid."""
        shapes = [
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],

            [[1, 1, 1],
             [1, 0, 0],
             [1, 0, 0]],

            [[1, 0, 0],
             [1, 0, 0],
             [1, 1, 1]],

            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],

            [[0, 0, 1],
             [0, 1, 0],
             [1, 0, 0]],

            [[0, 1, 1],
             [1, 1, 0]],

            [[1, 0, 0],
             [1, 1, 1]],

            [[0, 1, 0],
             [1, 1, 1]],

            [[1, 1]],

            [[1, 1, 1]],

            [[1, 1, 1, 1]],

            [[1, 1, 1, 1, 1]],

            [[1, 1],
             [0, 1],
             [0, 1]],

            [[0, 1],
             [1, 1],
             [1, 0]],

            [[0, 1],
             [1, 0]],

            [[0, 1],
             [1, 1],
             [0, 1]],

            [[1],
             [1]],

            [[1],
             [1],
             [1],
             [1]],

            [[1],
             [1],
             [1],
             [1],
             [1]],

            [[1, 1],
             [0, 1]],
        ]

        shape = np.array(shapes[np.random.randint(len(shapes))], dtype=int)
        self.grid.fill(0)
        self.grid[:shape.shape[0], :shape.shape[1]] = shape

    def draw(self, surface, offset_x=0, offset_y=0):
        """Draw only the shape (1’s), no grid outline."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    x = offset_x + c * self.cell_size
                    y = offset_y + r * self.cell_size
                    rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                    # Filled green block with outline
                    inner_rect = rect.inflate(-6, -6)  
                    pygame.draw.rect(surface, (0, 200, 0), inner_rect)
                    pygame.draw.rect(surface, (0, 255, 0), rect, 2)