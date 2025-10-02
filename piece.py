import numpy as np
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
            # 3x3 solid block
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],

            # 2x2 square
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

            # Horizontal 4-line
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

            # Vertical 4-line
            [[1],
             [1],
             [1],
             [1]],

            # Vertical 5-line
            [[1],
             [1],
             [1],
             [1],
             [1]],

            # L-shape
            [[1, 1],
             [0, 1]],
        ]

        shape = np.array(shapes[np.random.randint(len(shapes))], dtype=int)
        self.grid.fill(0)  # reset
        self.grid[:shape.shape[0], :shape.shape[1]] = shape