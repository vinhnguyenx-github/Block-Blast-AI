import numpy as np
import pygame
from config import *

class Grid:
    def __init__(self, rows, cols , cell_size):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.grid = np.zeros((rows, cols), dtype=int)

    def get_cell(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r, c]
        return None
    
    def get_cell_from_pos(self, x, y, offset_x=0, offset_y=0):
        """Return (row, col) if click is inside grid, else None"""
        col = (x - offset_x) // self.cell_size
        row = (y - offset_y) // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(row), int(col)
        return None

    def set_cell(self, r, c, value):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r, c] = value

    def clear(self):
        self.grid.fill(0)
        
    def draw(self, surface, offset_x=0, offset_y=0):
        for r in range(self.rows):
            for c in range(self.cols):
                x = offset_x + c * self.cell_size
                y = offset_y + r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                # Always draw the green outline
                pygame.draw.rect(surface, (0, 255, 0), rect, 1)

                # Draw a slightly smaller filled rect if cell is filled
                if self.grid[r, c] == 1:
                    inner_rect = rect.inflate(-8, -8)  # shrink inside by 8px
                    pygame.draw.rect(surface, (0, 200, 0), inner_rect)
