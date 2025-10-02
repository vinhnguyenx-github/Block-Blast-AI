import pygame
from config import *

class Button:
    def __init__(self, x, y, w, h, text, font,
                 bg_color=BLACK, border_color=(0, 255, 0),
                 text_color=(0, 255, 0), hover_color=(0, 200, 0)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.bg_color = bg_color
        self.border_color = border_color
        self.text_color = text_color
        self.hover_color = hover_color

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mouse_pos)

        # Hover makes border/text brighter
        border_col = self.hover_color if is_hover else self.border_color
        text_col = self.hover_color if is_hover else self.text_color

        # Draw background
        pygame.draw.rect(surface, self.bg_color, self.rect)
        # Draw border
        pygame.draw.rect(surface, border_col, self.rect, 2)

        # Render text
        text_surface = self.font.render(self.text, True, text_col)
        text_x = self.rect.centerx - text_surface.get_width() // 2
        text_y = self.rect.centery - text_surface.get_height() // 2
        surface.blit(text_surface, (text_x, text_y))

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False