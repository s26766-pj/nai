import pygame
import math

class HexBoard:
    def __init__(self, rows, cols, hex_size, offset_x=50, offset_y=50):
        self.rows = rows
        self.cols = cols
        self.hex_size = hex_size
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.font = pygame.font.SysFont("Arial", 16, bold=True)

        # stan planszy: None / "red" / "blue"
        self.state = {
            f"{chr(ord('A') + c)}{r+1}": None
            for r in range(rows) for c in range(cols)
        }

        # wstępnie oblicza przesunięcia dla liter (A-K)
        step_x = self.hex_size * math.sqrt(3)
        self.letters_positions = [
            (chr(ord("A") + c),
             self.offset_x + c * step_x + 10) for c in range(self.cols)
        ]

        # wstępnie oblicz przesunięcia dla cyfr (1-11)
        step_y = self.hex_size * 1.5
        self.numbers_positions = [
            (str(r + 1),
             self.offset_y + r * step_y) for r in range(self.rows)
        ]

    def hex_corner(self, x, y, i):
        angle_deg = 60 * i - 30
        angle_rad = math.pi / 180 * angle_deg
        return (x + self.hex_size * math.cos(angle_rad),
                y + self.hex_size * math.sin(angle_rad))

    # konwersja (row, col) -> (x, y)
    def hex_to_pixel(self, row, col):
        x = self.offset_x + (col + row * 0.5) * (self.hex_size * math.sqrt(3))
        y = self.offset_y + row * (self.hex_size * 1.5)
        return (x, y)

    def draw_hex(self, surface, x, y, color=(200, 200, 200)):
        points = [self.hex_corner(x, y, i) for i in range(6)]
        pygame.draw.polygon(surface, color, points, 0)  # wypełnienie
        pygame.draw.polygon(surface, (0, 0, 0), points, 1)  # obrys

    # rysuje litery A-K
    def draw_letters(self, surface):
        for letter, xpos in self.letters_positions:
            text = self.font.render(letter, True, (200, 0, 0))
            # rysuje litery A-K u góry
            surface.blit(text, (xpos - 13, self.offset_y - 50))
            # rysuje litery A-K na dole
            surface.blit(text, (xpos + 245, self.offset_y + 481))

    # rysuje pojedynczą cyfrę w zadanej pozycji
    def draw_number(self, surface, number, x, y):
        text = self.font.render(str(number), True, (0, 0, 200))
        surface.blit(text, (x, y))

    # rysuje cyfry 1-11
    def draw_numbers(self, surface):
        # Lewa strona
        x_left, y_left = 80, 90
        for number in range(1, 12):
            self.draw_number(surface, number, x_left, y_left)
            x_left += 25
            y_left += 45

        # Prawa strona
        x_right, y_right = 672, 90
        for number in range(1, 12):
            self.draw_number(surface, number, x_right, y_right)
            if number == 1:
                step_x = 28  # pierwszy skok 28
            else:
                step_x = 25  # kolejne skoki 25
            x_right += step_x
            y_right += 45


    def draw(self, surface):
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = self.hex_to_pixel(r, c)
                label = f"{chr(ord('A') + c)}{r+1}"

                # wybór koloru wg stanu
                if self.state[label] == "red":
                    color = (255, 100, 100)
                elif self.state[label] == "blue":
                    color = (100, 100, 255)
                else:
                    color = (220, 220, 220)

                self.draw_hex(surface, x, y, color=color)

        # rysuj oznaczenia wokół planszy
        self.draw_letters(surface)
        self.draw_numbers(surface)

    # zwraca nazwę klikniętego pola (np. 'A1') albo None
    def get_clicked_hex(self, pos):
        mx, my = pos
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = self.hex_to_pixel(r, c)
                dx, dy = mx - x, my - y
                if dx*dx + dy*dy < (self.hex_size*0.9)**2:  # proste wykrywanie kliknięcia
                    return f"{chr(ord('A') + c)}{r+1}"
        return None
