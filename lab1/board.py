import pygame
import math
from colors import (
    COLOR_RED,
    COLOR_BLUE,
    COLOR_RED_FILL,
    COLOR_BLUE_FILL,
    COLOR_EMPTY_HEX,
    COLOR_BLACK,
    COLOR_WHITE,
    COLOR_LINE_RED,
    COLOR_WINNING_PATH,
    COLOR_BANNER_BG,
    COLOR_TIE,
)
from game import get_scores

class HexBoard:
    """
    A class representing a hexagonal board for the Hex game.
    
    This class manages the game state, rendering, and interactions for a Hex board.
    It handles coordinate conversion, neighbor calculations, and visual representation.
    """
    
    def __init__(self, rows, cols, hex_size, offset_x=50, offset_y=50):
        """
        Initialize a Hex board.
        
        Args:
            rows (int): Number of rows in the board
            cols (int): Number of columns in the board  
            hex_size (int): Size of each hexagonal cell
            offset_x (int, optional): X offset for board positioning. Defaults to 50.
            offset_y (int, optional): Y offset for board positioning. Defaults to 50.
        """
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

        # winner state: None / "red" / "blue"
        self.winner = None
        # match scores (wins)
        self._scores = {"red": 0, "blue": 0}
        # winning path overlays for red
        self.red_path_top_bottom = None  # list of labels
        # winning path overlay for blue (left to right)
        self.blue_path_left_right = None  # list of labels
        # match number counter (increments on each new game)
        self.match_number = 1

    def label_to_rc(self, label):
        """
        Convert a board label (e.g., 'A1') to row, column coordinates.
        
        Args:
            label (str): Board position label like 'A1', 'B2', etc.
            
        Returns:
            tuple: (row, col) coordinates where row and col are 0-based indices
        """
        col_char = label[0]
        row_num = int(label[1:])
        col = ord(col_char) - ord('A')
        row = row_num - 1
        return (row, col)

    def rc_to_label(self, row, col):
        """
        Convert row, column coordinates to a board label.
        
        Args:
            row (int): Row index (0-based)
            col (int): Column index (0-based)
            
        Returns:
            str: Board position label like 'A1', 'B2', etc.
        """
        return f"{chr(ord('A') + col)}{row + 1}"

    def neighbors(self, row, col):
        """
        Get all valid neighboring positions for a given row, column.
        
        Args:
            row (int): Row index (0-based)
            col (int): Column index (0-based)
            
        Yields:
            tuple: (neighbor_row, neighbor_col) for each valid neighbor
        """
        deltas = [
            (0, 1),  # east
            (0, -1), # west
            (1, 0),  # south-east in axial r+1
            (-1, 0), # north-west in axial r-1
            (-1, 1), # north-east (r-1, c+1)
            (1, -1), # south-west (r+1, c-1)
        ]
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                yield (nr, nc)

    def red_neighbors(self, label):
        """
        Get all red pieces that are neighbors of the given position.
        
        Args:
            label (str): Board position label like 'A1'
            
        Returns:
            list: List of neighbor labels that contain red pieces
        """
        r, c = self.label_to_rc(label)
        result = []
        for nr, nc in self.neighbors(r, c):
            nlabel = self.rc_to_label(nr, nc)
            if self.state.get(nlabel) == "red":
                result.append(nlabel)
        return result

    def are_neighbors(self, label_a, label_b):
        """
        Check if two board positions are neighbors.
        
        Args:
            label_a (str): First board position label
            label_b (str): Second board position label
            
        Returns:
            bool: True if the positions are neighbors, False otherwise
        """
        ra, ca = self.label_to_rc(label_a)
        rb, cb = self.label_to_rc(label_b)
        for nr, nc in self.neighbors(ra, ca):
            if (nr, nc) == (rb, cb):
                return True
        return False

    def draw_hud(self, surface, current_player, turn_number):
        """
        Draw the heads-up display showing game information.
        
        Args:
            surface: Pygame surface to draw on
            current_player (str): Current player ("red" or "blue")
            turn_number (int): Current turn number
        """
        scores = get_scores(self)
        hud_font = self.font
        # Current player label
        player_text = f"Current: {'RED' if current_player == 'red' else 'BLUE'}"
        player_color = COLOR_RED if current_player == "red" else COLOR_BLUE
        player_surf = hud_font.render(player_text, True, player_color)
        
        # Informacje o grze
        red_surf = hud_font.render(f"Red: {scores['red']}", True, COLOR_RED)
        blue_surf = hud_font.render(f"Blue: {scores['blue']}", True, COLOR_BLUE)
        match_surf = hud_font.render(f"Match: {self.match_number}", True, COLOR_BLACK)
        turn_surf = hud_font.render(f"Turn: {turn_number}", True, COLOR_BLACK)

        # pozycje wyświetlania informacji o grze
        surface.blit(player_surf, (20, 350))
        surface.blit(match_surf, (20, 375))
        surface.blit(turn_surf, (20, 400))
        surface.blit(red_surf, (20, 425))
        surface.blit(blue_surf, (20, 450))
        

    def hex_corner(self, x, y, i):
        """
        Calculate the coordinates of a hexagon corner.
        
        Args:
            x (float): Center x coordinate of the hexagon
            y (float): Center y coordinate of the hexagon
            i (int): Corner index (0-5, going clockwise from right)
            
        Returns:
            tuple: (x, y) coordinates of the specified corner
        """
        angle_deg = 60 * i - 30
        angle_rad = math.pi / 180 * angle_deg
        return (x + self.hex_size * math.cos(angle_rad),
                y + self.hex_size * math.sin(angle_rad))

    def hex_to_pixel(self, row, col):
        """
        Convert board coordinates to pixel coordinates.
        
        Args:
            row (int): Row index (0-based)
            col (int): Column index (0-based)
            
        Returns:
            tuple: (x, y) pixel coordinates for the center of the hexagon
        """
        x = self.offset_x + (col + row * 0.5) * (self.hex_size * math.sqrt(3))
        y = self.offset_y + row * (self.hex_size * 1.5)
        return (x, y)

    def draw_hex(self, surface, x, y, color=COLOR_EMPTY_HEX):
        """
        Draw a single hexagon at the specified coordinates.
        
        Args:
            surface: Pygame surface to draw on
            x (float): Center x coordinate
            y (float): Center y coordinate
            color (tuple, optional): Color of the hexagon. Defaults to COLOR_EMPTY_HEX.
        """
        points = [self.hex_corner(x, y, i) for i in range(6)]
        pygame.draw.polygon(surface, color, points, 0)  # wypełnienie
        pygame.draw.polygon(surface, COLOR_BLACK, points, 1)  # obrys

    def draw_letters(self, surface):
        """
        Draw the column labels (A-K) on the board.
        
        Args:
            surface: Pygame surface to draw on
        """
        for letter, xpos in self.letters_positions:
            text = self.font.render(letter, True, COLOR_RED)
            # rysuje litery A-K u góry
            surface.blit(text, (xpos - 13, self.offset_y - 50))
            # rysuje litery A-K na dole
            surface.blit(text, (xpos + 245, self.offset_y + 481))

    def draw_number(self, surface, number, x, y):
        """
        Draw a single number at the specified position.
        
        Args:
            surface: Pygame surface to draw on
            number (int): Number to draw
            x (int): X coordinate
            y (int): Y coordinate
        """
        text = self.font.render(str(number), True, COLOR_BLUE)
        surface.blit(text, (x, y))

    def draw_numbers(self, surface):
        """
        Draw the row numbers (1-11) on the board.
        
        Args:
            surface: Pygame surface to draw on
        """
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
        """
        Draw the entire board including hexagons, labels, and game state.
        
        Args:
            surface: Pygame surface to draw on
        """
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = self.hex_to_pixel(r, c)
                label = f"{chr(ord('A') + c)}{r+1}"

                # wybór koloru wg stanu
                if self.state[label] == "red":
                    color = COLOR_RED_FILL
                elif self.state[label] == "blue":
                    color = COLOR_BLUE_FILL
                else:
                    color = COLOR_EMPTY_HEX

                self.draw_hex(surface, x, y, color=color)

        # rysuj oznaczenia wokół planszy
        self.draw_letters(surface)
        self.draw_numbers(surface)

        # draw winning paths will occur after winner banner (if any)

        # if someone won, draw an overlay banner
        if self.winner is not None:
        

                    
            line_color = COLOR_WINNING_PATH
            line_width = 6
            if self.winner == "red" and self.red_path_top_bottom:
                self.draw_winning_line(surface, self.red_path_top_bottom, line_color, line_width)
            if self.winner == "blue" and self.blue_path_left_right:
                self.draw_winning_line(surface, self.blue_path_left_right, line_color, line_width)


            if self.winner in ("red", "blue"):
                banner_text = f"Winner: {'RED' if self.winner == 'red' else 'BLUE'}"
                banner_color = COLOR_RED if self.winner == 'red' else COLOR_BLUE
            else:
                banner_text = "Tie: board is full"
                banner_color = COLOR_TIE

            # large centered banner
            big_font = pygame.font.SysFont("Arial", 48, bold=True)
            text_surf = big_font.render(banner_text, True, COLOR_WHITE)
            sw, sh = surface.get_size()
            rect = text_surf.get_rect(center=(sw // 2, sh // 2))
            bg_rect = rect.inflate(80, 40)
            pygame.draw.rect(surface, COLOR_BANNER_BG, bg_rect, 0)
            pygame.draw.rect(surface, banner_color, bg_rect, 4)
            surface.blit(text_surf, rect)



    def draw_winning_line(self, surface, labels, color, width):
        """
        Draw a line connecting the winning path.
        
        Args:
            surface: Pygame surface to draw on
            labels (list): List of position labels forming the winning path
            color (tuple): Color of the line
            width (int): Width of the line
        """
        points = [self.hex_to_pixel(*self.label_to_rc(lbl)) for lbl in labels]
        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, width)

