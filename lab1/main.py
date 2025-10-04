import pygame
from board import HexBoard

def main():
    pygame.init()
    screen = pygame.display.set_mode((1000, 650))
    pygame.display.set_caption("Hex Game")

    board = HexBoard(rows=11, cols=11, hex_size=30, offset_x=120, offset_y=100)

    clock = pygame.time.Clock()
    running = True

    current_player = "red"

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                clicked_hex = board.get_clicked_hex(pos)
                if clicked_hex and board.state[clicked_hex] is None:
                    # zaznacz pole
                    board.state[clicked_hex] = current_player

                    # wypisz do konsoli
                    gracz = "czerwony" if current_player == "red" else "niebieski"
                    print(f"Gracz {gracz} zaznaczył pole {clicked_hex}")

                    # zmiana gracza
                    current_player = "blue" if current_player == "red" else "red"

        screen.fill((255, 255, 255))
        board.draw(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
