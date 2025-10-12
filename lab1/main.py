import pygame
from board import HexBoard
from game import (
    get_clicked_hex,
    check_red_win,
    check_blue_win,
    is_full,
    reset_for_new_game,
)
from ai_minimax import compute_minimax_move
from menu import run_menu

def main():
    """
    Main game loop for the Hex game.
    
    Handles the complete game flow including:
    - Player selection via menu
    - Game board initialization
    - Turn-based gameplay
    - AI move computation
    - Win detection and game reset
    """
    # Run menu first to select players
    red_player, blue_player = run_menu()

    pygame.init()
    screen = pygame.display.set_mode((1000, 650))
    pygame.display.set_caption("Hex Game")

    board = HexBoard(rows=11, cols=11, hex_size=30, offset_x=120, offset_y=100)

    clock = pygame.time.Clock()
    running = True

    current_player = "red"
    turn_number = 1
    result_time_ms = None  # when winner/tie declared

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and board.winner is None:
                pos = pygame.mouse.get_pos()
                clicked_hex = get_clicked_hex(board, pos)
                if clicked_hex and board.state[clicked_hex] is None:
                    # place for current player
                    board.state[clicked_hex] = current_player

                    # print player and clicked hex
                    if current_player == "red":
                        print(f"Player red clicked {clicked_hex}")
                        if check_red_win(board):
                            print("Red wins!")
                            result_time_ms = pygame.time.get_ticks()
                    else:
                        print(f"Player blue clicked {clicked_hex}")
                        if check_blue_win(board):
                            print("Blue wins!")
                            result_time_ms = pygame.time.get_ticks()

                    # switch player; increment turn after blue's move
                    previous_player = current_player
                    current_player = "blue" if current_player == "red" else "red"
                    if previous_player == "blue":
                        turn_number += 1

                    # tie check (no winner yet and board full)
                    if board.winner is None and is_full(board):
                        print("Tie: board is full, no points awarded.")
                        board.winner = "tie"
                        result_time_ms = pygame.time.get_ticks()

        # If it's an AI player's turn and no result yet, make AI move automatically
        if board.winner is None:
            if current_player == "red" and red_player == "Minimax":
                ai_move = compute_minimax_move(board, "red", depth=2)
                if ai_move and board.state.get(ai_move) is None:
                    board.state[ai_move] = "red"
                    print(f"Player red clicked {ai_move}")
                    if check_red_win(board):
                        print("Red wins!")
                        result_time_ms = pygame.time.get_ticks()
                    # switch to blue
                    previous_player = current_player
                    current_player = "blue"
                    if previous_player == "blue":
                        turn_number += 1
            elif current_player == "blue" and blue_player == "Minimax":
                ai_move = compute_minimax_move(board, "blue", depth=2)
                if ai_move and board.state.get(ai_move) is None:
                    board.state[ai_move] = "blue"
                    print(f"Player blue clicked {ai_move}")
                    if check_blue_win(board):
                        print("Blue wins!")
                        result_time_ms = pygame.time.get_ticks()
                    previous_player = current_player
                    current_player = "red"
                    if previous_player == "blue":
                        turn_number += 1

        # auto-reset 5 seconds after result
        if board.winner is not None and result_time_ms is not None:
            if pygame.time.get_ticks() - result_time_ms >= 5000:
                reset_for_new_game(board)
                current_player = "red"
                turn_number = 1
                result_time_ms = None

        screen.fill((255, 255, 255))
        # HUD with current player, turn and scores
        board.draw_hud(screen, current_player, turn_number)
        board.draw(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
