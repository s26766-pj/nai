from typing import Optional, Tuple
import math
from game import get_legal_moves, check_red_win, check_blue_win


def evaluate(board, player_color: str) -> int:
    opponent = "blue" if player_color == "red" else "red"
    if board.winner == player_color:
        return 10_000
    if board.winner == opponent:
        return -10_000
    player_stones = sum(1 for v in board.state.values() if v == player_color)
    opp_stones = sum(1 for v in board.state.values() if v == opponent)
    return player_stones - opp_stones


def minimax(board, depth: int, maximizing: bool, player_color: str) -> Tuple[int, Optional[str]]:
    if board.winner is not None or depth == 0:
        return evaluate(board, player_color), None

    moves = get_legal_moves(board)
    if not moves:
        return evaluate(board, player_color), None

    current_color = player_color if maximizing else ("blue" if player_color == "red" else "red")

    best_move: Optional[str] = None
    if maximizing:
        best_score = -math.inf
        for m in moves:
            board.state[m] = current_color
            prev_winner = board.winner
            if current_color == "red":
                check_red_win(board, do_score=False)
            else:
                check_blue_win(board, do_score=False)
            score, _ = minimax(board, depth - 1, False, player_color)
            board.state[m] = None
            board.winner = prev_winner
            if score > best_score:
                best_score = score
                best_move = m
        return int(best_score), best_move
    else:
        best_score = math.inf
        for m in moves:
            board.state[m] = current_color
            prev_winner = board.winner
            if current_color == "red":
                check_red_win(board, do_score=False)
            else:
                check_blue_win(board, do_score=False)
            score, _ = minimax(board, depth - 1, True, player_color)
            board.state[m] = None
            board.winner = prev_winner
            if score < best_score:
                best_score = score
                best_move = m
        return int(best_score), best_move


def compute_minimax_move(board, player_color: str, depth: int = 2) -> Optional[str]:
    _, move = minimax(board, depth, True, player_color)
    return move


