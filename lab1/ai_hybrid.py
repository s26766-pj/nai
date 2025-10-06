from typing import Optional
import math
import random
from game import get_legal_moves, check_red_win, check_blue_win


def _clear_paths(board) -> None:
    if hasattr(board, "red_path_top_bottom"):
        board.red_path_top_bottom = None
    if hasattr(board, "red_path_left_right"):
        board.red_path_left_right = None
    if hasattr(board, "blue_path_left_right"):
        board.blue_path_left_right = None


def _random_rollout(board, current_turn: str) -> Optional[str]:
    winner_before = board.winner
    applied: list[tuple[str, str]] = []
    try:
        while board.winner is None:
            legal = get_legal_moves(board)
            if not legal:
                break
            mv = random.choice(legal)
            board.state[mv] = current_turn
            applied.append((mv, current_turn))
            if current_turn == "red":
                check_red_win(board, do_score=False)
            else:
                check_blue_win(board, do_score=False)
            current_turn = "blue" if current_turn == "red" else "red"
        return board.winner
    finally:
        for mv, _ in reversed(applied):
            board.state[mv] = None
        board.winner = winner_before
        _clear_paths(board)


def _monte_carlo_eval(board, root_player: str, to_move: str, rollouts: int = 24) -> float:
    wins = 0.0
    for _ in range(rollouts):
        w = _random_rollout(board, to_move)
        if w is None:
            wins += 0.5
        elif w == root_player:
            wins += 1.0
        else:
            wins += 0.0
    return wins / rollouts


def _alpha_beta_mc(board, depth: int, alpha: float, beta: float, maximizing: bool,
                   root_player: str, to_move: str, rollouts: int) -> tuple[float, Optional[str]]:
    if board.winner is not None:
        if board.winner == root_player:
            return 1.0, None
        elif board.winner is None:
            return 0.5, None
        else:
            return 0.0, None
    if depth == 0:
        return _monte_carlo_eval(board, root_player, to_move, rollouts), None

    legal = get_legal_moves(board)
    if not legal:
        return _monte_carlo_eval(board, root_player, to_move, rollouts), None

    best_move: Optional[str] = None
    if maximizing:
        value = -math.inf
        for mv in legal:
            board.state[mv] = to_move
            prev_winner = board.winner
            if to_move == "red":
                check_red_win(board, do_score=False)
            else:
                check_blue_win(board, do_score=False)
            score, _ = _alpha_beta_mc(
                board, depth - 1, alpha, beta, False,
                root_player, "blue" if to_move == "red" else "red", rollouts
            )
            board.state[mv] = None
            board.winner = prev_winner
            _clear_paths(board)
            if score > value:
                value = score
                best_move = mv
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return float(value), best_move
    else:
        value = math.inf
        for mv in legal:
            board.state[mv] = to_move
            prev_winner = board.winner
            if to_move == "red":
                check_red_win(board, do_score=False)
            else:
                check_blue_win(board, do_score=False)
            score, _ = _alpha_beta_mc(
                board, depth - 1, alpha, beta, True,
                root_player, "blue" if to_move == "red" else "red", rollouts
            )
            board.state[mv] = None
            board.winner = prev_winner
            _clear_paths(board)
            if score < value:
                value = score
                best_move = mv
            beta = min(beta, value)
            if alpha >= beta:
                break
        return float(value), best_move


def compute_minimax_mc_move(board, player_color: str, depth: int = 2, rollouts: int = 24) -> Optional[str]:
    score, move = _alpha_beta_mc(board, depth, -math.inf, math.inf, True, player_color, player_color, rollouts)
    return move


from ai_mcts import compute_mcts_move


def compute_hybrid_move(board, player_color: str) -> Optional[str]:
    legal = get_legal_moves(board)
    # Heuristic: use MCTS in early game (many moves), alpha-beta+MC later
    if len(legal) > 50:
        return compute_mcts_move(board, player_color, iterations=400)
    else:
        return compute_minimax_mc_move(board, player_color, depth=3, rollouts=32)


