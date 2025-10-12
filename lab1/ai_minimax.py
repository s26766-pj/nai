"""
Minimax AI implementation for the Hex game.

This module provides a minimax algorithm with alpha-beta pruning
for determining the best move in a Hex game position.
"""

from typing import Optional, Tuple
import math
from game import get_legal_moves, check_red_win, check_blue_win


def evaluate(board, player_color: str) -> int:
    """
    Evaluate the current board position from a player's perspective.
    
    Uses a simple piece counting heuristic that favors positions
    with more pieces for the evaluating player.
    
    Args:
        board: HexBoard instance
        player_color (str): Player color to evaluate for ("red" or "blue")
        
    Returns:
        int: Evaluation score (positive favors the player)
    """
    opponent = "blue" if player_color == "red" else "red"
    if board.winner == player_color:
        return 10_000
    if board.winner == opponent:
        return -10_000
    player_stones = sum(1 for v in board.state.values() if v == player_color)
    opp_stones = sum(1 for v in board.state.values() if v == opponent)
    return player_stones - opp_stones


def minimax(board, depth: int, maximizing: bool, player_color: str) -> Tuple[int, Optional[str]]:
    """
    Minimax algorithm with alpha-beta pruning for Hex game.
    
    Recursively evaluates the game tree to find the best move
    by assuming both players play optimally.
    
    Args:
        board: HexBoard instance
        depth (int): Remaining search depth
        maximizing (bool): True if maximizing player's turn
        player_color (str): Color of the player we're optimizing for
        
    Returns:
        Tuple[int, Optional[str]]: (score, best_move) pair
    """
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
    """
    Compute the best move for a player using the minimax algorithm.
    
    Uses minimax with alpha-beta pruning to find the optimal move
    by exploring the game tree to the specified depth.
    
    Args:
        board: HexBoard instance representing the current game state
        player_color (str): Player color ("red" or "blue")
        depth (int, optional): Search depth for minimax. Defaults to 2.
        
    Returns:
        str or None: Best move as a position label (e.g., "A1"), or None if no moves available
    """
    _, move = minimax(board, depth, True, player_color)
    return move


