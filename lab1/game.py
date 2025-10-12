"""
Game logic helpers that operate on a HexBoard instance.

This module contains functions for game mechanics, move validation,
win detection, and game state management for the Hex game.
"""

from typing import Optional, Dict, List
import math
import random

def get_legal_moves(board) -> List[str]:
    """
    Get all legal moves (empty positions) on the board.
    
    Args:
        board: HexBoard instance
        
    Returns:
        List[str]: List of position labels that are empty (None)
    """
    return [label for label, v in board.state.items() if v is None]

def get_clicked_hex(board, pos) -> Optional[str]:
    """
    Determine which hex was clicked based on mouse position.
    
    Args:
        board: HexBoard instance
        pos: Mouse position tuple (x, y)
        
    Returns:
        Optional[str]: Position label if a hex was clicked, None otherwise
    """
    mx, my = pos
    for r in range(board.rows):
        for c in range(board.cols):
            x, y = board.hex_to_pixel(r, c)
            dx, dy = mx - x, my - y
            if dx * dx + dy * dy < (board.hex_size * 0.9) ** 2:
                return f"{chr(ord('A') + c)}{r+1}"
    return None


def get_scores(board) -> Dict[str, int]:
    """
    Get the current match scores for both players.
    
    Args:
        board: HexBoard instance
        
    Returns:
        Dict[str, int]: Dictionary with 'red' and 'blue' scores
    """
    return {"red": board._scores["red"], "blue": board._scores["blue"]}


def register_win(board, winner: str) -> None:
    """
    Register a win for the specified player.
    
    Args:
        board: HexBoard instance
        winner (str): Winning player ("red" or "blue")
    """
    if winner in ("red", "blue"):
        board._scores[winner] += 1


def is_full(board) -> bool:
    """
    Check if the board is completely filled with pieces.
    
    Args:
        board: HexBoard instance
        
    Returns:
        bool: True if all positions are occupied, False otherwise
    """
    for v in board.state.values():
        if v is None:
            return False
    return True


def red_connects_row_1_to_11(board) -> bool:
    """
    Check if red pieces connect the top row to the bottom row.
    
    Uses breadth-first search to find if there's a path of connected red pieces
    from any position in the top row to any position in the bottom row.
    
    Args:
        board: HexBoard instance
        
    Returns:
        bool: True if red has a winning connection, False otherwise
    """
    from collections import deque
    visited = set()
    queue = deque()
    for c in range(board.cols):
        start_label = board.rc_to_label(0, c)
        if board.state.get(start_label) == "red":
            queue.append((0, c))
            visited.add((0, c))
    while queue:
        r, c = queue.popleft()
        if r == board.rows - 1:
            return True
        for nr, nc in board.neighbors(r, c):
            if (nr, nc) not in visited:
                nlabel = board.rc_to_label(nr, nc)
                if board.state.get(nlabel) == "red":
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def find_red_path_top_bottom(board) -> Optional[List[str]]:
    """
    Find the actual path of red pieces connecting top to bottom.
    
    Uses breadth-first search to find the shortest path of connected red pieces
    from the top row to the bottom row.
    
    Args:
        board: HexBoard instance
        
    Returns:
        Optional[List[str]]: List of position labels forming the winning path,
                            or None if no path exists
    """
    from collections import deque
    parents = {}
    visited = set()
    queue = deque()
    for c in range(board.cols):
        start = (0, c)
        if board.state.get(board.rc_to_label(*start)) == "red":
            queue.append(start)
            visited.add(start)
            parents[start] = None
    end_node = None
    while queue:
        r, c = queue.popleft()
        if r == board.rows - 1:
            end_node = (r, c)
            break
        for nr, nc in board.neighbors(r, c):
            node = (nr, nc)
            if node not in visited and board.state.get(board.rc_to_label(nr, nc)) == "red":
                visited.add(node)
                parents[node] = (r, c)
                queue.append(node)
    if end_node is None:
        return None
    path: List[str] = []
    cur = end_node
    while cur is not None:
        path.append(board.rc_to_label(*cur))
        cur = parents[cur]
    path.reverse()
    return path

def blue_connects_A_to_K(board) -> bool:
    """
    Check if blue pieces connect the left column to the right column.
    
    Uses breadth-first search to find if there's a path of connected blue pieces
    from any position in the leftmost column to any position in the rightmost column.
    
    Args:
        board: HexBoard instance
        
    Returns:
        bool: True if blue has a winning connection, False otherwise
    """
    from collections import deque
    visited = set()
    queue = deque()
    for r in range(board.rows):
        start_label = board.rc_to_label(r, 0)
        if board.state.get(start_label) == "blue":
            queue.append((r, 0))
            visited.add((r, 0))
    while queue:
        r, c = queue.popleft()
        if c == board.cols - 1:
            return True
        for nr, nc in board.neighbors(r, c):
            if (nr, nc) not in visited:
                nlabel = board.rc_to_label(nr, nc)
                if board.state.get(nlabel) == "blue":
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def find_blue_path_left_right(board) -> Optional[List[str]]:
    """
    Find the actual path of blue pieces connecting left to right.
    
    Uses breadth-first search to find the shortest path of connected blue pieces
    from the leftmost column to the rightmost column.
    
    Args:
        board: HexBoard instance
        
    Returns:
        Optional[List[str]]: List of position labels forming the winning path,
                            or None if no path exists
    """
    from collections import deque
    parents = {}
    visited = set()
    queue = deque()
    for r in range(board.rows):
        start = (r, 0)
        if board.state.get(board.rc_to_label(*start)) == "blue":
            queue.append(start)
            visited.add(start)
            parents[start] = None
    end_node = None
    while queue:
        r, c = queue.popleft()
        if c == board.cols - 1:
            end_node = (r, c)
            break
        for nr, nc in board.neighbors(r, c):
            node = (nr, nc)
            if node not in visited and board.state.get(board.rc_to_label(nr, nc)) == "blue":
                visited.add(node)
                parents[node] = (r, c)
                queue.append(node)
    if end_node is None:
        return None
    path: List[str] = []
    cur = end_node
    while cur is not None:
        path.append(board.rc_to_label(*cur))
        cur = parents[cur]
    path.reverse()
    return path


def check_red_win(board, do_score: bool = True) -> bool:
    """
    Check if red player has won the game.
    
    Args:
        board: HexBoard instance
        do_score (bool, optional): Whether to update the score. Defaults to True.
        
    Returns:
        bool: True if red has won, False otherwise
    """
    if board.winner is not None:
        return board.winner == "red"
    tb_path = find_red_path_top_bottom(board)
    if tb_path:
        board.red_path_top_bottom = tb_path
        board.winner = "red"
        if do_score:
            register_win(board, "red")
        return True
    return False


def check_blue_win(board, do_score: bool = True) -> bool:
    """
    Check if blue player has won the game.
    
    Args:
        board: HexBoard instance
        do_score (bool, optional): Whether to update the score. Defaults to True.
        
    Returns:
        bool: True if blue has won, False otherwise
    """
    if board.winner is not None:
        return board.winner == "blue"
    path = find_blue_path_left_right(board)
    if path:
        board.blue_path_left_right = path
        board.winner = "blue"
        if do_score:
            register_win(board, "blue")
        return True
    return False


def reset_for_new_game(board) -> None:
    """
    Reset the board state for a new game while preserving scores.
    
    Clears all pieces, resets winner state, and increments match number.
    Does not reset cumulative scores.
    
    Args:
        board: HexBoard instance
    """
    # Clear board state but keep cumulative scores
    for key in list(board.state.keys()):
        board.state[key] = None
    board.winner = None
    # clear stored paths
    if hasattr(board, "red_path_top_bottom"):
        board.red_path_top_bottom = None
    if hasattr(board, "red_path_left_right"):
        board.red_path_left_right = None
    if hasattr(board, "blue_path_left_right"):
        board.blue_path_left_right = None
    # increment match number if present
    if hasattr(board, "match_number"):
        board.match_number += 1


