# Game logic helpers that operate on a HexBoard instance

from typing import Optional, Dict, List
import math
import random

def get_legal_moves(board) -> List[str]:
    return [label for label, v in board.state.items() if v is None]

def get_clicked_hex(board, pos) -> Optional[str]:
    mx, my = pos
    for r in range(board.rows):
        for c in range(board.cols):
            x, y = board.hex_to_pixel(r, c)
            dx, dy = mx - x, my - y
            if dx * dx + dy * dy < (board.hex_size * 0.9) ** 2:
                return f"{chr(ord('A') + c)}{r+1}"
    return None


def get_scores(board) -> Dict[str, int]:
    return {"red": board._scores["red"], "blue": board._scores["blue"]}


def register_win(board, winner: str) -> None:
    if winner in ("red", "blue"):
        board._scores[winner] += 1


def is_full(board) -> bool:
    for v in board.state.values():
        if v is None:
            return False
    return True


def red_connects_row_1_to_11(board) -> bool:
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


