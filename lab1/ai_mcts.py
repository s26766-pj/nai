from typing import Optional
import math
import random
from game import get_legal_moves, check_red_win, check_blue_win


class MCTSNode:
    def __init__(self, player_to_move: str, move: Optional[str] = None, parent: Optional["MCTSNode"] = None):
        self.player_to_move = player_to_move
        self.move = move
        self.parent = parent
        self.children: list[MCTSNode] = []
        self._unexpanded_moves: Optional[list[str]] = None
        self.visits = 0
        self.wins = 0.0

    def unexpanded_moves(self, board) -> list[str]:
        if self._unexpanded_moves is None:
            self._unexpanded_moves = get_legal_moves(board)
        return self._unexpanded_moves


def _uct_value(child: "MCTSNode", total_visits: int, c: float = 1.41421356) -> float:
    if child.visits == 0:
        return math.inf
    exploitation = child.wins / child.visits
    exploration = c * math.sqrt(math.log(total_visits) / child.visits)
    return exploitation + exploration


def _clear_paths(board) -> None:
    if hasattr(board, "red_path_top_bottom"):
        board.red_path_top_bottom = None
    if hasattr(board, "red_path_left_right"):
        board.red_path_left_right = None
    if hasattr(board, "blue_path_left_right"):
        board.blue_path_left_right = None


def mcts_simulate(board, root_color: str, iterations: int = 500) -> Optional[str]:
    root = MCTSNode(root_color)

    for _ in range(iterations):
        node = root
        path: list[MCTSNode] = []
        applied_moves: list[tuple[str, str]] = []

        # Selection/Expansion
        while True:
            path.append(node)
            moves = node.unexpanded_moves(board)
            if moves:
                mv = random.choice(moves)
                moves.remove(mv)
                color = node.player_to_move
                board.state[mv] = color
                applied_moves.append((mv, color))
                if color == "red":
                    check_red_win(board, do_score=False)
                else:
                    check_blue_win(board, do_score=False)
                next_player = "blue" if color == "red" else "red"
                child = MCTSNode(next_player, move=mv, parent=node)
                node.children.append(child)
                node = child
                break
            else:
                if not node.children:
                    break
                total = sum(ch.visits for ch in node.children) + 1
                node = max(node.children, key=lambda ch: _uct_value(ch, total))
                mv = node.move
                color = "blue" if node.player_to_move == "red" else "red"
                if mv is not None and board.state.get(mv) is None:
                    board.state[mv] = color
                    applied_moves.append((mv, color))
                    if color == "red":
                        check_red_win(board, do_score=False)
                    else:
                        check_blue_win(board, do_score=False)

        # Rollout
        rollout_winner = board.winner
        turn = node.player_to_move
        while rollout_winner is None:
            legal = get_legal_moves(board)
            if not legal:
                break
            mv = random.choice(legal)
            board.state[mv] = turn
            applied_moves.append((mv, turn))
            if turn == "red":
                check_red_win(board, do_score=False)
            else:
                check_blue_win(board, do_score=False)
            rollout_winner = board.winner
            turn = "blue" if turn == "red" else "red"

        # Backpropagate
        reward = 1.0 if rollout_winner == root_color else (0.5 if rollout_winner is None else 0.0)
        for n in path:
            n.visits += 1
            n.wins += reward

        # Undo
        for mv, _c in reversed(applied_moves):
            board.state[mv] = None
        board.winner = None
        _clear_paths(board)

    if not root.children:
        legal = get_legal_moves(board)
        return random.choice(legal) if legal else None
    best = max(root.children, key=lambda ch: ch.visits)
    return best.move


def compute_mcts_move(board, player_color: str, iterations: int = 500) -> Optional[str]:
    return mcts_simulate(board, player_color, iterations)


