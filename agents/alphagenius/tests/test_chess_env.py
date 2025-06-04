import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
from agents.alphagenius.environment.chess import ChessEnv


def test_chess_env_eval_move():
    env = ChessEnv()
    score, goal, result = env.eval("board.push_san('e4')")
    assert score == 0.0
    assert env.board.move_stack[-1].uci() == "e2e4"
    assert goal == "execute code"


def test_chess_env_eval_error():
    env = ChessEnv()
    score, _, result = env.eval("board.push_san('invalid')")
    assert score == -1.0
    assert "Traceback" in result or "ValueError" in result
