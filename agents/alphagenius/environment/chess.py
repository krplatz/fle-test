from __future__ import annotations

import contextlib
import io
import traceback
from typing import Tuple

import chess

from .base import EnvironmentBase


class ChessEnv(EnvironmentBase):
    """Simple Python-chess environment wrapper."""

    def __init__(self):
        self.board = chess.Board()

    def eval(self, code: str, agent_idx: int = 0) -> Tuple[float, str, str]:
        """Execute ``code`` with a ``board`` variable from python-chess."""
        local_board = self.board.copy()
        env_globals = {"board": local_board, "chess": chess}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(stdout_capture),
                contextlib.redirect_stderr(stderr_capture),
            ):
                exec(code, env_globals)
            stderr_val = stderr_capture.getvalue()
            if stderr_val:
                return -1.0, "execute code", stderr_val
            self.board = local_board
            return 0.0, "execute code", stdout_capture.getvalue()
        except Exception as e:
            traceback.print_exc(file=stderr_capture)
            return -1.0, "execute code", stderr_capture.getvalue()

    def get_system_prompt(self, agent_idx: int = 0) -> str:
        return (
            "You can manipulate the variable 'board' from python-chess. "
            "Useful functions: board.push_san(move), board.is_game_over(), "
            "board.result(), board.board_fen()."
        )

    def reset(self) -> None:
        self.board.reset()
