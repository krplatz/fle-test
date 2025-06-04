from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

class EnvironmentBase(ABC):
    """Abstract environment interface for AlphaGeniusAgent.

    Different games or simulators should implement this interface so that
    AlphaGeniusAgent can interact with them in a uniform way.
    """

    @abstractmethod
    def eval(self, code: str, agent_idx: int = 0) -> Tuple[float, str, str]:
        """Execute a Python program within the environment.

        Args:
            code: Python source code to execute.
            agent_idx: Index of the agent, for multi-agent environments.

        Returns:
            A tuple ``(score, goal_description, result_str)`` representing the
            environment specific score, the goal that was evaluated and the raw
            result output (stdout/stderr combined) from execution.
        """

    @abstractmethod
    def get_system_prompt(self, agent_idx: int = 0) -> str:
        """Return an environment specific system prompt describing available tools."""

    def reset(self) -> None:  # pragma: no cover - optional
        """Reset the environment to its initial state."""
        pass
