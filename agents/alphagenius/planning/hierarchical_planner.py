"""Simple hierarchical planner for AlphaGenius.

This planner builds on the existing ``Planner`` class to recursively
break down tasks. It stops decomposing when a sub-goal appears to
match a skill provided in the :mod:`skill_library`.
"""

from __future__ import annotations

from typing import List

from .planner import Planner
from ..skill_library import __all__ as skill_list

class HierarchicalPlanner:
    """Recursively decompose tasks into executable sub-goals."""

    def __init__(self, planner: Planner, max_depth: int = 2):
        self.planner = planner
        self.max_depth = max_depth
        self.skill_names = set(skill_list)

    async def plan(self, task: str, observation: str, depth: int | None = None) -> List[str]:
        """Return a flat list of sub-goals for ``task``.

        Args:
            task: The high level goal to decompose.
            observation: Current observation/context for the LLM planner.
            depth: Optional recursion depth override.
        """
        if depth is None:
            depth = self.max_depth
        if depth <= 0:
            return [task]

        sub_goals = await self.planner.decompose_task(task, observation)
        final_goals: List[str] = []
        for sg in sub_goals:
            # If sub-goal corresponds to a known skill we stop recursing
            if any(name.replace('_', ' ') in sg.lower() for name in self.skill_names):
                final_goals.append(sg)
            else:
                final_goals.extend(await self.plan(sg, observation, depth - 1))
        return final_goals
