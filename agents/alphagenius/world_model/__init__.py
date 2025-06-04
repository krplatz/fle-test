"""Basic in-memory world model."""

from typing import Any, Dict, Optional

class WorldModel:
    """Simple key-value store for agent knowledge."""

    def __init__(self) -> None:
        self._facts: Dict[str, Any] = {}

    def set_fact(self, key: str, value: Any) -> None:
        """Record a fact in the world model."""
        self._facts[key] = value

    def get_fact(self, key: str) -> Optional[Any]:
        """Retrieve a fact by key."""
        return self._facts.get(key)

__all__ = ["WorldModel"]
