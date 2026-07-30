"""In-memory loop state store shared by the FastAPI app (v1)."""

from __future__ import annotations

from threading import Lock
from typing import Any

from agents.state import LoopState


class LoopStore:
    """Thread-safe snapshot of user loop states + latest uplift metrics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.users: list[LoopState] = []
        self.uplift: list[dict[str, Any]] = []
        self.last_step: str | None = None

    def set_users(self, users: list[LoopState], step: str) -> None:
        with self._lock:
            self.users = list(users)
            self.last_step = step

    def set_uplift(self, uplift: list[dict[str, Any]]) -> None:
        with self._lock:
            self.uplift = list(uplift)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "last_step": self.last_step,
                "n_users": len(self.users),
                "users": list(self.users),
                "uplift": list(self.uplift),
            }


store = LoopStore()
