from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Optional

from .state import AgentSessionState
from .tool_registry import ToolRegistry


@dataclass(slots=True)
class AgentBridgeSession:
    session_id: str
    state: AgentSessionState = field(default_factory=AgentSessionState)
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    upstream_state: Any = None
    model: Optional[str] = None
    account_label: Optional[str] = None
    turns: int = 0


class AgentSessionStore:
    """Small in-process store for logical Claude Code agent sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentBridgeSession] = {}
        self._lock = RLock()

    def get(self, session_id: str) -> AgentBridgeSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = AgentBridgeSession(session_id=session_id)
                self._sessions[session_id] = session
            return session

    def pop(self, session_id: str) -> Optional[AgentBridgeSession]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)
