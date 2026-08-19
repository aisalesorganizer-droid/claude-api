from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from anthropic_protocol.models import ToolInvocation


@dataclass(slots=True)
class PendingToolCall:
    id: str
    name: str
    input: dict[str, Any]
    status: str = "pending"
    result: Any = None

    def to_invocation(self) -> ToolInvocation:
        return ToolInvocation(
            id=self.id,
            name=self.name,
            input=self.input,
            status=self.status,
            result=self.result,
        )


@dataclass(slots=True)
class AgentSessionState:
    """Logical agent-turn state kept separate from Claude.ai account state."""

    account_id: Optional[str] = None
    conv_id: Optional[str] = None
    parent_message_uuid: Optional[str] = None
    canonical_history: list[dict[str, Any]] = field(default_factory=list)
    pending_tools: dict[str, PendingToolCall] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)
    turn_active: bool = False

    def register_tool(self, tool_id: str, name: str, input_data: dict[str, Any]) -> PendingToolCall:
        if tool_id in self.pending_tools:
            raise ValueError(f"duplicate tool_use_id: {tool_id}")
        call = PendingToolCall(id=tool_id, name=name, input=input_data)
        self.pending_tools[tool_id] = call
        return call

    def register_invocation(self, invocation: ToolInvocation) -> PendingToolCall:
        return self.register_tool(invocation.id, invocation.name, invocation.input)

    def resolve_tool(self, tool_id: str, result: Any, is_error: bool = False) -> PendingToolCall:
        call = self.pending_tools.get(tool_id)
        if call is None:
            raise KeyError(f"unknown tool_use_id: {tool_id}")
        call.result = result
        call.status = "failed" if is_error else "completed"
        return call

    def has_pending_tools(self) -> bool:
        return any(call.status == "pending" for call in self.pending_tools.values())
