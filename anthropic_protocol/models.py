from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class ContentBlock:
    """Canonical content block; raw preserves forward-compatible fields."""

    type: str
    index: Optional[int] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Any = None
    text: Optional[str] = None
    thinking: Optional[str] = None
    tool_use_id: Optional[str] = None
    content: Any = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamEvent:
    """One structured event from an upstream Anthropic-compatible SSE stream."""

    type: str
    raw: dict[str, Any]
    index: Optional[int] = None
    block: Optional[ContentBlock] = None
    delta_type: Optional[str] = None
    text: Optional[str] = None
    thinking: Optional[str] = None
    partial_json: Optional[str] = None
    stop_reason: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolInvocation:
    id: str
    name: str
    input: dict[str, Any]
    status: str = "pending"
    result: Any = None


@dataclass(slots=True)
class ToolResult:
    tool_use_id: str
    content: Any
    is_error: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
