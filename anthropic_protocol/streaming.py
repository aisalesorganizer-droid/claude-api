from __future__ import annotations

import json
from typing import Any, Iterable, Iterator, Optional

from .models import ContentBlock, StreamEvent


def parse_sse_event(event_name: Optional[str], data: str) -> Optional[StreamEvent]:
    """Parse one SSE data payload without discarding unknown fields."""
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    etype = payload.get("type") or event_name or "unknown"

    if etype == "content_block_start":
        block = payload.get("content_block") or {}
        canonical = ContentBlock(
            type=str(block.get("type", "unknown")),
            index=payload.get("index"),
            id=block.get("id"),
            name=block.get("name"),
            input=block.get("input"),
            text=block.get("text"),
            thinking=block.get("thinking"),
            tool_use_id=block.get("tool_use_id"),
            content=block.get("content"),
            raw=block,
        )
        return StreamEvent(
            type=etype,
            raw=payload,
            index=payload.get("index"),
            block=canonical,
        )

    if etype == "content_block_delta":
        delta = payload.get("delta") or {}
        dtype = delta.get("type")
        return StreamEvent(
            type=etype,
            raw=payload,
            index=payload.get("index"),
            delta_type=dtype,
            text=delta.get("text") if dtype == "text_delta" else None,
            thinking=delta.get("thinking") if dtype == "thinking_delta" else None,
            partial_json=(delta.get("partial_json") if dtype == "input_json_delta" else None),
        )

    if etype == "message_delta":
        delta = payload.get("delta") or {}
        return StreamEvent(
            type=etype,
            raw=payload,
            stop_reason=delta.get("stop_reason"),
        )

    if etype == "error":
        error = payload.get("error") or {}
        return StreamEvent(
            type=etype,
            raw=payload,
            error_type=error.get("type"),
            error_message=error.get("message"),
        )

    return StreamEvent(type=etype, raw=payload)


def parse_sse_lines(lines: Iterable[str | bytes]) -> Iterator[StreamEvent]:
    """Parse an SSE line stream, preserving event ordering and unknown events."""
    event_name: Optional[str] = None
    data_lines: list[str] = []

    def flush() -> Optional[StreamEvent]:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        event = parse_sse_event(event_name, "\n".join(data_lines).strip())
        event_name = None
        data_lines = []
        return event

    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if line.endswith("\r"):
            line = line[:-1]
        if line == "":
            event = flush()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].lstrip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        # Ignore id/retry for now but keep processing the event.

    event = flush()
    if event is not None:
        yield event
