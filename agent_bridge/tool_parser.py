from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass(slots=True, frozen=True)
class ParsedToolCall:
    """A syntactically valid model-emitted tool call, not yet schema-validated."""

    name: str
    input: dict[str, Any]
    raw: str
    start: int
    end: int


class ToolCallParseError(ValueError):
    """Raised when a complete tool_call block is present but malformed."""


def _decode_payload(payload: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"invalid tool-call JSON: {exc.msg}") from exc

    if not isinstance(value, dict):
        raise ToolCallParseError("tool-call payload must be a JSON object")

    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolCallParseError("tool-call payload requires a non-empty string 'name'")

    input_data = value.get("input", {})
    if not isinstance(input_data, dict):
        raise ToolCallParseError("tool-call payload 'input' must be a JSON object")

    # Normalize the name while keeping the payload otherwise untouched.
    return {"name": name.strip(), "input": input_data}


def extract_tool_calls(text: str) -> list[ParsedToolCall]:
    """Extract all complete <tool_call>{...}</tool_call> blocks from text.

    This stage is deliberately syntactic. It does not check whether the named
    tool exists or whether the input matches a JSON schema; that belongs to the
    Phase 3.4 validation layer.
    """
    if not text:
        return []

    calls: list[ParsedToolCall] = []
    cursor = 0
    while True:
        start = text.find(TOOL_CALL_OPEN, cursor)
        if start < 0:
            break
        payload_start = start + len(TOOL_CALL_OPEN)
        end = text.find(TOOL_CALL_CLOSE, payload_start)
        if end < 0:
            # Incomplete trailing call: leave it for the incremental detector.
            break

        raw = text[start : end + len(TOOL_CALL_CLOSE)]
        payload = text[payload_start:end].strip()
        decoded = _decode_payload(payload)
        calls.append(
            ParsedToolCall(
                name=decoded["name"],
                input=decoded["input"],
                raw=raw,
                start=start,
                end=end + len(TOOL_CALL_CLOSE),
            )
        )
        cursor = end + len(TOOL_CALL_CLOSE)

    return calls


@dataclass(slots=True)
class IncrementalToolCallDetector:
    """Incrementally detect complete tool-call blocks from text deltas.

    Text outside a complete tool call is retained as ordinary assistant text.
    A partial trailing block remains buffered until more text arrives.
    """

    buffer: str = ""
    scan_offset: int = 0

    def feed(self, chunk: str) -> tuple[list[ParsedToolCall], str]:
        if chunk:
            self.buffer += chunk

        calls: list[ParsedToolCall] = []
        plain_parts: list[str] = []

        while True:
            start = self.buffer.find(TOOL_CALL_OPEN)
            if start < 0:
                # Keep only a suffix that could be the beginning of the marker.
                keep = self._marker_prefix_suffix(self.buffer)
                plain_parts.append(self.buffer[:-keep] if keep else self.buffer)
                self.buffer = self.buffer[-keep:] if keep else ""
                break

            # Everything before a candidate marker is ordinary text.
            if start:
                plain_parts.append(self.buffer[:start])

            payload_start = start + len(TOOL_CALL_OPEN)
            end = self.buffer.find(TOOL_CALL_CLOSE, payload_start)
            if end < 0:
                # Preserve the incomplete candidate for the next feed().
                self.buffer = self.buffer[start:]
                break

            raw = self.buffer[start : end + len(TOOL_CALL_CLOSE)]
            payload = self.buffer[payload_start:end].strip()
            decoded = _decode_payload(payload)
            calls.append(
                ParsedToolCall(
                    name=decoded["name"],
                    input=decoded["input"],
                    raw=raw,
                    start=start,
                    end=end + len(TOOL_CALL_CLOSE),
                )
            )
            self.buffer = self.buffer[end + len(TOOL_CALL_CLOSE) :]

        return calls, "".join(plain_parts)

    def finish(self) -> str:
        """Flush remaining buffered text as ordinary text.

        A complete-looking opening marker without a closing tag is not silently
        converted into a tool call. It remains text at this phase.
        """
        remaining = self.buffer
        self.buffer = ""
        return remaining

    @staticmethod
    def _marker_prefix_suffix(text: str) -> int:
        max_len = min(len(text), len(TOOL_CALL_OPEN) - 1)
        for length in range(max_len, 0, -1):
            if text.endswith(TOOL_CALL_OPEN[:length]):
                return length
        return 0
