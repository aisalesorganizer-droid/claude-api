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

    return {"name": name.strip(), "input": input_data}


def _decode_json_block(payload: str) -> dict[str, Any]:
    """Decode standalone JSON blocks: {"action": "...", "parameters": {...}}."""
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"invalid JSON block: {exc.msg}") from exc

    if not isinstance(value, dict):
        raise ToolCallParseError("JSON block must be a JSON object")

    name = value.get("action") or value.get("name") or value.get("tool")
    if not isinstance(name, str) or not name.strip():
        raise ToolCallParseError("JSON block requires a non-empty string 'action', 'name', or 'tool'")  # noqa: E501

    input_data = value.get("parameters") or value.get("input") or value.get("args", {})
    if not isinstance(input_data, dict):
        raise ToolCallParseError("JSON block 'parameters'/'input' must be a JSON object")

    return {"name": name.strip(), "input": input_data}


def _find_json_objects(text: str) -> list[tuple[str, int, int]]:
    """Find all top-level JSON objects in text. Returns list of (json_str, start, end)."""
    results: list[tuple[str, int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                results.append((text[start:i+1], start, i+1))
                start = -1

    return results


def extract_tool_calls(text: str) -> list[ParsedToolCall]:
    """Extract all complete tool calls from text.

    Supports:
      1. <tool_call>{...}</tool_call> blocks (legacy format)
      2. Standalone JSON objects with "action"/"parameters" keys (new format)
    """
    if not text:
        return []

    calls: list[ParsedToolCall] = []

    # Phase 1: Extract <tool_call> blocks (original logic, raises on malformed)
    cursor = 0
    while True:
        start = text.find(TOOL_CALL_OPEN, cursor)
        if start < 0:
            break
        payload_start = start + len(TOOL_CALL_OPEN)
        end = text.find(TOOL_CALL_CLOSE, payload_start)
        if end < 0:
            # Incomplete trailing call: stop here
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

    # Phase 2: Extract standalone JSON blocks from text NOT covered by <tool_call>
    covered_ranges = [(c.start, c.end) for c in calls]
    incomplete_start = text.find(TOOL_CALL_OPEN, cursor)
    if incomplete_start >= 0:
        covered_ranges.append((incomplete_start, len(text)))

    for json_str, j_start, j_end in _find_json_objects(text):
        if any(j_start >= r[0] and j_end <= r[1] for r in covered_ranges):
            continue
        try:
            decoded = _decode_json_block(json_str)
        except ToolCallParseError:
            continue
        calls.append(
            ParsedToolCall(
                name=decoded["name"],
                input=decoded["input"],
                raw=json_str,
                start=j_start,
                end=j_end,
            )
        )

    calls.sort(key=lambda c: c.start)
    return calls


@dataclass(slots=True)
class IncrementalToolCallDetector:
    """Incrementally detect complete tool-call blocks from text deltas."""

    buffer: str = ""
    scan_offset: int = 0

    def feed(self, chunk: str) -> tuple[list[ParsedToolCall], str]:
        if chunk:
            self.buffer += chunk

        calls: list[ParsedToolCall] = []
        plain_parts: list[str] = []

        # Phase 1: <tool_call> blocks (original logic)
        while True:
            start = self.buffer.find(TOOL_CALL_OPEN)
            if start < 0:
                keep = self._marker_prefix_suffix(self.buffer)
                plain_parts.append(self.buffer[:-keep] if keep else self.buffer)
                self.buffer = self.buffer[-keep:] if keep else ""
                break

            if start:
                plain_parts.append(self.buffer[:start])

            payload_start = start + len(TOOL_CALL_OPEN)
            end = self.buffer.find(TOOL_CALL_CLOSE, payload_start)
            if end < 0:
                self.buffer = self.buffer[start:]
                break

            raw = self.buffer[start : end + len(TOOL_CALL_CLOSE)]
            payload = self.buffer[payload_start:end].strip()
            try:
                decoded = _decode_payload(payload)
            except ToolCallParseError:
                plain_parts.append(self.buffer[:start + len(TOOL_CALL_OPEN)])
                self.buffer = self.buffer[start + len(TOOL_CALL_OPEN):]
                continue

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

        # Phase 2: JSON blocks ONLY if no incomplete <tool_call> in buffer
        if not self.buffer:
            combined_plain = "".join(plain_parts)
            json_calls, remaining = self._extract_json_blocks(combined_plain)
            if json_calls:
                calls.extend(json_calls)
                plain_parts = [remaining] if remaining else []

        return calls, "".join(plain_parts)

    def finish(self) -> str:
        remaining = self.buffer
        self.buffer = ""
        return remaining

    def _extract_json_blocks(self, text: str) -> tuple[list[ParsedToolCall], str]:
        if not text:
            return [], ""

        calls: list[ParsedToolCall] = []
        last_end = 0

        for json_str, j_start, j_end in _find_json_objects(text):
            try:
                decoded = _decode_json_block(json_str)
            except ToolCallParseError:
                last_end = j_end
                continue

            calls.append(
                ParsedToolCall(
                    name=decoded["name"],
                    input=decoded["input"],
                    raw=json_str,
                    start=j_start,
                    end=j_end,
                )
            )
            last_end = j_end

        return calls, text[last_end:]

    @staticmethod
    def _marker_prefix_suffix(text: str) -> int:
        max_len = min(len(text), len(TOOL_CALL_OPEN) - 1)
        for length in range(max_len, 0, -1):
            if text.endswith(TOOL_CALL_OPEN[:length]):
                return length
        return 0