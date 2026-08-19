from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .state import AgentSessionState, PendingToolCall
from anthropic_protocol.models import ToolResult


class ToolResultError(ValueError):
    """Raised when a Claude Code tool_result block is malformed."""


@dataclass(slots=True, frozen=True)
class ResolvedToolResult:
    """A validated tool result after ledger correlation."""

    result: ToolResult
    call: PendingToolCall


def parse_tool_result_block(block: dict[str, Any]) -> ToolResult:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        raise ToolResultError("expected a tool_result content block")

    tool_use_id = block.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise ToolResultError("tool_result requires a non-empty tool_use_id")

    if "content" not in block:
        raise ToolResultError("tool_result requires a content field")

    is_error = block.get("is_error", False)
    if not isinstance(is_error, bool):
        raise ToolResultError("tool_result is_error must be boolean")

    return ToolResult(
        tool_use_id=tool_use_id.strip(),
        content=block.get("content"),
        is_error=is_error,
        raw=dict(block),
    )


def extract_tool_results(content: Any) -> list[ToolResult]:
    """Extract tool_result blocks from an Anthropic message content value."""
    if not isinstance(content, list):
        return []

    results: list[ToolResult] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            results.append(parse_tool_result_block(block))
    return results


def resolve_tool_results(
    state: AgentSessionState,
    results: Iterable[ToolResult],
) -> list[ResolvedToolResult]:
    """Correlate each tool_result with exactly one pending tool invocation."""
    resolved: list[ResolvedToolResult] = []
    for result in results:
        call = state.resolve_tool(
            result.tool_use_id,
            result.content,
            is_error=result.is_error,
        )
        resolved.append(ResolvedToolResult(result=result, call=call))
    return resolved


def _render_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            else:
                text_parts.append(json.dumps(block, ensure_ascii=False, separators=(",", ":")))
        return "".join(text_parts)
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def compile_tool_result_for_model(result: ToolResult) -> str:
    """Render one canonical tool result into the Claude.ai text prompt format."""
    payload = {
        "tool_use_id": result.tool_use_id,
        "is_error": result.is_error,
        "content": _render_result_content(result.content),
    }
    return "<tool_result>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</tool_result>"


def compile_tool_results_for_model(results: Iterable[ToolResult]) -> str:
    blocks = [compile_tool_result_for_model(result) for result in results]
    return "\n\n".join(blocks)
