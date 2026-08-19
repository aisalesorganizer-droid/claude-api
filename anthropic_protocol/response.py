from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from .models import StreamEvent, ToolInvocation


@dataclass(slots=True)
class _BlockState:
    index: int
    block_type: str
    started: bool = False
    stopped: bool = False
    has_delta: bool = False


@dataclass(slots=True)
class AnthropicResponseCompiler:
    """Compile canonical upstream StreamEvents into Anthropic SSE frames."""

    model: str
    message_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    _blocks: dict[int, _BlockState] = field(default_factory=dict)
    _message_started: bool = False
    _message_stopped: bool = False
    _last_stop_reason: Optional[str] = None

    def _event(self, event_name: str, payload: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def compile(self, events: Iterable[StreamEvent]) -> Iterator[str]:
        if not self._message_started:
            self._message_started = True
            yield self._event("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": self.output_tokens,
                    },
                },
            })

        for event in events:
            yield from self._compile_event(event)

        # Upstream should normally terminate with message_stop. Keep the
        # compiler safe for a truncated iterator without fabricating a
        # second message_delta.
        if not self._message_stopped:
            if self._last_stop_reason is None:
                self._last_stop_reason = "end_turn"
            yield self._event("message_delta", {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._last_stop_reason,
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": self.output_tokens},
            })
            yield self._event("message_stop", {"type": "message_stop"})
            self._message_stopped = True

    def compile_tool_turn(self, invocations: Iterable[ToolInvocation], prefix_text: str = "") -> Iterator[str]:
        """Compile a complete assistant turn containing optional text and one or more tool_use blocks."""
        invocations = list(invocations)
        if not invocations:
            raise ValueError("at least one tool invocation is required")
        if self._message_started or self._message_stopped:
            raise RuntimeError("tool turn compiler must start a fresh response")

        self._message_started = True
        yield self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
            },
        })
        next_index = 0
        if prefix_text:
            yield self._event("content_block_start", {
                "type": "content_block_start",
                "index": next_index,
                "content_block": {"type": "text", "text": ""},
            })
            yield self._event("ping", {"type": "ping"})
            yield self._event("content_block_delta", {
                "type": "content_block_delta",
                "index": next_index,
                "delta": {"type": "text_delta", "text": prefix_text},
            })
            yield self._event("content_block_stop", {"type": "content_block_stop", "index": next_index})
            next_index += 1
        else:
            yield self._event("ping", {"type": "ping"})

        for invocation in invocations:
            if not invocation.id.strip() or not invocation.name.strip():
                raise ValueError("tool invocation id and name must be non-empty")
            if not isinstance(invocation.input, dict):
                raise TypeError("tool invocation input must be a JSON object")
            yield self._event("content_block_start", {
                "type": "content_block_start",
                "index": next_index,
                "content_block": {
                    "type": "tool_use",
                    "id": invocation.id,
                    "name": invocation.name,
                    "input": {},
                },
            })
            partial_json = json.dumps(invocation.input, ensure_ascii=False, separators=(",", ":"))
            yield self._event("content_block_delta", {
                "type": "content_block_delta",
                "index": next_index,
                "delta": {"type": "input_json_delta", "partial_json": partial_json},
            })
            yield self._event("content_block_stop", {"type": "content_block_stop", "index": next_index})
            next_index += 1

        yield self._event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": self.output_tokens},
        })
        yield self._event("message_stop", {"type": "message_stop"})
        self._message_stopped = True

    def compile_tool_invocation(self, invocation: ToolInvocation) -> Iterator[str]:
        """Compile one validated ToolInvocation into a Claude Code tool-use turn.

        This is intentionally a response-only operation: it does not execute the
        tool or mutate the session ledger. The caller is responsible for having
        already validated and registered the invocation.
        """
        if not invocation.id.strip():
            raise ValueError("tool invocation id must be non-empty")
        if not invocation.name.strip():
            raise ValueError("tool invocation name must be non-empty")
        if not isinstance(invocation.input, dict):
            raise TypeError("tool invocation input must be a JSON object")

        if self._message_started or self._message_stopped:
            raise RuntimeError("tool invocation compiler must start a fresh response")

        self._message_started = True
        yield self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                },
            },
        })

        index = 0
        yield self._event("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": invocation.id,
                "name": invocation.name,
                "input": {},
            },
        })

        partial_json = json.dumps(invocation.input, ensure_ascii=False, separators=(",", ":"))
        yield self._event("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": partial_json,
            },
        })
        yield self._event("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        })
        yield self._event("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use",
                "stop_sequence": None,
            },
            "usage": {
                "output_tokens": self.output_tokens,
            },
        })
        yield self._event("message_stop", {"type": "message_stop"})
        self._message_stopped = True
        self._last_stop_reason = "tool_use"

    def _compile_event(self, event: StreamEvent) -> Iterator[str]:
        etype = event.type

        if etype == "message_start":
            # Preserve the upstream envelope where useful, but normalize the
            # message id/model to the gateway-owned values.
            if not self._message_started:
                self._message_started = True
                raw_message = event.raw.get("message") or {}
                message = dict(raw_message)
                message.update({
                    "id": self.message_id,
                    "type": "message",
                    "role": message.get("role", "assistant"),
                    "model": self.model,
                    "stop_reason": message.get("stop_reason"),
                    "stop_sequence": message.get("stop_sequence"),
                })
                usage = dict(message.get("usage") or {})
                usage.setdefault("input_tokens", self.input_tokens)
                usage.setdefault("output_tokens", self.output_tokens)
                message["usage"] = usage
                yield self._event("message_start", {
                    "type": "message_start",
                    "message": message,
                })
            return

        if etype == "content_block_start":
            index = event.index if event.index is not None else 0
            block = event.block
            raw_block = dict(block.raw) if block is not None else dict(event.raw.get("content_block") or {})
            block_type = str(raw_block.get("type", "unknown"))
            self._blocks[index] = _BlockState(index=index, block_type=block_type, started=True)
            yield self._event("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": raw_block,
            })
            return

        if etype == "content_block_delta":
            index = event.index if event.index is not None else 0
            state = self._blocks.get(index)
            if state is None:
                # Be tolerant of unusual upstream ordering.
                state = _BlockState(index=index, block_type="unknown", started=False)
                self._blocks[index] = state

            delta: dict
            if event.delta_type == "text_delta":
                delta = {"type": "text_delta", "text": event.text or ""}
                self.output_tokens += len((event.text or "").split())
            elif event.delta_type == "thinking_delta":
                delta = {"type": "thinking_delta", "thinking": event.thinking or ""}
            elif event.delta_type == "input_json_delta":
                delta = {"type": "input_json_delta", "partial_json": event.partial_json or ""}
            else:
                raw_delta = event.raw.get("delta")
                delta = dict(raw_delta) if isinstance(raw_delta, dict) else {"type": event.delta_type or "unknown"}

            state.has_delta = True
            yield self._event("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            })
            return

        if etype == "content_block_stop":
            index = event.index if event.index is not None else 0
            state = self._blocks.setdefault(index, _BlockState(index=index, block_type="unknown"))
            state.stopped = True
            yield self._event("content_block_stop", {
                "type": "content_block_stop",
                "index": index,
            })
            return

        if etype == "message_delta":
            stop_reason = event.stop_reason
            self._last_stop_reason = stop_reason or self._last_stop_reason or "end_turn"
            raw_delta = event.raw.get("delta") or {}
            delta = dict(raw_delta)
            delta["stop_reason"] = self._last_stop_reason
            delta.setdefault("stop_sequence", None)
            usage = dict(event.raw.get("usage") or {})
            usage.setdefault("output_tokens", self.output_tokens)
            yield self._event("message_delta", {
                "type": "message_delta",
                "delta": delta,
                "usage": usage,
            })
            return

        if etype == "ping":
            yield self._event("ping", {"type": "ping"})
            return

        if etype == "error":
            error_payload = event.raw.get("error") or {}
            yield self._event("error", {
                "type": "error",
                "error": {
                    "type": event.error_type or error_payload.get("type", "api_error"),
                    "message": event.error_message or error_payload.get("message", "Upstream error"),
                },
            })
            return

        if etype == "message_stop":
            if not self._message_stopped:
                self._message_stopped = True
                yield self._event("message_stop", {"type": "message_stop"})
            return

        # Forward unknown events unchanged. This keeps the bridge tolerant of
        # future upstream protocol additions.
        yield self._event(etype, event.raw)
