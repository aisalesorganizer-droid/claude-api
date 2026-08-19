# Phase 3.6 — Tool Result → Claude.ai

Implemented:
- Canonical `ToolResult` parsing for Claude Code `tool_result` blocks.
- Multiple tool-result extraction from Anthropic message content.
- Ledger correlation through `tool_use_id` with completed/failed states.
- Model-facing `<tool_result>` compiler for the Claude.ai text backend.
- `/v1/messages` prompt assembly now preserves tool-result content instead of discarding it.

Not implemented yet:
- Full multi-turn agent loop.
- Automatic registration of live tool invocations into a persistent AgentSessionState.
- Pseudo-tool generation/detection integration into the live request lifecycle.
- End-to-end Claude Code local execution test.
