# Phase 3.8 — Integration Test Checkpoint

## Status
COMPLETE

## Verified
- FastAPI `/v1/messages` agent path can be exercised without live credentials.
- Claude Code tool definitions register into the session registry.
- Model-emitted `<tool_call>` is parsed and schema-validated.
- A validated invocation becomes a real Anthropic `tool_use` SSE turn.
- The Claude Code `tool_result` continuation resolves against the same ledger.
- The model-facing continuation preserves both the preceding assistant `tool_use` and the subsequent `tool_result`.
- The same logical Claude Code session reuses the same upstream state across the tool turn.
- The final continuation emits `stop_reason=end_turn` and removes the completed agent session.
- Existing protocol, parser, registry, compiler, ledger, and result tests remain green.

## Regression result
35 passed

## Important boundary
This phase is an automated integration simulation of the complete protocol loop. It does not make live Railway/Claude.ai calls and does not require credentials.

The next live operation is a controlled Railway + Claude Code integration test using the Phase 3.8 build.
