# Phase 3.8 — Integration-Tested Agent Bridge

This checkpoint connects the Phase 3 protocol components and verifies a complete two-request local-tool loop.

Flow:

Claude Code request
→ tool registry
→ Claude.ai text response containing pseudo-tool call
→ parser
→ schema validation
→ invocation ledger
→ Anthropic `tool_use` SSE
→ Claude Code local execution
→ `tool_result`
→ preserved tool-use context + tool result
→ Claude.ai continuation
→ final `end_turn`

The integration suite uses credential-free stubs. No live account/session material is included.
