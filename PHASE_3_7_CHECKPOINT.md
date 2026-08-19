PHASE 3.7 CHECKPOINT

Implemented full agent-loop core:
- persistent Claude Code session state keyed by x-claude-code-session-id
- pinned upstream Claude.ai account/conversation per logical agent session
- tool registry refresh from Claude Code requests
- tool_result correlation through invocation ledger
- tool-result continuation prompts
- pseudo-tool parsing -> validated ToolInvocation -> tool_use SSE
- final model turn -> end_turn response

Tests: 34 passed

Note: tool-call turns are buffered until the upstream turn completes so the bridge can safely detect the pseudo-tool grammar before emitting tool_use. Normal text-only turns still use the structured compiler.
