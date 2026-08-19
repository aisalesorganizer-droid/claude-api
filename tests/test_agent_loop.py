import json

from agent_bridge.session import AgentSessionStore
from agent_bridge.tool_registry import ToolRegistry
from agent_bridge.tool_parser import extract_tool_calls
from anthropic_protocol.models import ToolDefinition, ToolInvocation
from anthropic_protocol.response import AnthropicResponseCompiler


def test_session_store_reuses_logical_session():
    store = AgentSessionStore()
    a = store.get("s1")
    b = store.get("s1")
    assert a is b
    assert store.size() == 1


def test_tool_turn_emits_tool_use_stop_reason():
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="Read",
        input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
    ))
    parsed = extract_tool_calls('<tool_call>{"name":"Read","input":{"file_path":"x.txt"}}</tool_call>')[0]
    invocation = registry.build_invocation(parsed, "toolu_test")
    events = list(AnthropicResponseCompiler("m", "msg").compile_tool_turn([invocation], "I will read it."))
    payloads = [json.loads(x.split("data: ",1)[1]) for x in events]
    assert any(p.get("type") == "content_block_start" and p["content_block"]["type"] == "tool_use" for p in payloads)
    delta = [p for p in payloads if p.get("type") == "message_delta"][0]
    assert delta["delta"]["stop_reason"] == "tool_use"
