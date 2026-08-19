import pytest

from agent_bridge.state import AgentSessionState
from agent_bridge.tool_results import (
    ToolResultError,
    compile_tool_result_for_model,
    compile_tool_results_for_model,
    extract_tool_results,
    parse_tool_result_block,
    resolve_tool_results,
)
from anthropic_protocol.models import ToolInvocation


def test_parse_tool_result_block():
    result = parse_tool_result_block({
        "type": "tool_result",
        "tool_use_id": "toolu_123",
        "content": "hello",
        "is_error": False,
    })
    assert result.tool_use_id == "toolu_123"
    assert result.content == "hello"
    assert result.is_error is False


def test_extract_multiple_tool_results():
    results = extract_tool_results([
        {"type": "text", "text": "ignored"},
        {"type": "tool_result", "tool_use_id": "a", "content": "A"},
        {"type": "tool_result", "tool_use_id": "b", "content": [{"type": "text", "text": "B"}]},
    ])
    assert [r.tool_use_id for r in results] == ["a", "b"]


def test_malformed_tool_result_rejected():
    with pytest.raises(ToolResultError):
        parse_tool_result_block({"type": "tool_result", "content": "x"})


def test_resolve_results_against_ledger():
    state = AgentSessionState()
    state.register_invocation(ToolInvocation("toolu_1", "Read", {"file_path": "x.txt"}))
    resolved = resolve_tool_results(state, [parse_tool_result_block({
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "file contents",
    })])
    assert len(resolved) == 1
    assert resolved[0].call.status == "completed"
    assert resolved[0].call.result == "file contents"
    assert not state.has_pending_tools()


def test_error_result_marks_ledger_failed():
    state = AgentSessionState()
    state.register_invocation(ToolInvocation("toolu_2", "Bash", {"command": "false"}))
    resolved = resolve_tool_results(state, [parse_tool_result_block({
        "type": "tool_result",
        "tool_use_id": "toolu_2",
        "content": "exit 1",
        "is_error": True,
    })])
    assert resolved[0].call.status == "failed"
    assert state.pending_tools["toolu_2"].result == "exit 1"


def test_unknown_tool_result_id_rejected_by_ledger():
    state = AgentSessionState()
    with pytest.raises(KeyError):
        resolve_tool_results(state, [parse_tool_result_block({
            "type": "tool_result",
            "tool_use_id": "missing",
            "content": "x",
        })])


def test_compile_tool_result_for_model():
    result = parse_tool_result_block({
        "type": "tool_result",
        "tool_use_id": "toolu_3",
        "content": "PROTOCOL_PROBE_SUCCESS",
    })
    rendered = compile_tool_result_for_model(result)
    assert rendered.startswith("<tool_result>\n")
    assert '"tool_use_id":"toolu_3"' in rendered
    assert '"content":"PROTOCOL_PROBE_SUCCESS"' in rendered


def test_compile_multiple_tool_results_for_model():
    results = extract_tool_results([
        {"type": "tool_result", "tool_use_id": "a", "content": "A"},
        {"type": "tool_result", "tool_use_id": "b", "content": "B", "is_error": True},
    ])
    rendered = compile_tool_results_for_model(results)
    assert rendered.count("<tool_result>") == 2
    assert '"tool_use_id":"a"' in rendered
    assert '"tool_use_id":"b"' in rendered
