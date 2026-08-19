import pytest

from anthropic_protocol.models import ToolDefinition
from agent_bridge.state import AgentSessionState
from agent_bridge.tool_parser import ParsedToolCall
from agent_bridge.tool_registry import (
    ToolInputValidationError,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
)


def read_tool_definition():
    return ToolDefinition(
        name="Read",
        description="Read a file",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1},
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
        raw={"name": "Read"},
    )


def test_registry_registers_and_validates_valid_input():
    registry = ToolRegistry()
    registry.register(read_tool_definition())

    registry.validate_input("Read", {"file_path": "CLAUDE.md"})


def test_registry_rejects_unknown_tool():
    registry = ToolRegistry()
    registry.register(read_tool_definition())

    with pytest.raises(ToolNotFoundError):
        registry.validate_input("Bash", {"command": "echo hi"})


def test_registry_rejects_invalid_input():
    registry = ToolRegistry()
    registry.register(read_tool_definition())

    with pytest.raises(ToolInputValidationError) as exc:
        registry.validate_input("Read", {"file_path": 123})

    assert "file_path" in str(exc.value)


def test_registry_rejects_extra_properties():
    registry = ToolRegistry()
    registry.register(read_tool_definition())

    with pytest.raises(ToolInputValidationError):
        registry.validate_input("Read", {"file_path": "x", "evil": True})


def test_registry_builds_invocation_after_validation():
    registry = ToolRegistry()
    registry.register(read_tool_definition())
    parsed = ParsedToolCall(
        name="Read",
        input={"file_path": "x.txt"},
        raw="<tool_call>...</tool_call>",
        start=0,
        end=25,
    )

    invocation = registry.build_invocation(parsed, "toolu_test_001")
    assert invocation.id == "toolu_test_001"
    assert invocation.name == "Read"
    assert invocation.status == "pending"


def test_session_ledger_registers_and_resolves_invocation():
    registry = ToolRegistry()
    registry.register(read_tool_definition())
    parsed = ParsedToolCall("Read", {"file_path": "x.txt"}, "raw", 0, 4)
    invocation = registry.build_invocation(parsed, "toolu_test_002")

    state = AgentSessionState()
    pending = state.register_invocation(invocation)
    assert pending.status == "pending"
    assert state.has_pending_tools()

    state.resolve_tool("toolu_test_002", "hello")
    assert state.pending_tools["toolu_test_002"].status == "completed"
    assert state.pending_tools["toolu_test_002"].result == "hello"
    assert not state.has_pending_tools()


def test_session_ledger_rejects_duplicate_tool_id():
    registry = ToolRegistry()
    registry.register(read_tool_definition())
    parsed = ParsedToolCall("Read", {"file_path": "x.txt"}, "raw", 0, 4)
    invocation = registry.build_invocation(parsed, "toolu_duplicate")

    state = AgentSessionState()
    state.register_invocation(invocation)
    with pytest.raises(ValueError):
        state.register_invocation(invocation)


def test_session_ledger_rejects_unknown_result_id():
    state = AgentSessionState()
    with pytest.raises(KeyError):
        state.resolve_tool("toolu_missing", "x")
