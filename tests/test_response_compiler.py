import pytest
from anthropic_protocol.models import StreamEvent
from anthropic_protocol.response import AnthropicResponseCompiler


def _evt(type_, raw, **kwargs):
    return StreamEvent(type=type_, raw=raw, **kwargs)


def test_text_response_compiles_to_anthropic_sse():
    events = [
        _evt("message_start", {
            "type": "message_start",
            "message": {"id": "upstream", "type": "message", "role": "assistant", "content": []},
        }),
        _evt("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }, index=0),
        _evt("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }, index=0, delta_type="text_delta", text="Hello"),
        _evt("content_block_stop", {"type": "content_block_stop", "index": 0}, index=0),
        _evt("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        }, stop_reason="end_turn"),
        _evt("message_stop", {"type": "message_stop"}),
    ]
    frames = list(AnthropicResponseCompiler("claude-sonnet", "msg_test").compile(events))
    names = [line[7:] for frame in frames for line in frame.splitlines() if line.startswith("event: ")]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert '"id": "msg_test"' in frames[0]
    assert '"stop_reason": "end_turn"' in frames[-2]


def test_tool_use_stream_stays_structured():
    events = [
        _evt("message_start", {"type": "message_start", "message": {"content": []}}),
        _evt("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
        }, index=0),
        _evt("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"file_path":"x.txt"}'},
        }, index=0, delta_type="input_json_delta", partial_json='{"file_path":"x.txt"}'),
        _evt("content_block_stop", {"type": "content_block_stop", "index": 0}, index=0),
        _evt("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}, stop_reason="tool_use"),
        _evt("message_stop", {"type": "message_stop"}),
    ]
    frames = list(AnthropicResponseCompiler("claude-sonnet", "msg_tool").compile(events))
    joined = "".join(frames)
    assert '"type": "tool_use"' in joined
    assert '"name": "Read"' in joined
    assert '"type": "input_json_delta"' in joined
    assert '"stop_reason": "tool_use"' in joined


def test_unknown_event_is_forwarded():
    events = [_evt("future_event", {"type": "future_event", "new_field": {"x": 1}})]
    frames = list(AnthropicResponseCompiler("claude-sonnet", "msg_future").compile(events))
    assert any('event: future_event' in frame for frame in frames)
    assert any('"new_field": {"x": 1}' in frame for frame in frames)


def test_validated_tool_invocation_compiles_to_tool_use_turn():
    from anthropic_protocol.models import ToolInvocation

    invocation = ToolInvocation(
        id="toolu_3_5",
        name="Read",
        input={"file_path": "protocol_probe.txt", "offset": 1},
    )
    frames = list(AnthropicResponseCompiler("claude-sonnet", "msg_tool_35").compile_tool_invocation(invocation))
    joined = "".join(frames)

    assert [
        line[7:] for frame in frames for line in frame.splitlines() if line.startswith("event: ")
    ] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert '"type": "tool_use"' in joined
    assert '"id": "toolu_3_5"' in joined
    assert '"name": "Read"' in joined
    assert '"type": "input_json_delta"' in joined
    assert 'file_path' in joined and 'protocol_probe.txt' in joined
    assert '"stop_reason": "tool_use"' in joined


def test_tool_invocation_requires_fresh_compiler():
    from anthropic_protocol.models import ToolInvocation

    compiler = AnthropicResponseCompiler("claude-sonnet", "msg_tool_35_guard")
    list(compiler.compile([_evt("message_start", {"type": "message_start", "message": {}})]))

    with pytest.raises(RuntimeError):
        list(compiler.compile_tool_invocation(ToolInvocation("toolu_2", "Read", {"file_path": "x"})))
