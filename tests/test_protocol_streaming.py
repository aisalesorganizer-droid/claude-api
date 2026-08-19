from anthropic_protocol.streaming import parse_sse_lines


def test_tool_stream_is_structured_and_ordered():
    lines = [
        'event: message_start',
        'data: {"type":"message_start","message":{"id":"msg_1"}}',
        '',
        'event: content_block_start',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read","input":{}}}',
        '',
        'event: content_block_delta',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\":\\"CLAUDE.md\\"}"}}',
        '',
        'event: content_block_stop',
        'data: {"type":"content_block_stop","index":0}',
        '',
        'event: message_delta',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
        '',
        'event: message_stop',
        'data: {"type":"message_stop"}',
        '',
    ]
    events = list(parse_sse_lines(lines))
    assert [e.type for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1].block is not None
    assert events[1].block.type == "tool_use"
    assert events[1].block.id == "toolu_1"
    assert events[2].delta_type == "input_json_delta"
    assert events[2].partial_json is not None
    assert events[4].stop_reason == "tool_use"


def test_unknown_event_is_preserved():
    lines = [
        'event: future_event',
        'data: {"type":"future_event","new_field":{"x":1}}',
        '',
    ]
    events = list(parse_sse_lines(lines))
    assert len(events) == 1
    assert events[0].type == "future_event"
    assert events[0].raw["new_field"] == {"x": 1}
