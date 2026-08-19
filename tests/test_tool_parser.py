from agent_bridge.tool_parser import (
    IncrementalToolCallDetector,
    ParsedToolCall,
    ToolCallParseError,
    extract_tool_calls,
)


def test_extract_single_tool_call():
    text = 'Before <tool_call>{"name":"Read","input":{"file_path":"x.txt"}}</tool_call> after'
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "Read"
    assert calls[0].input == {"file_path": "x.txt"}


def test_extract_multiple_tool_calls():
    text = (
        '<tool_call>{"name":"Read","input":{"file_path":"a.txt"}}</tool_call>'
        'middle'
        '<tool_call>{"name":"Grep","input":{"pattern":"TODO"}}</tool_call>'
    )
    calls = extract_tool_calls(text)
    assert [call.name for call in calls] == ["Read", "Grep"]


def test_incremental_detector_buffers_partial_marker():
    detector = IncrementalToolCallDetector()
    calls, plain = detector.feed("Thinking... <tool_")
    assert calls == []
    assert plain == "Thinking... "

    calls, plain = detector.feed(
        'call>{"name":"Read","input":{"file_path":"x.txt"}}</tool_call> done'
    )
    assert [call.name for call in calls] == ["Read"]
    assert plain == " done"
    assert detector.finish() == ""


def test_incremental_detector_supports_text_tool_text():
    detector = IncrementalToolCallDetector()
    calls, plain = detector.feed(
        'First <tool_call>{"name":"Read","input":{}}</tool_call> Second'
    )
    assert [call.name for call in calls] == ["Read"]
    assert plain == "First  Second"


def test_malformed_json_is_rejected():
    try:
        extract_tool_calls('<tool_call>{"name":"Read","input":</tool_call>')
    except ToolCallParseError as exc:
        assert "invalid tool-call JSON" in str(exc)
    else:
        raise AssertionError("expected ToolCallParseError")


def test_non_object_input_is_rejected():
    try:
        extract_tool_calls('<tool_call>{"name":"Read","input":[]}</tool_call>')
    except ToolCallParseError as exc:
        assert "input" in str(exc)
    else:
        raise AssertionError("expected ToolCallParseError")


def test_incomplete_tool_call_is_not_emitted():
    calls = extract_tool_calls('<tool_call>{"name":"Read","input":{}}')
    assert calls == []
