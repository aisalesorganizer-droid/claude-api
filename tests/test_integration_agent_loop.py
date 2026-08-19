import json
from types import SimpleNamespace


import main
from anthropic_protocol.models import StreamEvent


def _read_tool() -> dict:
    return {
        "name": "Read",
        "description": "Read a local file.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
            "additionalProperties": False,
        },
    }


class FakeRequest:
    def __init__(self, body, session_id="integration-session"):
        self._body = body
        self.headers = {"x-claude-code-session-id": session_id}
        self.client = SimpleNamespace(host="127.0.0.1")

    async def json(self):
        return self._body


async def _decode_sse(streaming_response):
    chunks = [chunk async for chunk in streaming_response.body_iterator]
    text = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks).decode()
    records = []
    for frame in [f for f in text.split("\n\n") if f.strip()]:
        event = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event:
            records.append((event, data))
    return records


def _configure(monkeypatch):
    monkeypatch.setattr(main, "_guard", lambda: None)
    monkeypatch.setattr(main, "_next_healthy_account", lambda: SimpleNamespace(label="account_test"))
    monkeypatch.setattr(main, "_fresh_slot_state", lambda label: {"label": label})
    monkeypatch.setattr(main, "_account_objects", [SimpleNamespace(label="account_test")])
    main.agent_session_store.clear()


def test_full_two_request_read_loop(monkeypatch):
    _configure(monkeypatch)
    upstream_prompts = []
    upstream_states = []
    call_count = {"n": 0}

    def fake_stream(state, prompt, model, attachments):
        upstream_states.append(state)
        upstream_prompts.append(prompt)
        call_count["n"] += 1
        if call_count["n"] == 1:
            text = 'I will inspect the file. <tool_call>{"name":"Read","input":{"file_path":"probe.txt"}}</tool_call>'
            return iter([
                StreamEvent(type="message_start", raw={"type": "message_start", "message": {"usage": {}}}),
                StreamEvent(type="content_block_start", raw={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}, index=0),
                StreamEvent(type="content_block_delta", raw={"type": "content_block_delta"}, index=0, delta_type="text_delta", text=text),
                StreamEvent(type="content_block_stop", raw={"type": "content_block_stop", "index": 0}, index=0),
                StreamEvent(type="message_delta", raw={"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, stop_reason="end_turn"),
                StreamEvent(type="message_stop", raw={"type": "message_stop"}),
            ])
        return iter([
            StreamEvent(type="message_start", raw={"type": "message_start", "message": {"usage": {}}}),
            StreamEvent(type="content_block_start", raw={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}, index=0),
            StreamEvent(type="content_block_delta", raw={"type": "content_block_delta"}, index=0, delta_type="text_delta", text="Done. I read the file successfully."),
            StreamEvent(type="content_block_stop", raw={"type": "content_block_stop", "index": 0}, index=0),
            StreamEvent(type="message_delta", raw={"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, stop_reason="end_turn"),
            StreamEvent(type="message_stop", raw={"type": "message_stop"}),
        ])

    monkeypatch.setattr(main, "_stream_events_on_slot", fake_stream)

    first = {
        "model": "claude-opus-5",
        "stream": True,
        "tools": [_read_tool()],
        "messages": [{"role": "user", "content": "Read probe.txt and tell me what you find."}],
    }
    first_response = __import__("asyncio").run(main.ant_messages(FakeRequest(first), None))
    first_events = __import__("asyncio").run(_decode_sse(first_response))

    starts = [d for _, d in first_events if d.get("type") == "content_block_start"]
    tool_starts = [d for d in starts if d["content_block"].get("type") == "tool_use"]
    assert tool_starts, starts
    tool_id = tool_starts[0]["content_block"]["id"]
    assert any(d.get("delta", {}).get("stop_reason") == "tool_use" for _, d in first_events)

    second = {
        "model": "claude-opus-5",
        "stream": True,
        "tools": [_read_tool()],
        "messages": [
            {"role": "user", "content": "Read probe.txt and tell me what you find."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": "Read", "input": {"file_path": "probe.txt"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "PROTOCOL_PROBE_SUCCESS", "is_error": False}]},
        ],
    }
    second_response = __import__("asyncio").run(main.ant_messages(FakeRequest(second), None))
    second_events = __import__("asyncio").run(_decode_sse(second_response))

    assert any("<tool_use>" in upstream_prompts[1] for _ in [0])
    assert tool_id in upstream_prompts[1]
    assert '"name":"Read"' in upstream_prompts[1]
    assert "<tool_result>" in upstream_prompts[1]
    assert "PROTOCOL_PROBE_SUCCESS" in upstream_prompts[1]
    assert upstream_states[0] is upstream_states[1]
    assert any(d.get("delta", {}).get("stop_reason") == "end_turn" for _, d in second_events)
    assert main.agent_session_store.size() == 0
