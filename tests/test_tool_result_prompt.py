from main import AntRequest, _ant_build_prompt


def test_tool_result_survives_prompt_compilation():
    req = AntRequest(
        model="claude-sonnet-4-6 High",
        system="system",
        messages=[
            {"role": "user", "content": "Read the file."},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"Read\",\"input\":{\"file_path\":\"x.txt\"}}</tool_call>"},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_123",
                "content": "HELLO FROM LOCAL FILE",
                "is_error": False,
            }]},
        ],
    )
    prompt = _ant_build_prompt(req)
    assert "<tool_result>" in prompt
    assert "toolu_123" in prompt
    assert "HELLO FROM LOCAL FILE" in prompt


def test_normal_text_still_compiles():
    req = AntRequest(
        model="claude-sonnet-4-6 High",
        messages=[{"role": "user", "content": "Hello"}],
    )
    prompt = _ant_build_prompt(req)
    assert "Human: Hello" in prompt
    assert prompt.endswith("\n\nAssistant:")
