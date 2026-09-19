"""解析器测试：思考过程 / 原生工具调用 / 文本协议回退 / 最终答案。

标签字面量统一由 parser 模块常量拼接生成，保持测试可读且不依赖魔法字符串。
"""
import pytest

from mini_agent.exceptions import LLMResponseError
from mini_agent.parser import (RAW_ARGS_KEY, THINK_CLOSE, THINK_OPEN,
                               TOOL_CLOSE, TOOL_OPEN, parse_assistant_message)


def wrap_think(text):
    return THINK_OPEN + text + THINK_CLOSE


def wrap_tool(json_text):
    return TOOL_OPEN + json_text + TOOL_CLOSE


# ======================================================================
# 思考过程
# ======================================================================
class TestThinking:
    def test_reasoning_content_field(self):
        parsed = parse_assistant_message(
            {"role": "assistant", "content": "答案是 4",
             "reasoning_content": "先算 2+2"})
        assert parsed.thinking == "先算 2+2"
        assert parsed.final_answer == "答案是 4"
        assert not parsed.need_tool_call

    def test_think_block_in_content(self):
        parsed = parse_assistant_message(
            {"role": "assistant", "content": wrap_think("思考中") + "最终答案"})
        assert parsed.thinking == "思考中"
        assert parsed.final_answer == "最终答案"

    def test_unclosed_think_block(self):
        parsed = parse_assistant_message(
            {"role": "assistant", "content": "前缀" + THINK_OPEN + "想到一半"})
        assert parsed.thinking == "想到一半"
        assert "想到一半" not in parsed.content

    def test_both_sources_merged(self):
        parsed = parse_assistant_message({
            "role": "assistant",
            "content": wrap_think("块内思考") + "答案",
            "reasoning_content": "字段思考",
        })
        assert "字段思考" in parsed.thinking and "块内思考" in parsed.thinking

    def test_no_thinking(self):
        parsed = parse_assistant_message({"role": "assistant", "content": "直接回答"})
        assert parsed.thinking is None


# ======================================================================
# 原生 function calling
# ======================================================================
class TestNativeToolCalls:
    def test_single_call(self):
        parsed = parse_assistant_message({
            "role": "assistant", "content": "我来算一下",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "calculator",
                             "arguments": "{\"expression\": \"1+2\"}"},
            }],
        })
        assert parsed.need_tool_call
        assert parsed.content == "我来算一下"
        tc = parsed.tool_calls[0]
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "1+2"}
        assert tc.source == "native"

    def test_parallel_calls(self):
        parsed = parse_assistant_message({
            "role": "assistant", "content": None,
            "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "calculator", "arguments": "{\"expression\": \"1\"}"}},
                {"id": "b", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}},
            ],
        })
        assert len(parsed.tool_calls) == 2
        assert [t.name for t in parsed.tool_calls] == ["calculator", "get_weather"]

    def test_invalid_arguments_json(self):
        parsed = parse_assistant_message({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "x", "type": "function",
                "function": {"name": "calculator", "arguments": "{bad json"},
            }],
        })
        assert RAW_ARGS_KEY in parsed.tool_calls[0].arguments

    def test_missing_id_synthesized(self):
        parsed = parse_assistant_message({
            "role": "assistant", "content": "",
            "tool_calls": [{"type": "function",
                            "function": {"name": "search", "arguments": "{\"query\": \"x\"}"}}],
        })
        assert parsed.tool_calls[0].id


# ======================================================================
# 文本协议回退
# ======================================================================
class TestTextProtocol:
    def test_tag_protocol(self):
        content = wrap_think("需要查天气") + wrap_tool(
            '{"name": "get_weather", "arguments": {"city": "北京"}}') + "以上。"
        parsed = parse_assistant_message({"role": "assistant", "content": content})
        assert parsed.need_tool_call
        assert parsed.thinking == "需要查天气"
        assert parsed.tool_calls[0].name == "get_weather"
        assert parsed.tool_calls[0].arguments == {"city": "北京"}
        assert parsed.tool_calls[0].source == "text"
        assert "get_weather" not in parsed.content  # 调用块已从正文剥离

    def test_fenced_protocol(self):
        content = "```tool_call\n{\"tool\": \"search\", \"args\": {\"query\": \"agent\"}}\n```"
        parsed = parse_assistant_message({"role": "assistant", "content": content})
        assert parsed.need_tool_call
        assert parsed.tool_calls[0].name == "search"
        assert parsed.tool_calls[0].arguments == {"query": "agent"}

    def test_bare_json_protocol(self):
        content = '{"name": "calculator", "arguments": {"expression": "1+1"}}'
        parsed = parse_assistant_message({"role": "assistant", "content": content})
        assert parsed.need_tool_call
        assert parsed.tool_calls[0].name == "calculator"

    def test_multiple_text_calls(self):
        content = (wrap_tool('{"name": "calculator", "arguments": {"expression": "1"}}')
                   + wrap_tool('{"name": "search", "arguments": {"query": "x"}}'))
        parsed = parse_assistant_message({"role": "assistant", "content": content})
        assert len(parsed.tool_calls) == 2

    def test_native_wins_over_text(self):
        # 原生与文本同时存在时，只信原生，避免重复执行
        content = wrap_tool('{"name": "calculator", "arguments": {"expression": "1"}}')
        parsed = parse_assistant_message({
            "role": "assistant", "content": content,
            "tool_calls": [{
                "id": "n1", "type": "function",
                "function": {"name": "search", "arguments": "{\"query\": \"q\"}"},
            }],
        })
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "search"

    def test_plain_text_is_final_answer(self):
        parsed = parse_assistant_message({"role": "assistant", "content": "2+2=4，不用工具"})
        assert not parsed.need_tool_call
        assert parsed.final_answer == "2+2=4，不用工具"


# ======================================================================
# 异常路径
# ======================================================================
class TestParseErrors:
    def test_not_assistant_role(self):
        with pytest.raises(LLMResponseError):
            parse_assistant_message({"role": "user", "content": "hi"})

    def test_not_dict(self):
        with pytest.raises(LLMResponseError):
            parse_assistant_message("assistant message")
