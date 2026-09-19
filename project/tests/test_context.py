"""Context 管理测试：组装 / 预算截断 / 工具结果截断 / 压缩。"""
import json

import pytest

from mini_agent.context import (ContextManager, SUMMARY_PREFIX, estimate_tokens,
                                messages_tokens, truncate_text)
from mini_agent.session import Session
from mini_agent.tools import create_default_registry
from tests.conftest import FakeLLM, final_msg, tool_msg


@pytest.fixture
def ctx(cfg):
    return ContextManager(cfg, create_default_registry(), llm=FakeLLM())


def make_session_with_rounds(n_rounds, per_round=2, chars=0):
    s = Session("ctx-test")
    fill = "数" * chars
    for i in range(n_rounds):
        s.add_user(f"第{i+1}轮问题{fill}")
        if per_round >= 3:
            s.add_assistant(tool_calls_api=[{
                "id": f"c{i}", "type": "function",
                "function": {"name": "calculator",
                             "arguments": json.dumps({"expression": "1+1"})}}])
            s.add_tool_result(f"c{i}", "calculator", f"2{fill}")
        s.add_assistant(content=f"第{i+1}轮回答{fill}")
    return s


# ======================================================================
# token 估算与截断
# ======================================================================
class TestEstimate:
    def test_ascii(self):
        assert estimate_tokens("abcdefgh") == 2

    def test_cjk(self):
        assert estimate_tokens("你好世界") == 4

    def test_mixed(self):
        t = estimate_tokens("你好abcd")
        assert t == 2 + 1  # 2个汉字 + 4字母=1 token

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_truncate(self):
        out = truncate_text("x" * 100, 10)
        assert out.startswith("xxxxxxxxxx") and "已截断" in out
        assert truncate_text("short", 10) == "short"


# ======================================================================
# build_messages
# ======================================================================
class TestBuild:
    def test_structure(self, ctx):
        s = make_session_with_rounds(2)
        msgs = ctx.build_messages(s)
        assert msgs[0]["role"] == "system"
        assert "calculator" in msgs[0]["content"]     # 工具一览进 system
        assert "todo" in msgs[0]["content"]
        assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user", "assistant"]

    def test_summary_included(self, ctx):
        s = make_session_with_rounds(1)
        s.summary = "用户问过天气"
        msgs = ctx.build_messages(s)
        assert any(m["role"] == "system" and m["content"].startswith(SUMMARY_PREFIX)
                   for m in msgs)

    def test_memory_hint_in_system(self, ctx):
        s = make_session_with_rounds(1)
        msgs = ctx.build_messages(s, memory_hint="\n[长期记忆召回]\n- 用户在上海工作")
        assert "用户在上海工作" in msgs[0]["content"]

    def test_max_history_messages(self, cfg):
        cfg.max_history_messages = 6
        ctx = ContextManager(cfg, create_default_registry(), llm=FakeLLM())
        s = make_session_with_rounds(10)          # 20 条消息
        msgs = ctx.build_messages(s)
        body = [m for m in msgs if m["role"] != "system"]
        assert len(body) <= 6
        assert body[0]["role"] == "user"          # 对齐到完整轮次

    def test_orphan_tool_result_dropped(self, cfg):
        """窗口头部不能出现没有配对 assistant 调用的 tool 结果。
        每轮 4 条消息（user/assistant(tool)/tool/answer），窗口取 5：
        头部必然切在轮次中间，必须对齐裁掉残轮。"""
        cfg.max_history_messages = 5
        ctx = ContextManager(cfg, create_default_registry(), llm=FakeLLM())
        s = make_session_with_rounds(3, per_round=3)
        msgs = ctx.build_messages(s)
        body = [m for m in msgs if m["role"] != "system"]
        assert body[0]["role"] == "user"
        assert len(body) == 4          # 最后一轮完整保留（user/assistant/tool/answer）
        # 消息序列中 assistant 工具调用与 tool 结果必须成对
        pending = set()
        for m in body:
            if m["role"] == "assistant" and m.get("tool_calls"):
                pending = {tc["id"] for tc in m["tool_calls"]}
            elif m["role"] == "tool":
                assert m["tool_call_id"] in pending

    def test_tool_result_truncated(self, cfg):
        cfg.max_tool_result_chars = 50
        ctx = ContextManager(cfg, create_default_registry(), llm=FakeLLM())
        s = Session("t")
        s.add_user("读文档")
        s.add_assistant(tool_calls_api=[{
            "id": "c1", "type": "function",
            "function": {"name": "read_docs", "arguments": "{}"}}])
        s.add_tool_result("c1", "read_docs", "字" * 500)
        msgs = ctx.build_messages(s)
        tool_msg_out = [m for m in msgs if m["role"] == "tool"][0]
        assert len(tool_msg_out["content"]) < 100
        assert "已截断" in tool_msg_out["content"]

    def test_hard_budget_drops_oldest_rounds(self, cfg):
        cfg.max_context_tokens = 700              # system 后剩余预算很小
        ctx = ContextManager(cfg, create_default_registry(), llm=FakeLLM())
        s = make_session_with_rounds(10, chars=100)   # 每条消息 ~100 token
        msgs = ctx.build_messages(s)
        body = [m for m in msgs if m["role"] != "system"]
        assert len(body) < 20                     # 确实被裁剪
        assert body[0]["role"] == "user"          # 丢整轮，从 user 开始
        assert messages_tokens(body) <= 600       # 保持在预算附近


# ======================================================================
# 压缩
# ======================================================================
class TestCompress:
    def test_not_triggered_below_threshold(self, ctx, cfg):
        s = make_session_with_rounds(3)
        before = len(s.messages)
        assert ctx.maybe_compress(s) is False
        assert len(s.messages) == before and s.summary == ""

    def test_compress_on_message_count(self, cfg):
        cfg.max_history_messages = 10
        cfg.keep_recent_messages = 4
        ctx = ContextManager(cfg, create_default_registry(), llm=FakeLLM())
        s = make_session_with_rounds(8, per_round=3)   # 24 条
        assert ctx.maybe_compress(s) is True
        assert s.summary                            # 生成了滚动摘要
        assert s.messages[0].role == "user"          # 保留段从完整轮次开始
        assert len(s.messages) <= 24 - 10 + 4 + 3    # 久远段被摘除
        assert len(s.archive) >= 10                  # 原文进审计存档
        # 压缩后再 build：system + 摘要 + 近期原文
        msgs = ctx.build_messages(s)
        assert any(m["content"].startswith(SUMMARY_PREFIX) for m in msgs)

    def test_compress_keeps_pairs_aligned(self, cfg):
        cfg.max_history_messages = 8
        cfg.keep_recent_messages = 3
        ctx = ContextManager(cfg, create_default_registry(), llm=FakeLLM())
        s = make_session_with_rounds(8, per_round=3)
        ctx.maybe_compress(s)
        # 保留段内 assistant 调用与 tool 结果必须配对完整
        pending = set()
        for m in s.messages:
            if m.role == "assistant" and m.tool_calls:
                pending = {tc["id"] for tc in m.tool_calls}
            elif m.role == "tool":
                assert m.tool_call_id in pending

    def test_compress_failure_keeps_history(self, cfg):
        class BrokenSummarizer(FakeLLM):
            def chat(self, messages, tools=None, temperature=None):
                system = "\n".join(str(m.get("content", ""))
                                   for m in messages if m.get("role") == "system")
                if "对话压缩助手" in system:
                    raise RuntimeError("summarize failed")
                return super().chat(messages, tools, temperature)

        cfg.max_history_messages = 6
        cfg.keep_recent_messages = 2
        ctx = ContextManager(cfg, create_default_registry(), llm=BrokenSummarizer())
        s = make_session_with_rounds(6, per_round=3)
        n_before = len(s.messages)
        assert ctx.maybe_compress(s) is False       # 失败：宁可不压
        assert len(s.messages) == n_before and s.summary == ""

    def test_compress_not_when_all_recent(self, cfg):
        cfg.max_history_messages = 100             # 不触发条数阈值
        cfg.compress_trigger_tokens = 10           # 触发 token 阈值
        cfg.keep_recent_messages = 1000            # 但保留窗口覆盖全部
        ctx = ContextManager(cfg, create_default_registry(), llm=FakeLLM())
        s = make_session_with_rounds(4)
        assert ctx.maybe_compress(s) is False
