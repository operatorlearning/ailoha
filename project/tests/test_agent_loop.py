"""Agent 主循环测试（FakeLLM 脚本化，覆盖题目全部核心要求）。

覆盖：
  基本循环：直接回复 / 单工具 / 并行工具 / 工具结果驱动下一轮
  异常处理：未知工具回喂自愈 / 参数错误自愈 / LLM API 失败 / 轮次上限
  追问支持：纯对话追问（历史进 context）/ 带工具追问（工具结果进 context）
  context：压缩在多轮后触发 / trace 完整记录
"""
import json
import os

import pytest

from mini_agent.config import AgentConfig
from mini_agent.exceptions import LLMAPIError
from mini_agent.tracing import Tracer
from tests.conftest import FakeLLM, final_msg, tool_msg


def read_trace(cfg, session_id):
    return Tracer(cfg.trace_dir, session_id).tail(500)


def events(trace):
    return [r["event"] for r in trace]


# ======================================================================
# 基本循环
# ======================================================================
class TestBasicLoop:
    def test_direct_reply(self, make_agent, fake_llm):
        fake_llm.responses = [final_msg("你好！我是 MiniAgent。")]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        result = agent.chat(s, "你好")
        assert result.answer == "你好！我是 MiniAgent。"
        assert result.tool_calls == []
        assert result.iterations == 1
        assert [m.role for m in s.messages] == ["user", "assistant"]

    def test_single_tool_roundtrip(self, make_agent, fake_llm):
        fake_llm.responses = [
            tool_msg(("calculator", {"expression": "1204/2"})),
            final_msg("1204 除以 2 等于 602。"),
        ]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        result = agent.chat(s, "1204除以2是多少")
        assert "602" in result.answer
        assert result.iterations == 2
        assert result.tool_calls[0]["name"] == "calculator"
        assert result.tool_calls[0]["ok"] is True
        # 消息序列：user -> assistant(调用) -> tool(结果) -> assistant(答案)
        roles = [m.role for m in s.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert s.messages[2].name == "calculator"
        assert "602" in s.messages[2].content
        # 上下文确实携带工具调用与结果
        msgs2 = fake_llm.loop_calls[1]["messages"]
        assert any(m["role"] == "tool" and "602" in str(m.get("content"))
                   for m in msgs2)

    def test_parallel_tools(self, make_agent, fake_llm):
        fake_llm.responses = [
            tool_msg([("calculator", {"expression": "1+1"}),
                      ("get_weather", {"city": "北京"})]),
            final_msg("算好了，天气也查到了。"),
        ]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        result = agent.chat(s, "1+1等于几？顺便看下北京天气")
        assert result.answer == "算好了，天气也查到了。"
        assert [t["name"] for t in result.tool_calls] == ["calculator", "get_weather"]
        tool_results = [m for m in s.messages if m.role == "tool"]
        assert len(tool_results) == 2

    def test_llm_receives_tool_schemas(self, make_agent, fake_llm):
        fake_llm.responses = [final_msg("ok")]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        agent.chat(s, "hi")
        tools = fake_llm.loop_calls[0]["tools"]
        assert tools and tools[0]["type"] == "function"
        names = {t["function"]["name"] for t in tools}
        assert "calculator" in names and "todo" in names


# ======================================================================
# 异常处理：错误回喂 + 自愈
# ======================================================================
class TestErrorRecovery:
    def test_unknown_tool_self_heals(self, make_agent, fake_llm):
        fake_llm.responses = [
            tool_msg(("no_such_tool", {"x": 1})),
            final_msg("没有这个工具，我直接告诉你：1+1=2。"),
        ]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        result = agent.chat(s, "帮我做点什么")
        assert "1+1=2" in result.answer
        assert result.tool_calls[0]["ok"] is False          # 第一次失败
        # 失败信息以 tool 结果形式回喂给 LLM
        msgs2 = fake_llm.loop_calls[1]["messages"]
        err_msgs = [m for m in msgs2 if m["role"] == "tool"
                    and "工具执行错误" in str(m.get("content"))]
        assert err_msgs

    def test_bad_arguments_self_heals(self, make_agent, fake_llm):
        fake_llm.responses = [
            tool_msg(("calculator", {"expr": "1+1"})),          # 参数名错误
            tool_msg(("calculator", {"expression": "1+1"})),    # 修正后重试
            final_msg("1+1=2。"),
        ]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        result = agent.chat(s, "1+1?")
        assert result.answer == "1+1=2。"
        assert result.tool_calls[0]["ok"] is False
        assert result.tool_calls[1]["ok"] is True

    def test_llm_api_error_friendly_message(self, make_agent):
        class ApiDown(FakeLLM):
            def chat(self, messages, tools=None, temperature=None):
                raise LLMAPIError("连接超时（mock）")
        agent = make_agent(ApiDown())
        s = agent.new_session("t")
        result = agent.chat(s, "你好")
        assert "抱歉" in result.answer
        assert result.error and "LLM" in result.error
        # 失败的这轮也被保存，可恢复
        assert os.path.isfile(os.path.join(agent.cfg.data_dir, "t.json"))
        assert [m.role for m in s.messages] == ["user"]

    def test_max_iterations_forced_final(self, tmp_path):
        cfg = AgentConfig(
            data_dir=str(tmp_path / "data"), trace_dir=str(tmp_path / "traces"),
            max_iterations=2)
        llm = FakeLLM(responses=[
            tool_msg(("calculator", {"expression": "1+1"})),
            tool_msg(("calculator", {"expression": "2+2"})),
            # 之后 FakeLLM 弹空 -> default_final（作为强制收尾的回复）
        ], default_final="基于已获得的信息：计算均已完成。")
        from mini_agent.agent import Agent
        from mini_agent.memory import MemoryStore
        from mini_agent.session import SessionManager
        agent = Agent(llm, cfg, SessionManager(cfg.data_dir),
                      memory=MemoryStore(str(tmp_path / "memory.json")))
        s = agent.new_session("t")
        result = agent.chat(s, "反复算")
        assert result.iterations == 2
        assert result.answer                      # 有兜底收尾，不是空
        assert len(result.tool_calls) == 2
        tr = read_trace(cfg, "t")
        assert "max_iterations" in events(tr)
        assert s.messages[-1].role == "assistant"


# ======================================================================
# 追问支持
# ======================================================================
class TestFollowUp:
    def test_pure_chat_followup_uses_history(self, make_agent, fake_llm):
        fake_llm.responses = [
            final_msg("我叫小 M，是你的助手。"),
            final_msg("我刚才说过，我叫小 M。"),
        ]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        agent.chat(s, "你叫什么名字？")
        result = agent.chat(s, "你再说一遍你叫什么？")
        assert "小 M" in result.answer
        # 第二轮请求确实带上了第一轮的原文
        msgs2 = fake_llm.loop_calls[1]["messages"]
        assert any(m["role"] == "user" and "叫什么名字" in str(m.get("content"))
                   for m in msgs2)

    def test_followup_with_tool_context(self, make_agent, fake_llm):
        """带工具的追问：上轮工具结果仍在上下文中，模型可直接引用。"""
        fake_llm.responses = [
            tool_msg(("todo", {"action": "add", "task": "写周报"})),
            final_msg("已添加待办：写周报。"),
            tool_msg(("todo", {"action": "list"})),
            final_msg("你目前有 1 条待办：写周报（来自上轮工具记录）。"),
        ]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        agent.chat(s, "加个待办：写周报")
        result = agent.chat(s, "我刚才让你加的待办是什么？")
        assert "写周报" in result.answer
        # 追问轮的请求里包含第一轮的 tool 结果
        msgs = fake_llm.loop_calls[2]["messages"]
        assert any(m["role"] == "tool" and "写周报" in str(m.get("content"))
                   for m in msgs)
        # todo 状态跨轮持续存在
        assert s.tool_state["todo"]["items"][0]["task"] == "写周报"


# ======================================================================
# context 压缩 & trace
# ======================================================================
class TestContextAndTrace:
    def test_compression_triggers_after_many_rounds(self, tmp_path):
        cfg = AgentConfig(
            data_dir=str(tmp_path / "data"), trace_dir=str(tmp_path / "traces"),
            max_history_messages=6, keep_recent_messages=3)
        # 每轮 user->tool->result->answer = 4 条消息，3 轮后超过 6 条
        responses = []
        for i in range(4):
            responses.append(tool_msg(("calculator", {"expression": f"{i}+1"})))
            responses.append(final_msg(f"第{i+1}轮答案"))
        llm = FakeLLM(responses=responses)
        from mini_agent.agent import Agent
        from mini_agent.memory import MemoryStore
        from mini_agent.session import SessionManager
        agent = Agent(llm, cfg, SessionManager(cfg.data_dir),
                      memory=MemoryStore(str(tmp_path / "memory.json")))
        s = agent.new_session("t")
        for i in range(4):
            agent.chat(s, f"第{i+1}轮：算 {i}+1")
        assert s.summary                        # 压缩已发生
        assert len(s.messages) <= 6
        assert s.messages[0].role == "user"
        tr = read_trace(cfg, "t")
        assert "compress" in events(tr)
        assert "session_saved" in events(tr)

    def test_trace_records_full_loop(self, make_agent, fake_llm):
        fake_llm.responses = [
            tool_msg(("calculator", {"expression": "2*3"}), thinking="需要算一下"),
            final_msg("2*3=6。"),
        ]
        agent = make_agent(fake_llm)
        s = agent.new_session("trace-test")
        agent.chat(s, "2*3=?")
        tr = read_trace(agent.cfg, "trace-test")
        evs = events(tr)
        for expected in ["user_input", "llm_request", "llm_response",
                         "tool_call", "tool_result", "final_answer",
                         "session_saved"]:
            assert expected in evs, f"trace 缺少事件 {expected}"
        # thinking 也进入 trace
        assert any(r.get("thinking") for r in tr if r["event"] == "llm_response")
        assert os.path.isfile(tr[0] and os.path.join(agent.cfg.trace_dir, "trace-test.jsonl"))

    def test_thinking_kept_in_session_not_in_api(self, make_agent, fake_llm):
        """思考过程存档（session）但不回传给 LLM（上下文省 token）。"""
        fake_llm.responses = [
            final_msg("答案是 6", thinking="先算 2*3"),
            final_msg("再见"),
        ]
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        agent.chat(s, "2*3?")
        assert s.messages[1].thinking == "先算 2*3"
        agent.chat(s, "拜拜")
        msgs2 = fake_llm.loop_calls[1]["messages"]
        assistant_msgs = [m for m in msgs2 if m["role"] == "assistant"]
        assert all("reasoning" not in m and "thinking" not in m
                   for m in assistant_msgs)


# ======================================================================
# memory 联动
# ======================================================================
class TestMemoryIntegration:
    def test_recall_injected_into_system(self, make_agent, fake_llm, tmp_path):
        from mini_agent.memory import MemoryStore
        store = MemoryStore(str(tmp_path / "memory.json"))
        store.add("用户的工作城市是上海")
        fake_llm.responses = [final_msg("你在上海工作。")]
        agent = make_agent(fake_llm)
        agent.memory = store
        s = agent.new_session("t")
        agent.chat(s, "我在哪个城市工作？")
        system = fake_llm.loop_calls[0]["messages"][0]["content"]
        assert "上海" in system and "长期记忆" in system

    def test_extraction_after_turn(self, make_agent, fake_llm, tmp_path):
        fake_llm.responses = [
            final_msg("好的，记住了。"),
        ]
        fake_llm.extract_response = '["用户的工作城市是上海"]'
        agent = make_agent(fake_llm)
        s = agent.new_session("t")
        agent.chat(s, "顺便记住：我在上海工作")
        assert any("上海" in i["content"] for i in agent.memory.items)
        tr = read_trace(agent.cfg, "t")
        assert "memory_extracted" in events(tr)
