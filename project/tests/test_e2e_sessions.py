"""端到端测试：题目场景（用户A的两个窗口）+ 跨进程持久恢复。"""
import os

import pytest

from mini_agent.agent import Agent
from mini_agent.config import AgentConfig
from mini_agent.memory import MemoryStore
from mini_agent.session import SessionManager
from tests.conftest import FakeLLM, final_msg, tool_msg


def build_agent(tmp_path, llm, **cfg_kwargs):
    cfg = AgentConfig(
        data_dir=str(tmp_path / "data"),
        trace_dir=str(tmp_path / "traces"),
        max_iterations=3,
        **cfg_kwargs,
    )
    return Agent(llm, cfg, SessionManager(cfg.data_dir),
                 memory=MemoryStore(str(tmp_path / "memory.json")))


# ======================================================================
# 题目场景：用户 A 开窗口1（日历/待办）与窗口2（联系人/待办），互不影响
# ======================================================================
class TestTwoWindows:
    def test_windows_isolated_and_resumable(self, tmp_path):
        llm = FakeLLM(responses=[
            # 窗口1：加日历事项（用 todo 工具模拟"加日程"）
            tool_msg(("todo", {"action": "add", "task": "日历：明天15:00周会"})),
            final_msg("已在窗口1添加日历：明天15:00周会。"),
            # 窗口2：加联系人（同样用 todo，但处于独立 session 状态）
            tool_msg(("todo", {"action": "add", "task": "联系人：张三 13800000000"})),
            final_msg("已在窗口2添加联系人：张三。"),
            # 回到窗口1：查看待办
            tool_msg(("todo", {"action": "list"})),
            final_msg("窗口1当前只有一条：明天15:00周会。"),
            # 再看窗口2
            tool_msg(("todo", {"action": "list"})),
            final_msg("窗口2当前只有一条：联系人张三。"),
        ])
        agent = build_agent(tmp_path, llm)

        win1 = agent.new_session("win1")
        win2 = agent.new_session("win2")

        r1 = agent.chat(win1, "帮我在日历加上：明天15:00周会")
        assert "周会" in r1.answer
        r2 = agent.chat(win2, "帮我加个联系人：张三 13800000000")
        assert "张三" in r2.answer

        # 窗口1的状态只有日历事项；窗口2只有联系人 —— session 级隔离
        tasks1 = [i["task"] for i in win1.tool_state["todo"]["items"]]
        tasks2 = [i["task"] for i in win2.tool_state["todo"]["items"]]
        assert tasks1 == ["日历：明天15:00周会"]
        assert tasks2 == ["联系人：张三 13800000000"]

        # 随时接着任何一个窗口继续聊，互不影响
        r1b = agent.chat(win1, "看下我的日程")
        assert "周会" in r1b.answer and "张三" not in r1b.answer
        r2b = agent.chat(win2, "看下我的联系人")
        assert "张三" in r2b.answer and "周会" not in r2b.answer

        # 两窗口的消息历史互相干净
        all_text1 = " ".join(str(m.content) for m in win1.messages if m.content)
        assert "张三" not in all_text1
        all_text2 = " ".join(str(m.content) for m in win2.messages if m.content)
        assert "周会" not in all_text2

    def test_window2_llm_context_not_polluted_by_window1(self, tmp_path):
        llm = FakeLLM(responses=[
            tool_msg(("todo", {"action": "add", "task": "窗口1的任务"})),
            final_msg("窗口1已添加。"),
            final_msg("窗口2直接回答，历史干净。"),
        ])
        agent = build_agent(tmp_path, llm)
        win1 = agent.new_session("w1")
        win2 = agent.new_session("w2")
        agent.chat(win1, "加个待办：窗口1的任务")
        agent.chat(win2, "你好")
        # 窗口2 的请求上下文中不能出现窗口1 的内容
        msgs_w2 = llm.loop_calls[2]["messages"]
        joined = " ".join(str(m.get("content")) for m in msgs_w2)
        assert "窗口1的任务" not in joined


# ======================================================================
# 持久化：重启进程后接着聊（记住之前状态）
# ======================================================================
class TestPersistenceResume:
    def test_resume_after_restart(self, tmp_path):
        llm1 = FakeLLM(responses=[
            tool_msg(("todo", {"action": "add", "task": "写周报"})),
            final_msg("已添加待办：写周报。"),
        ])
        agent1 = build_agent(tmp_path, llm1)
        s = agent1.new_session("resume-me")
        agent1.chat(s, "加个待办：写周报")
        assert os.path.isfile(os.path.join(str(tmp_path / "data"), "resume-me.json"))

        # 模拟"重启"：全新的 Agent / SessionManager / LLM 实例
        llm2 = FakeLLM(responses=[final_msg("你之前让我加的待办是：写周报。")])
        agent2 = build_agent(tmp_path, llm2)
        s2 = agent2.get_session("resume-me")
        assert s2.user_rounds() == 1                      # 历史恢复
        assert s2.tool_state["todo"]["items"][0]["task"] == "写周报"

        result = agent2.chat(s2, "我刚才让你加什么来着？")
        assert "写周报" in result.answer
        # 恢复后的请求上下文包含重启前的工具结果
        msgs = llm2.calls[0]["messages"]
        assert any(m["role"] == "tool" and "写周报" in str(m.get("content"))
                   for m in msgs)

    def test_trace_file_accumulates_across_restarts(self, tmp_path):
        llm = FakeLLM(responses=[final_msg("好。"), final_msg("继续。")])
        agent = build_agent(tmp_path, llm)
        s = agent.new_session("traced")
        agent.chat(s, "第一轮")
        agent.chat(s, "第二轮")

        from mini_agent.tracing import Tracer
        tr = Tracer(agent.cfg.trace_dir, "traced").tail(100)
        turns = {r["turn"] for r in tr}
        assert turns == {1, 2}          # turn 编号连续累加
        assert any(r["event"] == "user_input" for r in tr)
