"""Session 管理测试：独立会话 / 持久化恢复 / id 规范化 / 列表删除。"""
import json
import os
import time

import pytest

from mini_agent.exceptions import SessionIdError, SessionNotFoundError
from mini_agent.session import (Message, Session, SessionManager,
                                 make_api_tool_calls)


# ======================================================================
# Message
# ======================================================================
class TestMessage:
    def test_user_to_api(self):
        m = Message(role="user", content="你好")
        assert m.to_api() == {"role": "user", "content": "你好"}

    def test_assistant_tool_calls_to_api(self):
        api_calls = make_api_tool_calls([type("C", (), {
            "id": "c1", "name": "calculator", "arguments": {"expression": "1+1"}})()])
        m = Message(role="assistant", content=None, tool_calls=api_calls)
        d = m.to_api()
        assert d["role"] == "assistant"
        assert d["tool_calls"][0]["function"]["name"] == "calculator"
        assert json.loads(d["tool_calls"][0]["function"]["arguments"]) == {"expression": "1+1"}

    def test_tool_result_to_api(self):
        m = Message(role="tool", content="结果", tool_call_id="c1", name="calculator")
        d = m.to_api()
        assert d["tool_call_id"] == "c1" and d["name"] == "calculator"

    def test_dict_roundtrip(self):
        m = Message(role="assistant", content="答", thinking="想")
        m2 = Message.from_dict(m.to_dict())
        assert m2 == m or (m2.content == m.content and m2.thinking == m.thinking
                           and m2.role == m.role)


# ======================================================================
# Session
# ======================================================================
class TestSession:
    def test_add_and_rounds(self):
        s = Session("s1")
        assert s.user_rounds() == 0
        s.add_user("第一轮")
        s.add_assistant(content="回复")
        s.add_user("第二轮")
        assert s.user_rounds() == 2
        assert s.title == "第一轮"

    def test_tool_flow_messages(self):
        s = Session("s1")
        s.add_user("算 1+1")
        s.add_assistant(tool_calls_api=make_api_tool_calls([]) or [{
            "id": "c1", "type": "function",
            "function": {"name": "calculator", "arguments": "{}"}}])
        s.add_tool_result("c1", "calculator", "2")
        roles = [m.role for m in s.messages]
        assert roles == ["user", "assistant", "tool"]
        assert s.messages[-1].tool_call_id == "c1"

    def test_dict_roundtrip(self):
        s = Session("roundtrip")
        s.add_user("你好")
        s.add_assistant(content="在", thinking="思考内容")
        s.summary = "此前摘要"
        s.archive = [{"role": "user", "content": "旧"}]
        s.tool_state = {"todo": {"items": [{"id": 1, "task": "x", "done": False}],
                                 "next_id": 2}}
        s2 = Session.from_dict(s.to_dict())
        assert s2.id == "roundtrip"
        assert s2.summary == "此前摘要"
        assert s2.tool_state["todo"]["next_id"] == 2
        assert s2.messages[1].thinking == "思考内容"
        assert len(s2.archive) == 1


# ======================================================================
# SessionManager
# ======================================================================
class TestSessionManager:
    def test_sanitize(self):
        mgr = SessionManager("data")
        assert mgr.sanitize_id("win 1") == "win_1"
        assert mgr.sanitize_id("a/b\\c") == "a_b_c"
        with pytest.raises(SessionIdError):
            mgr.sanitize_id("   ")
        with pytest.raises(SessionIdError):
            mgr.sanitize_id("..")

    def test_auto_id_unique(self):
        assert SessionManager.auto_id() != SessionManager.auto_id()

    def test_save_load_roundtrip(self, tmp_path):
        mgr = SessionManager(str(tmp_path))
        s = mgr.get_or_create("win1")
        s.add_user("加个待办：写周报")
        s.add_assistant(content="已添加")
        mgr.save(s)
        assert os.path.isfile(str(tmp_path / "win1.json"))
        assert not os.path.isfile(str(tmp_path / "win1.json.tmp"))  # 原子写无残留

        s2 = mgr.load("win1")
        assert s2.id == "win1"
        assert [m.role for m in s2.messages] == ["user", "assistant"]
        assert s2.messages[0].content == "加个待办：写周报"

    def test_get_or_create(self, tmp_path):
        mgr = SessionManager(str(tmp_path))
        s = mgr.get_or_create("w1")
        s.add_user("hi")
        mgr.save(s)
        s_loaded = mgr.get_or_create("w1")
        assert s_loaded.user_rounds() == 1          # 恢复而非新建
        s_fresh = mgr.get_or_create("w2")
        assert s_fresh.user_rounds() == 0

    def test_load_missing(self, tmp_path):
        mgr = SessionManager(str(tmp_path))
        with pytest.raises(SessionNotFoundError):
            mgr.load("ghost")

    def test_two_sessions_independent(self, tmp_path):
        """题目场景：窗口1加日历、窗口2加联系人，互不干扰。"""
        mgr = SessionManager(str(tmp_path))
        cal = mgr.get_or_create("win1-calendar")
        contact = mgr.get_or_create("win2-contact")
        cal.add_user("日历加一条：明天 15:00 周会")
        contact.add_user("联系人加一条：张三 13800000000")
        mgr.save(cal)
        mgr.save(contact)

        cal2 = mgr.load("win1-calendar")
        contact2 = mgr.load("win2-contact")
        assert cal2.user_rounds() == 1
        assert contact2.user_rounds() == 1
        assert "周会" in cal2.messages[0].content
        assert "张三" in contact2.messages[0].content
        assert all("张三" not in (m.content or "") for m in cal2.messages)
        assert all("周会" not in (m.content or "") for m in contact2.messages)

    def test_list_and_delete(self, tmp_path):
        mgr = SessionManager(str(tmp_path))
        for sid, text in [("a", "第一"), ("b", "第二")]:
            s = mgr.get_or_create(sid)
            s.add_user(text)
            s.updated_at = time.time() + (1 if sid == "a" else 0)
            mgr.save(s)
        rows = mgr.list_sessions()
        assert [r["id"] for r in rows] == ["a", "b"]   # 按更新时间倒序
        assert rows[0]["rounds"] == 1
        mgr.delete("a")
        assert [r["id"] for r in mgr.list_sessions()] == ["b"]
        with pytest.raises(SessionNotFoundError):
            mgr.delete("a")
