"""长期记忆测试：存取 / 召回打分 / hint 渲染 / LLM 抽取。"""
import pytest

from mini_agent.memory import MemoryStore, extract_memories
from tests.conftest import FakeLLM


@pytest.fixture
def store(tmp_path):
    return MemoryStore(str(tmp_path / "memory.json"))


class TestStore:
    def test_add_and_recall(self, store):
        assert store.add("用户的工作城市是上海") is True
        assert store.add("用户喜欢喝美式咖啡") is True
        hits = store.recall("我在哪个城市工作")
        assert len(hits) == 1
        assert "上海" in hits[0]["content"]

    def test_add_dedup(self, store):
        store.add("用户喜欢简洁回答")
        assert store.add("用户喜欢简洁回答") is False   # 完全相同内容去重
        assert len(store.items) == 1

    def test_add_rejects_empty_or_long(self, store):
        assert store.add("   ") is False
        assert store.add("超" * 200) is False

    def test_recall_no_hit(self, store):
        store.add("用户喜欢喝美式咖啡")
        assert store.recall("今天天气如何") == []

    def test_recall_empty_query_or_store(self, store):
        assert store.recall("") == []
        empty = MemoryStore.__new__(MemoryStore)
        empty.items = []
        assert empty.recall("任意") == []

    def test_render_hint(self, store):
        store.add("用户的工作城市是上海")
        hint = store.render_hint("我在哪个城市工作")
        assert "上海" in hint
        assert "长期记忆" in hint and "仅供参考" in hint

    def test_render_hint_empty(self, store):
        store.add("用户喜欢喝美式咖啡")
        assert store.render_hint("讲个笑话") == ""

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "memory.json")
        s1 = MemoryStore(path)
        s1.add("用户讨厌啰嗦")
        s2 = MemoryStore(path)                 # 重新加载
        assert any("啰嗦" in i["content"] for i in s2.items)


class TestExtract:
    def test_extract_ok(self):
        llm = FakeLLM(extract_response='["用户的工作城市是上海", "用户偏好简洁回答"]')
        facts = extract_memories(llm, "我在上海上班", "了解，会尽量简洁")
        assert facts == ["用户的工作城市是上海", "用户偏好简洁回答"]

    def test_extract_with_noise(self):
        llm = FakeLLM(extract_response='好的，提取结果如下：\n["用户在上海上班"]\n以上。')
        facts = extract_memories(llm, "我在上海上班", "了解")
        assert facts == ["用户在上海上班"]

    def test_extract_empty(self):
        llm = FakeLLM(extract_response="[]")
        assert extract_memories(llm, "算了下 1+1", "等于 2") == []

    def test_extract_bad_json(self):
        llm = FakeLLM(extract_response="我觉得没什么要记的")
        assert extract_memories(llm, "hi", "hello") == []

    def test_extract_llm_error(self):
        class Boom(FakeLLM):
            def chat(self, messages, tools=None, temperature=None):
                raise RuntimeError("api down")
        assert extract_memories(Boom(), "hi", "hello") == []

    def test_extract_no_llm(self):
        assert extract_memories(None, "hi", "hello") == []
