"""工具层测试：注册机制 / Schema 校验 / 各内置工具 / 异常路径。"""
import json
import os

import pytest

from mini_agent.exceptions import (ToolArgumentError, ToolExecutionError,
                                   ToolNotFoundError)
from mini_agent.parser import ToolCall
from mini_agent.tools import (Tool, ToolRegistry, calculate,
                              create_default_registry, validate_schema)


def call(name, arguments, cid="c1"):
    return ToolCall(id=cid, name=name, arguments=arguments,
                    raw_arguments=json.dumps(arguments, ensure_ascii=False))


# ======================================================================
# 注册机制
# ======================================================================
class TestRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = Tool(name="echo", description="回声", parameters={"type": "object"},
                    func=lambda a, s: "ok")
        reg.register(tool)
        assert reg.get("echo") is tool
        assert reg.names() == ["echo"]
        assert reg.get("nope") is None

    def test_duplicate_register(self):
        reg = ToolRegistry()
        reg.register(Tool("x", "x", {"type": "object"}, lambda a, s: "ok"))
        with pytest.raises(ValueError):
            reg.register(Tool("x", "x", {"type": "object"}, lambda a, s: "ok"))

    def test_openai_schema_format(self):
        reg = create_default_registry()
        schemas = reg.to_openai_schemas()
        assert len(schemas) >= 5
        for s in schemas:
            assert s["type"] == "function"
            assert set(s["function"]) >= {"name", "description", "parameters"}
        names = {s["function"]["name"] for s in schemas}
        assert {"calculator", "search", "todo", "get_weather", "read_docs"} <= names

    def test_unknown_tool_raises(self):
        reg = create_default_registry()
        with pytest.raises(ToolNotFoundError):
            reg.execute(call("no_such_tool", {}))

    def test_invalid_json_arguments(self):
        from mini_agent.parser import RAW_ARGS_KEY
        reg = create_default_registry()
        with pytest.raises(ToolArgumentError):
            reg.execute(call("calculator", {RAW_ARGS_KEY: "not json"}))


# ======================================================================
# Schema 校验器
# ======================================================================
class TestValidator:
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list"]},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
    }

    def test_ok(self):
        assert validate_schema(
            {"action": "add", "count": 3, "ratio": 0.5, "flag": True, "tags": ["a"]},
            self.schema) == []

    def test_missing_required(self):
        errs = validate_schema({"count": 1}, self.schema)
        assert any("action" in e for e in errs)

    def test_wrong_type(self):
        errs = validate_schema({"action": "add", "count": "3"}, self.schema)
        assert any("count" in e for e in errs)

    def test_enum_violation(self):
        errs = validate_schema({"action": "delete"}, self.schema)
        assert any("枚举" in e for e in errs)

    def test_bool_is_not_integer(self):
        errs = validate_schema({"action": "add", "count": True}, self.schema)
        assert any("count" in e for e in errs)

    def test_array_items(self):
        errs = validate_schema({"action": "add", "tags": [1, 2]}, self.schema)
        assert any("tags" in e for e in errs)


# ======================================================================
# calculator
# ======================================================================
class TestCalculator:
    @pytest.mark.parametrize("expr,expected", [
        ("1+2", 3),
        ("(12+8)*3.5", 70),
        ("2**10", 1024),
        ("10/4", 2.5),
        ("-5+3", -2),
        ("7 % 3", 1),
        ("sqrt(16)+1", 5),
        ("abs(-3)*2", 6),
        ("min(3, 1, 2) + max(1, 5)", 6),
        ("2*(3+4)", 14),
        ("1.5*2", 3),
        ("pi*0", 0),
        ("2^3", 8),
    ])
    def test_correct(self, expr, expected):
        assert calculate(expr) == expected

    @pytest.mark.parametrize("expr", ["1/0", "abc$(" , "import os", "1+", "()", "foo(1)"])
    def test_errors(self, expr):
        with pytest.raises(ToolExecutionError):
            calculate(expr)

    def test_via_registry(self):
        reg = create_default_registry()
        out = json.loads(reg.execute(call("calculator", {"expression": "1204/2"})))
        assert out["result"] == 602

    def test_schema_rejects_missing_expression(self):
        reg = create_default_registry()
        with pytest.raises(ToolArgumentError):
            reg.execute(call("calculator", {}))


# ======================================================================
# search（mock）
# ======================================================================
class TestSearch:
    def test_hit(self):
        reg = create_default_registry()
        out = json.loads(reg.execute(call("search", {"query": "function calling"})))
        assert out["query"] == "function calling"
        assert len(out["results"]) >= 1
        assert "title" in out["results"][0]
        assert any("Function" in r["title"] for r in out["results"])

    def test_chinese_query(self):
        reg = create_default_registry()
        out = json.loads(reg.execute(call("search", {"query": "记忆分层"})))
        assert len(out["results"]) >= 1

    def test_empty_query(self):
        reg = create_default_registry()
        with pytest.raises(ToolArgumentError):
            reg.execute(call("search", {"query": "  "}))


# ======================================================================
# todo（session 级状态）
# ======================================================================
class TestTodo:
    def test_add_list_complete(self):
        reg = create_default_registry()
        states = {}
        out = json.loads(reg.execute(call("todo", {"action": "add", "task": "写周报"}), states))
        assert out["added"]["id"] == 1 and out["added"]["task"] == "写周报"
        reg.execute(call("todo", {"action": "add", "task": "review PR"}), states)

        out = json.loads(reg.execute(call("todo", {"action": "list"}), states))
        assert len(out["items"]) == 2 and out["pending"] == 2

        out = json.loads(reg.execute(call("todo", {"action": "complete", "id": 1}), states))
        assert out["completed"]["done"] is True

        out = json.loads(reg.execute(call("todo", {"action": "list"}), states))
        assert out["pending"] == 1

    def test_state_isolation_between_sessions(self):
        reg = create_default_registry()
        s1, s2 = {}, {}
        reg.execute(call("todo", {"action": "add", "task": "窗口1的任务"}), s1)
        reg.execute(call("todo", {"action": "add", "task": "窗口2的任务"}), s2)
        r1 = json.loads(reg.execute(call("todo", {"action": "list"}), s1))
        r2 = json.loads(reg.execute(call("todo", {"action": "list"}), s2))
        assert [i["task"] for i in r1["items"]] == ["窗口1的任务"]
        assert [i["task"] for i in r2["items"]] == ["窗口2的任务"]

    def test_errors(self):
        reg = create_default_registry()
        states = {}
        with pytest.raises(ToolArgumentError):   # add 缺 task
            reg.execute(call("todo", {"action": "add"}), states)
        with pytest.raises(ToolArgumentError):   # 非法 action
            reg.execute(call("todo", {"action": "clear"}), states)
        with pytest.raises(ToolExecutionError):  # complete 不存在 id
            reg.execute(call("todo", {"action": "complete", "id": 99}), states)


# ======================================================================
# get_weather / read_docs
# ======================================================================
class TestWeather:
    def test_deterministic(self):
        reg = create_default_registry()
        r1 = reg.execute(call("get_weather", {"city": "上海"}))
        r2 = reg.execute(call("get_weather", {"city": "上海"}))
        assert r1 == r2
        data = json.loads(r1)
        assert data["city"] == "上海" and data["mock"] is True
        assert "condition" in data and "temp_c" in data


class TestReadDocs:
    def test_read_readme(self):
        reg = create_default_registry()
        if not os.path.isfile("README.md"):
            pytest.skip("README.md 尚未生成")
        out = json.loads(reg.execute(call("read_docs", {"filename": "README.md"})))
        assert out["filename"] == "README.md" and out["length"] > 0

    def test_path_traversal_blocked(self):
        reg = create_default_registry()
        with pytest.raises(ToolExecutionError):
            reg.execute(call("read_docs", {"filename": "../main.py"}))

    def test_missing_file(self):
        reg = create_default_registry()
        with pytest.raises(ToolExecutionError):
            reg.execute(call("read_docs", {"filename": "docs/不存在的文件.md"}))

    def test_bad_extension(self):
        reg = create_default_registry()
        with pytest.raises(ToolExecutionError):
            reg.execute(call("read_docs", {"filename": "main.py"}))
