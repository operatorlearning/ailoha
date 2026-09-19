"""工具层：注册机制 + 内置工具 + 轻量 JSON Schema 校验。

每个工具包含：名称、描述、参数 JSON Schema、执行函数。
LLM 基于这些 Schema 自主决策调用（function calling）。

内置工具：
  calculator   精确算术计算（自实现递归下降解析器，绝不使用 eval）
  search       Mock 搜索（内置小型知识库）
  todo         待办管理（session 级状态：不同窗口的待办互相隔离）
  get_weather  Mock 天气查询（确定性伪随机）
  read_docs    读取项目文档（README / docs 目录，路径白名单校验）
"""
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exceptions import ToolArgumentError, ToolExecutionError, ToolNotFoundError
from .parser import RAW_ARGS_KEY, ToolCall

# ======================================================================
# 工具定义与注册机制
# ======================================================================

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]           # JSON Schema
    func: Callable[[Dict[str, Any], Optional[dict]], str]
    session_local: bool = False          # True: 每个session独立状态（如 todo）


class ToolRegistry:
    """工具注册中心：注册 / 查询 / Schema 导出 / 校验 / 执行。"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册：{tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def to_openai_schemas(self) -> List[dict]:
        """导出为 OpenAI-compatible tools 参数。"""
        return [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        } for t in self._tools.values()]

    # ------------------------------------------------------------------
    def execute(self, call: ToolCall, states: Optional[dict] = None) -> str:
        """执行一次工具调用。所有失败都以异常抛出，由 Agent 转成
        '错误描述'形式的工具结果回喂给 LLM（自愈），而不是崩溃。

        states: session 级工具状态表（session.tool_state）。
        session_local 工具从这里取自己的独立状态（如 todo）。
        """
        tool = self.get(call.name)
        if tool is None:
            raise ToolNotFoundError(
                f"工具不存在：{call.name}。可用工具：{', '.join(self.names())}")

        if RAW_ARGS_KEY in call.arguments:
            raise ToolArgumentError(
                f"工具 {call.name} 的参数不是合法 JSON：{call.raw_arguments[:200]}")

        errors = validate_schema(call.arguments, tool.parameters)
        if errors:
            raise ToolArgumentError(
                f"工具 {call.name} 参数校验失败：{'; '.join(errors)}")

        state = None
        if tool.session_local and states is not None:
            state = states.setdefault(tool.name, {})
        try:
            result = tool.func(call.arguments, state)
        except (ToolExecutionError, ToolArgumentError):
            raise
        except Exception as e:  # 工具内部任何异常都不击穿主循环
            raise ToolExecutionError(f"工具 {call.name} 执行出错：{type(e).__name__}: {e}")
        return result


# ======================================================================
# 轻量 JSON Schema 校验（支持本项目所需子集，零依赖）
# ======================================================================

_TYPE_MAP = {
    "object": dict, "array": list, "string": str,
    "integer": int, "number": (int, float), "boolean": bool, "null": type(None),
}


def validate_schema(obj: Any, schema: dict, path: str = "$") -> List[str]:
    """校验 obj 是否符合 schema（支持 type/required/properties/enum/items），
    返回错误信息列表（空列表 = 通过）。"""
    errors: List[str] = []
    t = schema.get("type")
    if t:
        py = _TYPE_MAP.get(t)
        if py is None:
            errors.append(f"{path}: 未知类型 {t}")
        elif isinstance(obj, bool) and t not in ("boolean",):
            # Python 的 bool 是 int 子类，需特判
            errors.append(f"{path}: 期望 {t}，实际 boolean")
        elif not isinstance(obj, py) or (t in ("integer",) and isinstance(obj, bool)):
            errors.append(f"{path}: 期望 {t}，实际 {type(obj).__name__}")
    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: {obj!r} 不在枚举 {schema['enum']} 中")
    if t == "object" and isinstance(obj, dict):
        for key in schema.get("required", []):
            if key not in obj:
                errors.append(f"{path}: 缺少必填字段 {key}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in obj:
                errors.extend(validate_schema(obj[key], sub, f"{path}.{key}"))
    if t == "array" and isinstance(obj, list) and "items" in schema:
        for i, item in enumerate(obj):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


# ======================================================================
# 查询打分（search 与 memory 召回共用）：ASCII 词项 + 中文 bigram
# ======================================================================

_ASCII_TERM_RE = re.compile(r"[a-z0-9]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def query_score(query: str, text: str) -> int:
    """query 与 text 的相关性粗打分。
    - ASCII 词项（空格分隔）命中计 3 分；
    - 中文串切 bigram，每个命中计 2 分（单个汉字时按单字 1 分）。
    """
    q = (query or "").lower()
    text = (text or "").lower()
    score = 0
    for t in re.split(r"\s+", q):
        if t and _ASCII_TERM_RE.search(t):
            score += text.count(t) * 3
    for run in _CJK_RUN_RE.findall(q):
        if len(run) == 1:
            score += text.count(run)
        else:
            score += sum(text.count(run[i:i+2]) * 2 for i in range(len(run) - 1))
    return score


# ======================================================================
# calculator：自实现词法 + 递归下降语法分析（不使用 eval）
# ======================================================================

_FUNCS = {
    "sqrt": lambda x: math.sqrt(x), "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log10, "ln": math.log,
    "pow": pow, "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<op>//|\*\*|[-+*/%^(),]))")


def _tokenize(src: str) -> List[Tuple[str, Any]]:
    tokens, pos = [], 0
    while pos < len(src):
        if src[pos].isspace():   # 跳过任意位置的白字符（含末尾）
            pos += 1
            continue
        m = _TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            raise ToolExecutionError(f"表达式含非法字符：{src[pos]!r}（位置 {pos}）")
        pos = m.end()
        if m.group("num") is not None:
            text = m.group("num")
            tokens.append(("num", float(text) if ("." in text or "e" in text.lower()) else int(text)))
        elif m.group("name") is not None:
            tokens.append(("name", m.group("name")))
        else:
            tokens.append(("op", m.group("op")))
    tokens.append(("end", None))
    return tokens


class _CalcParser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def next(self):
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expect_op(self, op):
        kind, val = self.next()
        if kind != "op" or val != op:
            raise ToolExecutionError(f"期望 '{op}'，得到 '{val}'")

    # expr := term (('+'|'-') term)*
    def expr(self):
        val = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            _, op = self.next()
            rhs = self.term()
            val = val + rhs if op == "+" else val - rhs
        return val

    # term := unary (('*'|'/'|'%'|'//') unary)*
    def term(self):
        val = self.unary()
        while self.peek()[1] in ("*", "/", "%", "//") and self.peek()[0] == "op":
            _, op = self.next()
            rhs = self.unary()
            try:
                if op == "*":
                    val = val * rhs
                elif op == "/":
                    val = val / rhs
                elif op == "%":
                    val = val % rhs
                else:
                    val = val // rhs
            except ZeroDivisionError:
                raise ToolExecutionError("除数为零")
        return val

    # unary := ('+'|'-') unary | power
    def unary(self):
        if self.peek() in (("op", "+"), ("op", "-")):
            _, op = self.next()
            val = self.unary()
            return val if op == "+" else -val
        return self.power()

    # power := primary (('**'|'^') unary)?   右结合
    def power(self):
        base = self.primary()
        if self.peek() in (("op", "**"), ("op", "^")):
            self.next()
            exp = self.unary()   # 右结合：2**3**2 = 2**(3**2)
            try:
                return base ** exp
            except (OverflowError, ZeroDivisionError) as e:
                raise ToolExecutionError(f"幂运算出错：{e}")
        return base

    def primary(self):
        kind, val = self.next()
        if kind == "num":
            return val
        if kind == "name":
            if val in _CONSTS and self.peek() != ("op", "("):
                return _CONSTS[val]
            if val in _FUNCS:
                self.expect_op("(")
                args = [self.expr()]
                while self.peek() == ("op", ","):
                    self.next()
                    args.append(self.expr())
                self.expect_op(")")
                try:
                    return _FUNCS[val](*args)
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    raise ToolExecutionError(f"函数 {val} 计算出错：{e}")
            raise ToolExecutionError(f"未知标识符：{val}")
        if (kind, val) == ("op", "("):
            val = self.expr()
            self.expect_op(")")
            return val
        raise ToolExecutionError(f"意外的符号：{val}")


def calculate(expression: str):
    tokens = _tokenize(expression)
    parser = _CalcParser(tokens)
    value = parser.expr()
    if parser.peek()[0] != "end":
        raise ToolExecutionError(f"表达式末尾有多余内容：{parser.peek()[1]!r}")
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            value = int(value)
        else:
            value = round(value, 10)
    return value


def _tool_calculator(args, _state) -> str:
    expression = str(args["expression"]).strip()
    if not expression:
        raise ToolArgumentError("expression 不能为空")
    value = calculate(expression)
    return json.dumps({"expression": expression, "result": value}, ensure_ascii=False)


# ======================================================================
# search：Mock 搜索（内置知识库 + 简单打分）
# ======================================================================

_MOCK_KB = [
    ("MiniAgent 架构", "MiniAgent 是一个从零实现的最小可用 Agent Runtime，核心组件包括："
     "LLM 客户端、输出解析器、Agent 主循环、工具注册中心、Session 管理、Context 管理与 Tracer。"
     "主循环为：接收输入 - LLM 决策 - 工具调用 - 结果判断 - 继续或返回。"),
    ("Function Calling 原理", "Function calling 让 LLM 基于 JSON Schema 描述的工具自主决策调用。"
     "模型输出 tool_calls 结构，Runtime 执行真实工具并把结果以 role=tool 消息回喂，"
     "模型再基于结果生成最终回答。OpenAI / GLM / DeepSeek 等均兼容该协议。"),
    ("ReAct 范式", "ReAct = Reasoning + Acting。模型交替输出思考(Thought)与行动(Action)，"
     "行动结果(Observation)进入上下文，循环直到得出最终答案。"
     "在缺乏原生 function calling 的模型上，可用文本标签协议模拟 ReAct。"),
    ("Context 管理与压缩", "长对话的 context 管理通常采用：滑动窗口保留近期原文 + "
     "把久远消息压缩为滚动摘要 + 关键事实单独结构化保存。压缩需在完整轮次边界切分，"
     "避免拆散 assistant 工具调用与 tool 结果的配对。"),
    ("Agent Memory 分层", "Agent 记忆一般分为：working memory（上下文窗口）、"
     "episodic memory（对话存档）、semantic memory（用户画像/事实卡）、"
     "procedural memory（操作偏好）。召回时机通常是 query-time 检索注入 system prompt。"),
    ("MCP 协议", "MCP（Model Context Protocol）是 Anthropic 提出的工具/上下文接入协议，"
     "把工具、资源、提示统一为标准接口，便于 Agent 接入外部系统，生态正在快速成长。"),
    ("TTFT 优化", "首 token 延迟（TTFT）优化手段：prompt prefix caching（KV 缓存复用）、"
     "输入预压缩、空闲预热 prefill、流式输出骨架反馈。多模态输入可先异步抽取为文本。"),
    ("Long-horizon 任务", "长程任务中模型容易遗忘目标，工程解法包括：目标卡重注入、"
     "外部计划文件（plan file）、子任务编排、检查点自校验等，混合使用效果最好。"),
]


def _score(doc: Tuple[str, str], query: str) -> int:
    return query_score(query, doc[0] + " " + doc[1])


def _tool_search(args, _state) -> str:
    query = str(args["query"]).strip()
    if not query:
        raise ToolArgumentError("query 不能为空")
    scored = sorted(((s, i, doc) for i, doc in enumerate(_MOCK_KB)
                     for s in [_score(doc, query)]), key=lambda x: (-x[0], x[1]))
    hits = [d for s, _, d in scored[:3] if s > 0]
    if not hits:
        hits = _MOCK_KB[:3]
        note = "（未找到强相关结果，返回默认条目——mock 搜索）"
    else:
        note = ""
    results = [{
        "title": t,
        "snippet": b[:120] + ("..." if len(b) > 120 else ""),
        "source": f"mock-kb://{i}",
    } for i, (t, b) in enumerate(hits)]
    return json.dumps({"query": query, "results": results, "note": note},
                      ensure_ascii=False)


# ======================================================================
# todo：session 级状态工具（多窗口隔离的关键演示）
# ======================================================================

def _tool_todo(args, state) -> str:
    state = state if state is not None else {"items": [], "next_id": 1}
    items, next_id = state.get("items", []), state.get("next_id", 1)
    action = args["action"]

    if action == "add":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ToolArgumentError("action=add 时必须提供非空 task")
        item = {"id": next_id, "task": task, "done": False,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        items.append(item)
        state["items"], state["next_id"] = items, next_id + 1
        return json.dumps({"ok": True, "added": item, "total": len(items),
                           "pending": sum(1 for x in items if not x["done"])},
                          ensure_ascii=False)

    if action == "list":
        return json.dumps({"items": items,
                           "pending": sum(1 for x in items if not x["done"])},
                          ensure_ascii=False)

    if action == "complete":
        tid = args.get("id")
        for item in items:
            if item["id"] == tid:
                item["done"] = True
                return json.dumps({"ok": True, "completed": item}, ensure_ascii=False)
        raise ToolExecutionError(f"找不到待办 id={tid}，当前共 {len(items)} 条")

    raise ToolArgumentError(f"未知 action：{action}（支持 add/list/complete）")


# ======================================================================
# get_weather：Mock 天气（按城市名确定性生成）
# ======================================================================

_W_COND = ["晴", "多云", "阴", "小雨", "雷阵雨", "小雪"]
_W_WIND = ["无风", "微风", "3-4级", "4-5级"]


def _tool_weather(args, _state) -> str:
    city = str(args["city"]).strip()
    if not city:
        raise ToolArgumentError("city 不能为空")
    h = sum(ord(c) for c in city)  # 确定性：同名城市永远同结果
    data = {
        "city": city,
        "condition": _W_COND[h % len(_W_COND)],
        "temp_c": 5 + (h % 31),
        "humidity": 40 + (h % 51),
        "wind": _W_WIND[h % len(_W_WIND)],
        "mock": True,
    }
    return json.dumps(data, ensure_ascii=False)


# ======================================================================
# read_docs：读取项目文档（路径白名单，防目录穿越）
# ======================================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALLOWED_EXT = {".md", ".txt"}


def _tool_read_docs(args, _state) -> str:
    filename = str(args["filename"]).strip().replace("\\", "/")
    if not filename:
        raise ToolArgumentError("filename 不能为空")
    base = os.path.normpath(_PROJECT_ROOT)
    target = os.path.normpath(os.path.join(base, filename))
    if not (target == base or target.startswith(base + os.sep)):
        raise ToolExecutionError(f"非法路径：{filename}（只允许项目目录内）")
    if os.path.splitext(target)[1].lower() not in _ALLOWED_EXT:
        raise ToolExecutionError(f"只支持读取 .md/.txt 文件，收到：{filename}")
    if not os.path.isfile(target):
        raise ToolExecutionError(
            f"文件不存在：{filename}。可用：README.md 或 docs/ 目录下的文档")
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(6000)   # 单次最多读取，进一步截断交给 context 层
    except OSError as e:
        raise ToolExecutionError(f"读取文件失败：{e}")
    return json.dumps({"filename": filename, "length": len(text), "content": text},
                      ensure_ascii=False)


# ======================================================================
# 默认注册表
# ======================================================================

def create_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="calculator",
        description="精确算术计算器。支持 + - * / // % ** ^ 括号，"
                    "以及 sqrt/abs/round/floor/ceil/sin/cos/tan/log/ln/pow/min/max 和常量 pi/e。"
                    "任何数学计算都必须使用本工具，禁止心算。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "算术表达式，例如 (1+2)*3.5"},
            },
            "required": ["expression"],
        },
        func=_tool_calculator,
    ))
    reg.register(Tool(
        name="search",
        description="搜索知识库（当前为 mock 数据，覆盖 Agent/LLM 相关主题），"
                    "返回最相关的条目标题与摘要。用于回答资料类问题。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
        func=_tool_search,
    ))
    reg.register(Tool(
        name="todo",
        description="管理当前会话的待办清单。action=add 新增（需 task），"
                    "action=list 列出全部，action=complete 完成（需 id）。"
                    "待办保存在当前 session 中，跨轮次有效。",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "complete"],
                           "description": "待办操作"},
                "task": {"type": "string", "description": "add 时必填：任务内容"},
                "id": {"type": "integer", "description": "complete 时必填：待办编号"},
            },
            "required": ["action"],
        },
        func=_tool_todo,
        session_local=True,
    ))
    reg.register(Tool(
        name="get_weather",
        description="查询指定城市的当前天气（mock 数据，仅用于演示）。",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，例如 北京"},
            },
            "required": ["city"],
        },
        func=_tool_weather,
    ))
    reg.register(Tool(
        name="read_docs",
        description="读取本项目内的文档内容（README.md 或 docs/ 目录下 .md/.txt），"
                    "用于回答关于本项目的问题。",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "相对项目根的文件路径"},
            },
            "required": ["filename"],
        },
        func=_tool_read_docs,
    ))
    return reg
