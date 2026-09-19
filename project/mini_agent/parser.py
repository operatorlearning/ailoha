"""LLM 输出解析：从 assistant message 中提取 思考过程 / 工具调用 / 最终答案。

支持两类协议（自动识别）：
1. 原生 function calling：message.tool_calls（OpenAI-compatible 标准字段）
2. 文本协议回退：模型在 content 里输出 think 块与 tool_call 块
   （标签字面量见下方 THINK_OPEN / TOOL_OPEN 常量，由片段拼接而成）

思考过程来源：
- reasoning_content 字段（GLM-4.5+ / DeepSeek-R1 等思考模型的输出）
- content 中的 think 块
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .exceptions import LLMResponseError

RAW_ARGS_KEY = "__raw_arguments__"   # 标记"模型给的工具参数不是合法 JSON"

# 标签常量（拼接构造，避免字面量）
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
TOOL_OPEN = "<" + "tool_call>"
TOOL_CLOSE = "</" + "tool_call>"

_THINK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.DOTALL)
_TOOL_TAG_RE = re.compile(
    re.escape(TOOL_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CLOSE), re.DOTALL)
_TOOL_FENCE_RE = re.compile(
    r"```(?:tool_call|tool|json)\s*\n(\s*\{.*?\})\s*\n```", re.DOTALL)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    source: str = "native"   # native / text


@dataclass
class Parsed:
    thinking: Optional[str]
    content: str                      # 去掉 think 块与工具调用块后的正文
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def need_tool_call(self) -> bool:
        return bool(self.tool_calls)

    @property
    def final_answer(self) -> str:
        return self.content.strip()


def parse_assistant_message(msg: dict) -> Parsed:
    """解析一条 assistant message -> Parsed。"""
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        raise LLMResponseError(f"期望 assistant message，得到：{str(msg)[:200]}")

    content = msg.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)

    # 1) 思考过程：reasoning_content 字段 + think 块
    thinking_parts = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        thinking_parts.append(reasoning.strip())
    block_thinking, content = _strip_think_blocks(content)
    if block_thinking:
        thinking_parts.append(block_thinking)
    thinking = "\n".join(p for p in thinking_parts if p) or None

    # 2) 工具调用：优先原生 tool_calls
    tool_calls = _parse_native_tool_calls(msg.get("tool_calls") or [])

    # 3) 无原生调用时，尝试从正文解析文本协议工具调用
    if not tool_calls:
        text_calls, content = _extract_text_tool_calls(content)
        tool_calls = text_calls

    return Parsed(thinking=thinking, content=content.strip(),
                  tool_calls=tool_calls, raw=msg)


# ----------------------------------------------------------------------
def _strip_think_blocks(content: str) -> Tuple[str, str]:
    """抽出 think 块，返回 (思考文本, 去块后的正文)。兼容未闭合的 think 块。"""
    blocks = _THINK_RE.findall(content)
    stripped = _THINK_RE.sub("", content)
    if not blocks and THINK_OPEN in stripped:
        # 未闭合：think 开始符之后全部视为思考
        head, _, tail = stripped.partition(THINK_OPEN)
        blocks = [tail]
        stripped = head
    text = "\n".join(b.strip() for b in blocks if b and b.strip())
    return text, stripped


def _parse_native_tool_calls(raw_calls) -> List[ToolCall]:
    calls = []
    for i, tc in enumerate(raw_calls):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or ""
        raw_args = fn.get("arguments", "")
        if isinstance(raw_args, dict):  # 少数网关直接给 dict
            arguments, raw_args = raw_args, json.dumps(raw_args, ensure_ascii=False)
        else:
            try:
                arguments = json.loads(raw_args) if raw_args else {}
                if not isinstance(arguments, dict):
                    arguments = {RAW_ARGS_KEY: str(raw_args)[:200]}
            except (ValueError, TypeError):
                arguments = {RAW_ARGS_KEY: str(raw_args)[:200]}
        calls.append(ToolCall(
            id=str(tc.get("id") or f"native_{i+1}"),
            name=str(name),
            arguments=arguments,
            raw_arguments=str(raw_args or ""),
            source="native",
        ))
    return calls


def _extract_text_tool_calls(content: str) -> Tuple[List[ToolCall], str]:
    """解析文本协议工具调用，返回 (calls, 去块后的正文)。"""
    spans = []
    for regex in (_TOOL_TAG_RE, _TOOL_FENCE_RE):
        for m in regex.finditer(content):
            spans.append((m.start(), m.end(), m.group(1)))

    if not spans and content.strip().startswith("{"):
        # 整个回复就是一个 JSON 对象（最宽松情形）
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and ("name" in obj or "tool" in obj):
                spans = [(0, len(content), content)]
        except ValueError:
            pass

    parsed_objs = []
    for start, end, snippet in spans:
        try:
            obj = json.loads(snippet)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or ""
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        if not name:
            continue
        if not isinstance(args, dict):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {RAW_ARGS_KEY: str(args)[:200]}
        parsed_objs.append((start, end, ToolCall(
            id=f"text_call_{len(parsed_objs)+1}",
            name=str(name), arguments=args,
            raw_arguments=json.dumps(args, ensure_ascii=False), source="text")))

    # 从正文中移除已识别的块（倒序删除保持偏移有效）
    for start, end, _ in sorted(parsed_objs, key=lambda x: -x[0]):
        content = content[:start] + content[end:]
    calls = [c for _, _, c in parsed_objs]
    return calls, content.strip()
