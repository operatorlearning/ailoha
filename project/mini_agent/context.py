"""Context 管理：决定"什么信息进入上下文"以及"上下文过长时如何压缩"。

进入上下文的信息（决策与理由见 README 系统设计一节）：
  - system prompt（角色/规则/工具一览/当前时间/长期记忆召回）
  - 滚动摘要（压缩产物，若有）
  - 近期消息原文：用户输入 / assistant 工具调用 / 工具结果（截断）/ assistant 最终回答
不进入上下文（只进 trace 与 session 存档）：
  - assistant 的完整思考过程（省 token；部分 API 也不接受历史中的 reasoning 字段）

预算控制：
  - max_history_messages：消息条数上限（触发压缩）
  - compress_trigger_tokens：token 估算上限（触发压缩）
  - max_context_tokens：build 时的硬预算（超出则丢弃最老轮次）
压缩策略（基础版）：
  - 只压缩"久远"消息，保留最近 keep_recent_messages 条原文
  - 在完整轮次边界切分（user 消息开头），保证 assistant 工具调用与
    tool 结果配对不被拆散
  - 旧消息交给 LLM 生成滚动摘要；压缩失败则保持原状（宁可不压也不丢信息）
"""
import re
import time
from typing import List, Optional

from .config import AgentConfig
from .session import Message, Session
from .tools import ToolRegistry

# 中文按 1 字 ≈ 1 token，其它按 4 字符 ≈ 1 token 的粗略启发式
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + (other // 4) + (1 if other % 4 else 0)


def messages_tokens(msgs) -> int:
    total = 0
    for m in msgs:
        if isinstance(m, Message):
            total += estimate_tokens(m.content or "")
            if m.tool_calls:
                total += estimate_tokens(str(m.tool_calls))
        elif isinstance(m, dict):
            total += estimate_tokens(str(m.get("content") or ""))
    return total


def truncate_text(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"……（已截断，原长 {len(s)} 字符）"


SUMMARY_PREFIX = "[对话摘要] 以下是本 session 更早对话的滚动摘要，请结合它理解上下文：\n"

_SYSTEM_TEMPLATE = """你是 MiniAgent，一个务实、简洁的中文智能助手，可以调用工具帮助用户完成任务。

可用工具（详细参数 Schema 已通过 API 下发）：
{tool_lines}

工作规则：
1. 涉及精确计算、资料检索、待办管理、天气查询、项目文档阅读的任务，必须调用对应工具完成，不要凭记忆回答数字或事实。
2. 工具返回错误时，请阅读错误信息并修正参数重试，或换一种方式；禁止编造工具结果。
3. 无需工具时直接、简洁地回答；有先前的对话摘要或历史消息时，结合它们理解用户意图（支持追问）。
4. 当前时间：{now}。

（备用协议）若当前接口不支持 function calling，可在正文使用成对的 tool_call 标签（或 tool_call 代码块）包裹 JSON 发起调用，例如：
```tool_call
{{"name": "工具名", "arguments": {{"参数": "值"}}}}
```
思考过程请放在 think 块或 reasoning 字段中，与最终答案分开。"""

_MEMORY_HINT_TEMPLATE = """
[长期记忆召回]（来自更早会话的用户画像/事实，仅供参考；如与当前对话冲突，以当前对话为准）：
{mem_lines}"""

_SUMMARIZE_SYSTEM = """你是对话压缩助手。请把下面的对话历史压缩成一段简洁的中文滚动摘要，供后续对话延续上下文使用。要求：
1. 保留：用户目标与偏好、关键决策、重要数字/时间/人名/地名、工具调用及其结果要点、未完成事项；
2. 第三人称陈述，不要逐轮复述，不要对话痕迹；
3. 不超过 300 字；直接输出摘要正文。"""

_COMPRESS_MARKER = "[已有摘要]"


def _render_for_summary(msgs: List[Message]) -> str:
    lines = []
    for m in msgs:
        if m.role == "user":
            lines.append(f"user: {truncate_text(m.content or '', 300)}")
        elif m.role == "assistant":
            if m.tool_calls:
                for tc in m.tool_calls:
                    fn = tc.get("function", {})
                    lines.append(f"assistant 调用 {fn.get('name')} 参数 {fn.get('arguments')}")
            if m.content:
                lines.append(f"assistant: {truncate_text(m.content, 300)}")
        elif m.role == "tool":
            lines.append(f"tool[{m.name}] 结果: {truncate_text(m.content or '', 300)}")
    return "\n".join(lines)


class ContextManager:
    def __init__(self, cfg: AgentConfig, registry: ToolRegistry, llm=None, tracer=None):
        self.cfg = cfg
        self.registry = registry
        self.llm = llm
        self.tracer = tracer

    # ------------------------------------------------------------------
    def system_prompt(self, memory_hint: str = "") -> str:
        tool_lines = "\n".join(
            f"- {t.name}: {t.description.split('。')[0]}。" for t in self.registry.all())
        prompt = _SYSTEM_TEMPLATE.format(
            tool_lines=tool_lines,
            now=time.strftime("%Y-%m-%d %H:%M %A"))
        if memory_hint:
            prompt += memory_hint
        return prompt

    def build_messages(self, session: Session, memory_hint: str = "") -> List[dict]:
        """构造发给 LLM 的完整 messages（system + 摘要 + 近期原文）。"""
        msgs: List[dict] = [{"role": "system", "content": self.system_prompt(memory_hint)}]
        if session.summary:
            msgs.append({"role": "system", "content": SUMMARY_PREFIX + session.summary})

        recent = session.messages[-self.cfg.max_history_messages:]
        recent = self._align_to_user_boundary(recent)
        if not recent and session.messages:
            # 窗口容不下任何完整轮次：退化为保留最后一整轮
            user_positions = [i for i, m in enumerate(session.messages)
                              if m.role == "user"]
            if user_positions:
                recent = session.messages[user_positions[-1]:]
            else:
                recent = session.messages[-self.cfg.max_history_messages:]

        # 硬预算：估算总 token 超限时从最老轮次开始丢弃（丢整轮）
        budget = self.cfg.max_context_tokens - estimate_tokens(
            msgs[0]["content"]) - estimate_tokens(msgs[1]["content"] if len(msgs) > 1 else "")
        while messages_tokens(recent) > max(budget, 500) and len(recent) > 1:
            trimmed, ok = self._drop_oldest_round(recent)
            if not ok:
                break
            recent = trimmed

        for m in recent:
            api = m.to_api()
            if m.role == "tool" and m.content:
                api["content"] = truncate_text(m.content, self.cfg.max_tool_result_chars)
            msgs.append(api)
        return msgs

    # ------------------------------------------------------------------
    def maybe_compress(self, session: Session) -> bool:
        """超过阈值则压缩久远消息为滚动摘要。返回是否发生了压缩。"""
        n = len(session.messages)
        est = messages_tokens(session.messages)
        if n <= self.cfg.max_history_messages and est <= self.cfg.compress_trigger_tokens:
            return False
        if n <= self.cfg.keep_recent_messages:
            return False  # 全部都是"近期"，没什么可压

        # 切分点：从"倒数第 keep_recent 条"前移到最近的 user 消息（完整轮次边界）
        n_msg = len(session.messages)
        idx = max(0, n_msg - self.cfg.keep_recent_messages)
        while idx < n_msg and session.messages[idx].role != "user":
            idx += 1
        if idx >= n_msg:
            # 尾部窗口里找不到轮次边界：退化为保留最后一整轮
            user_positions = [i for i, m in enumerate(session.messages)
                              if m.role == "user"]
            if not user_positions:
                return False
            idx = user_positions[-1]
        if idx == 0:
            return False  # 整个历史都是近期窗口
        older, recent = session.messages[:idx], session.messages[idx:]
        if not older:
            return False

        summary = self._summarize(older, session.summary)
        if not summary:
            if self.tracer:
                self.tracer.log("compress", ok=False, reason="摘要生成失败，保留原文")
            return False

        session.archive.extend(m.to_dict() for m in older)
        if len(session.archive) > 400:          # 审计存档也设上限
            session.archive = session.archive[-400:]
        session.summary = summary
        session.messages = recent
        session.touch()
        if self.tracer:
            self.tracer.log("compress", ok=True, archived=len(older),
                            kept=len(recent), summary=truncate_text(summary, 120))
        return True

    # ------------------------------------------------------------------
    def _summarize(self, older: List[Message], prev_summary: str) -> str:
        if self.llm is None:
            return ""
        user_content = _COMPRESS_MARKER + "\n" + (prev_summary or "（无）") + \
            "\n[对话历史]\n" + _render_for_summary(older)
        messages = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        try:
            msg = self.llm.chat(messages)   # 压缩调用不带工具
            content = msg.get("content") or ""
            return content.strip()[:600]
        except Exception:
            return ""   # 压缩失败：宁可不压，也不丢历史

    # ------------------------------------------------------------------
    @staticmethod
    def _align_to_user_boundary(msgs: List[Message]) -> List[Message]:
        """丢弃头部残轮（从非 user 消息开始的部分），保证窗口从完整轮次开始。"""
        i = 0
        while i < len(msgs) and msgs[i].role != "user":
            i += 1
        return msgs[i:]

    @staticmethod
    def _drop_oldest_round(msgs: List[Message]) -> tuple:
        user_idx = [i for i, m in enumerate(msgs) if m.role == "user"]
        if len(user_idx) < 2:
            return msgs, False
        return msgs[user_idx[1]:], True
