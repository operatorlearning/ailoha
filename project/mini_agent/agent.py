"""Agent 核心主循环（本项目的"Runtime"心脏，完全手写，不依赖 agent 框架）。

    ┌────────────────────────── Agent.chat(session, user_input) ─────────────────────┐
    │                                                                                │
    │  1. 记录 user_input（session + trace）                                          │
    │  2. memory 召回（query-time，注入 system prompt）                               │
    │  3. LOOP（最多 max_iterations 轮）:                                             │
    │     a. build_messages(system + 摘要 + 近期原文)                                 │
    │     b. LLM 决策（携带工具 Schema）                                               │
    │     c. 解析输出 -> thinking / tool_calls / final_answer                          │
    │     d. 无 tool_calls -> 记录最终回答，返回用户                                    │
    │     e. 有 tool_calls -> 逐个执行（异常转为错误结果回喂），继续 LOOP               │
    │  4. 超过轮次上限 -> 强制一次"无工具"收尾调用                                      │
    │  5. 每轮结束：长期记忆抽取(best-effort) -> context 压缩 -> session 落盘          │
    │                                                                                │
    └────────────────────────────────────────────────────────────────────────────────┘
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import __version__
from .config import AgentConfig, load_agent_config
from .context import ContextManager, truncate_text
from .exceptions import LLMAPIError, LLMResponseError, MiniAgentError
from .llm import LLMClient
from .memory import MemoryStore, extract_memories
from .parser import parse_assistant_message
from .session import Session, SessionManager, make_api_tool_calls
from .tools import ToolRegistry, create_default_registry
from .tracing import Tracer

_FORCED_FINAL_PROMPT = (
    "（系统提示：已达到本轮工具调用次数上限，请基于已获得的信息直接给出结论，"
    "不要再调用任何工具。）")


@dataclass
class AgentResult:
    answer: str
    error: Optional[str] = None
    iterations: int = 0
    tool_calls: List[dict] = field(default_factory=list)   # {name, ok, elapsed_ms}
    trace_path: Optional[str] = None


class Agent:
    def __init__(self, llm: LLMClient, cfg: Optional[AgentConfig] = None,
                 session_manager: Optional[SessionManager] = None,
                 registry: Optional[ToolRegistry] = None,
                 memory: Optional[MemoryStore] = None):
        self.llm = llm
        self.cfg = cfg or load_agent_config()
        self.registry = registry or create_default_registry()
        self.session_manager = session_manager or SessionManager(self.cfg.data_dir)
        self.memory = memory
        self.ctx = ContextManager(self.cfg, self.registry, llm=llm)

    # ------------------------------------------------------------------
    def new_session(self, session_id: Optional[str] = None) -> Session:
        return self.session_manager.get_or_create(session_id)

    def get_session(self, session_id: str) -> Session:
        return self.session_manager.load(session_id)

    # ==================================================================
    # 主入口
    # ==================================================================
    def chat(self, session: Session, user_input: str) -> AgentResult:
        tracer = Tracer(self.cfg.trace_dir, session.id, self.cfg.trace_also_stdout)
        tracer.turn = session.user_rounds()
        tracer.new_turn()
        tracer.log("user_input", session=session.id, version=__version__,
                   text=truncate_text(user_input, 500))
        session.add_user(user_input)

        result = AgentResult(answer="", trace_path=tracer.path)
        try:
            answer = self._loop(session, user_input, tracer, result)
            result.answer = answer
            self._maybe_remember(session, user_input, answer, tracer)
        except (LLMAPIError, LLMResponseError) as e:
            result.error = f"LLM 服务异常：{e}"
            result.answer = f"抱歉，{result.error}请稍后重试或检查 API 配置。"
            tracer.log("error", stage="llm", type=type(e).__name__, message=str(e))
        except MiniAgentError as e:
            result.error = str(e)
            result.answer = f"抱歉，处理时出现问题：{e}"
            tracer.log("error", stage="loop", type=type(e).__name__, message=str(e))
        except Exception as e:  # 兜底：runtime 自身缺陷也不向用户抛栈
            result.error = f"{type(e).__name__}: {e}"
            result.answer = "抱歉，发生未预期的内部错误，请重试。"
            tracer.log("error", stage="runtime", type=type(e).__name__, message=str(e))

        # 无论成败：压缩 + 落盘（保证崩溃后也能恢复会话）
        self.ctx.tracer = tracer
        try:
            if self.ctx.maybe_compress(session):
                pass  # compress 事件已在 Tracer 内记录
        except Exception as e:
            tracer.log("error", stage="compress", message=str(e))
        try:
            self.session_manager.save(session)
            tracer.log("session_saved", rounds=session.user_rounds(),
                       messages=len(session.messages), compressed=bool(session.summary))
        except Exception as e:
            tracer.log("error", stage="save", message=str(e))
        return result

    # ==================================================================
    # 主循环
    # ==================================================================
    def _loop(self, session: Session, user_input: str, tracer: Tracer,
              result: AgentResult) -> str:
        hint = ""
        if self.memory is not None:
            hint = self.memory.render_hint(user_input, self.cfg.memory_top_k)
            if hint:
                tracer.log("memory_recall", hint=truncate_text(hint, 300))

        schemas = self.registry.to_openai_schemas()

        for iteration in range(1, self.cfg.max_iterations + 1):
            result.iterations = iteration
            messages = self.ctx.build_messages(session, hint)

            # Step 2: LLM 决策（直接回复 or 调用工具）
            tracer.next_step()
            tracer.log("llm_request", model=self._model_name(),
                       messages=self._compact(messages))
            raw = self.llm.chat(messages, tools=schemas)

            # Step 2b: 解析（思考过程 / 工具调用 / 最终答案）
            parsed = parse_assistant_message(raw)
            tracer.next_step()
            tracer.log("llm_response",
                       thinking=truncate_text(parsed.thinking or "", 200),
                       content=truncate_text(parsed.content, 300),
                       tool_calls=[{"name": c.name, "arguments": c.arguments}
                                   for c in parsed.tool_calls])

            if not parsed.need_tool_call:
                # Step 4b: 返回结果给用户
                answer = parsed.final_answer or "（模型返回了空回复，请换个问法重试）"
                session.add_assistant(content=answer, thinking=parsed.thinking)
                tracer.next_step()
                tracer.log("final_answer", iteration=iteration,
                           text=truncate_text(answer, 500))
                return answer

            # Step 3: 调用工具（assistant 的调用意图进入上下文）
            session.add_assistant(
                content=parsed.content or None,
                tool_calls_api=make_api_tool_calls(parsed.tool_calls),
                thinking=parsed.thinking)
            for tc in parsed.tool_calls:
                self._exec_tool(session, tc, tracer, result)
            # Step 4a: 携带工具结果继续 loop

        tracer.log("max_iterations", limit=self.cfg.max_iterations)
        return self._force_final(session, hint, tracer)

    # ------------------------------------------------------------------
    def _exec_tool(self, session: Session, tc, tracer: Tracer, result: AgentResult):
        t0 = time.time()
        tracer.next_step()
        tracer.log("tool_call", id=tc.id, name=tc.name, arguments=tc.arguments)
        ok, err = True, None
        try:
            output = self.registry.execute(tc, session.tool_state)
        except MiniAgentError as e:
            ok, err, output = False, str(e), f"工具执行错误：{e}"
            tracer.next_step()
            tracer.log("error", stage="tool", tool=tc.name, message=str(e))
        elapsed = round((time.time() - t0) * 1000)
        # 完整结果进 trace；截断版进入上下文
        session.add_tool_result(tc.id, tc.name,
                                truncate_text(output, self.cfg.max_tool_result_chars))
        result.tool_calls.append({"name": tc.name, "ok": ok, "elapsed_ms": elapsed})
        tracer.next_step()
        tracer.log("tool_result", id=tc.id, name=tc.name, ok=ok,
                   elapsed_ms=elapsed, result=truncate_text(output, 400))
        return output

    # ------------------------------------------------------------------
    def _force_final(self, session: Session, hint: str, tracer: Tracer) -> str:
        """达到轮次上限后的兜底：不带工具强收尾。"""
        messages = self.ctx.build_messages(session, hint)
        messages.append({"role": "user", "content": _FORCED_FINAL_PROMPT})
        try:
            raw = self.llm.chat(messages, tools=None)
            parsed = parse_assistant_message(raw)
            answer = parsed.final_answer or "我已达到本轮工具调用上限，请开新的一轮继续。"
        except MiniAgentError as e:
            tracer.log("error", stage="force_final", message=str(e))
            answer = "我已达到本轮工具调用上限，暂时无法收尾，请重新描述需求。"
        session.add_assistant(content=answer)
        tracer.log("final_answer", forced=True, text=truncate_text(answer, 500))
        return answer

    # ------------------------------------------------------------------
    def _maybe_remember(self, session: Session, user_input: str,
                        answer: str, tracer: Tracer):
        if self.memory is None:
            return
        try:
            facts = extract_memories(self.llm, user_input, answer)
            added = [f for f in facts if self.memory.add(f, source_session=session.id)]
            if added:
                tracer.log("memory_extracted", facts=added)
        except Exception as e:
            tracer.log("error", stage="memory", message=str(e))

    # ------------------------------------------------------------------
    def _model_name(self) -> str:
        cfg = getattr(self.llm, "cfg", None)
        return getattr(cfg, "model", self.llm.name)

    @staticmethod
    def _compact(messages) -> list:
        """trace 用的消息摘要（避免日志爆炸）。"""
        out = []
        for m in messages:
            item = {"role": m.get("role"), "content": truncate_text(str(m.get("content") or ""), 120)}
            if m.get("tool_calls"):
                item["tool_calls"] = [
                    tc.get("function", {}).get("name") for tc in m["tool_calls"]]
            out.append(item)
        return out
