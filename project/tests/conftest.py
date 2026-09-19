"""测试公共设施：FakeLLM（脚本化 LLM）与 fixtures。

FakeLLM 与真实 OpenAICompatLLM 同构（同一 chat 接口），
通过"内部任务标记"自动应答压缩/记忆抽取请求，其余按脚本出牌。
"""
import json
from collections import deque

import pytest

from mini_agent.agent import Agent
from mini_agent.config import AgentConfig
from mini_agent.llm import LLMClient
from mini_agent.memory import MemoryStore
from mini_agent.session import SessionManager


# ----------------------------------------------------------------------
class FakeLLM(LLMClient):
    """脚本化 LLM：按顺序弹出 responses；内部任务按标记自动应答。

    responses 支持直接赋值 list（自动转为 deque）。
    每次调用记录到 self.calls（带 kind: loop/compress/extract），
    方便断言"哪些内容真的进了上下文"。
    """

    name = "fake"

    def __init__(self, responses=None, extract_response="[]",
                 summary_response="（测试摘要）用户此前添加了待办并完成了计算任务。",
                 default_final="（兜底最终回复）"):
        self._responses = deque(responses or [])
        self.extract_response = extract_response
        self.summary_response = summary_response
        self.default_final = default_final
        self.calls = []

    @property
    def responses(self):
        return self._responses

    @responses.setter
    def responses(self, value):
        self._responses = deque(value or [])

    @property
    def loop_calls(self):
        """只看主循环的 LLM 调用（排除压缩/记忆抽取等内部调用）。"""
        return [c for c in self.calls if c["kind"] == "loop"]

    def chat(self, messages, tools=None, temperature=None):
        system = "\n".join(str(m.get("content", ""))
                           for m in messages if m.get("role") == "system")
        if "对话压缩助手" in system:
            self.calls.append({"kind": "compress", "messages": list(messages),
                               "tools": tools})
            return {"role": "assistant", "content": self.summary_response}
        if "记忆提取助手" in system:
            self.calls.append({"kind": "extract", "messages": list(messages),
                               "tools": tools})
            return {"role": "assistant", "content": self.extract_response}
        self.calls.append({"kind": "loop", "messages": [dict(m) for m in messages],
                           "tools": tools})
        if self._responses:
            return self._responses.popleft()
        return {"role": "assistant", "content": self.default_final}


def tool_msg(calls, content="好的，我需要调用工具。", thinking=None):
    """构造一次（可含多个工具调用的）assistant 响应。calls: [(name, args), ...]"""
    if isinstance(calls, tuple):
        calls = [calls]
    tool_calls = [{
        "id": f"call_{i+1}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    } for i, (name, args) in enumerate(calls)]
    msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    if thinking:
        msg["reasoning_content"] = thinking
    return msg


def final_msg(text, thinking=None):
    msg = {"role": "assistant", "content": text}
    if thinking:
        msg["reasoning_content"] = thinking
    return msg


# ----------------------------------------------------------------------
@pytest.fixture
def cfg(tmp_path):
    return AgentConfig(
        data_dir=str(tmp_path / "data"),
        trace_dir=str(tmp_path / "traces"),
        max_iterations=3,
    )


@pytest.fixture
def make_agent(tmp_path, cfg):
    """Agent 工厂：make_agent(llm, cfg=..., memory=True/False)。"""
    default_cfg = cfg

    def _make(llm, cfg=None, memory=True):
        c = cfg or default_cfg
        mem = MemoryStore(str(tmp_path / "memory.json")) if memory else None
        return Agent(llm, c, SessionManager(c.data_dir), memory=mem)
    return _make


@pytest.fixture
def fake_llm():
    return FakeLLM()
