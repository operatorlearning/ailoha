"""真实 LLM API 集成测试（可选）。

默认跳过；设置环境变量后运行::

    set LLM_API_KEY=xxx
    set LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 可选，默认 GLM
    set LLM_MODEL=glm-4.6                                    # 可选
    pytest -m integration tests/test_integration_llm.py
"""
import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("LLM_API_KEY"),
                       reason="未设置 LLM_API_KEY，跳过真实 API 集成测试"),
]

from mini_agent.agent import Agent
from mini_agent.config import load_agent_config, load_llm_config
from mini_agent.llm import OpenAICompatLLM
from mini_agent.memory import MemoryStore
from mini_agent.session import SessionManager


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("integration")
    cfg = load_agent_config(data_dir=str(tmp / "data"),
                            trace_dir=str(tmp / "traces"))
    return Agent(OpenAICompatLLM(load_llm_config()), cfg,
                 SessionManager(cfg.data_dir),
                 memory=MemoryStore(str(tmp / "memory.json")))


def test_direct_reply(agent):
    s = agent.new_session("it-direct")
    result = agent.chat(s, "用一句话介绍什么是 function calling")
    assert result.error is None
    assert len(result.answer) > 10
    assert result.tool_calls == []


def test_tool_roundtrip_calculator(agent):
    s = agent.new_session("it-calc")
    result = agent.chat(s, "请严格使用工具计算 (1204+96)/2，然后告诉我结果")
    assert result.error is None
    assert any(tc["name"] == "calculator" and tc["ok"] for tc in result.tool_calls)
    assert "650" in result.answer


def test_session_followup(agent):
    s = agent.new_session("it-followup")
    r1 = agent.chat(s, "记住这个暗号：蓝色橘子")
    assert r1.error is None
    r2 = agent.chat(s, "我刚才说的暗号是什么？只回答暗号本身")
    assert "蓝色橘子" in r2.answer
