"""配置：从环境变量加载，全部有默认值（api_key 除外）。

用法（真实 API，任何 OpenAI-compatible 服务均可）::

    # GLM（默认）
    set LLM_API_KEY=你的key
    set LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
    set LLM_MODEL=glm-4.6

    # 或 DeepSeek / OpenAI / Qwen / 本地 vLLM ...
    set LLM_BASE_URL=https://api.deepseek.com/v1
    set LLM_MODEL=deepseek-chat
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from .exceptions import LLMAPIError


DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4.6"


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    timeout: float = 60.0
    max_retries: int = 2


@dataclass
class AgentConfig:
    # --- 主循环 ---
    max_iterations: int = 8            # 单次用户输入内，LLM 循环最大轮数（防死循环）
    # --- context 管理 ---
    max_history_messages: int = 40     # session 内消息条数超过则触发压缩
    keep_recent_messages: int = 16     # 压缩时保留原文的最近消息条数（完整轮次边界）
    compress_trigger_tokens: int = 4500  # 估算 token 超过则触发压缩
    max_context_tokens: int = 6000     # build_messages 硬预算（超出再截断最老消息）
    max_tool_result_chars: int = 2000  # 单条工具结果最大字符数
    # --- memory ---
    memory_top_k: int = 4
    # --- 落盘 ---
    data_dir: str = "data"
    trace_dir: str = "logs/traces"
    trace_also_stdout: bool = False
    extra: dict = field(default_factory=dict)


def load_llm_config(env=None) -> LLMConfig:
    env = env if env is not None else os.environ
    api_key = env.get("LLM_API_KEY", "")
    if not api_key:
        raise LLMAPIError(
            "缺少 LLM_API_KEY 环境变量。请设置后重试，"
            "（或使用 `python main.py --mock` 体验离线演示模式）"
        )
    return LLMConfig(
        api_key=api_key,
        base_url=env.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        model=env.get("LLM_MODEL", DEFAULT_MODEL),
        temperature=float(env.get("LLM_TEMPERATURE", "0.7")),
        timeout=float(env.get("LLM_TIMEOUT", "60")),
        max_retries=int(env.get("LLM_MAX_RETRIES", "2")),
    )


def load_agent_config(env=None, **overrides) -> AgentConfig:
    env = env if env is not None else os.environ
    cfg = AgentConfig(
        max_iterations=int(env.get("AGENT_MAX_ITERATIONS", "8")),
        max_history_messages=int(env.get("AGENT_MAX_HISTORY_MESSAGES", "40")),
        keep_recent_messages=int(env.get("AGENT_KEEP_RECENT_MESSAGES", "16")),
        compress_trigger_tokens=int(env.get("AGENT_COMPRESS_TRIGGER_TOKENS", "4500")),
        max_context_tokens=int(env.get("AGENT_MAX_CONTEXT_TOKENS", "6000")),
        max_tool_result_chars=int(env.get("AGENT_MAX_TOOL_RESULT_CHARS", "2000")),
        memory_top_k=int(env.get("AGENT_MEMORY_TOP_K", "4")),
        data_dir=env.get("AGENT_DATA_DIR", "data"),
        trace_dir=env.get("AGENT_TRACE_DIR", "logs/traces"),
    )
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
