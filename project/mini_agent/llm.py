"""LLM 客户端：手写的 OpenAI-compatible chat/completions HTTP 调用。

- 不使用任何 SDK / agent 框架，只用 requests 直连 REST API；
- 适用于 GLM / DeepSeek / OpenAI / Qwen / Kimi / 本地 vLLM 等任何
  兼容 OpenAI Chat Completions 协议的服务；
- 返回"原始 assistant message dict"，解析交给 parser 模块；
- MockLLM 提供离线演示与测试替身（真实主循环完全一致，仅替换 LLM）。
"""
import json
import re
import time
from typing import Dict, List, Optional

import requests

from .config import LLMConfig
from .exceptions import LLMAPIError, LLMResponseError

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMClient:
    """LLM 抽象接口：输入 messages（+tools schema），输出 assistant message dict。"""

    name = "base"

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None,
             temperature: Optional[float] = None) -> dict:
        raise NotImplementedError

    def usage_stats(self) -> Dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


class OpenAICompatLLM(LLMClient):
    """真实 API 客户端：requests + 重试退避。"""

    name = "openai_compat"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    # ------------------------------------------------------------------
    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None,
             temperature: Optional[float] = None) -> dict:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=self.cfg.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = f"网络错误：{type(e).__name__}: {e}"
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt * 0.8, 6))
                    continue
                raise LLMAPIError(f"LLM 请求失败（已重试{self.cfg.max_retries}次）：{last_err}")
            except requests.RequestException as e:
                raise LLMAPIError(f"LLM 请求异常：{e}")

            if resp.status_code == 200:
                return self._extract_message(resp)

            if resp.status_code in RETRYABLE_STATUS and attempt < self.cfg.max_retries:
                time.sleep(min(2 ** attempt * 0.8, 6))
                continue
            # 不可重试错误：尽量带上服务端返回的错误信息
            detail = ""
            try:
                detail = str(resp.json().get("error", resp.text[:300]))
            except Exception:
                detail = resp.text[:300]
            raise LLMAPIError(
                f"LLM API 返回 {resp.status_code}：{detail}", {"status": resp.status_code}
            )
        raise LLMAPIError(f"LLM 请求失败：{last_err or 'unknown'}")

    # ------------------------------------------------------------------
    def _extract_message(self, resp: requests.Response) -> dict:
        try:
            data = resp.json()
        except ValueError as e:
            raise LLMResponseError(f"LLM 返回非 JSON 内容：{e}")
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMResponseError(f"LLM 返回结构异常：{e}；body={str(data)[:300]}")
        # usage 统计（可选字段）
        usage = data.get("usage") or {}
        self._usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self._usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        self._usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        self._usage["calls"] += 1
        if not isinstance(msg, dict):
            raise LLMResponseError("LLM message 不是 dict")
        return msg

    def usage_stats(self):
        return dict(self._usage)


# ======================================================================
# Mock LLM：离线演示 / 测试替身。行为与真实客户端完全同构。
# ======================================================================
_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "武汉", "南京", "重庆", "天津"]
_ARITH_RE = re.compile(r"[-+*/%().\d\s]{3,}")


def _extract_arithmetic(text: str) -> str:
    m = _ARITH_RE.findall(text)
    if not m:
        return "1+1"
    return max(m, key=len).strip()


class MockLLM(LLMClient):
    """关键词规则驱动的假 LLM：走同一套主循环，无需 API key。

    规则（针对最后一条用户消息）：
      含"天气"      -> 调 weather 工具
      含"计算/算"   -> 调 calculator 工具
      含"待办/todo" -> 调 todo 工具（add/list）
      含"搜索/搜"   -> 调 search 工具
      含"文档"      -> 调 read_docs 工具
      其余          -> 直接回复
      上一条是工具结果 -> 用工具结果组织最终回答
    压缩/记忆提取等内部请求按固定格式返回。
    """

    name = "mock"

    def __init__(self, think_tag: bool = False):
        self.think_tag = think_tag  # True 时在回复里附带 <think> 块，用于演示思考过程解析

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None,
             temperature: Optional[float] = None) -> dict:
        system = ""
        for m in messages:
            if m.get("role") == "system":
                system += "\n" + str(m.get("content", ""))
        # 内部任务：对话压缩
        if "压缩" in system and "摘要" in system:
            return self._final("（Mock摘要）此前对话中用户添加了待办与日历事项，助手通过工具完成并给出了结果。")
        # 内部任务：记忆抽取
        if "记忆提取" in system:
            return self._final("[]")

        last = messages[-1] if messages else {}
        # 工具结果 -> 基于结果回答
        if last.get("role") == "tool":
            tool_name = last.get("name", "tool")
            result = str(last.get("content", ""))[:200]
            return self._final(f"（Mock回复）{tool_name} 的执行结果是：{result}")
        # 无工具直接答
        user_text = str(last.get("content", "")) if last.get("role") == "user" else ""
        if not user_text:
            return self._final("（Mock回复）你好，我是 MockLLM 驱动的 MiniAgent。")
        return self._route(user_text)

    # ------------------------------------------------------------------
    def _final(self, text: str) -> dict:
        content = f"<think>mock thinking</think>{text}" if self.think_tag else text
        return {"role": "assistant", "content": content}

    def _tool_call(self, name: str, arguments: dict, call_id: str = "mock_call_1") -> dict:
        return {
            "role": "assistant",
            "content": "好的，我需要调用工具来处理。",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }],
        }

    def _route(self, text: str) -> dict:
        if "天气" in text:
            city = next((c for c in _CITIES if c in text), "北京")
            return self._tool_call("get_weather", {"city": city})
        if "计算" in text or "算算" in text or re.search(r"\d\s*[-+*/]\s*\d", text):
            return self._tool_call("calculator", {"expression": _extract_arithmetic(text)})
        if "待办" in text or "todo" in text.lower():
            if "列表" in text or "查看" in text or "清单" in text:
                return self._tool_call("todo", {"action": "list"})
            task = re.sub(r"^(帮我)?(添加|加个?|记一下|新增)?(待办|todo)[:：,，]?\s*", "", text).strip() or text
            return self._tool_call("todo", {"action": "add", "task": task})
        if "搜索" in text or "搜一下" in text or "查查" in text:
            q = re.sub(r"^(帮我)?(搜索|搜一下|查查)[:：,，]?\s*", "", text).strip() or "agent"
            return self._tool_call("search", {"query": q})
        if "文档" in text or "README" in text.upper():
            return self._tool_call("read_docs", {"filename": "README.md"})
        return self._final(f"（Mock回复）收到：{text}（离线演示模式，接入真实 API 可获得完整能力）")

    def usage_stats(self):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
