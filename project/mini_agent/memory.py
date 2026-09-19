"""长期记忆（跨 session）：轻量关键词召回 + LLM 事实抽取。

定位：与 per-session 的 context（近期原文 + 滚动摘要）互补，
memory 存的是跨会话仍然有效的"用户画像 / 偏好 / 事实"。

召回时机：每次用户输入后、build_messages 之前（query-time recall）。
放置方式：注入 system prompt 尾部（明确标注"仅供参考，冲突以当前对话为准"）。
写入时机：每轮结束后的 best-effort 抽取（失败不影响主流程）。

本实现刻意保持简单（JSON 文件 + 关键词打分），向量检索的演进方向见 README。
"""
import json
import os
import re
import time
from typing import List, Optional

from .context import _MEMORY_HINT_TEMPLATE
from .tools import query_score

_EXTRACT_SYSTEM = """你是记忆提取助手。从下面这轮用户输入与助手回答中，提取值得长期记住的用户事实或偏好。
要求：
1. 只提取跨会话仍有价值的信息（用户身份、偏好、约定、长期目标），忽略一次性任务细节；
2. 每条一句中文短句（不超过 40 字）；
3. 以 JSON 数组输出，无值得记住的信息则输出 []；
4. 直接输出 JSON，不要任何解释。"""


class MemoryStore:
    """JSON 文件存储的长期记忆（生产可换 SQLite / 向量库）。"""

    def __init__(self, path: str = "data/memory.json", max_items: int = 200):
        self.path = path
        self.max_items = max_items
        self.items: List[dict] = []
        self._load()

    def _load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.items = data
            except (OSError, ValueError):
                self.items = []

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    def add(self, content: str, source_session: str = "") -> bool:
        content = (content or "").strip()
        if not content or len(content) > 100:
            return False
        if any(i["content"] == content for i in self.items):   # 内容去重
            return False
        self.items.append({
            "content": content,
            "ts": time.strftime("%Y-%m-%d %H:%M"),
            "session": source_session,
        })
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        self._save()
        return True

    # ------------------------------------------------------------------
    def recall(self, query: str, top_k: int = 4) -> List[dict]:
        """关键词打分召回（ASCII 词项 + 中文 bigram，复用 tools.query_score），
        分数 > 0 才算命中。"""
        if not self.items or not query.strip():
            return []

        def score(item) -> int:
            return query_score(query, item["content"])

        scored = sorted(((score(i), i) for i in self.items),
                        key=lambda x: -x[0])
        return [i for s, i in scored[:top_k] if s > 0]

    def render_hint(self, query: str, top_k: int = 4) -> str:
        hits = self.recall(query, top_k)
        if not hits:
            return ""
        lines = "\n".join(f"- {h['content']}（{h['ts']}）" for h in hits)
        return _MEMORY_HINT_TEMPLATE.format(mem_lines=lines)


def extract_memories(llm, user_text: str, assistant_text: str) -> List[str]:
    """让 LLM 从一轮对话里抽取值得长期记住的事实。任何失败都返回 []。"""
    if not llm:
        return []
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": f"[用户输入]\n{user_text[:500]}\n[助手回答]\n{assistant_text[:500]}"},
    ]
    try:
        msg = llm.chat(messages)
        content = (msg.get("content") or "").strip()
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if not m:
            return []
        arr = json.loads(m.group(0))
        if not isinstance(arr, list):
            return []
        return [str(x).strip() for x in arr
                if isinstance(x, str) and x.strip() and len(x.strip()) <= 60][:5]
    except Exception:
        return []
