"""Session 管理：多窗口独立会话 + 落盘持久化 + 随时恢复。

设计要点：
- 每个窗口（终端/会话）对应一个独立 session，互相隔离；
- 消息历史、滚动摘要、session 级工具状态（如 todo）都保存在 session 内；
- JSON 文件持久化（原子写），重启后可无缝接着聊；
- assistant 的 thinking 会被保存（审计/回放用），但 to_api() 不回传给 LLM
  （省 token，且部分 API 拒绝历史里的 reasoning 字段）。
"""
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exceptions import SessionIdError, SessionNotFoundError


@dataclass
class Message:
    """统一消息模型（同时覆盖 user / assistant / tool 三种角色）。"""
    role: str                                   # user / assistant / tool
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None     # API 格式（assistant 发起的调用）
    tool_call_id: Optional[str] = None          # role=tool 时对应的调用 id
    name: Optional[str] = None                  # role=tool 时的工具名
    thinking: Optional[str] = None              # assistant 思考过程（仅存档，不进 API）
    ts: float = field(default_factory=time.time)

    # -- 转成发给 LLM 的 dict ------------------------------------------
    def to_api(self) -> dict:
        d: Dict[str, Any] = {"role": self.role}
        if self.role == "assistant":
            d["content"] = self.content if self.content is not None else ""
            if self.tool_calls:
                d["tool_calls"] = self.tool_calls
        elif self.role == "tool":
            d["content"] = self.content or ""
            d["tool_call_id"] = self.tool_call_id
            if self.name:
                d["name"] = self.name
        else:
            d["content"] = self.content or ""
        return d

    # -- 落盘序列化 ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "role": self.role, "content": self.content,
            "tool_calls": self.tool_calls, "tool_call_id": self.tool_call_id,
            "name": self.name, "thinking": self.thinking, "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d.get("role", "user"), content=d.get("content"),
            tool_calls=d.get("tool_calls"), tool_call_id=d.get("tool_call_id"),
            name=d.get("name"), thinking=d.get("thinking"), ts=d.get("ts", time.time()),
        )


def make_api_tool_calls(calls) -> List[dict]:
    """把 parser 的 ToolCall 列表转成 API 存储格式（arguments 序列化为字符串）。"""
    api_calls = []
    for c in calls:
        api_calls.append({
            "id": c.id,
            "type": "function",
            "function": {
                "name": c.name,
                "arguments": json.dumps(c.arguments, ensure_ascii=False),
            },
        })
    return api_calls


class Session:
    """一个独立会话：消息历史 + 滚动摘要 + 工具状态。"""

    def __init__(self, session_id: str):
        self.id = session_id
        self.created_at = time.time()
        self.updated_at = time.time()
        self.title: str = ""
        self.messages: List[Message] = []
        self.summary: str = ""                  # 压缩产生的滚动摘要
        self.archive: List[dict] = []           # 被压缩掉的原始消息（审计用）
        self.tool_state: Dict[str, Any] = {}    # session 级工具状态（如 todo）

    # -- 消息追加 --------------------------------------------------------
    def add_user(self, text: str) -> Message:
        msg = Message(role="user", content=text)
        self.messages.append(msg)
        if not self.title:
            self.title = text.strip().replace("\n", " ")[:30]
        self.touch()
        return msg

    def add_assistant(self, content: Optional[str] = None,
                      tool_calls_api: Optional[List[dict]] = None,
                      thinking: Optional[str] = None) -> Message:
        msg = Message(role="assistant", content=content,
                      tool_calls=tool_calls_api, thinking=thinking)
        self.messages.append(msg)
        self.touch()
        return msg

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> Message:
        msg = Message(role="tool", content=content,
                      tool_call_id=tool_call_id, name=name)
        self.messages.append(msg)
        self.touch()
        return msg

    # -- 统计 -------------------------------------------------------------
    def user_rounds(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    def touch(self):
        self.updated_at = time.time()

    # -- 序列化 ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "summary": self.summary,
            "archive": self.archive,
            "tool_state": self.tool_state,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        s = cls(d["id"])
        s.title = d.get("title", "")
        s.created_at = d.get("created_at", time.time())
        s.updated_at = d.get("updated_at", time.time())
        s.summary = d.get("summary", "")
        s.archive = d.get("archive", [])
        s.tool_state = d.get("tool_state", {})
        s.messages = [Message.from_dict(m) for m in d.get("messages", [])]
        return s


class SessionManager:
    """session 生命周期管理：创建 / 保存 / 加载 / 列出 / 删除。"""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    @staticmethod
    def sanitize_id(session_id: str) -> str:
        raw = (session_id or "").strip()
        if not raw:
            raise SessionIdError("session id 不能为空")
        if ".." in raw:   # 先于替换检查，防止路径穿越
            raise SessionIdError(f"非法 session id：{session_id}")
        sid = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]", "_", raw)
        if not sid.strip("_"):
            raise SessionIdError(f"非法 session id：{session_id}")
        return sid

    @staticmethod
    def auto_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]

    def path_for(self, session_id: str) -> str:
        return os.path.join(self.base_dir, self.sanitize_id(session_id) + ".json")

    # ------------------------------------------------------------------
    def exists(self, session_id: str) -> bool:
        return os.path.isfile(self.path_for(session_id))

    def save(self, session: Session):
        """原子写：先写临时文件再替换，避免进程中断产生半截文件。"""
        path = self.path_for(session.id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    def load(self, session_id: str) -> Session:
        path = self.path_for(session_id)
        if not os.path.isfile(path):
            raise SessionNotFoundError(
                f"session 不存在：{session_id}（可先用 /list 查看）")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = Session.from_dict(data)
        session.id = session_id  # 使用规范化后的 id
        return session

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        if session_id:
            if self.exists(session_id):
                return self.load(session_id)
            return Session(self.sanitize_id(session_id))
        return Session(self.auto_id())

    def list_sessions(self) -> List[dict]:
        out = []
        if os.path.isdir(self.base_dir):
            for fn in sorted(os.listdir(self.base_dir)):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(self.base_dir, fn), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    out.append({
                        "id": data.get("id", fn[:-5]),
                        "title": data.get("title", "")[:30],
                        "rounds": sum(1 for m in data.get("messages", [])
                                      if m.get("role") == "user"),
                        "updated_at": data.get("updated_at", 0),
                        "compressed": bool(data.get("summary")),
                    })
                except (OSError, ValueError):
                    continue
        out.sort(key=lambda x: x["updated_at"], reverse=True)
        return out

    def delete(self, session_id: str):
        path = self.path_for(session_id)
        if not os.path.isfile(path):
            raise SessionNotFoundError(f"session 不存在：{session_id}")
        os.remove(path)
