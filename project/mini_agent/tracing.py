"""执行日志（trace）：每个 session 一个 JSONL 文件，记录完整执行轨迹。

事件类型：
  user_input / llm_request / llm_response / thinking /
  tool_call / tool_result / final_answer / error /
  compress / memory_recall / max_iterations / session_saved

lines 可直接 `Get-Content logs/traces/<session>.jsonl` 查看，
也可 `python main.py --trace <session>` 按轮次友好展示。
"""
import json
import os
import time
from typing import Optional


class Tracer:
    """JSONL trace 记录器（线程不安全，本 runtime 为单线程同步模型）。"""

    def __init__(self, trace_dir: str, session_id: str, also_stdout: bool = False):
        self.trace_dir = trace_dir
        self.session_id = session_id
        self.also_stdout = also_stdout
        self.path = None  # type: Optional[str]
        self.turn = 0
        self.step = 0
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            self.path = os.path.join(trace_dir, f"{session_id}.jsonl")

    # -- 轮次/步骤计数 ------------------------------------------------
    def new_turn(self):
        self.turn += 1
        self.step = 0

    def next_step(self) -> int:
        self.step += 1
        return self.step

    # -- 记录 ----------------------------------------------------------
    def log(self, event: str, **payload):
        rec = {
            "ts": round(time.time(), 3),
            "turn": self.turn,
            "step": self.step,
            "event": event,
        }
        rec.update(payload)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass  # 日志失败不能影响主流程
        if self.also_stdout:
            print("[trace]", line)
        return rec

    def tail(self, n: int = 50):
        if not self.path or not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(x) for x in lines[-n:] if x.strip()]
