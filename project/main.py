"""MiniAgent CLI 入口。

用法：
  python main.py                      交互模式（真实 LLM，需环境变量 LLM_API_KEY）
  python main.py --mock               离线演示模式（MockLLM，无需 API key）
  python main.py --session win1       指定 / 恢复某个 session（多窗口隔离的关键）
  python main.py --ask "12*8是多少"    单次问答
  python main.py --list               列出所有 session
  python main.py --trace win1         查看某 session 的执行 trace
  python main.py --verbose            在终端实时打印 trace（调试用）

交互命令：
  /new [id]   新建会话   /switch id  切换/恢复会话   /list  列出会话
  /history    查看当前会话历史        /tools  列出工具  /trace [n]  查看trace
  /help  帮助  /exit  退出
"""
import argparse
import json
import os
import sys

from mini_agent import __version__
from mini_agent.agent import Agent
from mini_agent.config import load_agent_config, load_llm_config
from mini_agent.exceptions import MiniAgentError
from mini_agent.llm import MockLLM, OpenAICompatLLM
from mini_agent.memory import MemoryStore
from mini_agent.session import SessionManager
from mini_agent.tracing import Tracer

BANNER = f"""
====================================================
 MiniAgent v{__version__} - 从零实现的最小可用 Agent
 主循环手写 | 工具注册/解析/多session/压缩/trace 全内置
====================================================
"""


# ----------------------------------------------------------------------
def build_agent(args, need_llm: bool = True) -> Agent:
    # --list / --trace 等管理命令不需要 LLM，避免无 key 时无法使用
    if not need_llm or args.mock:
        llm = MockLLM()
    else:
        llm = OpenAICompatLLM(load_llm_config())   # 缺 key 会在这里抛友好错误
    cfg = load_agent_config(trace_also_stdout=args.verbose)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.trace_dir:
        cfg.trace_dir = args.trace_dir
    return Agent(
        llm, cfg,
        session_manager=SessionManager(cfg.data_dir),
        memory=MemoryStore(os.path.join(cfg.data_dir, "memory.json")),
    )


def show_answer(result):
    print("\n" + result.answer)
    if result.tool_calls:
        parts = []
        for tc in result.tool_calls:
            mark = "OK" if tc["ok"] else "FAIL"
            parts.append(f"{tc['name']}[{mark} {tc['elapsed_ms']}ms]")
        print(f"（本轮工具调用：{' -> '.join(parts)}，trace: {result.trace_path}）")
    if result.error:
        print(f"（错误详情：{result.error}）")


def show_sessions(mgr: SessionManager):
    rows = mgr.list_sessions()
    if not rows:
        print("（暂无 session，直接开聊会自动创建）")
        return
    print(f"{'SESSION ID':<24}{'轮次':<6}{'摘要':<6}标题")
    for r in rows:
        print(f"{r['id']:<24}{r['rounds']:<6}"
              f"{'有' if r['compressed'] else '无':<6}{r['title'] or '-'}")


def show_history(session):
    print(f"== session {session.id}（{session.user_rounds()} 轮，"
          f"消息 {len(session.messages)} 条，已压缩 {len(session.archive)} 条）==")
    if session.summary:
        print(f"[滚动摘要] {session.summary[:160]}...")
    for m in session.messages:
        if m.role == "user":
            print(f"  [用户] {truncate(m.content)}")
        elif m.role == "assistant":
            if m.tool_calls:
                names = ",".join(tc.get("function", {}).get("name", "?")
                                 for tc in m.tool_calls)
                print(f"  [助手] (调用工具: {names})")
            if m.content:
                print(f"  [助手] {truncate(m.content)}")
        elif m.role == "tool":
            print(f"  [工具 {m.name}] {truncate(m.content)}")
    if session.tool_state.get("todo"):
        items = session.tool_state["todo"].get("items", [])
        print(f"  [待办] 共 {len(items)} 条，未完成 "
              f"{sum(1 for x in items if not x['done'])} 条")


def truncate(s, n=80):
    s = str(s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def show_trace(agent: Agent, session_id: str, n=20):
    tracer = Tracer(agent.cfg.trace_dir, session_id)
    rows = tracer.tail(n)
    if not rows:
        print(f"（session {session_id} 暂无 trace 记录）")
        return
    for r in rows:
        head = (f"#{r.get('turn')}#{r.get('step')} {r.get('event')}")
        rest = {k: v for k, v in r.items()
                if k not in ("ts", "turn", "step", "event")}
        print(f"{head:<24}{json.dumps(rest, ensure_ascii=False)[:200]}")


def show_tools(agent: Agent):
    for t in agent.registry.all():
        scope = "session级" if t.session_local else "全局"
        print(f"- {t.name}（{scope}）: {t.description[:60]}...")


# ----------------------------------------------------------------------
def interactive(agent: Agent, session):
    print(BANNER)
    print(f"当前会话：{session.id}（输入 /help 查看命令，/exit 退出）")
    if agent.llm.name != "mock":
        cfg = getattr(agent.llm, "cfg", None)
        print(f"LLM：{getattr(cfg, 'model', '?')} @ {getattr(cfg, 'base_url', '?')}")
    while True:
        try:
            line = input(f"\nMiniAgent [{session.id}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        if not line:
            continue

        # --- 会话管理命令 ---
        if line == "/exit" or line == "/quit":
            print("再见！")
            break
        if line == "/help":
            print(__doc__)
            continue
        if line == "/list":
            show_sessions(agent.session_manager)
            continue
        if line == "/history":
            show_history(session)
            continue
        if line == "/tools":
            show_tools(agent)
            continue
        if line.startswith("/new"):
            name = line[4:].strip() or None
            session = agent.new_session(name)
            print(f"已新建会话：{session.id}")
            continue
        if line.startswith("/switch"):
            sid = line[7:].strip()
            if not sid:
                print("用法：/switch 会话id（先用 /list 查看）")
                continue
            try:
                session = agent.get_session(sid)
                print(f"已切换到会话：{session.id}（{session.user_rounds()} 轮历史）")
                show_history(session)
            except MiniAgentError as e:
                print(f"切换失败：{e}")
            continue
        if line.startswith("/trace"):
            arg = line[6:].strip()
            n = 20
            if arg.isdigit():
                n = int(arg)
                arg = session.id
            show_trace(agent, arg or session.id, n)
            continue
        if line.startswith("/"):
            print(f"未知命令：{line}（/help 查看可用命令）")
            continue

        # --- 正常对话 ---
        try:
            result = agent.chat(session, line)
            show_answer(result)
        except MiniAgentError as e:
            print(f"出错了：{e}")


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="mini-agent", description="从零实现的最小可用 Agent")
    ap.add_argument("--mock", action="store_true", help="离线演示模式（MockLLM，无需 API key）")
    ap.add_argument("--session", help="指定/恢复 session id")
    ap.add_argument("--ask", help="单次问答后退出")
    ap.add_argument("--list", action="store_true", help="列出所有 session")
    ap.add_argument("--trace", metavar="SESSION_ID", help="查看某 session 的 trace")
    ap.add_argument("--tail", type=int, default=20, help="配合 --trace：显示条数")
    ap.add_argument("--verbose", action="store_true", help="终端实时打印 trace")
    ap.add_argument("--data-dir", help="session 存储目录（默认 data）")
    ap.add_argument("--trace-dir", help="trace 存储目录（默认 logs/traces）")
    ap.add_argument("--version", action="version", version=f"mini-agent {__version__}")
    args = ap.parse_args(argv)

    try:
        agent = build_agent(args, need_llm=not (args.list or args.trace))
    except MiniAgentError as e:
        print(f"启动失败：{e}")
        sys.exit(1)

    if args.list:
        show_sessions(agent.session_manager)
        return
    if args.trace:
        show_trace(agent, args.trace, args.tail)
        return

    try:
        session = agent.new_session(args.session)
    except MiniAgentError as e:
        print(f"会话加载失败：{e}")
        sys.exit(1)

    if args.ask is not None:
        result = agent.chat(session, args.ask)
        show_answer(result)
        return

    interactive(agent, session)


if __name__ == "__main__":
    main()
