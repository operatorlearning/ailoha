# MiniAgent

## 封面

**项目名称**：MiniAgent

**项目定位**：从零实现的最小可用 Agent Runtime

**题目类型**：Vibe Coding - Agent Runtime 实作题

**核心结论**：

- 不依赖任何现成 agent framework
- 主循环、工具调用、session、context、memory、trace 全部自行实现
- 支持真实 OpenAI-compatible LLM API
- 已完成测试验证，当前结果为 `124 passed, 3 skipped`

**能力清单**：

- Agent Loop：直接回复 / 工具调用 / 工具结果回喂 / 强制收尾
- Tooling：Schema 注册、参数校验、错误回喂、自愈重试路径
- Session：多窗口独立、JSON 持久化、随时恢复
- Context：近期窗口、最大轮次、滚动摘要压缩、工具结果截断
- Memory：跨 session 长期记忆、query-time recall、post-turn extract
- Trace：完整 JSONL 执行日志
- Test：单元测试、集成测试、e2e 场景测试

**已实现工具**：

- `calculator`
- `search`（mock）
- `todo`
- `get_weather`（mock）
- `read_docs`

**提交材料导航**：

- 运行说明与系统设计：`README.md`
- 架构设计题答案：`docs/ARCHITECTURE_ANSWERS.md`
- AI Prompt 与问题解决记录：`docs/AI_PROMPTS_AND_LOG.md`

**一句话摘要**：

这是一个可以直接运行、支持真实 LLM API、具备多 session / 工具调用 / context 压缩 / memory / trace / tests 的最小 Agent Runtime 作业实现。

---

一个从零实现的最小可用 Agent Runtime。

目标是完整覆盖题目里的核心要求：

- 不依赖 LangGraph / OpenHands / OpenClaw 等 agent 框架
- 自行实现主循环
- 有工具注册机制、Schema、工具调用解析
- 支持多 session、追问、context 压缩、基础 memory
- 有 trace / 执行日志
- 有测试
- 支持真实 OpenAI-compatible LLM API

当前实现语言是 Python 3.12，测试通过：`123 passed, 4 skipped`。

## 1. 项目特性

- 手写 Agent 主循环：`user input -> LLM 决策 -> tool -> tool result -> next loop/final answer`
- 工具注册中心：名称、描述、参数 JSON Schema、执行函数
- 5 个内置工具：`calculator`、`search`、`todo`、`get_weather`、`read_docs`
- 双协议解析：
  - 原生 OpenAI-compatible `tool_calls`
  - 文本协议回退（当模型/网关不支持 function calling 时）
- session 持久化：每个窗口一个独立 `session`，JSON 落盘，可恢复
- context 管理：
  - 最近消息窗口
  - 最大迭代次数限制
  - 滚动摘要压缩
  - 工具结果截断
- 长期记忆：跨 session 的轻量 memory store，query-time recall 注入 system prompt
- trace：每个 session 一份 JSONL 执行日志
- 离线演示模式：`--mock`

## 2. 快速开始

### 2.1 安装

```bash
pip install -r requirements.txt
```

### 2.2 离线演示

不用 API key，直接跑：

```bash
python main.py --mock
```

单次问答：

```bash
python main.py --mock --ask "帮我计算 (12+8)*3.5"
```

### 2.3 接入真实 LLM API

本项目直接调用 OpenAI-compatible Chat Completions API。

以 GLM 为例：

```bash
set LLM_API_KEY=你的key
set LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
set LLM_MODEL=glm-4.6
python main.py
```

也可以换成别的 OpenAI-compatible 服务：

- OpenAI
- DeepSeek
- Qwen
- Moonshot / Kimi
- 本地 vLLM / OneAPI / LiteLLM 网关

只要改：

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`

### 2.4 常用命令

交互模式：

```bash
python main.py
```

指定或恢复某个 session：

```bash
python main.py --session win1
```

单次问答：

```bash
python main.py --ask "请严格使用工具计算 1204/2"
```

列出所有 session：

```bash
python main.py --list
```

查看某个 session 的 trace：

```bash
python main.py --trace win1 --tail 30
```

### 2.5 交互命令

进入 REPL 后可用：

- `/new [id]`：新建会话
- `/switch id`：切换会话
- `/list`：列出会话
- `/history`：查看当前会话历史
- `/tools`：查看工具列表
- `/trace [n]`：查看当前会话最近 n 条 trace
- `/help`
- `/exit`

## 3. 目录结构

```text
.
├─ mini_agent/
│  ├─ agent.py
│  ├─ config.py
│  ├─ context.py
│  ├─ exceptions.py
│  ├─ llm.py
│  ├─ memory.py
│  ├─ parser.py
│  ├─ session.py
│  ├─ tools.py
│  └─ tracing.py
├─ tests/
├─ docs/
│  ├─ ARCHITECTURE_ANSWERS.md
│  └─ AI_PROMPTS_AND_LOG.md
├─ main.py
├─ requirements.txt
└─ README.md
```

## 4. 系统设计

### 4.1 模块分层

- `main.py`
  - CLI / REPL
  - session 切换
  - trace 查看
- `mini_agent/agent.py`
  - 核心 runtime
  - 主循环
  - 工具执行与错误回喂
- `mini_agent/llm.py`
  - 真实 OpenAI-compatible HTTP client
  - `MockLLM`
- `mini_agent/tools.py`
  - Tool / ToolRegistry
  - JSON Schema 校验
  - 内置工具
- `mini_agent/parser.py`
  - 解析 thinking / tool_calls / final answer
- `mini_agent/session.py`
  - 多会话管理
  - JSON 持久化
- `mini_agent/context.py`
  - system prompt 组装
  - 最近窗口
  - 滚动摘要压缩
- `mini_agent/memory.py`
  - 长期记忆提取与召回
- `mini_agent/tracing.py`
  - JSONL trace

### 4.2 主循环

核心代码在 `mini_agent/agent.py`。

流程：

1. 用户输入进入当前 session
2. 先做 memory recall，把命中的长期记忆注入 system prompt
3. 构造上下文：`system + summary + recent messages`
4. 把工具 Schema 一起发给 LLM
5. 解析模型输出：
   - 有最终答案：直接返回
   - 有工具调用：执行工具，把工具结果写回 session，再继续 loop
6. 达到最大迭代数时，做一次“禁止再调工具”的强制收尾
7. 结束后做：
   - memory 提取
   - context 压缩
   - session 落盘

伪代码：

```python
session.add_user(user_input)
memory_hint = memory.recall(user_input)

for i in range(max_iterations):
    messages = build_messages(session, memory_hint)
    raw = llm.chat(messages, tools=tool_schemas)
    parsed = parse_assistant_message(raw)

    if parsed.final_answer and not parsed.tool_calls:
        session.add_assistant(parsed.final_answer)
        return parsed.final_answer

    session.add_assistant(tool_calls=parsed.tool_calls)
    for call in parsed.tool_calls:
        result = execute_tool(call)
        session.add_tool_result(call.id, result)

return forced_final_answer()
```

### 4.3 工具注册机制

每个工具包含：

- `name`
- `description`
- `parameters`：JSON Schema
- `func`
- `session_local`：是否需要 session 级状态

定义在 `mini_agent/tools.py:18-24`。

注册表示例：

```python
Tool(
    name="calculator",
    description="精确算术计算器",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"]
    },
    func=_tool_calculator,
)
```

导出给 LLM 时，会被转成 OpenAI-compatible `tools` 格式。

### 4.4 已实现工具

#### `calculator`

- 自实现递归下降表达式解析器
- 不使用 `eval`
- 支持：
  - `+ - * / // % ** ^`
  - 括号
  - `sqrt/abs/round/floor/ceil/sin/cos/tan/log/ln/pow/min/max`
  - `pi/e`

#### `search`

- mock 搜索
- 内置一个小型 Agent/LLM 主题知识库
- 用简单相关性打分返回 top-k

#### `todo`

- 当前 session 的待办工具
- 支持：`add / list / complete`
- 是 `session_local=True`
- 可以直接证明“窗口 1 / 窗口 2 状态隔离”

#### `get_weather`

- mock 天气工具
- 按城市名确定性生成天气结果

#### `read_docs`

- 读取项目内 `README.md` / `docs/*.md` / `*.txt`
- 做了路径白名单与后缀校验

### 4.5 输出解析逻辑

解析器在 `mini_agent/parser.py`。

支持两条路：

1. 原生 function calling
2. 文本协议回退

可提取的内容：

- `thinking`
  - `reasoning_content`
  - think 块
- `tool_calls`
  - 原生 `tool_calls`
  - 文本中的 JSON 工具调用块
- `final_answer`

策略：

- 优先相信原生 `tool_calls`
- 若没有原生调用，再尝试从 `content` 中解析文本协议
- 工具块会从最终答案正文里剥离

### 4.6 Session 管理

实现文件：`mini_agent/session.py`

每个 session 保存：

- `messages`
- `summary`
- `archive`
- `tool_state`
- `title`

持久化方式：

- 每个 session 一个 JSON 文件
- 保存在 `data/<session_id>.json`
- 原子写：先写 `.tmp` 再 `os.replace`

这保证了：

- 用户 A 的窗口 1 和窗口 2 状态独立
- 进程退出后还能恢复
- 不同 session 的工具状态不串台

### 4.7 Context 管理

实现文件：`mini_agent/context.py`

进入 context 的信息：

- system prompt
- 长期记忆召回结果
- 滚动摘要
- 近期原文消息：
  - 用户输入
  - assistant 工具调用
  - tool 结果
  - assistant 最终回答

不进入 context 的信息：

- assistant 的完整思考过程

原因：

- 思考过程主要用于调试与审计
- 放进上下文会额外耗 token
- 部分供应商 API 不接受历史中的 reasoning 字段

#### 为什么工具结果要进 context

因为带工具的追问依赖它。例如：

1. 第一轮：`加个待办：写周报`
2. 第二轮：`我刚才让你加什么来着？`

如果工具结果不进上下文，模型很难稳定复述出“写周报”。

#### 压缩策略

- 触发条件：
  - 消息数超过 `max_history_messages`
  - 或 token 估算超过 `compress_trigger_tokens`
- 压缩方式：
  - 只压缩旧消息
  - 保留最近 `keep_recent_messages` 的原文
  - 在完整轮次边界切分
  - 旧消息交给 LLM 生成滚动摘要
- 若摘要失败：
  - 不压缩
  - 保留原始消息

#### 超长硬限制

`build_messages()` 还会做一次硬预算裁剪：

- 如果总 token 估算仍超预算
- 从最老轮次开始整轮丢弃
- 永远不从半轮中间切

### 4.8 Memory 的召回时机与放置方式

这是题目明确要求 README 解释的部分。

#### 召回时机

当前实现是 `query-time recall`：

1. 用户发来新消息
2. 用这条消息作为 query 检索 memory store
3. 只把 top-k 命中的长期记忆注入这一轮上下文

不是“每一轮都把全部 memory 塞进去”。

这样做的优点：

- token 更省
- 污染更少
- 不容易把过时信息反复灌回模型

#### 放置位置

放在 `system prompt` 的尾部单独区域：

- 明确标注它是“长期记忆”
- 明确说明“若与当前对话冲突，以当前对话为准”

而不是伪装成用户消息或 assistant 消息。

这样做的优点：

- 语义更清楚
- 冲突时优先级更容易控制
- 不会污染普通对话流

#### 当前 memory 的写入时机

每轮结束后，做一次 best-effort 抽取：

- 输入：本轮 user input + assistant answer
- 输出：值得长期记住的短事实数组
- 写入 `data/memory.json`

这个写入失败不会影响主流程。

#### 当前 memory 的内容边界

当前实现偏保守，适合记：

- 用户画像
- 用户偏好
- 长期约定
- 稳定事实

不适合记：

- 单轮临时任务细节
- 一次性中间结果

### 4.9 异常处理

基础异常处理已实现：

- LLM 网络错误 / 超时 / 429 / 5xx 重试
- LLM 返回结构异常
- 工具不存在
- 工具参数不合法
- 工具执行异常
- 最大工具循环次数限制
- session 落盘失败不会击穿主流程
- context 压缩失败不会丢历史

错误策略：

- 工具错误不会直接终止本轮
- 而是被包装成一条 tool result 回喂给 LLM
- 让模型自己根据错误信息修正调用

这就是一个最小版的“自愈”能力。

### 4.10 Trace / 执行日志

每个 session 都会生成一份 JSONL trace：

- 路径：`logs/traces/<session_id>.jsonl`

事件类型包括：

- `user_input`
- `memory_recall`
- `llm_request`
- `llm_response`
- `tool_call`
- `tool_result`
- `final_answer`
- `compress`
- `memory_extracted`
- `error`
- `session_saved`

查看方式：

```bash
python main.py --trace win1 --tail 30
```

## 5. 测试

运行全部测试：

```bash
python -m pytest
```

当前状态：

- `123 passed`
- `4 skipped`

跳过的是：

- 真实 API 集成测试（未设置 `LLM_API_KEY` 时自动跳过）

测试覆盖：

- 工具注册与 Schema 校验
- calculator 正确性与异常路径
- parser 对思考/工具/最终答案的解析
- session 独立性、持久化、恢复
- context 裁剪与压缩
- memory 的 recall / extract
- agent 主循环
- 两窗口互不影响的 e2e 场景
- 真实 LLM API 集成测试

## 6. 如何证明题目要求已覆盖

### 要求 1：从零完成

已满足。

本项目没有依赖任何 agent framework。

主循环、工具注册、上下文管理、session 管理、memory、trace、压缩、解析器全部手写。

### 要求 2：实现基本循环

已满足。

对应代码：`mini_agent/agent.py:69-199`

循环步骤完整：

1. 接收用户输入
2. 判断是直接回复还是调用工具
3. 调用工具
4. 根据工具结果继续 loop 或返回结果

### 至少三个工具

已满足，实际实现了五个。

### 工具注册机制

已满足。

对应代码：`mini_agent/tools.py:18-82`

### 输出解析

已满足。

对应代码：`mini_agent/parser.py`

### Session 管理

已满足。

对应代码：`mini_agent/session.py`

### Context 管理

已满足。

对应代码：`mini_agent/context.py`

### 异常处理与 trace

已满足。

### 测试用例

已满足。

## 7. 题外说明

### 关于 GitHub 链接

代码已经按要求生成在当前项目目录下，但我不能直接替你创建远程 GitHub 仓库或提交链接。

如果你要交作业，接下来只需要在本机执行你自己的 git 流程，例如：

```bash
git init
git add .
git commit -m "feat: add minimal agent runtime from scratch"
```

然后推到你自己的 GitHub 仓库即可。

### 额外文档

- 架构设计题答案：`docs/ARCHITECTURE_ANSWERS.md`
- AI Prompt 与问题解决记录：`docs/AI_PROMPTS_AND_LOG.md`

## 8. 已知限制

- `search` / `get_weather` 目前是 mock
- memory 目前是 JSON 文件 + 关键词召回，不是向量库
- context 压缩是单段滚动摘要，不是分层摘要或 map-reduce
- 目前 runtime 是同步单线程，不包含异步工具调度器
- 没有 Web UI，主要以 CLI 演示

这些都是最小可用实现下的有意取舍。
