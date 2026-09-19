# AI Prompt 与问题解决记录

## 1. 原始任务

本项目基于如下任务实现：

- 从零实现一个最小可用 Agent
- 不依赖现有 agent framework 完成主流程
- 必须有工具循环、session、context 管理、测试
- 使用真实 LLM API
- 需要 README、架构题答案、问题解决记录

## 2. 开发中的关键 Prompt 方向

开发过程中，AI 协作主要围绕以下几个子问题展开：

1. 如何设计最小但完整的 runtime 分层
2. 工具注册机制如何定义
3. 如何在不依赖框架的情况下手写主循环
4. 如何解析 thinking / tool call / final answer
5. 如何做 session 落盘与恢复
6. context 过长时如何做基础压缩
7. memory 应该何时召回，放到哪里
8. 如何设计 tests 覆盖题目要求

## 3. 关键工程决策

### 决策 1：使用 Python + requests

原因：

- 题目重点是 runtime，而不是语言本身
- Python 对 HTTP、测试、CLI 足够快
- 可以用最少依赖把主循环写清楚

### 决策 2：真实 API 直接走 OpenAI-compatible HTTP

原因：

- 不引入 SDK 级 agent 封装
- 兼容 GLM / DeepSeek / OpenAI / Qwen / 本地网关
- 最能证明“主流程是自己写的”

### 决策 3：工具错误回喂，而不是直接中断

原因：

- 能体现 agent 的自愈能力
- 与真实 agent runtime 的行为更接近
- 更容易覆盖“参数错了以后模型自己修正”这个场景

### 决策 4：思考过程保存，但不回注到上下文

原因：

- 题目要求能提取思考过程
- 但不要求必须把思考内容持续塞回模型
- trace 和 session 存档里保留就够了

### 决策 5：最小 memory 做成 JSON + 关键词召回

原因：

- 题目重点不是向量数据库集成
- JSON 文件便于直接展示“memory 的召回时机与放置方式”
- 后续演进到向量库很自然

## 4. 真正遇到的问题与修复

### 问��� 1：FakeLLM 在测试里被重新赋值成普通 list，导致 `.popleft()` 崩溃

现象：

- 多个 agent loop 测试报“内部错误”

原因：

- `FakeLLM.responses` 原本是 `deque`
- 测试里直接做了 `fake_llm.responses = [...]`
- 把它覆盖成了普通 `list`

修复：

- 给 `responses` 加 property setter
- 无论赋值什么，都自动转成 `deque`

### 问题 2：中文 memory recall 基本召回不到

现象：

- `我在哪个城市工作` 无法召回 `用户的工作城市是上海`

原因：

- 原实现简单 `split()`
- 中文没有空格，整句变成一个大 token

修复：

- 加了共享 `query_score()`
- ASCII 用词项匹配
- 中文串切 bigram 匹配

### 问题 3：`sanitize_id("..")` 没有真正拦住非法 id

现象：

- 测试期望异常，但没有抛

原因：

- 先做了字符替换，再检查 `..`
- `..` 已经被替换成 `__`

修复：

- 对原始 id 先做 `..` 检查，再进入替换流程

### 问题 4：超小窗口下，context 对齐会把整个 body 清空

现象：

- 当 `max_history_messages` 小于一整轮消息数时
- `_align_to_user_boundary()` 可能把窗口裁成空列表

原因：

- 最近窗口从半轮开始，且窗口太小，找不到 user 边界

修复：

- 增加 fallback：保留最后一整轮

### 问题 5：压缩切点在尾部窗口内找不到 user 边界时，会把所有消息都归档

现象：

- 某些轮次压缩后 recent 为空

原因：

- 从 `n - keep_recent_messages` 往后找 user
- 有时尾部窗口刚好不包含 user 起点

修复：

- 如果没找到，就退化为“保留最后一整轮”

### 问题 6：`--list` / `--trace` 不应要求 LLM API key

现象：

- 只是想看 session 列表，也会因为没配 key 启动失败

修复：

- CLI 改成延迟加载 LLM
- 管理命令只用 `MockLLM` 占位，不发请求

### 问题 7：Windows 上 `pytest.ini` 含中文导致解析失败

现象：

- `iniconfig` 读取 `pytest.ini` 时抛 `UnicodeDecodeError`

原因：

- Windows 环境下按本地编码读取 ini
- 中文内容导致解码失败

修复：

- `pytest.ini` 改成纯 ASCII

## 5. 为什么这份实现是“最小可用”而不是“过度设计”

有意没有做的东西：

- 没有引入消息队列
- 没有引入数据库
- 没有引入向量库
- 没有引入异步调度器
- 没有引入 Web UI
- 没有引入复杂 planner

原因很简单：

- 题目要求的是最小可用 Agent
- 不是完整生产平台

我保留的能力都是直接对应题目要求的：

- agent loop
- tool registry
- parser
- session
- context
- memory
- trace
- tests

## 6. 如果继续演进，我会怎么做

优先级从高到低：

1. 把 memory 从 JSON 升级到 SQLite + 向量检索
2. 把 `search` / `get_weather` 换成真实外部 API
3. 为 async tool 增加 job runtime
4. 增加 Web UI / REST API
5. 增加 summary 质量自检
6. 增加用户级多租户隔离

## 7. 最终验证记录

本地验证已完成：

- `python -m pytest`
- 结果：`123 passed, 4 skipped`
- `python main.py --mock --ask "帮我计算 (12+8)*3.5"`
- `python main.py --mock --session smoke1 --ask "上海天气怎么样"`

说明：

- 离线演示正常
- 主循环正常
- tool trace 正常
- session 恢复正常
- tests 正常
