"""统一异常体系：所有可预期的错误都转成带用户可读信息的异常。"""


class MiniAgentError(Exception):
    """mini_agent 基础异常。message 面向调用方（可直接展示给用户）。"""

    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail or {}


class LLMAPIError(MiniAgentError):
    """LLM HTTP 调用失败（网络/超时/鉴权/限流，重试后仍失败）。"""


class LLMResponseError(MiniAgentError):
    """LLM 返回结构异常，无法解析出有效消息。"""


class ToolNotFoundError(MiniAgentError):
    """LLM 调用了不存在的工具。"""


class ToolArgumentError(MiniAgentError):
    """工具参数不符合 JSON Schema。"""


class ToolExecutionError(MiniAgentError):
    """工具执行过程中抛出的异常。"""


class SessionNotFoundError(MiniAgentError):
    """指定 session 不存在。"""


class SessionIdError(MiniAgentError):
    """session id 不合法。"""
