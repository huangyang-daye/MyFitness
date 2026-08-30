"""工具基础设施 — LangGraph / LangChain `@tool` 注册与调用约定。

本项目所有「可被 Agent / 图节点调用」的能力都修饰为 `@tool`（见各 tools 子模块），
并通过本模块的 `invoke_tool` 统一调用，符合 LangGraph 的规范：

- 工具的 `session` / `user_id` 等运行时上下文用 `InjectedToolArg` 标注，
  因此**不会**出现在暴露给 LLM 的 `tool.args` 模式中（LLM 只应提供业务参数）；
- 图节点 / 服务内部调用工具时一律走 `tool.invoke({...}, config=...)`，
  本模块的 `invoke_tool` 负责把 `session` / `user_id` 注入到 `RunnableConfig`
  并透传给工具，避免散落各处的手动参数拼装。

```python
from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.query_tools import query_body_metrics

result = invoke_tool(query_body_metrics, session, user_id, start_date=d, end_date=d)
```

如需把工具绑定到 LLM，使用 `bind_tools(llm, ALL_TOOLS)`（见 `__init__.py`）。
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from myfitness.debug import trace_tool_call


def tool_config(session: Session, user_id: int) -> RunnableConfig:
    """构造注入 `session` / `user_id` 的 RunnableConfig。"""
    return RunnableConfig(configurable={"session": session, "user_id": user_id})


def invoke_tool(tool, session: Session, user_id: int, /, **kwargs):
    """按 LangGraph 约定调用一个 `@tool` 工具。

    `session` / `user_id` 作为 `InjectedToolArg` 注入到工具执行上下文，
    不在业务参数中暴露；其余关键字参数即工具的 LLM 可调用参数。

    Args:
        tool: 被 `@tool` 修饰的工具对象。
        session: 当前数据库会话（注入，非 LLM 参数）。
        user_id: 当前用户 ID（注入，非 LLM 参数）。
        kwargs: 工具的其余业务参数（如 start_date / end_date / metric 等）。
    """
    return trace_tool_call(
        tool,
        user_id,
        kwargs,
        lambda: tool.invoke(
            {"session": session, "user_id": user_id, **kwargs},
            config=tool_config(session, user_id),
        ),
    )


def bind_tools(llm, tools):
    """把工具列表绑定到 LLM，供工具调用（function calling）模式使用。"""
    return llm.bind_tools(tools)
