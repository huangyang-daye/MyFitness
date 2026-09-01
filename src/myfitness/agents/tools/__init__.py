"""Agent 工具包。

所有「可被 Agent / 图节点调用」的能力都用 LangChain `@tool` 修饰（见各子模块），
运行时上下文 `session` / `user_id` 通过 `InjectedToolArg` 注入，不会进入 LLM 入参模式。
调用一律走 `invoke_tool`（见 `base.py`）。

- `ALL_TOOLS`：可直接绑定到 LLM 的规范工具集合（入参 / 返回值均可序列化）；
- `build_*_chart` / `insert_chart_into_document` 同样是 `@tool`，但返回 ChartSpec /
  操作文件，主要在图内部通过 `invoke_tool` 调用，未纳入 `ALL_TOOLS`。
"""

from myfitness.agents.tools.base import bind_tools, invoke_tool, tool_config
from myfitness.agents.tools.chart_tools import (
    ChartRequest,
    ChartSeries,
    ChartSpec,
    build_body_metric_chart,
    build_body_trend_charts,
    build_chart,
    build_nutrition_chart,
    build_training_chart,
    generate_chart,
    insert_chart_into_document,
    is_chart_request,
    parse_chart_request,
    render_chart_document,
    write_chart_document,
)
from myfitness.agents.tools.query_format import format_query_results
from myfitness.agents.tools.query_planner import (
    QueryPlan,
    needs_database_query,
    parse_date_range_text,
    parse_single_date,
)
from myfitness.agents.tools.query_tools import (
    execute_query_plan,
    query_body_metrics,
    query_nutrition_logs,
    query_training_logs,
)
from myfitness.agents.tools.schedule_tools import (
    apply_schedule_cancel,
    apply_schedule_upsert,
    list_scheduled_tasks,
)
from myfitness.agents.tools.web_search import (
    build_search_query,
    format_web_search_results,
    is_web_search_request,
    needs_web_search,
    search_web,
    web_search,
)
from myfitness.agents.tools.write_tools import apply_body_manual_write, apply_nutrition_manual_write

# 可直接绑定到 LLM 的规范工具集合（入参 / 返回值均可序列化）。
ALL_TOOLS = [
    query_body_metrics,
    query_nutrition_logs,
    query_training_logs,
    execute_query_plan,
    apply_body_manual_write,
    apply_nutrition_manual_write,
    apply_schedule_upsert,
    apply_schedule_cancel,
    list_scheduled_tasks,
    generate_chart,
    insert_chart_into_document,
    web_search,
]

# 供测试 / 调试：按名称索引。
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "ChartRequest",
    "ChartSeries",
    "ChartSpec",
    "QueryPlan",
    "apply_body_manual_write",
    "apply_nutrition_manual_write",
    "apply_schedule_cancel",
    "apply_schedule_upsert",
    "bind_tools",
    "build_body_metric_chart",
    "build_body_trend_charts",
    "build_chart",
    "build_nutrition_chart",
    "build_training_chart",
    "execute_query_plan",
    "format_query_results",
    "generate_chart",
    "insert_chart_into_document",
    "invoke_tool",
    "is_chart_request",
    "list_scheduled_tasks",
    "needs_database_query",
    "parse_chart_request",
    "parse_date_range_text",
    "parse_single_date",
    "query_body_metrics",
    "query_nutrition_logs",
    "query_training_logs",
    "render_chart_document",
    "tool_config",
    "write_chart_document",
    "web_search",
    "search_web",
    "needs_web_search",
    "is_web_search_request",
    "build_search_query",
    "format_web_search_results",
]
