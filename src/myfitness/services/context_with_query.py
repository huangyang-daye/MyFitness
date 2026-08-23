"""加载上下文并在需要时先执行 DB 查询。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from myfitness.agents.tools.query_planner import QueryPlan, build_query_plan
from myfitness.agents.tools.query_tools import execute_query_plan
from myfitness.graph.progress import ProgressCallback, emit, label_for
from myfitness.schemas.state import ContextSnapshot, Intent
from myfitness.services.context_loader import load_context_snapshot


def load_context_for_turn(
    session: Session,
    user_id: int,
    message: str,
    intent: Intent,
    domain: str | None = None,
    on_progress: ProgressCallback | None = None,
    plan: QueryPlan | None = None,
) -> tuple[ContextSnapshot, list[str]]:
    """构建 Agent 上下文；若问题涉及历史数据则先查 DB。"""
    tools_invoked: list[str] = []
    plan = plan or build_query_plan(message, intent, domain)

    if plan:
        query_results = execute_query_plan(
            session,
            user_id,
            list(plan.domains),
            plan.start_date,
            plan.end_date,
            metric_type=plan.metric_type,
            meal_type=plan.meal_type,
            on_progress=on_progress,
        )
        for domain_key, result in query_results.items():
            tools_invoked.append(result.get("tool", f"query_{domain_key}"))
        emit(on_progress, f"{label_for('load_context')}…")
        context = load_context_snapshot(
            session,
            user_id,
            end_date=plan.end_date,
            lookback_days=plan.lookback_days,
            query_results=query_results,
        )
        return context.model_copy(update={"query_results": query_results}), tools_invoked

    emit(on_progress, f"{label_for('load_context')}…")
    context = load_context_snapshot(session, user_id)
    return context, tools_invoked
