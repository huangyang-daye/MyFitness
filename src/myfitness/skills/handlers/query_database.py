"""query-database Skill：从用户问题规划并执行本地 DB 查询。"""

from __future__ import annotations

from sqlalchemy import func, select

from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.query_planner import QueryPlan, build_query_plan, needs_database_query
from myfitness.agents.tools.query_tools import execute_query_plan
from myfitness.db.models import BodyMetric
from myfitness.graph.progress import emit, label_for
from myfitness.schemas.state import Intent
from myfitness.skills.models import SkillContext, SkillResult

SKILL_NAME = "query-database"
_PROGRESS_HINT = ("进度", "趋势", "变化", "到今天", "至今", "减肥", "减脂")


def should_run(ctx: SkillContext) -> bool:
    if ctx.plan is not None:
        return True
    return needs_database_query(ctx.intent, ctx.message)


def run(ctx: SkillContext) -> SkillResult:
    plan = ctx.plan or build_query_plan(
        ctx.message,
        ctx.intent,
        ctx.domain,
        today=ctx.today or ctx.end_date,
        start_date=ctx.start_date,
        end_date=ctx.end_date,
    )
    if plan is None:
        return SkillResult(name=SKILL_NAME)

    plan = _maybe_widen_progress_plan(ctx, plan)
    query_results = invoke_tool(
        execute_query_plan,
        ctx.session,
        ctx.user_id,
        domains=list(plan.domains),
        start_date=plan.start_date,
        end_date=plan.end_date,
        metric_type=plan.metric_type,
        meal_type=plan.meal_type,
        on_progress=ctx.on_progress,
        include_latest_body=plan.include_latest_body,
        muscle_group=plan.muscle_group,
    )
    tools_invoked = [
        result.get("tool", f"query_{domain}")
        for domain, result in query_results.items()
        if isinstance(result, dict)
    ]
    emit(ctx.on_progress, f"{label_for('load_context')}…")
    return SkillResult(
        name=SKILL_NAME,
        data=query_results,
        tools_invoked=tools_invoked,
        extra={"plan": plan},
    )


def _maybe_widen_progress_plan(ctx: SkillContext, plan: QueryPlan) -> QueryPlan:
    """进度/趋势类问题：把查询起点拉到已有身体数据的最早日期。"""
    if ctx.intent != Intent.TREND_ANALYSIS:
        return plan
    if not any(keyword in ctx.message for keyword in _PROGRESS_HINT):
        return plan
    if "body" not in plan.domains:
        return plan

    earliest = ctx.session.scalar(
        select(func.min(BodyMetric.record_date)).where(BodyMetric.user_id == ctx.user_id)
    )
    if earliest is None or earliest >= plan.start_date:
        return plan
    return QueryPlan(
        start_date=earliest,
        end_date=plan.end_date,
        domains=plan.domains,
        metric_type=plan.metric_type,
        meal_type=plan.meal_type,
        include_latest_body=plan.include_latest_body,
        muscle_group=plan.muscle_group,
    )
