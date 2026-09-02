"""Orchestrator — 按 Planner 任务图执行，支持无依赖任务并行与 Judge 重做循环。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.orm import Session

from myfitness.agents.body_monitor import run_body_agent
from myfitness.agents.fitness_planner import run_fitness_agent
from myfitness.agents.manual_parser import (
    format_body_confirmation,
    format_nutrition_confirmation,
    parse_body_entry,
    parse_goal_weight,
    parse_nutrition_entry,
)
from myfitness.agents.nutritionist import run_nutrition_agent
from myfitness.agents.summary import run_summary_agent, should_stream_summary
from myfitness.agents.tools.query_planner import QueryPlan, build_query_plan
from myfitness.db.repositories.goals import UserGoalRepository
from myfitness.graph.context_reflection import reflect_before_answer
from myfitness.graph.judge import judge_turn, max_judge_attempts
from myfitness.graph.planner import build_task_plan
from myfitness.graph.progress import ProgressCallback, emit_plan, emit_step, emit_task_status, label_for
from myfitness.graph.task_plan import ExecutionResult, JudgeVerdict, PlannedTask, TaskPlan, TaskResult
from myfitness.memory.manager import attach_memory
from myfitness.memory.types import MemoryBundle
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import Intent, MyFitnessGraphState, PendingConfirmation, RouteResult
from myfitness.services.context_with_query import load_context_for_turn

logger = logging.getLogger(__name__)

_ACTION_ONLY = {
    Intent.SYNC_TRIGGER,
    Intent.REPORT_TRIGGER,
    Intent.CHART_TRIGGER,
    Intent.SCHEDULE_MANAGE,
    Intent.CONFIRMATION_RESPONSE,
}


def run_orchestrated_turn(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult,
    memory_bundle: MemoryBundle,
    *,
    on_progress: ProgressCallback | None = None,
    plan: TaskPlan | None = None,
    use_llm: bool | None = None,
) -> tuple[ExecutionResult, bool]:
    """执行 Planner 任务并在 Judge 通过后生成 Summary。

    返回 (execution, stream_summary)。
    """
    llm_enabled = use_llm
    task_plan = plan or build_task_plan(state.user_message, route, use_llm=llm_enabled)
    emit_plan(on_progress, task_plan)
    execution = ExecutionResult()
    retry_ids: set[str] = set()
    fetch_task_ids = [
        task.id
        for task in task_plan.tasks
        if task.intent == Intent.DATA_QUERY and task.params.get("include_latest_body")
    ]

    for attempt in range(1, max_judge_attempts() + 1):
        emit_step(on_progress, f"{label_for('planner')}（第 {attempt} 轮）…")
        execution = _execute_plan(
            session,
            state,
            route,
            task_plan,
            memory_bundle,
            on_progress=on_progress,
            retry_task_ids=retry_ids,
        )
        if execution.needs_confirmation:
            return execution, False

        emit_step(on_progress, f"{label_for('context_reflection')}…")
        reflection = reflect_before_answer(
            state.user_message,
            execution,
            fetch_task_ids=fetch_task_ids,
            use_llm=llm_enabled,
        )
        if reflection.confirmed_notes and execution.context:
            execution.context = execution.context.model_copy(
                update={"reflection_notes": reflection.confirmed_notes}
            )

        emit_step(on_progress, f"{label_for('judge')}…")
        if not reflection.ready:
            verdict = JudgeVerdict(
                satisfied=False,
                feedback=reflection.feedback or "个体数据尚未从数据库确认",
                missing=reflection.missing_fetches,
                retry_task_ids=reflection.retry_task_ids or fetch_task_ids,
            )
        else:
            verdict = judge_turn(
                state.user_message,
                task_plan,
                execution,
                attempt=attempt,
                use_llm=llm_enabled,
            )
        if verdict.satisfied:
            break
        retry_ids = set(verdict.retry_task_ids)
        if not retry_ids:
            retry_ids = {
                task.id
                for task in task_plan.tasks
                if task.intent not in {Intent.MANUAL_ENTRY, Intent.CONFIRMATION_RESPONSE}
            }
        execution.errors.append(verdict.feedback or "Judge 认为结果未满足用户需求")
        logger.info("Judge 未通过（第 %s 轮）: %s", attempt, verdict.feedback)
        if attempt >= max_judge_attempts():
            break

    emit_step(on_progress, f"{label_for('summary')}…")
    execution.agent_outputs.summary = run_summary_agent(
        execution.agent_outputs,
        execution.context,
        task_plan.primary_intent,
        state.user_message,
    )
    execution.agents_invoked.append("summary")
    if execution.reply_parts:
        execution.reply_parts.append(execution.agent_outputs.summary.content_md)
    return execution, bool(llm_enabled) and should_stream_summary(
        task_plan.primary_intent, state.user_message
    )


def _execute_plan(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult,
    plan: TaskPlan,
    memory_bundle: MemoryBundle,
    *,
    on_progress: ProgressCallback | None,
    retry_task_ids: set[str],
) -> ExecutionResult:
    execution = ExecutionResult()
    completed: set[str] = set()
    if retry_task_ids:
        completed = {task.id for task in plan.tasks if task.id not in retry_task_ids}
    levels = _task_levels(plan.tasks)

    for level in levels:
        for task in level:
            if not all(dep in completed for dep in task.depends_on):
                continue
            if retry_task_ids and task.id not in retry_task_ids:
                continue
            emit_task_status(on_progress, task.id, "running", description=task.description)
            result = _execute_task(
                session, state, route, plan, task, memory_bundle, execution, on_progress
            )
            execution.task_results.append(result)
            emit_task_status(
                on_progress,
                task.id,
                result.status,
                description=result.summary or task.description,
            )
            if result.status == "pending_confirmation":
                execution.needs_confirmation = True
                return execution
            if result.status == "success":
                completed.add(task.id)
            elif result.status == "failed":
                execution.errors.append(result.error or result.summary)

    state.agent_outputs = execution.agent_outputs
    state.context = execution.context
    state.metadata.tools_invoked = execution.tools_invoked
    state.metadata.agents_invoked = execution.agents_invoked
    return execution


def _execute_task(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult,
    plan: TaskPlan,
    task: PlannedTask,
    memory_bundle: MemoryBundle,
    execution: ExecutionResult,
    on_progress: ProgressCallback | None,
) -> TaskResult:
    if task.intent == Intent.MANUAL_ENTRY:
        return _run_manual_entry(session, state, plan, task)
    if task.intent == Intent.GOAL_SETTING:
        return _run_goal_setting(session, state, task)
    if task.intent in _ACTION_ONLY:
        return TaskResult(
            task_id=task.id,
            intent=task.intent,
            status="skipped",
            summary="动作类任务由专用 handler 处理",
        )
    return _run_analysis_task(
        session,
        state,
        route,
        task,
        memory_bundle,
        execution,
        on_progress,
    )


def _run_manual_entry(
    session: Session,
    state: MyFitnessGraphState,
    plan: TaskPlan,
    task: PlannedTask,
) -> TaskResult:
    domain = task.domain or "body"
    target = task.start_date or date.today()
    if domain == "body":
        payload = parse_body_entry(state.user_message, target)
        if not payload:
            return TaskResult(
                task_id=task.id,
                intent=task.intent,
                status="failed",
                error="未能解析体重/体脂，请明确写出「体重 130kg」「体脂 37%」等",
            )
        summary = format_body_confirmation(payload)
        remaining = [item.to_dict() for item in plan.tasks if item.id != task.id]
        state.pending_confirmation = PendingConfirmation(
            action_type="db_write",
            summary=summary,
            payload={**payload, "_pending_plan": {"tasks": remaining, "plan_meta": plan.to_dict()}},
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
            domain="body",
        )
        state.pending_plan = {
            "tasks": remaining,
            "plan_meta": plan.to_dict(),
        }
        state.reply = summary
        return TaskResult(
            task_id=task.id,
            intent=task.intent,
            status="pending_confirmation",
            summary=summary,
        )

    payload = parse_nutrition_entry(state.user_message, target)
    if not payload:
        return TaskResult(
            task_id=task.id,
            intent=task.intent,
            status="failed",
            error="未能解析饮食记录",
        )
    summary = format_nutrition_confirmation(payload)
    remaining = [item.to_dict() for item in plan.tasks if item.id != task.id]
    state.pending_confirmation = PendingConfirmation(
        action_type="db_write",
        summary=summary,
        payload={**payload, "_pending_plan": {"tasks": remaining, "plan_meta": plan.to_dict()}},
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        domain="nutrition",
    )
    state.pending_plan = {"tasks": remaining, "plan_meta": plan.to_dict()}
    state.reply = summary
    return TaskResult(
        task_id=task.id,
        intent=task.intent,
        status="pending_confirmation",
        summary=summary,
    )


def _run_goal_setting(session: Session, state: MyFitnessGraphState, task: PlannedTask) -> TaskResult:
    target_weight = task.params.get("target_weight_kg")
    if target_weight is None:
        target_weight = parse_goal_weight(state.user_message)
    if target_weight is None:
        return TaskResult(
            task_id=task.id,
            intent=task.intent,
            status="skipped",
            summary="未识别到明确目标体重",
        )

    repo = UserGoalRepository(session, state.user_id)
    repo.create(
        goal_type="weight",
        target_value=float(target_weight),
        start_date=date.today(),
    )
    summary = f"已记录目标体重 {target_weight} kg"
    state.reply = f"{state.reply}\n\n{summary}".strip() if state.reply else summary
    return TaskResult(task_id=task.id, intent=task.intent, status="success", summary=summary)


def _run_analysis_task(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult,
    task: PlannedTask,
    memory_bundle: MemoryBundle,
    execution: ExecutionResult,
    on_progress: ProgressCallback | None,
) -> TaskResult:
    from dataclasses import replace

    from datetime import date as date_cls

    today = date_cls.today()
    emit_step(on_progress, f"{label_for('load_context')}…")
    query_plan = build_query_plan(
        state.user_message,
        task.intent,
        task.domain or route.domain,
        start_date=task.start_date or route.start_date,
        end_date=task.end_date or route.end_date,
    )
    if task.params.get("include_training_history"):
        muscle = task.params.get("muscle_group")
        if query_plan is None:
            query_plan = QueryPlan(
                start_date=task.start_date or today - timedelta(days=29),
                end_date=task.end_date or today,
                domains=("training",),
                muscle_group=muscle,
            )
        else:
            domains = query_plan.domains
            if "training" not in domains:
                domains = (*domains, "training")
            query_plan = replace(
                query_plan,
                domains=domains,
                start_date=task.start_date or query_plan.start_date,
                end_date=task.end_date or query_plan.end_date,
                muscle_group=muscle or query_plan.muscle_group,
            )
    if task.params.get("include_latest_body") and query_plan is not None:
        domains = query_plan.domains
        if "body" not in domains:
            domains = ("body", *domains)
        query_plan = replace(
            query_plan,
            include_latest_body=True,
            domains=domains,
        )
    context, tools = load_context_for_turn(
        session,
        state.user_id,
        state.user_message,
        task.intent,
        task.domain or route.domain,
        on_progress=on_progress,
        plan=query_plan,
        start_date=task.start_date or route.start_date,
        end_date=task.end_date or route.end_date,
    )
    if memory_bundle.short_term or memory_bundle.long_term:
        context = attach_memory(context, memory_bundle)
        tools.append("memory")

    _merge_execution_context(execution, context)
    for tool in tools:
        if tool not in execution.tools_invoked:
            execution.tools_invoked.append(tool)

    agents = _agents_for_task(task, query_plan)
    _run_specialists(agents, context, execution, on_progress, parallel=len(agents) > 1)
    return TaskResult(
        task_id=task.id,
        intent=task.intent,
        status="success",
        summary=task.description or "分析完成",
    )


def _run_specialists(
    agents: list[str],
    context,
    execution: ExecutionResult,
    on_progress: ProgressCallback | None,
    *,
    parallel: bool,
) -> None:
    if not agents:
        return

    def _run_one(name: str):
        if name == "body":
            emit_step(on_progress, f"{label_for('body_monitor')}…")
            return name, run_body_agent(context)
        if name == "nutrition":
            emit_step(on_progress, f"{label_for('nutritionist')}…")
            return name, run_nutrition_agent(context)
        emit_step(on_progress, f"{label_for('fitness_planner')}…")
        return name, run_fitness_agent(context)

    if parallel and len(agents) > 1:
        with ThreadPoolExecutor(max_workers=min(3, len(agents))) as pool:
            futures = [pool.submit(_run_one, name) for name in agents]
            for future in as_completed(futures):
                name, output = future.result()
                _store_agent_output(name, output, execution)
    else:
        for name in agents:
            agent_name, output = _run_one(name)
            _store_agent_output(agent_name, output, execution)


def _store_agent_output(name: str, output, execution: ExecutionResult) -> None:
    label = {"body": "body_monitor", "nutrition": "nutritionist", "fitness": "fitness_planner"}[name]
    if name == "body":
        execution.agent_outputs.body = output
    elif name == "nutrition":
        execution.agent_outputs.nutrition = output
    else:
        execution.agent_outputs.fitness = output
    if label not in execution.agents_invoked:
        execution.agents_invoked.append(label)


def _agents_for_task(task: PlannedTask, plan: QueryPlan | None) -> list[str]:
    if task.params.get("scope") == "confirm":
        return ["body"]
    if task.params.get("scope") == "training_history":
        return ["fitness"]
    domain_to_agent = {
        "body": "body",
        "nutrition": "nutrition",
        "training": "fitness",
        "fitness": "fitness",
    }
    if plan and plan.domains:
        agents: list[str] = []
        for domain in plan.domains:
            agent = domain_to_agent.get(domain)
            if agent and agent not in agents:
                agents.append(agent)
        if agents:
            return agents
    if task.domain:
        agent = domain_to_agent.get(task.domain)
        return [agent] if agent else ["body"]
    return ["body", "nutrition", "fitness"]


def _merge_execution_context(execution: ExecutionResult, new_context) -> None:
    """合并多轮任务产生的查询结果与上下文。"""
    if execution.context is None:
        execution.context = new_context
        return

    old = execution.context
    merged_qr = {**(old.query_results or {}), **(new_context.query_results or {})}
    if "body" in (old.query_results or {}) and "body" in (new_context.query_results or {}):
        body = {**(old.query_results or {})["body"], **new_context.query_results["body"]}
        latest = new_context.query_results["body"].get("latest_metrics")
        if latest:
            body["latest_metrics"] = latest
        merged_qr["body"] = body

    execution.context = new_context.model_copy(
        update={
            "query_results": merged_qr,
            "reflection_notes": new_context.reflection_notes or old.reflection_notes,
            "user_goals": new_context.user_goals or old.user_goals,
            "memory_long_term": new_context.memory_long_term or old.memory_long_term,
            "memory_short_term": new_context.memory_short_term or old.memory_short_term,
        }
    )


def _task_levels(tasks: list[PlannedTask]) -> list[list[PlannedTask]]:
    remaining = {task.id: task for task in tasks}
    done: set[str] = set()
    levels: list[list[PlannedTask]] = []
    while remaining:
        level = [
            task
            for task in remaining.values()
            if all(dep in done for dep in task.depends_on)
        ]
        if not level:
            levels.append(list(remaining.values()))
            break
        levels.append(level)
        for task in level:
            done.add(task.id)
            del remaining[task.id]
    return levels


def resume_pending_plan(
    session: Session,
    state: MyFitnessGraphState,
    memory_bundle: MemoryBundle,
    *,
    on_progress: ProgressCallback | None = None,
) -> tuple[ExecutionResult, bool]:
    """手动录入确认后继续执行剩余任务。"""
    raw = state.pending_plan
    if not raw:
        return ExecutionResult(), False

    tasks = [PlannedTask.from_dict(item) for item in raw.get("tasks") or []]
    meta = raw.get("plan_meta") or {}
    plan = TaskPlan(
        tasks=tasks,
        user_requirements=str(meta.get("user_requirements") or state.user_message),
        primary_intent=Intent(str(meta.get("primary_intent") or Intent.TREND_ANALYSIS.value)),
        domain=meta.get("domain"),
    )
    route = RouteResult(intents=[task.intent for task in tasks], domain=plan.domain)
    state.pending_plan = None
    return run_orchestrated_turn(
        session,
        state,
        route,
        memory_bundle,
        on_progress=on_progress,
        plan=plan,
    )
