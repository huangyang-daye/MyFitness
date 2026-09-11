"""LangGraph 分析子图 — plan → execute_ready → reflect → judge → summary。"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from sqlalchemy.orm import Session

from myfitness.agents.summary import run_summary_agent, should_stream_summary
from myfitness.graph.context_reflection import reflect_before_answer
from myfitness.graph.judge import judge_turn, max_judge_attempts
from myfitness.graph.orchestrator import execute_plan
from myfitness.graph.planner import build_task_plan
from myfitness.graph.progress import ProgressCallback, emit_plan, emit_step, label_for
from myfitness.graph.task_plan import ExecutionResult, JudgeVerdict, TaskPlan
from myfitness.memory.types import MemoryBundle
from myfitness.schemas.state import Intent, MyFitnessGraphState, RouteResult

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_ANALYSIS_GRAPH = None
_CHECKPOINTER = None
_CHECKPOINTER_BACKEND: str | None = None


def reset_checkpointer_cache() -> None:
    """测试或切换 Redis 配置后重置全局图 / checkpointer。"""
    global _ANALYSIS_GRAPH, _CHECKPOINTER, _CHECKPOINTER_BACKEND
    _ANALYSIS_GRAPH = None
    _CHECKPOINTER = None
    _CHECKPOINTER_BACKEND = None


def get_checkpointer():
    """返回全局 checkpointer：配置了 REDIS_URL 时用 Redis，否则 MemorySaver。"""
    global _CHECKPOINTER, _CHECKPOINTER_BACKEND
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:
        raise ImportError('pip install -e ".[agents]"') from exc

    backend = "memory"
    saver = None
    try:
        from myfitness.config import get_settings

        settings = get_settings()
        redis_url = settings.resolved_redis_url()
        if redis_url:
            from myfitness.graph.redis_checkpointer import build_redis_checkpointer

            ttl = int(getattr(settings, "checkpoint_ttl_seconds", None) or settings.memory_working_ttl_seconds)
            saver = build_redis_checkpointer(redis_url, ttl_seconds=ttl)
            backend = "redis"
            logger.info("LangGraph checkpointer 使用 Redis")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis checkpointer 不可用，回退 MemorySaver: %s", exc)
        saver = None

    if saver is None:
        saver = MemorySaver()
    _CHECKPOINTER = saver
    _CHECKPOINTER_BACKEND = backend
    return _CHECKPOINTER


def checkpointer_backend() -> str:
    get_checkpointer()
    return _CHECKPOINTER_BACKEND or "memory"


class AnalysisGraphState(TypedDict, total=False):
    """可序列化的控制流状态；Session / progress / execution 经 Runtime.context 注入。"""

    task_plan: dict[str, Any]
    judge_attempt: int
    retry_task_ids: list[str]
    needs_confirmation: bool
    stream_summary: bool
    use_llm: bool | None
    skip_plan: bool
    fetch_task_ids: list[str]
    judge_satisfied: bool
    reflection_ready: bool
    reflection_feedback: str
    reflection_missing: list[str]
    reflection_retry_ids: list[str]


@dataclass
class AnalysisContext:
    """单次 invoke 的运行时依赖，不进入 checkpointer。"""

    session: Session
    chat_state: MyFitnessGraphState
    route: RouteResult
    memory_bundle: MemoryBundle
    on_progress: ProgressCallback | None = None
    execution: ExecutionResult = field(default_factory=ExecutionResult)


_ANALYSIS_CTX: ContextVar[AnalysisContext | None] = ContextVar("mf_analysis_ctx", default=None)


def _ctx(runtime: Runtime[AnalysisContext] | None = None) -> AnalysisContext:
    """从 ContextVar / LangGraph Runtime 取出分析上下文。

    LangGraph ≥0.6 可通过 runtime.context 注入；0.5.x 不支持 invoke(context=)，
    统一走 ContextVar，保证跨版本可用。
    """
    bound = _ANALYSIS_CTX.get()
    if isinstance(bound, AnalysisContext):
        return bound
    context = getattr(runtime, "context", None)
    if isinstance(context, AnalysisContext):
        return context
    try:
        from langgraph.runtime import get_runtime

        context = get_runtime(AnalysisContext).context
    except Exception:
        context = None
    if isinstance(context, AnalysisContext):
        return context
    raise RuntimeError("Analysis graph requires AnalysisContext (ContextVar or Runtime.context)")


def plan_node(
    state: AnalysisGraphState, runtime: Runtime[AnalysisContext] | None = None
) -> dict[str, Any]:
    ctx = _ctx(runtime)
    chat_state = ctx.chat_state
    route = ctx.route
    on_progress = ctx.on_progress
    use_llm = state.get("use_llm")

    if state.get("skip_plan") and state.get("task_plan"):
        task_plan = TaskPlan.from_dict(state["task_plan"])
    else:
        task_plan = build_task_plan(chat_state.user_message, route, use_llm=use_llm)

    emit_plan(on_progress, task_plan)
    fetch_task_ids = [
        task.id
        for task in task_plan.tasks
        if task.intent == Intent.DATA_QUERY and task.params.get("include_latest_body")
    ]
    return {
        "task_plan": task_plan.to_dict(),
        "fetch_task_ids": fetch_task_ids,
        "judge_attempt": 0,
        "retry_task_ids": [],
        "needs_confirmation": False,
        "judge_satisfied": False,
        "stream_summary": False,
    }


def execute_ready_node(
    state: AnalysisGraphState, runtime: Runtime[AnalysisContext] | None = None
) -> dict[str, Any]:
    ctx = _ctx(runtime)
    chat_state = ctx.chat_state
    route = ctx.route
    memory_bundle = ctx.memory_bundle
    session = ctx.session
    on_progress = ctx.on_progress

    attempt = int(state.get("judge_attempt") or 0) + 1
    emit_step(on_progress, f"{label_for('planner')}（第 {attempt} 轮）…")

    task_plan = TaskPlan.from_dict(state["task_plan"])
    retry_ids = set(state.get("retry_task_ids") or [])
    execution = execute_plan(
        session,
        chat_state,
        route,
        task_plan,
        memory_bundle,
        on_progress=on_progress,
        retry_task_ids=retry_ids,
    )
    ctx.execution = execution
    return {
        "judge_attempt": attempt,
        "needs_confirmation": bool(execution.needs_confirmation),
        "retry_task_ids": [],
    }


def reflect_node(
    state: AnalysisGraphState, runtime: Runtime[AnalysisContext] | None = None
) -> dict[str, Any]:
    ctx = _ctx(runtime)
    chat_state = ctx.chat_state
    execution = ctx.execution
    on_progress = ctx.on_progress
    use_llm = state.get("use_llm")
    fetch_task_ids = list(state.get("fetch_task_ids") or [])

    emit_step(on_progress, f"{label_for('context_reflection')}…")
    reflection = reflect_before_answer(
        chat_state.user_message,
        execution,
        fetch_task_ids=fetch_task_ids,
        use_llm=use_llm,
    )
    if reflection.confirmed_notes and execution.context:
        execution.context = execution.context.model_copy(
            update={"reflection_notes": reflection.confirmed_notes}
        )
        chat_state.context = execution.context

    return {
        "reflection_ready": bool(reflection.ready),
        "reflection_feedback": reflection.feedback or "",
        "reflection_missing": list(reflection.missing_fetches or []),
        "reflection_retry_ids": list(reflection.retry_task_ids or []),
    }


def judge_node(
    state: AnalysisGraphState, runtime: Runtime[AnalysisContext] | None = None
) -> dict[str, Any]:
    ctx = _ctx(runtime)
    chat_state = ctx.chat_state
    execution = ctx.execution
    on_progress = ctx.on_progress
    use_llm = state.get("use_llm")
    task_plan = TaskPlan.from_dict(state["task_plan"])
    attempt = int(state.get("judge_attempt") or 1)
    fetch_task_ids = list(state.get("fetch_task_ids") or [])

    emit_step(on_progress, f"{label_for('judge')}…")
    if not state.get("reflection_ready", True):
        verdict = JudgeVerdict(
            satisfied=False,
            feedback=state.get("reflection_feedback") or "个体数据尚未从数据库确认",
            missing=list(state.get("reflection_missing") or []),
            retry_task_ids=list(state.get("reflection_retry_ids") or fetch_task_ids),
        )
    else:
        verdict = judge_turn(
            chat_state.user_message,
            task_plan,
            execution,
            attempt=attempt,
            use_llm=use_llm,
        )

    if verdict.satisfied:
        return {"judge_satisfied": True, "retry_task_ids": []}

    retry_ids = set(verdict.retry_task_ids)
    if not retry_ids:
        retry_ids = {
            task.id
            for task in task_plan.tasks
            if task.intent not in {Intent.MANUAL_ENTRY, Intent.CONFIRMATION_RESPONSE}
        }
    execution.errors.append(verdict.feedback or "Judge 认为结果未满足用户需求")
    logger.info("Judge 未通过（第 %s 轮）: %s", attempt, verdict.feedback)
    return {
        "judge_satisfied": False,
        "retry_task_ids": sorted(retry_ids),
    }


def summary_node(
    state: AnalysisGraphState, runtime: Runtime[AnalysisContext] | None = None
) -> dict[str, Any]:
    ctx = _ctx(runtime)
    chat_state = ctx.chat_state
    execution = ctx.execution
    on_progress = ctx.on_progress
    task_plan = TaskPlan.from_dict(state["task_plan"])
    use_llm = state.get("use_llm")

    emit_step(on_progress, f"{label_for('summary')}…")
    execution.agent_outputs.summary = run_summary_agent(
        execution.agent_outputs,
        execution.context,
        task_plan.primary_intent,
        chat_state.user_message,
    )
    if "summary" not in execution.agents_invoked:
        execution.agents_invoked.append("summary")
    if execution.reply_parts:
        execution.reply_parts.append(execution.agent_outputs.summary.content_md)

    chat_state.agent_outputs = execution.agent_outputs
    chat_state.context = execution.context
    chat_state.metadata.tools_invoked = execution.tools_invoked
    chat_state.metadata.agents_invoked = execution.agents_invoked

    stream = bool(use_llm) and should_stream_summary(
        task_plan.primary_intent, chat_state.user_message
    )
    return {"stream_summary": stream, "needs_confirmation": False}


def route_after_execute(state: AnalysisGraphState) -> Literal["reflect", "__end__"]:
    if state.get("needs_confirmation"):
        return "__end__"
    return "reflect"


def route_after_judge(state: AnalysisGraphState) -> Literal["execute_ready", "summary"]:
    if state.get("judge_satisfied"):
        return "summary"
    attempt = int(state.get("judge_attempt") or 0)
    if attempt >= max_judge_attempts():
        return "summary"
    if state.get("retry_task_ids"):
        return "execute_ready"
    return "summary"


def build_analysis_graph():
    """编译分析子图：plan → execute_ready → reflect → judge ⇄ execute_ready → summary。"""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ImportError('pip install -e ".[agents]"') from exc

    graph = StateGraph(AnalysisGraphState, context_schema=AnalysisContext)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_ready", execute_ready_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("judge", judge_node)
    graph.add_node("summary", summary_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_ready")
    graph.add_conditional_edges(
        "execute_ready",
        route_after_execute,
        {"reflect": "reflect", "__end__": END},
    )
    graph.add_edge("reflect", "judge")
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {"execute_ready": "execute_ready", "summary": "summary"},
    )
    graph.add_edge("summary", END)
    return graph.compile(checkpointer=get_checkpointer())


def get_analysis_graph():
    global _ANALYSIS_GRAPH
    if _ANALYSIS_GRAPH is None:
        _ANALYSIS_GRAPH = build_analysis_graph()
    return _ANALYSIS_GRAPH


def get_graph_state(thread_id: str) -> dict[str, Any] | None:
    """获取某 thread 的最新图状态快照（用于断点恢复检查）。"""
    try:
        graph = get_analysis_graph()
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        if snapshot is None or not snapshot.values:
            return None
        return dict(snapshot.values)
    except Exception:
        return None


def has_incomplete_checkpoint(thread_id: str) -> bool:
    """是否存在未跑完的图断点（get_state().next 非空）。"""
    try:
        graph = get_analysis_graph()
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        return bool(snapshot and snapshot.next)
    except Exception:
        return False


def clear_graph_state(thread_id: str) -> None:
    """清除指定 thread 的图状态（用于重新编辑问题时重置）。"""
    try:
        checkpointer = get_checkpointer()
        delete_thread = getattr(checkpointer, "delete_thread", None)
        if callable(delete_thread):
            delete_thread(thread_id)
            return
        # 兼容旧 MemorySaver 直接清 storage
        storage = getattr(checkpointer, "storage", None)
        if storage is not None:
            keys_to_del = [k for k in list(storage.keys()) if str(k).startswith(thread_id) or k == thread_id]
            for k in keys_to_del:
                del storage[k]
    except Exception as exc:
        logger.warning("clear_graph_state failed for %s: %s", thread_id, exc)


def _invoke_analysis_graph(graph, payload, config: dict[str, Any], ctx: AnalysisContext):
    """兼容 LangGraph 0.5（无 context=）与 0.6+（可选 context=）。"""
    token = _ANALYSIS_CTX.set(ctx)
    try:
        try:
            return graph.invoke(payload, config=config, context=ctx)
        except TypeError as exc:
            if "context" not in str(exc):
                raise
            return graph.invoke(payload, config=config)
    finally:
        _ANALYSIS_CTX.reset(token)


def run_analysis_graph(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult,
    memory_bundle: MemoryBundle,
    *,
    on_progress: ProgressCallback | None = None,
    plan: TaskPlan | None = None,
    use_llm: bool | None = None,
    thread_id: str | None = None,
    resume: bool = False,
) -> tuple[ExecutionResult, bool]:
    """Invoke 分析子图，返回 (execution, stream_summary)。

    thread_id 用于 checkpointer 持久化，默认取 state.session_id。
    resume=True 时从 Redis/Memory 断点继续（不重新 plan）。
    """
    effective_thread_id = thread_id or state.session_id or "default"
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": effective_thread_id,
        }
    }
    graph = get_analysis_graph()

    if resume or (
        getattr(state.metadata, "graph_incomplete", False)
        and has_incomplete_checkpoint(effective_thread_id)
    ):
        ctx = AnalysisContext(
            session=session,
            chat_state=state,
            route=route,
            memory_bundle=memory_bundle,
            on_progress=on_progress,
        )
        emit_step(on_progress, "resume", "从断点恢复分析…")
        try:
            final_state = _invoke_analysis_graph(graph, None, config, ctx)
        except InterruptedError:
            state.metadata.graph_incomplete = True
            raise
        state.metadata.graph_incomplete = False
        execution = ctx.execution
        stream = bool((final_state or {}).get("stream_summary"))
        if execution.needs_confirmation:
            return execution, False
        return execution, stream

    # 新一轮：清掉可能残留的未完成断点，避免串台
    if has_incomplete_checkpoint(effective_thread_id):
        clear_graph_state(effective_thread_id)

    ctx = AnalysisContext(
        session=session,
        chat_state=state,
        route=route,
        memory_bundle=memory_bundle,
        on_progress=on_progress,
    )
    initial: AnalysisGraphState = {
        "use_llm": use_llm,
        "skip_plan": plan is not None,
        "judge_attempt": 0,
        "retry_task_ids": [],
        "needs_confirmation": False,
        "stream_summary": False,
        "judge_satisfied": False,
        "fetch_task_ids": [],
    }
    if plan is not None:
        initial["task_plan"] = plan.to_dict()
        initial["fetch_task_ids"] = [
            task.id
            for task in plan.tasks
            if task.intent == Intent.DATA_QUERY and task.params.get("include_latest_body")
        ]

    try:
        final_state = _invoke_analysis_graph(graph, initial, config, ctx)
    except InterruptedError:
        state.metadata.graph_incomplete = True
        raise
    state.metadata.graph_incomplete = False
    execution = ctx.execution
    stream = bool(final_state.get("stream_summary"))
    if execution.needs_confirmation:
        return execution, False
    return execution, stream
