"""LangGraph 分析子图 — plan → execute_ready → reflect → judge → summary。"""

from __future__ import annotations

import logging
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


def get_checkpointer():
    """返回全局 MemorySaver checkpointer（进程内内存持久化）。"""
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError as exc:
            raise ImportError('pip install -e ".[agents]"') from exc
        _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER


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


def _ctx(runtime: Runtime[AnalysisContext] | None) -> AnalysisContext:
    """从 LangGraph Runtime 取出分析上下文。

    LangGraph ≥0.6 只向参数名 `runtime`（或类型为 RunnableConfig 的 `config`）注入；
    自定义 `config: dict` 不会收到配置。
    """
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
    raise RuntimeError("Analysis graph requires Runtime.context (AnalysisContext)")


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
        checkpointer = get_checkpointer()
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = checkpointer.get(config)
        if snapshot is None:
            return None
        return dict(snapshot.channel_values or {})
    except Exception:
        return None


def clear_graph_state(thread_id: str) -> None:
    """清除指定 thread 的图状态（用于重新编辑问题时重置）。"""
    try:
        checkpointer = get_checkpointer()
        # MemorySaver 直接清除存储
        storage = getattr(checkpointer, "storage", None)
        if storage is not None:
            keys_to_del = [k for k in list(storage.keys()) if str(k).startswith(thread_id)]
            for k in keys_to_del:
                del storage[k]
    except Exception as exc:
        logger.warning("clear_graph_state failed for %s: %s", thread_id, exc)


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
) -> tuple[ExecutionResult, bool]:
    """Invoke 分析子图，返回 (execution, stream_summary)。

    thread_id 用于 checkpointer 持久化，默认取 state.session_id。
    """
    effective_thread_id = thread_id or state.session_id or "default"
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

    graph = get_analysis_graph()
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": effective_thread_id,
        }
    }
    final_state = graph.invoke(initial, config=config, context=ctx)
    execution = ctx.execution
    stream = bool(final_state.get("stream_summary"))
    if execution.needs_confirmation:
        return execution, False
    return execution, stream
