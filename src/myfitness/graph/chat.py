"""LangGraph Chat 工作流 — M2 对话编排。"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.orm import Session

from myfitness.agents.body_monitor import run_body_agent
from myfitness.agents.fitness_planner import run_fitness_agent
from myfitness.agents.manual_parser import (
    format_body_confirmation,
    format_nutrition_confirmation,
    parse_body_entry,
    parse_nutrition_entry,
)
from myfitness.agents.nutritionist import run_nutrition_agent
from myfitness.agents.schedule_parser import (
    format_schedule_confirmation,
    format_schedule_list,
    parse_schedule_request,
)
from myfitness.agents.summary import iter_summary_reply, run_summary_agent, should_stream_summary
from myfitness.agents.tools.query_planner import QueryPlan, build_query_plan, parse_single_date
from myfitness.agents.tools.schedule_tools import (
    apply_schedule_cancel,
    apply_schedule_upsert,
    list_scheduled_tasks,
)
from myfitness.agents.tools.write_tools import apply_body_manual_write, apply_nutrition_manual_write
from myfitness.graph.progress import ProgressCallback, emit, label_for
from myfitness.graph.router import RouteResult, agents_for_intent, classify_intent
from myfitness.llm.factory import is_llm_configured
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.constants import DISCLAIMER
from myfitness.schemas.state import (
    ChatMessage,
    GraphMetadata,
    Intent,
    MyFitnessGraphState,
    PendingConfirmation,
    RunMode,
)
from myfitness.services.context_with_query import load_context_for_turn
from myfitness.services.daily_report import run_daily_report
from myfitness.sync.orchestrator import run_sync


@dataclass
class ChatTurnResult:
    state: MyFitnessGraphState
    stream: bool = False


def new_chat_state(user_id: int = 1, session_id: str | None = None) -> MyFitnessGraphState:
    return MyFitnessGraphState(
        user_id=user_id,
        session_id=session_id or str(uuid.uuid4()),
        mode=RunMode.CHAT,
        metadata=GraphMetadata(started_at=datetime.now(UTC), agents_invoked=[]),
    )


def run_chat_turn(
    session: Session,
    state: MyFitnessGraphState,
    message: str,
    on_progress: ProgressCallback | None = None,
) -> MyFitnessGraphState:
    result = prepare_chat_turn(session, state, message, on_progress=on_progress)
    if result.stream:
        parts: list[str] = []
        for chunk in iter_chat_reply(result.state):
            parts.append(chunk)
        result.state.reply = "".join(parts)
        _append_assistant(result.state)
    elif not result.state.reply:
        _finalize_rule_summary(result.state)
        _append_assistant(result.state)
    return result.state


def prepare_chat_turn(
    session: Session,
    state: MyFitnessGraphState,
    message: str,
    on_progress: ProgressCallback | None = None,
) -> ChatTurnResult:
    state.user_message = message.strip()
    state.messages.append(
        ChatMessage(role="user", content=state.user_message, timestamp=datetime.now(UTC))
    )
    state.errors = []
    state.agent_outputs = AgentOutputs()

    if state.pending_confirmation and _is_expired(state.pending_confirmation):
        state.pending_confirmation = None
        state.reply = "确认已过期，请重新发起录入。"
        return ChatTurnResult(state=state, stream=False)

    emit(on_progress, f"{label_for('classify_intent')}…")
    route = classify_intent(
        state.user_message,
        state.pending_confirmation,
        use_llm=is_llm_configured(),
    )
    state.intent = route.intent
    state.intent_domain = route.domain

    if route.has(Intent.CONFIRMATION_RESPONSE) and state.pending_confirmation:
        emit(on_progress, f"{label_for('confirmation')}…")
        _handle_confirmation(session, state, route.confirmation_action or "cancel")
        return ChatTurnResult(state=state, stream=False)

    if route.has(Intent.MANUAL_ENTRY):
        emit(on_progress, f"{label_for('manual_entry')}…")
        _handle_manual_entry(session, state, route.domain or "nutrition")
        return ChatTurnResult(state=state, stream=False)

    # 组合意图：先同步数据，再基于新数据生成日报
    if route.has(Intent.SYNC_TRIGGER) and route.has(Intent.REPORT_TRIGGER):
        emit(on_progress, f"{label_for('sync')}…")
        emit(on_progress, f"{label_for('daily_report')}…")
        _handle_sync_and_report(session, state, route)
        return ChatTurnResult(state=state, stream=False)

    if route.has(Intent.SYNC_TRIGGER):
        emit(on_progress, f"{label_for('sync')}…")
        _handle_sync(session, state, route)
        return ChatTurnResult(state=state, stream=False)

    if route.has(Intent.SCHEDULE_MANAGE):
        emit(on_progress, f"{label_for('schedule')}…")
        _handle_schedule(session, state)
        return ChatTurnResult(state=state, stream=False)

    if route.has(Intent.REPORT_TRIGGER):
        emit(on_progress, f"{label_for('daily_report')}…")
        _handle_report(session, state, route)
        return ChatTurnResult(state=state, stream=False)

    plan = build_query_plan(state.user_message, route.intent, route.domain)
    state.context, tools_invoked = load_context_for_turn(
        session,
        state.user_id,
        state.user_message,
        route.intent,
        route.domain,
        on_progress=on_progress,
        plan=plan,
    )
    state.metadata.tools_invoked = tools_invoked
    invoked: list[str] = []

    for agent in _agents_for_turn(route, plan):
        if agent == "body":
            emit(on_progress, f"{label_for('body_monitor')}…")
            state.agent_outputs.body = run_body_agent(state.context)
            invoked.append("body_monitor")
        elif agent == "nutrition":
            emit(on_progress, f"{label_for('nutritionist')}…")
            state.agent_outputs.nutrition = run_nutrition_agent(state.context)
            invoked.append("nutritionist")
        elif agent == "fitness":
            emit(on_progress, f"{label_for('fitness_planner')}…")
            state.agent_outputs.fitness = run_fitness_agent(state.context)
            invoked.append("fitness_planner")

    if not invoked and route.intent != Intent.GENERAL:
        invoked.append("summary")

    invoked.append("summary")
    state.metadata.agents_invoked = invoked
    emit(on_progress, f"{label_for('summary')}…")

    use_stream = (
        is_llm_configured()
        and should_stream_summary(route.intent)
    )
    return ChatTurnResult(state=state, stream=use_stream)


def iter_chat_turn(
    session: Session,
    state: MyFitnessGraphState,
    message: str,
    on_progress: ProgressCallback | None = None,
) -> tuple[MyFitnessGraphState, Iterator[str]]:
    result = prepare_chat_turn(session, state, message, on_progress=on_progress)
    if result.stream:
        return result.state, iter_chat_reply(result.state)
    if result.state.reply:
        return result.state, iter([result.state.reply])
    _finalize_rule_summary(result.state)
    _append_assistant(result.state)
    return result.state, iter([result.state.reply])


def finalize_streamed_reply(state: MyFitnessGraphState, reply: str) -> MyFitnessGraphState:
    """流式输出结束后写入 state.reply 与对话历史。"""
    state.reply = reply
    if not state.messages or state.messages[-1].role != "assistant":
        _append_assistant(state)
    else:
        state.messages[-1].content = reply
    return state


def iter_chat_reply(state: MyFitnessGraphState) -> Iterator[str]:
    """在 prepare_chat_turn(stream=True) 之后调用，yield 回复正文 + 免责声明。"""
    for chunk in iter_summary_reply(
        state.agent_outputs,
        state.context,
        state.intent or Intent.GENERAL,
        state.user_message,
    ):
        yield chunk
    yield f"\n\n_{DISCLAIMER}_"


def _finalize_rule_summary(state: MyFitnessGraphState) -> None:
    state.agent_outputs.summary = run_summary_agent(
        state.agent_outputs,
        state.context,
        state.intent or Intent.GENERAL,
        state.user_message,
    )
    summary = state.agent_outputs.summary
    state.reply = f"{summary.content_md}\n\n_{summary.disclaimer}_"


def _agents_for_turn(route: RouteResult, plan: QueryPlan | None) -> list[str]:
    """按查询计划收窄 Specialist Agent，避免无关域拖慢首响。"""
    domain_to_agent = {
        "body": "body",
        "nutrition": "nutrition",
        "training": "fitness",
        "fitness": "fitness",
    }
    if plan and plan.domains:
        agents: list[str] = []
        for d in plan.domains:
            agent = domain_to_agent.get(d)
            if agent and agent not in agents:
                agents.append(agent)
        if agents:
            return agents
    return agents_for_intent(route.intent, route.domain)


def _handle_manual_entry(session: Session, state: MyFitnessGraphState, domain: str) -> None:
    target = state.target_date or date.today()
    if domain == "body":
        payload = parse_body_entry(state.user_message, target)
        if not payload:
            state.reply = "未能解析体重/体脂，请例如：记录体重 72.5kg"
            _append_assistant(state)
            return
        summary = format_body_confirmation(payload)
        state.pending_confirmation = PendingConfirmation(
            action_type="db_write",
            summary=summary,
            payload=payload,
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
            domain="body",
        )
        state.reply = summary
    else:
        payload = parse_nutrition_entry(state.user_message, target)
        if not payload:
            state.reply = "未能解析饮食，请例如：午餐 鸡胸肉 200g 苹果 1个"
            _append_assistant(state)
            return
        summary = format_nutrition_confirmation(payload)
        state.pending_confirmation = PendingConfirmation(
            action_type="db_write",
            summary=summary,
            payload=payload,
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
            domain="nutrition",
        )
        state.reply = summary

    _append_assistant(state)


def _handle_confirmation(
    session: Session,
    state: MyFitnessGraphState,
    action: str,
) -> None:
    pending = state.pending_confirmation
    if not pending:
        state.reply = "当前没有待确认的操作。"
        _append_assistant(state)
        return

    if action == "cancel":
        state.pending_confirmation = None
        state.reply = "已取消，未写入任何数据。"
        _append_assistant(state)
        return

    domain = pending.domain or "nutrition"
    if pending.action_type == "schedule_upsert":
        state.reply = apply_schedule_upsert(session, state.user_id, pending.payload)
    elif pending.action_type == "schedule_cancel":
        state.reply = apply_schedule_cancel(
            session, state.user_id, pending.payload.get("task_type", "daily_report")
        )
    elif domain == "body":
        written = apply_body_manual_write(session, state.user_id, pending.payload)
        state.reply = "已写入身体数据：\n" + "\n".join(f"- {w}" for w in written)
    else:
        written = apply_nutrition_manual_write(session, state.user_id, pending.payload)
        state.reply = "已写入饮食记录：\n" + "\n".join(f"- {w}" for w in written)

    state.pending_confirmation = None
    if pending.action_type.startswith("schedule_"):
        _append_assistant(state)
        return

    state.context, _ = load_context_for_turn(
        session, state.user_id, state.user_message, Intent.MANUAL_ENTRY, "nutrition"
    )
    state.agent_outputs.nutrition = run_nutrition_agent(state.context)
    state.agent_outputs.summary = run_summary_agent(
        state.agent_outputs, state.context, Intent.MANUAL_ENTRY
    )
    state.reply += f"\n\n_{DISCLAIMER}_"
    _append_assistant(state)


def _handle_sync(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult | None = None,
) -> None:
    """同步训记数据；消息中指明日期时按该范围同步，否则默认最近 7 天。"""
    start_date = route.start_date if route else None
    end_date = route.end_date if route else None
    try:
        if start_date and end_date:
            result = run_sync(
                session, state.user_id, start_date=start_date, end_date=end_date
            )
        else:
            result = run_sync(session, state.user_id, days=7)
        state.reply = (
            f"同步完成：{result['status']}，"
            f"范围 {result['start_date']} ~ {result['end_date']}。"
        )
    except Exception as exc:
        state.errors.append(str(exc))
        state.reply = f"同步失败：{exc}"
    _append_assistant(state)


def _handle_sync_and_report(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult,
) -> None:
    """组合意图：先同步指定日期数据，再基于新数据生成该日日报。"""
    report_date = route.end_date or route.start_date
    if report_date is None:
        report_date = parse_single_date(
            state.user_message,
            default=date.today() - timedelta(days=1),
        )
        assert report_date is not None
    sync_start = route.start_date or report_date
    sync_end = route.end_date or report_date

    parts: list[str] = []
    try:
        sync_result = run_sync(
            session, state.user_id, start_date=sync_start, end_date=sync_end
        )
        parts.append(
            f"同步完成：{sync_result['status']}，"
            f"范围 {sync_result['start_date']} ~ {sync_result['end_date']}。"
        )
    except Exception as exc:
        state.errors.append(str(exc))
        parts.append(f"同步失败：{exc}（继续使用已有数据生成日报）")

    try:
        result = run_daily_report(
            session,
            state.user_id,
            report_date=report_date,
            sync_first=False,  # 已在上方显式同步
        )
        path = result.get("file_path") or "（未写入文件）"
        parts.append(
            f"已生成 {result['report_date']} 日报。\n"
            f"文件：{path}\n\n"
            f"{result['content_md'][:2000]}"
        )
        if len(result["content_md"]) > 2000:
            parts.append("\n\n…（已截断，完整内容见文件）")
    except Exception as exc:
        state.errors.append(str(exc))
        parts.append(f"生成日报失败：{exc}")

    state.reply = "\n\n".join(parts)
    _append_assistant(state)


def _handle_schedule(session: Session, state: MyFitnessGraphState) -> None:
    parsed = parse_schedule_request(state.user_message)
    if not parsed:
        state.reply = (
            "未能解析定时任务。示例：\n"
            "- 每天早上7点生成日报\n"
            "- 每天 8:00 同步训记数据\n"
            "- 查看定时任务\n"
            "- 取消日报定时任务"
        )
        _append_assistant(state)
        return

    action = parsed["action"]
    if action == "list":
        tasks = list_scheduled_tasks(session, state.user_id)
        state.reply = format_schedule_list(tasks)
        _append_assistant(state)
        return

    if action == "cancel":
        summary = (
            f"请确认停用定时任务：{parsed['task_type']}\n\n"
            "回复「确认」停用，或「取消」放弃。"
        )
        state.pending_confirmation = PendingConfirmation(
            action_type="schedule_cancel",
            summary=summary,
            payload={"task_type": parsed["task_type"]},
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
            domain="schedule",
        )
        state.reply = summary
        _append_assistant(state)
        return

    summary = format_schedule_confirmation(parsed)
    state.pending_confirmation = PendingConfirmation(
        action_type="schedule_upsert",
        summary=summary,
        payload=parsed,
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        domain="schedule",
    )
    state.reply = summary
    _append_assistant(state)


def _handle_report(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult | None = None,
) -> None:
    report_date: date | None = None
    if route:
        report_date = route.end_date or route.start_date
    if report_date is None:
        report_date = parse_single_date(
            state.user_message,
            default=date.today() - timedelta(days=1),
        )
    assert report_date is not None
    try:
        result = run_daily_report(
            session,
            state.user_id,
            report_date=report_date,
            sync_first=False,
        )
        path = result.get("file_path") or "（未写入文件）"
        state.reply = (
            f"已生成 {result['report_date']} 日报。\n"
            f"文件：{path}\n\n"
            f"{result['content_md'][:2000]}"
        )
        if len(result["content_md"]) > 2000:
            state.reply += "\n\n…（已截断，完整内容见文件）"
    except Exception as exc:
        state.errors.append(str(exc))
        state.reply = f"生成日报失败：{exc}"
    _append_assistant(state)


def _append_assistant(state: MyFitnessGraphState) -> None:
    state.messages.append(
        ChatMessage(role="assistant", content=state.reply, timestamp=datetime.now(UTC))
    )


def _is_expired(pending: PendingConfirmation) -> bool:
    return datetime.now(UTC) > pending.expires_at.replace(tzinfo=UTC)


def build_langgraph_app(session: Session):
    """LangGraph 包装 — 供后续 Checkpoint / 并行流扩展。"""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError('pip install -e ".[agents]"') from exc

    def chat_node(graph_state: dict) -> dict:
        inner = MyFitnessGraphState.model_validate(graph_state["state"])
        updated = run_chat_turn(session, inner, graph_state["message"])
        return {"state": updated.model_dump(mode="json")}

    graph = StateGraph(dict)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()
