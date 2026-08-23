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
from myfitness.agents.summary import iter_summary_reply, run_summary_agent, should_stream_summary
from myfitness.agents.tools.write_tools import apply_body_manual_write, apply_nutrition_manual_write
from myfitness.graph.router import agents_for_intent, classify_intent
from myfitness.llm.factory import is_llm_configured
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.constants import DISCLAIMER
from myfitness.schemas.state import (
    ChatMessage,
    ContextSnapshot,
    GraphMetadata,
    Intent,
    MyFitnessGraphState,
    PendingConfirmation,
    RunMode,
)
from myfitness.services.context_with_query import load_context_for_turn
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
) -> MyFitnessGraphState:
    result = prepare_chat_turn(session, state, message)
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

    route = classify_intent(state.user_message, state.pending_confirmation)
    state.intent = route.intent
    state.intent_domain = route.domain

    if route.intent == Intent.CONFIRMATION_RESPONSE and state.pending_confirmation:
        _handle_confirmation(session, state, route.confirmation_action or "cancel")
        return ChatTurnResult(state=state, stream=False)

    if route.intent == Intent.MANUAL_ENTRY:
        _handle_manual_entry(session, state, route.domain or "nutrition")
        return ChatTurnResult(state=state, stream=False)

    if route.intent == Intent.SYNC_TRIGGER:
        _handle_sync(session, state)
        return ChatTurnResult(state=state, stream=False)

    state.context, tools_invoked = load_context_for_turn(
        session,
        state.user_id,
        state.user_message,
        route.intent,
        route.domain,
    )
    state.metadata.tools_invoked = tools_invoked
    invoked: list[str] = []

    for agent in agents_for_intent(route.intent, route.domain):
        if agent == "body":
            state.agent_outputs.body = run_body_agent(state.context)
            invoked.append("body_monitor")
        elif agent == "nutrition":
            state.agent_outputs.nutrition = run_nutrition_agent(state.context)
            invoked.append("nutritionist")
        elif agent == "fitness":
            state.agent_outputs.fitness = run_fitness_agent(state.context)
            invoked.append("fitness_planner")

    if not invoked and route.intent != Intent.GENERAL:
        invoked.append("summary")

    invoked.append("summary")
    state.metadata.agents_invoked = invoked

    use_stream = (
        is_llm_configured()
        and should_stream_summary(route.intent)
    )
    return ChatTurnResult(state=state, stream=use_stream)


def iter_chat_turn(
    session: Session,
    state: MyFitnessGraphState,
    message: str,
) -> tuple[MyFitnessGraphState, Iterator[str]]:
    result = prepare_chat_turn(session, state, message)
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
    if domain == "body":
        written = apply_body_manual_write(session, state.user_id, pending.payload)
        state.reply = "已写入身体数据：\n" + "\n".join(f"- {w}" for w in written)
    else:
        written = apply_nutrition_manual_write(session, state.user_id, pending.payload)
        state.reply = "已写入饮食记录：\n" + "\n".join(f"- {w}" for w in written)

    state.pending_confirmation = None
    state.context, _ = load_context_for_turn(
        session, state.user_id, state.user_message, Intent.MANUAL_ENTRY, "nutrition"
    )
    state.agent_outputs.nutrition = run_nutrition_agent(state.context)
    state.agent_outputs.summary = run_summary_agent(
        state.agent_outputs, state.context, Intent.MANUAL_ENTRY
    )
    state.reply += f"\n\n_{DISCLAIMER}_"
    _append_assistant(state)


def _handle_sync(session: Session, state: MyFitnessGraphState) -> None:
    try:
        result = run_sync(session, state.user_id, days=7)
        state.reply = (
            f"同步完成：{result['status']}，"
            f"范围 {result['start_date']} ~ {result['end_date']}。"
        )
    except Exception as exc:
        state.errors.append(str(exc))
        state.reply = f"同步失败：{exc}"
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
