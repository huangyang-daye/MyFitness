"""LangGraph Chat 工作流 — M2 对话编排。"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

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
from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.chart_tools import (
    generate_chart,
    parse_chart_request,
)
from myfitness.agents.tools.query_planner import (
    QueryPlan,
    build_query_plan,
    parse_date_range_text,
)
from myfitness.agents.tools.schedule_tools import (
    apply_schedule_cancel,
    apply_schedule_upsert,
    list_scheduled_tasks,
)
from myfitness.agents.tools.write_tools import apply_body_manual_write, apply_nutrition_manual_write
from myfitness.config import get_settings
from myfitness.graph.progress import ProgressCallback, emit, label_for
from myfitness.graph.router import RouteResult, agents_for_intent, classify_intent
from myfitness.llm.factory import is_llm_configured
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import (
    Artifact,
    ChatMessage,
    GraphMetadata,
    Intent,
    MyFitnessGraphState,
    PendingConfirmation,
    RunMode,
)
from myfitness.services.context_with_query import load_context_for_turn
from myfitness.services.daily_report import run_daily_report
from myfitness.services.period_report import run_period_report
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
    state.pending_artifacts = []

    if state.pending_confirmation and _is_expired(state.pending_confirmation):
        state.pending_confirmation = None
        state.reply = "确认已过期，请重新发起录入。"
        return ChatTurnResult(state=state, stream=False)

    if state.pending_confirmation and _handle_pending_clarification(session, state):
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

    has_chart = route.has(Intent.CHART_TRIGGER)

    # 组合意图：先同步数据，再基于新数据生成日报 / 周期报表（可选再插入统计图）
    if route.has(Intent.SYNC_TRIGGER) and route.has(Intent.REPORT_TRIGGER):
        emit(on_progress, f"{label_for('sync')}…")
        emit(on_progress, f"{label_for('daily_report')}…")
        if has_chart:
            emit(on_progress, f"{label_for('chart')}…")
        result = _handle_sync_and_report(session, state, route)
        if has_chart:
            _append_chart(session, state, route, target_document=(result or {}).get("file_path"))
        return ChatTurnResult(state=state, stream=False)

    if route.has(Intent.SYNC_TRIGGER):
        emit(on_progress, f"{label_for('sync')}…")
        _handle_sync(session, state, route)
        if has_chart:
            emit(on_progress, f"{label_for('chart')}…")
            _append_chart(session, state, route)
        return ChatTurnResult(state=state, stream=False)

    if route.has(Intent.SCHEDULE_MANAGE):
        emit(on_progress, f"{label_for('schedule')}…")
        _handle_schedule(session, state)
        return ChatTurnResult(state=state, stream=False)

    if route.has(Intent.REPORT_TRIGGER):
        emit(on_progress, f"{label_for('daily_report')}…")
        if has_chart:
            emit(on_progress, f"{label_for('chart')}…")
        result = _handle_report(session, state, route)
        if has_chart:
            _append_chart(session, state, route, target_document=(result or {}).get("file_path"))
        return ChatTurnResult(state=state, stream=False)

    if has_chart:
        emit(on_progress, f"{label_for('chart')}…")
        _handle_chart(session, state, route)
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
    """在 prepare_chat_turn(stream=True) 之后调用，yield 回复正文。"""
    yield from iter_summary_reply(
        state.agent_outputs,
        state.context,
        state.intent or Intent.GENERAL,
        state.user_message,
    )


def _finalize_rule_summary(state: MyFitnessGraphState) -> None:
    state.agent_outputs.summary = run_summary_agent(
        state.agent_outputs,
        state.context,
        state.intent or Intent.GENERAL,
        state.user_message,
    )
    summary = state.agent_outputs.summary
    state.reply = summary.content_md


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
        state.reply = invoke_tool(
            apply_schedule_upsert, session, state.user_id, payload=pending.payload
        )
    elif pending.action_type == "schedule_cancel":
        state.reply = invoke_tool(
            apply_schedule_cancel,
            session,
            state.user_id,
            task_type=pending.payload.get("task_type", "daily_report"),
        )
    elif domain == "body":
        written = invoke_tool(
            apply_body_manual_write, session, state.user_id, payload=pending.payload
        )
        state.reply = "已写入身体数据：\n" + "\n".join(f"- {w}" for w in written)
    else:
        written = invoke_tool(
            apply_nutrition_manual_write, session, state.user_id, payload=pending.payload
        )
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
    _append_assistant(state)


def _handle_pending_clarification(session: Session, state: MyFitnessGraphState) -> bool:
    pending = state.pending_confirmation
    if not pending or pending.action_type not in {
        "report_date_clarification",
        "sync_report_date_clarification",
    }:
        return False

    text = state.user_message
    lower = text.lower()
    if any(w in lower or w in text for w in ("取消", "不要", "算了", "no", "cancel")):
        state.pending_confirmation = None
        state.reply = "已取消，未生成日报。"
        _append_assistant(state)
        return True

    start_date, end_date = parse_date_range_text(text)
    if start_date is None:
        state.reply = (
            "还需要一个具体日期。请回复例如：昨天、今天、8月24日、2026-08-24，"
            "或一个区间如「8月20日到8月25日」。"
        )
        _append_assistant(state)
        return True

    end_date = end_date or start_date
    action_type = pending.action_type
    state.pending_confirmation = None
    if action_type == "sync_report_date_clarification":
        route = RouteResult(
            intents=[Intent.SYNC_TRIGGER, Intent.REPORT_TRIGGER],
            start_date=start_date,
            end_date=end_date,
        )
        _handle_sync_and_report(session, state, route)
    else:
        route = RouteResult(
            intents=[Intent.REPORT_TRIGGER],
            start_date=start_date,
            end_date=end_date,
        )
        _handle_report(session, state, route)
    return True


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
        state.reply = _format_sync_result(result)
    except Exception as exc:
        state.errors.append(str(exc))
        state.reply = f"同步失败：{exc}"
    _append_assistant(state)


def _handle_sync_and_report(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult,
) -> dict | None:
    """组合意图：先同步指定日期（区间）数据，再基于新数据生成日报 / 周期报表。"""
    report_date = route.end_date or route.start_date
    if report_date is None:
        _ask_report_date(
            state,
            action_type="sync_report_date_clarification",
            prompt=(
                "你想生成哪天的日报？请回复具体日期，例如：昨天、今天、8月24日，"
                "或 2026-08-24；也可以给一个区间，例如「8月20日到8月25日」。"
                "确认日期后我会先同步该日数据再生成日报。"
            ),
        )
        return None
    sync_start = route.start_date or report_date
    sync_end = route.end_date or report_date

    parts: list[str] = []
    try:
        sync_result = run_sync(
            session, state.user_id, start_date=sync_start, end_date=sync_end
        )
        parts.append(_format_sync_result(sync_result))
    except Exception as exc:
        state.errors.append(str(exc))
        parts.append(f"同步失败：{exc}（继续使用已有数据生成日报）")

    result: dict | None = None
    try:
        result = _generate_report(session, state, sync_start, sync_end, route)
        _record_report_artifact(state, result)
        parts.append(_format_report_reply(result))
    except Exception as exc:
        state.errors.append(str(exc))
        parts.append(f"生成日报失败：{exc}")

    state.reply = "\n\n".join(parts)
    _append_assistant(state)
    return result


def _format_sync_result(result: dict) -> str:
    status = result.get("status", "failed")
    date_range = f"范围 {result.get('start_date')} ~ {result.get('end_date')}。"
    errors = [str(item) for item in result.get("errors", []) if item]
    if status == "success":
        return f"同步完成：{date_range}"
    if status == "partial":
        heading = f"同步部分完成：{date_range}"
    else:
        heading = f"同步失败：{date_range}"
    if not errors:
        return heading
    details = "\n".join(f"- {item}" for item in dict.fromkeys(errors))
    return f"{heading}\n{details}"


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
        tasks = invoke_tool(list_scheduled_tasks, session, state.user_id)
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
) -> dict | None:
    """生成日报（单日）或周期报表（多日区间）。"""
    start_date: date | None = route.start_date if route else None
    end_date: date | None = (route.end_date or route.start_date) if route else None
    if start_date is None:
        _ask_report_date(
            state,
            action_type="report_date_clarification",
            prompt=(
                "你想生成哪天的日报？请回复具体日期，例如：昨天、今天、8月24日，"
                "或 2026-08-24；也可以给一个区间，例如「8月20日到8月25日」。"
            ),
        )
        return None

    result: dict | None = None
    try:
        result = _generate_report(session, state, start_date, end_date or start_date, route)
        _record_report_artifact(state, result)
        state.reply = _format_report_reply(result)
    except Exception as exc:
        state.errors.append(str(exc))
        state.reply = f"生成日报失败：{exc}"
    _append_assistant(state)
    return result


def _generate_report(
    session: Session,
    state: MyFitnessGraphState,
    start_date: date,
    end_date: date,
    route: RouteResult | None = None,
) -> dict:
    """单日 → 日报；多日 → 周期报表（含身体数据趋势图）。"""
    domain = route.domain if route else None
    if start_date == end_date:
        return run_daily_report(
            session,
            state.user_id,
            report_date=end_date,
            sync_first=False,
            user_message=state.user_message,
            domain=domain,
        )
    return run_period_report(
        session,
        state.user_id,
        start_date=start_date,
        end_date=end_date,
        sync_first=False,
        user_message=state.user_message,
        domain=domain,
    )


def _format_report_reply(result: dict) -> str:
    """把报表结果整理成对话回复（长内容截断，提示见文件）。"""
    path = result.get("file_path") or "（未写入文件）"
    if result.get("report_kind") == "period":
        label = (
            f"{result.get('period_start')} ~ {result.get('period_end')}"
            f"（{result.get('period_days')} 天）"
        )
        kind = "周期报表"
    else:
        label = str(result.get("report_date", ""))
        kind = "日报"

    content = result.get("content_md", "") or ""
    text = f"已生成 {label} {kind}。\n文件：{path}\n\n{content[:2000]}"
    if len(content) > 2000:
        text += "\n\n…（已截断，完整内容见文件）"
    return text


def _handle_chart(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult | None = None,
    *,
    target_document: str | Path | None = None,
) -> None:
    """统计图意图：生成 mermaid 图表，按需要内联 / 生成文档 / 插入文档。"""
    _append_chart(session, state, route, target_document=target_document)


def _append_chart(
    session: Session,
    state: MyFitnessGraphState,
    route: RouteResult | None = None,
    *,
    target_document: str | Path | None = None,
) -> None:
    settings = get_settings()
    request = parse_chart_request(
        state.user_message,
        reports_dir=settings.daily_report_output_dir,
    )
    # 意图识别解析出的日期范围优先于请求文案解析
    if route and route.start_date and route.end_date:
        request.start_date, request.end_date = route.start_date, route.end_date
    if target_document:
        request.output_mode = "insert"
        request.target_path = Path(target_document)

    _record_tool(state, "chart_tools")
    try:
        result = invoke_tool(
            generate_chart,
            session,
            state.user_id,
            domain=request.domain,
            metric=request.metric,
            start_date=request.start_date,
            end_date=request.end_date,
            chart_type=request.chart_type,
            output_mode=request.output_mode,
            target_path=str(request.target_path) if request.target_path else None,
            anchor=request.anchor,
            title=request.title,
        )
    except Exception as exc:
        state.errors.append(str(exc))
        _append_to_reply(state, f"生成统计图失败：{exc}")
        return

    message = result.get("message") or ""
    markdown = result.get("markdown") or ""
    # 独立图表文档 → 记为会话产物；insert 模式改的是报告文档，由报告卡片覆盖，不重复登记
    if result.get("path") and result.get("output_mode") == "document":
        _record_artifact(
            state,
            kind="chart",
            title=f"{result.get('metric_label') or '统计'}趋势图",
            subtitle=Path(result["path"]).name,
            path=result["path"],
        )
    if result.get("draw_skipped") or result.get("duplicate"):
        # 无数据 / 数据点不足 / 文档已存在同指标图：仅给出说明，不附图表
        _append_to_reply(state, message)
    else:
        # 内联 / 文档 / 插入：说明 + 完整 Mermaid 片段一并返回
        _append_to_reply(state, f"{message}\n\n{markdown}" if markdown else message)


def _record_tool(state: MyFitnessGraphState, name: str) -> None:
    if name not in state.metadata.tools_invoked:
        state.metadata.tools_invoked.append(name)


def _record_artifact(
    state: MyFitnessGraphState,
    *,
    kind: str,
    title: str,
    subtitle: str = "",
    path: str | Path | None,
) -> None:
    """登记本轮产生的产物，前端据此在会话里渲染卡片。"""
    if not path:
        return
    state.pending_artifacts.append(
        Artifact(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            title=title,
            subtitle=subtitle,
            path=str(path),
            created_at=datetime.now(UTC),
        )
    )


def _record_report_artifact(state: MyFitnessGraphState, result: dict | None) -> None:
    if not result or not result.get("file_path"):
        return
    if result.get("report_kind") == "period":
        title = f"{result.get('period_start')} ~ {result.get('period_end')} 周期报表"
        subtitle = f"{result.get('period_days')} 天"
    else:
        title = f"{result.get('report_date')} 健康日报"
        subtitle = "单日汇总"
    _record_artifact(
        state, kind="report", title=title, subtitle=subtitle, path=result["file_path"]
    )


def _append_to_reply(state: MyFitnessGraphState, extra: str) -> None:
    """在本轮回复后追加内容（并同步最后一条 assistant 消息）。"""
    state.reply = f"{state.reply}\n\n{extra}" if state.reply else extra
    if state.messages and state.messages[-1].role == "assistant":
        state.messages[-1].content = state.reply
        # 追加内容时可能又产生了新产物（例如先出报表再插图），同步到已落地的消息
        state.messages[-1].artifacts = list(state.pending_artifacts)
        state.pending_artifacts = []
    else:
        _append_assistant(state)


def _ask_report_date(state: MyFitnessGraphState, action_type: str, prompt: str) -> None:
    state.pending_confirmation = PendingConfirmation(
        action_type=action_type,
        summary=prompt,
        payload={"original_message": state.user_message},
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        domain="report",
    )
    state.reply = prompt
    _append_assistant(state)


def _append_assistant(state: MyFitnessGraphState) -> None:
    state.messages.append(
        ChatMessage(
            role="assistant",
            content=state.reply,
            timestamp=datetime.now(UTC),
            artifacts=list(state.pending_artifacts),
        )
    )
    state.pending_artifacts = []


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
