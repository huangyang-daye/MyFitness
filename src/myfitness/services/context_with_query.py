"""加载上下文并在需要时先执行 DB 查询与 RAG 检索。"""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.query_planner import QueryPlan, build_query_plan
from myfitness.agents.tools.query_tools import execute_query_plan
from myfitness.agents.tools.web_search import build_search_query, needs_web_search, search_web
from myfitness.agents.tools.document_tools import (
    extract_document_path,
    needs_document_read,
    read_document_file,
)
from myfitness.db.models import BodyMetric
from myfitness.graph.progress import ProgressCallback, emit, label_for
from myfitness.rag.pipeline import retrieve_for_turn
from myfitness.schemas.state import ContextSnapshot, Intent
from myfitness.services.context_loader import load_context_snapshot

_PROGRESS_HINT = ("进度", "趋势", "变化", "到今天", "至今", "减肥", "减脂")


def _chunk_to_dict(chunk) -> dict:
    return {
        "id": chunk.id,
        "source_type": chunk.source_type,
        "source_id": chunk.source_id,
        "domain": chunk.domain,
        "title": chunk.title,
        "content": chunk.content,
        "record_date": chunk.record_date.isoformat() if chunk.record_date else None,
        "similarity": chunk.similarity,
        "metadata": chunk.metadata,
    }


def load_context_for_turn(
    session: Session,
    user_id: int,
    message: str,
    intent: Intent,
    domain: str | None = None,
    on_progress: ProgressCallback | None = None,
    plan: QueryPlan | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[ContextSnapshot, list[str]]:
    """构建 Agent 上下文；若问题涉及历史数据则先查 DB，再语义检索。"""
    tools_invoked: list[str] = []
    plan = plan or build_query_plan(
        message,
        intent,
        domain,
        start_date=start_date,
        end_date=end_date,
    )
    if plan:
        plan = _maybe_widen_progress_plan(session, user_id, plan, message, intent)
        query_results = invoke_tool(
            execute_query_plan,
            session,
            user_id,
            domains=list(plan.domains),
            start_date=plan.start_date,
            end_date=plan.end_date,
            metric_type=plan.metric_type,
            meal_type=plan.meal_type,
            on_progress=on_progress,
            include_latest_body=plan.include_latest_body,
            muscle_group=plan.muscle_group,
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
        context = context.model_copy(update={"query_results": query_results})
    else:
        emit(on_progress, f"{label_for('load_context')}…")
        context = load_context_snapshot(session, user_id)

    emit(on_progress, "语义检索…")
    retrieved = retrieve_for_turn(
        session,
        user_id,
        message,
        intent,
        domain=domain,
        plan=plan,
        start_date=start_date,
        end_date=end_date,
    )
    if retrieved:
        tools_invoked.append("rag_retriever")
        context = context.model_copy(
            update={"retrieved_chunks": [_chunk_to_dict(item) for item in retrieved]}
        )

    if needs_web_search(message, intent):
        emit(on_progress, f"{label_for('web_search')}…")
        search_result = search_web(build_search_query(message))
        hits = search_result.get("results") or []
        if hits:
            tools_invoked.append("web_search")
            context = context.model_copy(update={"web_search_results": hits})
        elif search_result.get("error"):
            tools_invoked.append("web_search")

    doc_path = extract_document_path(message)
    if doc_path or needs_document_read(message):
        emit(on_progress, f"{label_for('read_document')}…")
        if doc_path:
            try:
                doc_result = read_document_file(doc_path)
                tools_invoked.append("read_document")
                context = context.model_copy(
                    update={
                        "query_results": {
                            **(context.query_results or {}),
                            "document": doc_result,
                        }
                    }
                )
            except Exception as exc:  # noqa: BLE001
                tools_invoked.append("read_document")
                context = context.model_copy(
                    update={
                        "data_gaps": [
                            *(context.data_gaps or []),
                            f"读取文档失败：{exc}",
                        ]
                    }
                )

    return context, tools_invoked


def _maybe_widen_progress_plan(
    session: Session,
    user_id: int,
    plan: QueryPlan,
    message: str,
    intent: Intent,
) -> QueryPlan:
    """进度/趋势类问题：把查询起点拉到已有身体数据的最早日期。"""
    if intent != Intent.TREND_ANALYSIS:
        return plan
    if not any(keyword in message for keyword in _PROGRESS_HINT):
        return plan
    if "body" not in plan.domains:
        return plan

    earliest = session.scalar(
        select(func.min(BodyMetric.record_date)).where(BodyMetric.user_id == user_id)
    )
    if earliest is None or earliest >= plan.start_date:
        return plan
    return QueryPlan(
        start_date=earliest,
        end_date=plan.end_date,
        domains=plan.domains,
        metric_type=plan.metric_type,
        meal_type=plan.meal_type,
    )
