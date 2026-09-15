"""加载上下文并在需要时先执行 Skill 查询、RAG 检索与其它资料源。"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from myfitness.agents.tools.document_tools import (
    extract_document_path,
    needs_document_read,
    read_document_file,
)
from myfitness.agents.tools.query_planner import QueryPlan
from myfitness.agents.tools.web_search import build_search_query, needs_web_search, search_web
from myfitness.graph.progress import ProgressCallback, emit, label_for
from myfitness.rag.pipeline import retrieve_for_turn
from myfitness.schemas.state import ContextSnapshot, Intent
from myfitness.services.context_loader import load_context_snapshot
from myfitness.skills.models import SkillContext
from myfitness.skills.runtime import run_context_skills


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
    """构建 Agent 上下文：先跑匹配的 context Skill（含数据库查询），再双路检索。"""
    tools_invoked: list[str] = []
    skill_results = run_context_skills(
        SkillContext(
            session=session,
            user_id=user_id,
            message=message,
            intent=intent,
            domain=domain,
            start_date=start_date,
            end_date=end_date,
            plan=plan,
            on_progress=on_progress,
        )
    )
    query_results: dict = {}
    for result in skill_results:
        if result.error:
            continue
        tools_invoked.extend(result.tools_invoked)
        if result.data:
            query_results.update(result.data)
        extra_plan = result.extra.get("plan")
        if extra_plan is not None:
            plan = extra_plan

    if query_results and plan is not None:
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
        if query_results:
            context = context.model_copy(update={"query_results": query_results})

    emit(on_progress, "检索中…")
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
