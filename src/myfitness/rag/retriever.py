"""RAG 检索编排。"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from myfitness.agents.tools.query_planner import QueryPlan
from myfitness.config import get_settings
from myfitness.rag.schemas import RetrievedChunk
from myfitness.rag.store import search_chunks
from myfitness.schemas.state import Intent


_RAG_INTENTS = {
    Intent.DATA_QUERY,
    Intent.TREND_ANALYSIS,
    Intent.GENERAL,
    Intent.GOAL_SETTING,
    Intent.REPORT_TRIGGER,
    Intent.WEB_SEARCH,
}


def should_retrieve(intent: Intent) -> bool:
    settings = get_settings()
    return settings.rag_enabled and intent in _RAG_INTENTS


def retrieve_for_query(
    session: Session,
    user_id: int,
    query: str,
    *,
    intent: Intent,
    domain: str | None = None,
    plan: QueryPlan | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[RetrievedChunk]:
    if not should_retrieve(intent):
        return []

    rag_domain = _map_domain(domain)
    if plan:
        start_date = start_date or plan.start_date
        end_date = end_date or plan.end_date
        if not rag_domain and len(plan.domains) == 1:
            rag_domain = _map_domain(plan.domains[0])

    return search_chunks(
        session,
        user_id,
        query,
        start_date=start_date,
        end_date=end_date,
        domain=rag_domain,
    )


def _map_domain(domain: str | None) -> str | None:
    if not domain:
        return None
    if domain in {"body", "nutrition", "report", "knowledge", "memory"}:
        return domain
    if domain in {"fitness", "training"}:
        return "fitness"
    return None
