"""RAG 对外入口。"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from myfitness.agents.tools.query_planner import QueryPlan
from myfitness.rag.indexer import index_user_data
from myfitness.rag.retriever import retrieve_for_query
from myfitness.rag.schemas import RetrievedChunk
from myfitness.schemas.state import Intent

__all__ = [
    "index_user_data",
    "maybe_index_after_sync",
    "maybe_index_after_report",
    "retrieve_for_turn",
]


def retrieve_for_turn(
    session: Session,
    user_id: int,
    message: str,
    intent: Intent,
    *,
    domain: str | None = None,
    plan: QueryPlan | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[RetrievedChunk]:
    return retrieve_for_query(
        session,
        user_id,
        message,
        intent=intent,
        domain=domain,
        plan=plan,
        start_date=start_date,
        end_date=end_date,
    )


def maybe_index_after_sync(
    session: Session,
    user_id: int,
    start_date: date | None,
    end_date: date | None,
) -> None:
    index_user_data(session, user_id, start_date=start_date, end_date=end_date)


def maybe_index_after_report(session: Session, user_id: int, report_date: date) -> None:
    index_user_data(session, user_id, start_date=report_date, end_date=report_date)
