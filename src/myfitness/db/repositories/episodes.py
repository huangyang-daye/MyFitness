"""情景记忆 Repository — PostgreSQL 中的对话摘要。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from myfitness.db.models import EpisodicMemory


class EpisodeRepository:
    def __init__(self, session: Session, user_id: int) -> None:
        self.session = session
        self.user_id = user_id

    def add_if_absent(
        self,
        session_id: str,
        summary: str,
        *,
        turn_start: int,
        turn_end: int,
    ) -> EpisodicMemory:
        existing = self.session.scalar(
            select(EpisodicMemory).where(
                EpisodicMemory.user_id == self.user_id,
                EpisodicMemory.session_id == session_id,
                EpisodicMemory.turn_end == turn_end,
            )
        )
        if existing is not None:
            return existing
        row = EpisodicMemory(
            user_id=self.user_id,
            session_id=session_id,
            summary=summary.strip(),
            turn_start=turn_start,
            turn_end=turn_end,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def list_recent(self, *, limit: int = 8) -> list[EpisodicMemory]:
        rows = list(
            self.session.scalars(
                select(EpisodicMemory)
                .where(EpisodicMemory.user_id == self.user_id)
                .order_by(EpisodicMemory.created_at.desc())
                .limit(limit)
            ).all()
        )
        rows.reverse()
        return rows
