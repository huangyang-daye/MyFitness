"""知识库条目 Repository。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from myfitness.db.models import KnowledgeEntry


class KnowledgeRepository:
    def __init__(self, session: Session, user_id: int) -> None:
        self.session = session
        self.user_id = user_id

    def list_all(self) -> list[KnowledgeEntry]:
        return list(
            self.session.scalars(
                select(KnowledgeEntry)
                .where(KnowledgeEntry.user_id == self.user_id)
                .order_by(KnowledgeEntry.updated_at.desc())
            ).all()
        )

    def get(self, entry_id: int) -> KnowledgeEntry | None:
        return self.session.scalar(
            select(KnowledgeEntry).where(
                KnowledgeEntry.user_id == self.user_id,
                KnowledgeEntry.id == entry_id,
            )
        )

    def create(self, title: str, content: str, *, kind: str = "user") -> KnowledgeEntry:
        row = KnowledgeEntry(user_id=self.user_id, title=title, content=content, kind=kind)
        self.session.add(row)
        self.session.flush()
        return row

    def upsert_memory(self, title: str, content: str) -> KnowledgeEntry:
        row = self.session.scalar(
            select(KnowledgeEntry).where(
                KnowledgeEntry.user_id == self.user_id,
                KnowledgeEntry.kind == "memory",
                KnowledgeEntry.title == title,
            )
        )
        if row is None:
            return self.create(title, content, kind="memory")
        row.content = content
        self.session.flush()
        return row

    def delete(self, entry: KnowledgeEntry) -> None:
        self.session.delete(entry)
