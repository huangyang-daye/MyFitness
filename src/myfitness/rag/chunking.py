"""从 DB 数据源构建 RAG 文本块。"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from myfitness.db.models import BodyMetric, DailyReport, KnowledgeEntry, NutritionLog, TrainingLog
from myfitness.rag.schemas import ChunkDocument
from myfitness.xunji.parsers.training import format_movement_sets

MEAL_LABELS = {
    "breakfast": "早餐",
    "lunch": "午餐",
    "dinner": "晚餐",
    "snack": "加餐",
    "other": "其他",
}

BODY_LABELS = {
    "weight": "体重",
    "bodyfat": "体脂率",
    "weist": "腰围",
}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_chunks(
    session: Session,
    user_id: int,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[ChunkDocument]:
    chunks: list[ChunkDocument] = []
    chunks.extend(_collect_body_chunks(session, user_id, start_date, end_date))
    chunks.extend(_collect_nutrition_chunks(session, user_id, start_date, end_date))
    chunks.extend(_collect_training_chunks(session, user_id, start_date, end_date))
    chunks.extend(_collect_report_chunks(session, user_id, start_date, end_date))
    if start_date is None and end_date is None:
        chunks.extend(_collect_knowledge_chunks(session, user_id))
    return chunks


def entry_to_chunks(entry: KnowledgeEntry) -> list[ChunkDocument]:
    """单条知识库条目 → 一个或多个向量块。"""
    sections = _split_report_sections(entry.content)
    if not sections:
        sections = [entry.content.strip()]
    record_day = None
    if entry.updated_at:
        record_day = entry.updated_at.date()
    elif entry.created_at:
        record_day = entry.created_at.date()

    documents: list[ChunkDocument] = []
    for index, section in enumerate(sections):
        text = section.strip()
        if not text:
            continue
        if index == 0 and not text.startswith(entry.title):
            text = f"{entry.title}\n\n{text}"
        documents.append(
            ChunkDocument(
                source_type="knowledge",
                source_id=f"{entry.id}:{index}",
                domain="memory" if getattr(entry, "kind", "user") == "memory" else "knowledge",
                title=entry.title,
                content=text,
                record_date=record_day,
                metadata={
                    "knowledge_id": entry.id,
                    "section_index": index,
                    "kind": getattr(entry, "kind", "user"),
                },
            )
        )
    return documents


def _collect_knowledge_chunks(session: Session, user_id: int) -> list[ChunkDocument]:
    rows = session.scalars(
        select(KnowledgeEntry)
        .where(KnowledgeEntry.user_id == user_id)
        .order_by(KnowledgeEntry.updated_at.desc())
    ).all()
    chunks: list[ChunkDocument] = []
    for row in rows:
        chunks.extend(entry_to_chunks(row))
    return chunks


def _date_filter(column, start_date: date | None, end_date: date | None):
    clauses = []
    if start_date is not None:
        clauses.append(column >= start_date)
    if end_date is not None:
        clauses.append(column <= end_date)
    return clauses


def _collect_body_chunks(
    session: Session,
    user_id: int,
    start_date: date | None,
    end_date: date | None,
) -> list[ChunkDocument]:
    stmt = select(BodyMetric).where(BodyMetric.user_id == user_id)
    for clause in _date_filter(BodyMetric.record_date, start_date, end_date):
        stmt = stmt.where(clause)
    rows = session.scalars(stmt.order_by(BodyMetric.record_date)).all()

    by_date: dict[date, list[BodyMetric]] = defaultdict(list)
    for row in rows:
        by_date[row.record_date].append(row)

    chunks: list[ChunkDocument] = []
    for day, metrics in sorted(by_date.items()):
        parts = []
        for metric in metrics:
            label = BODY_LABELS.get(metric.metric_type, metric.metric_type)
            parts.append(f"{label} {float(metric.value)}{metric.unit}")
        text = f"{day.isoformat()} 身体数据：" + "，".join(parts)
        chunks.append(
            ChunkDocument(
                source_type="body_daily",
                source_id=day.isoformat(),
                domain="body",
                title=f"{day.isoformat()} 身体数据",
                content=text,
                record_date=day,
                metadata={"metric_count": len(metrics)},
            )
        )
    return chunks


def _collect_nutrition_chunks(
    session: Session,
    user_id: int,
    start_date: date | None,
    end_date: date | None,
) -> list[ChunkDocument]:
    stmt = select(NutritionLog).where(NutritionLog.user_id == user_id)
    for clause in _date_filter(NutritionLog.record_date, start_date, end_date):
        stmt = stmt.where(clause)
    rows = session.scalars(stmt.order_by(NutritionLog.record_date)).all()

    by_key: dict[tuple[date, str], list[NutritionLog]] = defaultdict(list)
    for row in rows:
        by_key[(row.record_date, row.meal_type)].append(row)

    chunks: list[ChunkDocument] = []
    for (day, meal_type), entries in sorted(by_key.items()):
        meal = MEAL_LABELS.get(meal_type, meal_type)
        lines = [f"{day.isoformat()} {meal}："]
        total_cal = 0.0
        total_protein = 0.0
        for entry in entries:
            ntr = entry.nutrients_snapshot or {}
            cal = float(ntr.get("cal") or ntr.get("calories") or 0)
            protein = float(ntr.get("protein") or ntr.get("protein_g") or 0)
            total_cal += cal
            total_protein += protein
            lines.append(
                f"- {entry.food_name} {float(entry.amount)}{entry.unit}，"
                f"{cal:.0f} kcal，蛋白 {protein:.0f}g"
            )
        lines.append(f"合计 {total_cal:.0f} kcal，蛋白 {total_protein:.0f}g")
        text = "\n".join(lines)
        chunks.append(
            ChunkDocument(
                source_type="nutrition_meal",
                source_id=f"{day.isoformat()}:{meal_type}",
                domain="nutrition",
                title=f"{day.isoformat()} {meal}",
                content=text,
                record_date=day,
                metadata={"meal_type": meal_type, "entry_count": len(entries)},
            )
        )
    return chunks


def _collect_training_chunks(
    session: Session,
    user_id: int,
    start_date: date | None,
    end_date: date | None,
) -> list[ChunkDocument]:
    stmt = select(TrainingLog).where(TrainingLog.user_id == user_id)
    for clause in _date_filter(TrainingLog.record_date, start_date, end_date):
        stmt = stmt.where(clause)
    rows = session.scalars(stmt.order_by(TrainingLog.record_date)).all()

    chunks: list[ChunkDocument] = []
    for row in rows:
        text = _format_training_chunk(row)
        chunks.append(
            ChunkDocument(
                source_type="training_session",
                source_id=str(row.id),
                domain="fitness",
                title=f"{row.record_date.isoformat()} {row.title or '训练'}",
                content=text,
                record_date=row.record_date,
                metadata={"training_log_id": row.id, "title": row.title},
            )
        )
    return chunks


def _format_training_chunk(row: TrainingLog) -> str:
    payload = row.raw_payload or {}
    title = row.title or payload.get("title") or "训练"
    lines = [f"{row.record_date.isoformat()} 训练：{title}"]
    movements = payload.get("movements") or []
    for movement in movements:
        name = movement.get("name") or "动作"
        sets_text = format_movement_sets(movement.get("sets") or [])
        lines.append(f"- {name}：{sets_text}" if sets_text else f"- {name}")
    if not movements:
        lines.append("（无动作明细）")
    return "\n".join(lines)


def _collect_report_chunks(
    session: Session,
    user_id: int,
    start_date: date | None,
    end_date: date | None,
) -> list[ChunkDocument]:
    stmt = select(DailyReport).where(DailyReport.user_id == user_id)
    for clause in _date_filter(DailyReport.report_date, start_date, end_date):
        stmt = stmt.where(clause)
    rows = session.scalars(stmt.order_by(DailyReport.report_date)).all()

    chunks: list[ChunkDocument] = []
    for row in rows:
        period = (row.agent_outputs or {}).get("period") or {}
        period_start = period.get("start_date")
        period_end = period.get("end_date")
        if period_start and period_end and period_start != period_end:
            source_id = f"{period_start}_{period_end}"
            title = f"周期报告 {period_start} ~ {period_end}"
        else:
            source_id = row.report_date.isoformat()
            title = f"日报 {row.report_date.isoformat()}"

        for index, section in enumerate(_split_report_sections(row.content_md)):
            section = section.strip()
            if len(section) < 20:
                continue
            chunks.append(
                ChunkDocument(
                    source_type="report",
                    source_id=f"{source_id}:{index}",
                    domain="report",
                    title=title,
                    content=section,
                    record_date=row.report_date,
                    metadata={"report_id": row.id, "section_index": index},
                )
            )
    return chunks


def _split_report_sections(content_md: str, *, max_chars: int = 1200) -> list[str]:
    """按 Markdown 二级标题切分；过长段落再按长度切。"""
    text = content_md.strip()
    if not text:
        return []

    parts = re.split(r"\n(?=## )", text)
    sections: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            sections.append(part)
            continue
        for i in range(0, len(part), max_chars):
            sections.append(part[i : i + max_chars])
    if not sections:
        sections.append(text[:max_chars])
    return sections
