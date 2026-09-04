"""从 DB 数据源构建 RAG 文本块。"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import date

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
            ntr = entry.nutrients_snapshot if isinstance(entry.nutrients_snapshot, dict) else {}
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
    payload = row.raw_payload if isinstance(row.raw_payload, dict) else {}
    title = row.title or payload.get("title") or "训练"
    lines = [f"{row.record_date.isoformat()} 训练：{title}"]
    movements = payload.get("movements") or [] if isinstance(payload, dict) else []
    for movement in movements:
        if not isinstance(movement, dict):
            continue
        name = movement.get("name") or "动作"
        sets_text = format_movement_sets(movement)
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
        outputs = row.agent_outputs if isinstance(row.agent_outputs, dict) else {}
        period = outputs.get("period") if isinstance(outputs.get("period"), dict) else {}
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


_HEADING_SPLIT = re.compile(r"\n(?=#{1,3} (?!第\s*\d+\s*页))")
_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")
_SENTENCE_SPLIT = re.compile(
    r"(?<=[。！？；…])"
    r"|(?<![A-Za-z0-9])(?<=[.!?])(?=\s+[A-Z\u4e00-\u9fff「『“（(])"
)
_LIST_LINE = re.compile(r"^([-*●•]\s+|\d+[.、,，]\s*)")
_PAGE_HEADING_LINE = re.compile(r"^#{1,3} 第\s*\d+\s*页\s*$")
_COMPLETE_END = re.compile(r"[。！？；…!?][」』”’）)\s]*$")


def _split_report_sections(content_md: str, *, max_chars: int = 1800, min_chars: int = 200) -> list[str]:
    """按标题打包，过长时再按段落 / 句子切，避免从句子中间断开。"""
    text = content_md.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    sections = [part.strip() for part in _HEADING_SPLIT.split(text) if part.strip()]
    packed: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for section in sections:
        extra = len(section) + (2 if buf else 0)
        if buf and buf_len + extra > max_chars:
            packed.append("\n\n".join(buf))
            buf = [section]
            buf_len = len(section)
        else:
            buf.append(section)
            buf_len += extra
    if buf:
        packed.append("\n\n".join(buf))

    chunks: list[str] = []
    for block in packed:
        if len(block) <= max_chars:
            chunks.append(block)
        else:
            chunks.extend(_split_overflow(block, max_chars))
    merged = _merge_small_chunks(chunks, min_chars=min_chars, max_chars=max_chars)
    stitched = _stitch_incomplete_sentences(merged, max_chars=max_chars)
    return stitched or [text[:max_chars]]


def _split_overflow(text: str, max_chars: int) -> list[str]:
    heading = _leading_heading(text)
    raw_paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT.split(text) if part.strip()]
    paragraphs = _fold_page_headings(raw_paragraphs)
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            sentences = [part.strip() for part in _SENTENCE_SPLIT.split(paragraph) if part.strip()]
            units.extend(sentences or [paragraph])

    packed = _greedy_pack(units, max_chars, sep="\n\n")
    chunks: list[str] = []
    for block in packed:
        if len(block) <= max_chars:
            chunks.append(_with_heading(block, heading))
            continue
        for piece in _split_on_soft_breaks(block, max_chars):
            chunks.append(_with_heading(piece, heading))
    return chunks


def _greedy_pack(pieces: list[str], max_chars: int, *, sep: str) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    sep_len = len(sep)
    for piece in pieces:
        extra = len(piece) + (sep_len if buf else 0)
        if buf and size + extra > max_chars:
            chunks.append(sep.join(buf))
            buf = [piece]
            size = len(piece)
        else:
            buf.append(piece)
            size += extra
    if buf:
        chunks.append(sep.join(buf))
    return chunks


def _split_on_soft_breaks(text: str, max_chars: int) -> list[str]:
    """单句仍超长时，优先在标点处切开；换行多半是 PDF 折行，放到最后。"""
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        remain = length - start
        if remain <= max_chars:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break
        window_end = start + max_chars
        floor = start + max(1, max_chars // 2)
        cut = _find_soft_cut(text, floor, window_end)
        if cut is None:
            cut = window_end
        piece = text[start:cut].strip()
        if piece:
            chunks.append(piece)
        start = cut
        while start < length and text[start] in " \t":
            start += 1
    return chunks


def _find_soft_cut(text: str, floor: int, window_end: int) -> int | None:
    for chars in ("。！？；…!?", "，、；,;：:", " ", "\n"):
        for index in range(window_end, floor - 1, -1):
            if text[index - 1] in chars:
                return index
    return None


def _leading_heading(text: str) -> str:
    first = text.split("\n", 1)[0].strip()
    if first.startswith("#") and not _PAGE_HEADING_LINE.match(first):
        return first
    return ""


def _with_heading(text: str, heading: str) -> str:
    if not heading:
        return text
    first = text.split("\n", 1)[0].strip()
    if first == heading or first.startswith("#"):
        return text
    return f"{heading}\n\n{text}"


def _merge_small_chunks(chunks: list[str], *, min_chars: int, max_chars: int) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    merged: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        previous = merged[-1]
        combined = len(previous) + 2 + len(chunk)
        too_small = len(chunk) < min_chars or len(previous) < min_chars
        if too_small and combined <= max_chars:
            merged[-1] = previous.rstrip() + "\n\n" + chunk.lstrip()
        else:
            merged.append(chunk)
    return merged


def _looks_incomplete(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    last_line = stripped.split("\n")[-1].strip()
    if _PAGE_HEADING_LINE.match(last_line):
        return True
    if _LIST_LINE.match(last_line):
        return False
    return _COMPLETE_END.search(last_line) is None


def _fold_page_headings(paragraphs: list[str]) -> list[str]:
    """PDF「第 N 页」只是分页标记，并入下一段，不当成独立切点。"""
    folded: list[str] = []
    carry = ""
    for paragraph in paragraphs:
        if _PAGE_HEADING_LINE.match(paragraph.strip()):
            carry = paragraph.strip()
            continue
        if carry:
            paragraph = f"{carry}\n\n{paragraph}"
            carry = ""
        folded.append(paragraph)
    if carry and folded:
        folded[-1] = f"{folded[-1]}\n\n{carry}"
    elif carry:
        folded.append(carry)
    return folded


def _lstrip_page_headings(text: str) -> str:
    lines = text.split("\n")
    index = 0
    while index < len(lines) and (
        not lines[index].strip() or _PAGE_HEADING_LINE.match(lines[index].strip())
    ):
        index += 1
    return "\n".join(lines[index:]).strip()


def _take_first_sentence(text: str) -> tuple[str, str]:
    match = _SENTENCE_SPLIT.search(text)
    if match is None:
        return text, ""
    cut = match.end()
    return text[:cut], text[cut:]


def _stitch_incomplete_sentences(chunks: list[str], *, max_chars: int) -> list[str]:
    """上一块若停在句中（常见于 PDF 分页），把下一块的第一句补回来。"""
    pending = list(chunks)
    result: list[str] = []
    overflow = int(max_chars * 1.25)
    while pending:
        current = pending.pop(0)
        while pending and _looks_incomplete(current):
            nxt = pending[0]
            first_line = nxt.lstrip().split("\n", 1)[0].strip()
            if first_line.startswith("#") and not _PAGE_HEADING_LINE.match(first_line):
                break
            body = _lstrip_page_headings(nxt)
            if not body:
                pending.pop(0)
                continue
            take, rest = _take_first_sentence(body)
            if not take.strip():
                break
            candidate = (current.rstrip() + "\n" + take.lstrip()).strip()
            if len(candidate) > overflow:
                break
            current = candidate
            if rest.strip():
                pending[0] = rest.strip()
            else:
                pending.pop(0)
            if not _looks_incomplete(current):
                break
        result.append(current)
    return result
