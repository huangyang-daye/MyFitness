"""RAG 评测共用：语料构建、关键词向量、内存检索。"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from myfitness.config import Settings
from myfitness.db.models import Base, BodyMetric, NutritionLog, RagChunk, TrainingLog, User
from myfitness.rag.bm25 import bm25_document_text
from myfitness.rag.chunking import collect_chunks
from myfitness.rag.hybrid import hybrid_rank
from myfitness.rag.store import upsert_chunks

KEYWORD_AXES: tuple[tuple[str, int], ...] = (
    ("体重", 0),
    ("体脂", 1),
    ("腰围", 2),
    ("蛋白质", 3),
    ("蛋白", 3),
    ("热量", 4),
    ("午餐", 5),
    ("晚餐", 6),
    ("早餐", 5),
    ("饮食", 7),
    ("深蹲", 8),
    ("卧推", 9),
    ("背部", 10),
    ("胸部", 10),
    ("训练", 11),
    ("腿部", 11),
    ("减脂", 12),
    ("鸡胸肉", 3),
    ("牛肉", 7),
    ("硬拉", 8),
    ("引体", 10),
    ("划船", 10),
    ("跑步", 11),
    ("有氧", 11),
    ("鸡蛋", 3),
    ("燕麦", 7),
    ("三文鱼", 3),
    ("米饭", 7),
    ("豆腐", 7),
    ("碳水", 4),
) + tuple((f"8月{day}日", 19 + day) for day in range(1, 21)) + (
    ("上旬", 40),
    ("中旬", 41),
    ("下旬", 42),
)

VECTOR_DIM = 64
TOP_K = 5
MIN_SIMILARITY = 0.35

_LUNCH_FOODS = ("鸡胸肉", "牛肉饭", "鸡蛋", "燕麦", "三文鱼", "米饭", "豆腐")
_DINNER_FOODS = ("牛肉饭", "三文鱼", "豆腐", "鸡胸肉", "米饭")
_TRAINING_PLANS: tuple[tuple[str, str], ...] = (
    ("腿部训练", "深蹲"),
    ("胸部训练", "卧推"),
    ("背部训练", "硬拉"),
    ("背部训练", "引体向上"),
    ("上肢训练", "杠铃划船"),
    ("有氧训练", "跑步"),
    ("腿部训练", "深蹲"),
    ("胸部训练", "卧推"),
    ("全身训练", "硬拉"),
    ("背部训练", "引体向上"),
    ("腿部训练", "深蹲"),
    ("胸部训练", "卧推"),
    ("有氧训练", "跑步"),
    ("上肢训练", "杠铃划船"),
    ("腿部训练", "深蹲"),
)


def keyword_vector(text: str, *, dim: int = VECTOR_DIM) -> list[float]:
    vec = [0.0] * dim
    for keyword, axis in KEYWORD_AXES:
        if keyword in text:
            vec[axis] += 1.0
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0:
        return vec
    return [value / norm for value in vec]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0 or norm_r == 0:
        return 0.0
    return dot / (norm_l * norm_r)


def _embed_texts_for_test(texts: list[str], **_kwargs) -> list[list[float]]:
    return [keyword_vector(text) for text in texts]


def build_corpus_session(*, expanded: bool = True):
    """构建评测语料；expanded=True 时生成 8 月多域数据。"""
    test_settings = Settings(embedding_dimensions=VECTOR_DIM, rag_enabled=True)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))

    rows: list = []
    if expanded:
        base = date(2026, 8, 1)
        for offset in range(20):
            day = base + timedelta(days=offset)
            rows.append(
                BodyMetric(
                    user_id=1,
                    record_date=day,
                    metric_type="weight",
                    value=round(73.0 - offset * 0.15, 1),
                    unit="kg",
                    source="manual",
                )
            )
            if offset % 5 == 4:
                rows.append(
                    BodyMetric(
                        user_id=1,
                        record_date=day,
                        metric_type="bodyfat",
                        value=round(18.0 - offset * 0.05, 1),
                        unit="%",
                        source="manual",
                    )
                )
            if offset in {9, 19}:
                rows.append(
                    BodyMetric(
                        user_id=1,
                        record_date=day,
                        metric_type="weist",
                        value=round(82.0 - offset * 0.1, 1),
                        unit="cm",
                        source="manual",
                    )
                )

        for offset in range(15):
            day = base + timedelta(days=offset)
            lunch_food = _LUNCH_FOODS[offset % len(_LUNCH_FOODS)]
            dinner_food = _DINNER_FOODS[offset % len(_DINNER_FOODS)]
            rows.append(
                NutritionLog(
                    user_id=1,
                    record_date=day,
                    meal_type="lunch",
                    food_name=lunch_food,
                    amount=200,
                    unit="g",
                    nutrients_snapshot={"cal": 300 + offset * 10, "protein": 40 + offset},
                    source="manual",
                )
            )
            rows.append(
                NutritionLog(
                    user_id=1,
                    record_date=day,
                    meal_type="dinner",
                    food_name=dinner_food,
                    amount=1,
                    unit="份",
                    nutrients_snapshot={"cal": 500 + offset * 8, "protein": 25 + offset},
                    source="manual",
                )
            )
            if offset % 4 == 0:
                rows.append(
                    NutritionLog(
                        user_id=1,
                        record_date=day,
                        meal_type="breakfast",
                        food_name="燕麦",
                        amount=80,
                        unit="g",
                        nutrients_snapshot={"cal": 280, "protein": 10},
                        source="manual",
                    )
                )

        for offset, (title, movement) in enumerate(_TRAINING_PLANS):
            day = base + timedelta(days=offset)
            rows.append(
                TrainingLog(
                    user_id=1,
                    record_date=day,
                    title=title,
                    raw_payload={
                        "movements": [
                            {
                                "name": movement,
                                "sets": [
                                    {
                                        "done": True,
                                        "weight": str(60 + offset),
                                        "unit": "kg",
                                        "reps": "8",
                                    }
                                ],
                            }
                        ]
                    },
                    source="xunji",
                )
            )
    else:
        rows.extend(_minimal_corpus_rows())

    session.add_all(rows)
    session.flush()

    start = date(2026, 8, 1)
    end = date(2026, 8, 31)
    documents = collect_chunks(session, 1, start_date=start, end_date=end)
    enriched: list = []
    for doc in documents:
        if doc.record_date is not None:
            day_offset = (doc.record_date - start).days
            period = "上旬" if day_offset < 7 else "中旬" if day_offset < 14 else "下旬"
            prefix = f"{doc.record_date.month}月{doc.record_date.day}日 {period} "
            enriched.append(
                replace(
                    doc,
                    title=prefix + doc.title,
                    content=prefix + doc.content,
                )
            )
        else:
            enriched.append(doc)
    documents = enriched
    with (
        patch("myfitness.rag.store.get_settings", return_value=test_settings),
        patch("myfitness.rag.store.rag_is_available", return_value=True),
        patch("myfitness.rag.store.embed_texts", side_effect=_embed_texts_for_test),
    ):
        upsert_chunks(session, 1, documents)
    session.flush()
    return session


def _minimal_corpus_rows() -> list:
    return [
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 20),
            metric_type="weight",
            value=72.5,
            unit="kg",
            source="manual",
        ),
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 22),
            metric_type="weight",
            value=71.8,
            unit="kg",
            source="manual",
        ),
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 22),
            metric_type="bodyfat",
            value=17.0,
            unit="%",
            source="manual",
        ),
        NutritionLog(
            user_id=1,
            record_date=date(2026, 8, 20),
            meal_type="lunch",
            food_name="鸡胸肉",
            amount=200,
            unit="g",
            nutrients_snapshot={"cal": 330, "protein": 62},
            source="manual",
        ),
        NutritionLog(
            user_id=1,
            record_date=date(2026, 8, 21),
            meal_type="dinner",
            food_name="牛肉饭",
            amount=1,
            unit="份",
            nutrients_snapshot={"cal": 650, "protein": 28},
            source="manual",
        ),
        TrainingLog(
            user_id=1,
            record_date=date(2026, 8, 18),
            title="腿部训练",
            raw_payload={
                "movements": [
                    {
                        "name": "深蹲",
                        "sets": [{"done": True, "weight": "80", "unit": "kg", "reps": "5"}],
                    }
                ]
            },
            source="xunji",
        ),
        TrainingLog(
            user_id=1,
            record_date=date(2026, 8, 19),
            title="胸部训练",
            raw_payload={
                "movements": [
                    {
                        "name": "卧推",
                        "sets": [{"done": True, "weight": "60", "unit": "kg", "reps": "8"}],
                    }
                ]
            },
            source="xunji",
        ),
    ]


def memory_search(
    session,
    user_id: int,
    query: str,
    *,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    domain: str | None = None,
) -> list[tuple[str, float]]:
    stmt = select(RagChunk).where(RagChunk.user_id == user_id)
    if domain:
        stmt = stmt.where(RagChunk.domain == domain)

    rows = list(session.scalars(stmt).all())
    if not rows:
        return []

    documents = [
        (row.id, bm25_document_text(row.title or "", row.content or "")) for row in rows
    ]
    query_vec = keyword_vector(query)
    vector_hits: list[tuple[int, float]] = []
    for row in rows:
        if not row.embedding:
            continue
        similarity = cosine_similarity(query_vec, row.embedding)
        if similarity > 0:
            vector_hits.append((row.id, similarity))
    vector_hits.sort(key=lambda item: item[1], reverse=True)

    ranked = hybrid_rank(
        query,
        documents,
        vector_hits,
        recall_k=max(20, top_k),
        top_k=top_k,
        rrf_k=60,
        min_score=min_similarity,
    )
    by_id = {row.id: row for row in rows}
    return [(by_id[hit.doc_id].source_id, hit.score) for hit in ranked]


def training_id_map(session) -> dict[int, str]:
    ids = [
        row.id
        for row in session.scalars(
            select(TrainingLog).where(TrainingLog.user_id == 1).order_by(TrainingLog.id)
        ).all()
    ]
    return {index: str(training_id) for index, training_id in enumerate(ids)}


def resolve_placeholders(session, samples: list[dict]) -> list[dict]:
    train_map = training_id_map(session)
    resolved: list[dict] = []
    for item in samples:
        ids: list[str] = []
        for sid in item["relevant_source_ids"]:
            if sid.startswith("{train:"):
                index = int(sid.removeprefix("{train:").removesuffix("}"))
                ids.append(train_map[index])
            elif sid == "{squat_id}":
                ids.append(train_map[0])
            elif sid == "{bench_id}":
                ids.append(train_map[1])
            else:
                ids.append(sid)
        resolved.append({**item, "relevant_source_ids": ids})
    return resolved
