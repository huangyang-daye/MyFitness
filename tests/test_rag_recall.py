"""RAG 召回率评测 — 确定性关键词向量 + 内存余弦检索，无需外部 Embedding API。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from myfitness.config import Settings
from myfitness.db.models import Base, BodyMetric, NutritionLog, RagChunk, TrainingLog, User
from myfitness.rag.chunking import collect_chunks
from myfitness.rag.store import upsert_chunks

# 关键词 → 向量维度（查询与文档共用，保证同词高相似）
_KEYWORD_AXES: tuple[tuple[str, int], ...] = (
    ("体重", 0),
    ("体脂", 1),
    ("腰围", 2),
    ("蛋白质", 3),
    ("蛋白", 3),
    ("热量", 4),
    ("午餐", 5),
    ("晚餐", 6),
    ("饮食", 7),
    ("深蹲", 8),
    ("卧推", 9),
    ("背部", 10),
    ("训练", 11),
    ("减脂", 12),
)

_VECTOR_DIM = 64
_MIN_SIMILARITY = 0.35
_TOP_K = 5
_RECALL_THRESHOLD = 0.8


@dataclass(frozen=True)
class RecallCase:
    query: str
    relevant_source_ids: frozenset[str]
    domain: str | None = None


def keyword_vector(text: str, *, dim: int = _VECTOR_DIM) -> list[float]:
    """将文本映射为归一化关键词向量（测试用，可复现）。"""
    vec = [0.0] * dim
    for keyword, axis in _KEYWORD_AXES:
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


def recall_at_k(retrieved_source_ids: list[str], relevant: set[str], k: int) -> float:
    """Recall@K = |TopK ∩ 相关集| / |相关集|。"""
    if not relevant:
        return 1.0
    top_k = set(retrieved_source_ids[:k])
    return len(top_k & relevant) / len(relevant)


def memory_search_source_ids(
    session,
    user_id: int,
    query: str,
    *,
    top_k: int = _TOP_K,
    min_similarity: float = _MIN_SIMILARITY,
    domain: str | None = None,
) -> list[str]:
    """在 SQLite 测试库中用 Python 余弦相似度模拟 pgvector 检索。"""
    stmt = select(RagChunk).where(
        RagChunk.user_id == user_id,
        RagChunk.embedding.isnot(None),
    )
    if domain:
        stmt = stmt.where(RagChunk.domain == domain)

    query_vec = keyword_vector(query)
    scored: list[tuple[float, str]] = []
    for row in session.scalars(stmt).all():
        if not row.embedding:
            continue
        similarity = cosine_similarity(query_vec, row.embedding)
        if similarity < min_similarity:
            continue
        scored.append((similarity, row.source_id))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [source_id for _, source_id in scored[:top_k]]


def _embed_texts_for_test(texts: list[str], **_kwargs) -> list[list[float]]:
    return [keyword_vector(text) for text in texts]


@pytest.fixture
def rag_corpus_session(monkeypatch):
    """构造多域语料：身体 / 饮食 / 训练各若干条。"""
    test_settings = Settings(embedding_dimensions=_VECTOR_DIM, rag_enabled=True)
    monkeypatch.setattr("myfitness.rag.store.get_settings", lambda: test_settings)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))

    session.add_all(
        [
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
    )
    session.flush()

    documents = collect_chunks(session, 1, start_date=date(2026, 8, 1), end_date=date(2026, 8, 31))
    with (
        patch("myfitness.rag.store.rag_is_available", return_value=True),
        patch("myfitness.rag.store.embed_texts", side_effect=_embed_texts_for_test),
    ):
        stats = upsert_chunks(session, 1, documents)
    session.flush()
    assert stats["indexed"] > 0

    yield session
    session.close()


RECALL_GOLDEN_CASES: tuple[RecallCase, ...] = (
    RecallCase(
        query="最近体重变化怎么样",
        relevant_source_ids=frozenset({"2026-08-20", "2026-08-22"}),
        domain="body",
    ),
    RecallCase(
        query="午餐蛋白质摄入多少",
        relevant_source_ids=frozenset({"2026-08-20:lunch"}),
        domain="nutrition",
    ),
    RecallCase(
        query="深蹲训练记录",
        relevant_source_ids=frozenset({"{squat_id}"}),
        domain="fitness",
    ),
    RecallCase(
        query="卧推胸部训练",
        relevant_source_ids=frozenset({"{bench_id}"}),
        domain="fitness",
    ),
)


def _resolve_golden_cases(session) -> list[RecallCase]:
    """训练日志 id 在 SQLite 中自增，运行时解析 source_id。"""
    training_ids = [
        row.id
        for row in session.scalars(
            select(TrainingLog).where(TrainingLog.user_id == 1).order_by(TrainingLog.id)
        ).all()
    ]
    squat_id, bench_id = str(training_ids[0]), str(training_ids[1])
    resolved: list[RecallCase] = []
    for case in RECALL_GOLDEN_CASES:
        ids = {
            squat_id if item == "{squat_id}" else bench_id if item == "{bench_id}" else item
            for item in case.relevant_source_ids
        }
        resolved.append(
            RecallCase(query=case.query, relevant_source_ids=frozenset(ids), domain=case.domain)
        )
    return resolved


def test_rag_recall_at_k_on_golden_queries(rag_corpus_session):
    """金标查询集 Recall@5 应达到阈值。"""
    session = rag_corpus_session
    cases = _resolve_golden_cases(session)
    recalls: list[float] = []

    for case in cases:
        retrieved = memory_search_source_ids(
            session,
            1,
            case.query,
            top_k=_TOP_K,
            min_similarity=_MIN_SIMILARITY,
            domain=case.domain,
        )
        score = recall_at_k(retrieved, set(case.relevant_source_ids), _TOP_K)
        recalls.append(score)
        assert score > 0, f"查询「{case.query}」未召回任何相关块，TopK={retrieved}"

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= _RECALL_THRESHOLD, (
        f"平均 Recall@{_TOP_K}={mean_recall:.2f} 低于阈值 {_RECALL_THRESHOLD}，"
        f"各查询={list(zip([c.query for c in cases], recalls, strict=True))}"
    )


def test_rag_domain_filter_improves_cross_domain_separation(rag_corpus_session):
    """带 domain 过滤时，不应把无关域块排在相关块之前。"""
    session = rag_corpus_session
    squat_id = str(
        session.scalar(
            select(TrainingLog.id).where(TrainingLog.title == "腿部训练")
        )
    )

    without_domain = memory_search_source_ids(session, 1, "深蹲训练记录", top_k=3)
    with_domain = memory_search_source_ids(
        session, 1, "深蹲训练记录", top_k=3, domain="fitness"
    )

    assert squat_id in with_domain
    assert with_domain[0] == squat_id
    # 无 domain 时仍应能召回训练块（可能混有其它域，但首条仍应相关）
    assert squat_id in without_domain
