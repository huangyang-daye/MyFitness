"""双路召回：BM25 倒排、RRF 融合、BM25 精排。"""

from datetime import date
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.config import Settings
from myfitness.db.models import Base, RagChunk, User
from myfitness.rag.bm25 import BM25Index, tokenize
from myfitness.rag.fusion import reciprocal_rank_fusion
from myfitness.rag.hybrid import hybrid_rank
from myfitness.rag.store import search_chunks


def test_tokenize_chinese_and_latin():
    tokens = tokenize("深蹲 squat 80kg")
    assert "深蹲" in tokens
    assert "深" in tokens
    assert "蹲" in tokens
    assert "squat" in tokens
    assert "80kg" in tokens


def test_bm25_ranks_keyword_overlap_higher():
    index = BM25Index.build(
        [
            (1, "2026-08-18 训练：腿部训练\n- 深蹲：80kg x 5"),
            (2, "2026-08-20 身体数据：体重 72.5kg"),
            (3, "2026-08-21 晚餐：牛肉饭"),
        ]
    )
    hits = index.search("深蹲重量", top_k=3)
    assert hits
    assert hits[0][0] == 1
    assert hits[0][1] > hits[-1][1]


def test_rrf_prefers_docs_in_both_lists():
    fused = reciprocal_rank_fusion(
        [
            [1, 2, 3],
            [2, 4, 5],
        ],
        k=60,
    )
    assert fused[0][0] == 2
    scores = dict(fused)
    assert scores[2] > scores[1]
    assert scores[2] > scores[4]


def test_hybrid_keyword_path_finds_doc_without_vectors():
    ranked = hybrid_rank(
        "午餐鸡胸肉蛋白质",
        [
            (1, "2026-08-20 午餐：鸡胸肉 200g，330 kcal，蛋白 62g"),
            (2, "2026-08-18 训练：腿部训练 深蹲"),
            (3, "2026-08-22 身体数据：体重 71.8kg"),
        ],
        vector_hits=[],
        recall_k=10,
        top_k=3,
        rrf_k=60,
        min_score=0.0,
    )
    assert ranked
    assert ranked[0].doc_id == 1
    assert ranked[0].bm25_score > 0
    assert ranked[0].score == 1.0


def test_hybrid_rrf_then_bm25_rerank():
    ranked = hybrid_rank(
        "深蹲",
        [
            (1, "今天天气不错，去公园散步"),
            (2, "深蹲 80kg 五组"),
            (3, "卧推 60kg"),
        ],
        vector_hits=[(1, 0.92), (3, 0.80), (2, 0.40)],
        recall_k=10,
        top_k=3,
        rrf_k=60,
        min_score=0.0,
    )
    assert ranked[0].doc_id == 2
    assert ranked[0].vector_score == 0.40
    assert ranked[0].rrf_score > 0


def test_hybrid_minscore_drops_weak_bm25():
    ranked = hybrid_rank(
        "深蹲",
        [
            (1, "深蹲 80kg"),
            (2, "今天吃了米饭"),
        ],
        vector_hits=[(1, 0.9), (2, 0.85)],
        recall_k=10,
        top_k=5,
        rrf_k=60,
        min_score=0.5,
    )
    ids = [hit.doc_id for hit in ranked]
    assert 1 in ids
    assert 2 not in ids


def test_hybrid_falls_back_to_vector_when_bm25_is_zero():
    ranked = hybrid_rank(
        "xyzzy",
        [
            (1, "体重 72kg"),
            (2, "深蹲训练"),
        ],
        vector_hits=[(1, 0.8), (2, 0.2)],
        recall_k=10,
        top_k=2,
        rrf_k=60,
        min_score=0.35,
    )
    assert [hit.doc_id for hit in ranked] == [1]
    assert ranked[0].score == 0.8
    assert ranked[0].bm25_score == 0.0


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()
    return session


def _add_chunk(
    session,
    *,
    chunk_id: int,
    source_id: str,
    domain: str,
    title: str,
    content: str,
    embedding: list[float] | None = None,
) -> None:
    session.add(
        RagChunk(
            id=chunk_id,
            user_id=1,
            source_type=domain,
            source_id=source_id,
            domain=domain,
            record_date=date(2026, 8, 20),
            title=title,
            content=content,
            content_hash=f"hash-{chunk_id}",
            embedding=embedding,
        )
    )


def test_search_chunks_bm25_works_without_embeddings():
    session = _session()
    _add_chunk(
        session,
        chunk_id=1,
        source_id="2026-08-20:lunch",
        domain="nutrition",
        title="午餐",
        content="2026-08-20 午餐：鸡胸肉 200g，蛋白 62g",
    )
    _add_chunk(
        session,
        chunk_id=2,
        source_id="2026-08-18",
        domain="fitness",
        title="腿部训练",
        content="2026-08-18 训练：深蹲 80kg",
    )
    session.flush()
    settings = Settings(rag_enabled=True, rag_top_k=5, rag_min_similarity=0.0)
    with patch("myfitness.rag.store.get_settings", return_value=settings):
        hits = search_chunks(session, 1, "鸡胸肉蛋白质")
    assert hits
    assert hits[0].source_id == "2026-08-20:lunch"
    assert hits[0].metadata is not None
    assert hits[0].metadata["bm25_score"] > 0
    assert hits[0].metadata["vector_score"] is None


def test_search_chunks_hybrid_with_mocked_vectors():
    session = _session()
    _add_chunk(
        session,
        chunk_id=1,
        source_id="squat",
        domain="fitness",
        title="腿部训练",
        content="深蹲 80kg x 5",
        embedding=[1.0, 0.0, 0.0],
    )
    _add_chunk(
        session,
        chunk_id=2,
        source_id="walk",
        domain="fitness",
        title="散步",
        content="公园散步 30 分钟",
        embedding=[0.0, 1.0, 0.0],
    )
    session.flush()
    settings = Settings(rag_enabled=True, rag_top_k=2, rag_min_similarity=0.0)

    def fake_embed(text: str, settings=None):
        if "深蹲" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    with (
        patch("myfitness.rag.store.get_settings", return_value=settings),
        patch("myfitness.rag.embedding.embed_text", side_effect=fake_embed),
    ):
        hits = search_chunks(session, 1, "今天深蹲多重")
    assert hits[0].source_id == "squat"
    assert hits[0].metadata["vector_score"] is not None
    assert hits[0].metadata["rrf_score"] > 0


def test_search_chunks_disabled_returns_empty():
    session = _session()
    _add_chunk(
        session,
        chunk_id=1,
        source_id="a",
        domain="body",
        title="体重",
        content="体重 70kg",
    )
    session.flush()
    settings = Settings(rag_enabled=False)
    with patch("myfitness.rag.store.get_settings", return_value=settings):
        assert search_chunks(session, 1, "体重") == []
