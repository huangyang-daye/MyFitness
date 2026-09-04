"""双路召回 + RRF 融合 + BM25 精排 + minScore 过滤。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from myfitness.rag.bm25 import BM25Index
from myfitness.rag.fusion import reciprocal_rank_fusion


@dataclass(frozen=True)
class RankedHit:
    doc_id: int
    score: float
    vector_score: float | None
    bm25_score: float
    rrf_score: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    return dot / (norm_l * norm_r)


def vector_knn(
    query_vector: list[float],
    corpus: list[tuple[int, list[float]]],
    top_k: int,
) -> list[tuple[int, float]]:
    """内存余弦 KNN，返回 (doc_id, cosine) 降序。"""
    if not query_vector or top_k <= 0:
        return []
    scored: list[tuple[int, float]] = []
    for doc_id, vector in corpus:
        similarity = cosine_similarity(query_vector, vector)
        if similarity > 0:
            scored.append((doc_id, similarity))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def hybrid_rank(
    query: str,
    documents: list[tuple[int, str]],
    vector_hits: list[tuple[int, float]],
    *,
    recall_k: int,
    top_k: int,
    rrf_k: int,
    min_score: float,
) -> list[RankedHit]:
    """向量 KNN 与 BM25 双路召回，RRF 融合后再用 BM25 精排并按 minScore 过滤。"""
    if not query.strip() or top_k <= 0:
        return []

    index = BM25Index.build(documents)
    bm25_hits = index.search(query, top_k=max(recall_k, 1))
    vector_ids = [doc_id for doc_id, _ in vector_hits[:recall_k]]
    bm25_ids = [doc_id for doc_id, _ in bm25_hits]

    fused = reciprocal_rank_fusion(
        [ranking for ranking in (vector_ids, bm25_ids) if ranking],
        k=rrf_k,
    )
    if not fused:
        return []

    query_tokens_scores = {doc_id: index.score(query, doc_id) for doc_id, _ in fused}
    reranked = sorted(
        fused,
        key=lambda item: query_tokens_scores.get(item[0], 0.0),
        reverse=True,
    )

    vector_map = dict(vector_hits)
    raw_scores = [query_tokens_scores.get(doc_id, 0.0) for doc_id, _ in reranked]
    max_bm25 = max(raw_scores, default=0.0)

    results: list[RankedHit] = []
    for (doc_id, rrf_score), raw_bm25 in zip(reranked, raw_scores, strict=True):
        vector_score = vector_map.get(doc_id)
        if max_bm25 > 0:
            norm = raw_bm25 / max_bm25
        else:
            norm = float(vector_score or 0.0)
        if norm < min_score:
            continue
        results.append(
            RankedHit(
                doc_id=doc_id,
                score=norm,
                vector_score=vector_score,
                bm25_score=raw_bm25,
                rrf_score=rrf_score,
            )
        )
        if len(results) >= top_k:
            break
    return results
