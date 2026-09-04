"""Reciprocal Rank Fusion（RRF）多路结果融合。"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
) -> list[tuple[int, float]]:
    """按 `1 / (k + rank)` 累加各路排名，返回 (doc_id, rrf_score) 降序。"""
    if k < 1:
        raise ValueError("RRF k 须 >= 1")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
