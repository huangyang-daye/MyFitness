"""BM25 倒排索引与打分。

中文用字 unigram + bigram，英文/数字按词切分，不引入分词依赖。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

_LATIN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """把查询或文档切成 BM25 词项。"""
    if not text or not text.strip():
        return []
    tokens = _LATIN.findall(text.lower())
    chars = _CJK_CHAR.findall(text)
    tokens.extend(chars)
    tokens.extend(left + right for left, right in pairwise(chars))
    return tokens


def bm25_document_text(title: str, content: str) -> str:
    """标题重复一次，提高标题命中权重。"""
    title = (title or "").strip()
    content = (content or "").strip()
    if title and title not in content:
        return f"{title}\n{title}\n{content}"
    if title:
        return f"{title}\n{content}"
    return content


@dataclass
class BM25Index:
    """内存倒排索引 + Okapi BM25。"""

    k1: float = 1.5
    b: float = 0.75
    n: int = 0
    avgdl: float = 0.0
    doc_len: dict[int, int] = field(default_factory=dict)
    df: dict[str, int] = field(default_factory=dict)
    postings: dict[str, dict[int, int]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        documents: list[tuple[int, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        postings: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        df: dict[str, int] = defaultdict(int)
        doc_len: dict[int, int] = {}
        for doc_id, text in documents:
            tokens = tokenize(text)
            doc_len[doc_id] = len(tokens)
            seen: set[str] = set()
            for token in tokens:
                postings[token][doc_id] += 1
                if token not in seen:
                    df[token] += 1
                    seen.add(token)
        n = len(doc_len)
        return cls(
            k1=k1,
            b=b,
            n=n,
            avgdl=(sum(doc_len.values()) / n) if n else 0.0,
            doc_len=doc_len,
            df=dict(df),
            postings={term: dict(tf) for term, tf in postings.items()},
        )

    def idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        return math.log(1.0 + (self.n - n_q + 0.5) / (n_q + 0.5))

    def score_tokens(self, query_tokens: list[str], doc_id: int) -> float:
        dl = self.doc_len.get(doc_id)
        if not query_tokens or not dl or self.avgdl <= 0:
            return 0.0
        score = 0.0
        for term in set(query_tokens):
            tf = self.postings.get(term, {}).get(doc_id, 0)
            if tf <= 0:
                continue
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            score += self.idf(term) * (tf * (self.k1 + 1.0)) / denom
        return score

    def score(self, query: str, doc_id: int) -> float:
        return self.score_tokens(tokenize(query), doc_id)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """倒排召回：只扫描查询词命中的文档。"""
        tokens = tokenize(query)
        if not tokens or self.n == 0 or top_k <= 0:
            return []
        candidates: set[int] = set()
        for term in set(tokens):
            candidates.update(self.postings.get(term, ()))
        scored = [(doc_id, self.score_tokens(tokens, doc_id)) for doc_id in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(doc_id, score) for doc_id, score in scored[:top_k] if score > 0]
