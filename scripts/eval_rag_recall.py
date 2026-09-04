"""RAG 检索准确率与召回率评测 — 确定性关键词向量 + 内存余弦检索。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_eval_support import (
    MIN_SIMILARITY,
    TOP_K,
    VECTOR_DIM,
    build_corpus_session,
    memory_search,
    resolve_placeholders,
)

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "rag_eval_dataset_100.json"
REPORT_PATH = ROOT / "docs" / "rag_recall_evaluation_report.md"
CSV_PATH = ROOT / "docs" / "rag_recall_evaluation_results.csv"

CSV_COLUMNS = [
    "id",
    "query",
    "difficulty",
    "domain",
    "expected_source_ids",
    "retrieved_source_ids",
    "retrieved_similarities",
    "recall_at_k",
    "precision_at_k",
    "hit_at_k",
    "mrr",
    "top1_ok",
]


@dataclass
class RagMetrics:
    name: str
    evaluated: int
    hit_rate: float
    recall: float
    precision: float
    f1: float
    mrr: float
    top1_accuracy: float

    def as_row(self) -> str:
        return (
            f"| {self.name} | {self.evaluated} | {self.hit_rate:.1%} | "
            f"{self.recall:.1%} | {self.precision:.1%} | {self.f1:.1%} | "
            f"{self.mrr:.3f} | {self.top1_accuracy:.1%} |"
        )


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> bool:
    return bool(set(retrieved[:k]) & relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, source_id in enumerate(retrieved, start=1):
        if source_id in relevant:
            return 1.0 / rank
    return 0.0


def _score_retrieval(
    retrieved: list[tuple[str, float]], relevant: set[str], *, k: int
) -> dict[str, float | bool]:
    source_ids = [sid for sid, _ in retrieved]
    rec = recall_at_k(source_ids, relevant, k)
    prec = precision_at_k(source_ids, relevant, k)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "hit": hit_at_k(source_ids, relevant, k),
        "mrr": mrr(source_ids, relevant),
        "top1_ok": bool(source_ids and source_ids[0] in relevant),
    }


def _aggregate(name: str, scores: list[dict[str, float | bool]]) -> RagMetrics:
    if not scores:
        return RagMetrics(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(scores)
    return RagMetrics(
        name=name,
        evaluated=n,
        hit_rate=sum(bool(s["hit"]) for s in scores) / n,
        recall=sum(float(s["recall"]) for s in scores) / n,
        precision=sum(float(s["precision"]) for s in scores) / n,
        f1=sum(float(s["f1"]) for s in scores) / n,
        mrr=sum(float(s["mrr"]) for s in scores) / n,
        top1_accuracy=sum(bool(s["top1_ok"]) for s in scores) / n,
    )


def evaluate(
    session,
    samples: list[dict],
    *,
    top_k: int,
    min_similarity: float,
    use_domain_filter: bool,
) -> tuple[RagMetrics, dict[str, RagMetrics], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    all_scores: list[dict[str, float | bool]] = []
    by_difficulty: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, float | bool]]] = defaultdict(list)

    for item in samples:
        domain = item.get("domain") if use_domain_filter else None
        retrieved = memory_search(
            session,
            1,
            item["query"],
            top_k=top_k,
            min_similarity=min_similarity,
            domain=domain,
        )
        relevant = set(item["relevant_source_ids"])
        score = _score_retrieval(retrieved, relevant, k=top_k)
        all_scores.append(score)
        by_difficulty[item["difficulty"]].append(score)
        if item.get("domain"):
            by_domain[item["domain"]].append(score)

        records.append(
            {
                "id": item["id"],
                "query": item["query"],
                "difficulty": item["difficulty"],
                "domain": item.get("domain") or "",
                "expected_source_ids": "|".join(item["relevant_source_ids"]),
                "retrieved_source_ids": "|".join(sid for sid, _ in retrieved),
                "retrieved_similarities": "|".join(f"{sim:.3f}" for _, sim in retrieved),
                "recall_at_k": score["recall"],
                "precision_at_k": score["precision"],
                "hit_at_k": score["hit"],
                "mrr": score["mrr"],
                "top1_ok": score["top1_ok"],
            }
        )

    overall = _aggregate("整体", all_scores)
    diff_metrics = {d: _aggregate(d, s) for d, s in by_difficulty.items()}
    domain_metrics = {d: _aggregate(d, s) for d, s in by_domain.items()}
    return overall, {**diff_metrics, **domain_metrics}, records


def write_csv(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_report(
    overall: RagMetrics,
    breakdown: dict[str, RagMetrics],
    *,
    sample_count: int,
    fixture_name: str,
    top_k: int,
    min_similarity: float,
    use_domain_filter: bool,
) -> str:
    mode = "带 domain 过滤" if use_domain_filter else "无 domain 过滤"
    lines = [
        "# RAG 检索准确率与召回率评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 测评集：`{fixture_name}`（{sample_count} 条）",
        f"- 检索模式：**{mode}**",
        f"- Top-K：{top_k}，最低相似度：{min_similarity}",
        f"- 向量方案：确定性关键词向量（{VECTOR_DIM} 维，余弦相似度，模拟 pgvector HNSW）",
        f"- 明细 CSV：`rag_recall_evaluation_results.csv`",
        "",
        "## 指标说明",
        "",
        "- **Hit@K（命中率/准确率）**：Top-K 中至少召回 1 个相关块",
        "- **Recall@K**：`|TopK ∩ 相关集| / |相关集|`",
        "- **Precision@K**：`|TopK ∩ 相关集| / |TopK|`",
        "- **Top-1 准确率**：排名第一的块属于相关集",
        "- **MRR**：首个相关块排名的倒数均值",
        "",
        "## 整体结果",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| Hit@{top_k} | {overall.hit_rate:.1%} |",
        f"| Recall@{top_k} | {overall.recall:.1%} |",
        f"| Precision@{top_k} | {overall.precision:.1%} |",
        f"| F1 | {overall.f1:.1%} |",
        f"| MRR | {overall.mrr:.3f} |",
        f"| Top-1 准确率 | {overall.top1_accuracy:.1%} |",
        "",
        "## 汇总表",
        "",
        "| 分组 | 样本数 | Hit@K | Recall@K | Precision@K | F1 | MRR | Top-1 |",
        "|------|--------|-------|----------|-------------|-----|-----|-------|",
        overall.as_row(),
    ]
    for key in ("simple", "medium", "complex", "body", "nutrition", "fitness"):
        metrics = breakdown.get(key)
        if metrics:
            label = {
                "simple": "简单",
                "medium": "中等",
                "complex": "复杂",
                "body": "身体",
                "nutrition": "饮食",
                "fitness": "训练",
            }.get(key, key)
            lines.append(metrics.as_row().replace(f"| {metrics.name} |", f"| {label} |", 1))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索准确率与召回率评测")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=FIXTURE_PATH,
        help="测评集 JSON 路径",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K, help="检索 Top-K")
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY, help="最低余弦相似度")
    parser.add_argument(
        "--no-domain-filter",
        action="store_true",
        help="不使用 domain 过滤（默认按标注 domain 过滤）",
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="明细 CSV 输出路径")
    args = parser.parse_args()

    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    session = build_corpus_session(expanded=len(samples) > 20)
    try:
        resolved = resolve_placeholders(session, samples)
        overall, breakdown, records = evaluate(
            session,
            resolved,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            use_domain_filter=not args.no_domain_filter,
        )
    finally:
        session.close()

    report = write_report(
        overall,
        breakdown,
        sample_count=len(samples),
        fixture_name=args.dataset.name,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        use_domain_filter=not args.no_domain_filter,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    write_csv(records, args.csv)

    print(report)
    print()
    print(f"Report saved: {REPORT_PATH}")
    print(f"CSV saved:   {args.csv}")


if __name__ == "__main__":
    main()
