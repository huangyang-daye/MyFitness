"""按实际意图识别流水线分步评测准确率与召回率（300 条测评集）。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myfitness.graph.router import (
    IntentClassificationTrace,
    RouteResult,
    trace_classify_intent,
)
from myfitness.llm.factory import is_llm_configured
from myfitness.schemas.state import Intent, PendingConfirmation

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "intent_dataset_300.json"
REPORT_PATH = ROOT / "docs" / "intent_pipeline_evaluation_report.md"
CSV_PATH = ROOT / "docs" / "intent_pipeline_evaluation_results.csv"
TODAY = date(2026, 8, 23)

CSV_COLUMNS = [
    "id",
    "text",
    "difficulty",
    "expected_intents",
    "expected_domain",
    "keyword_intents",
    "keyword_domain",
    "keyword_start_date",
    "keyword_end_date",
    "llm_intents",
    "llm_domain",
    "llm_start_date",
    "llm_end_date",
    "reconciled_intents",
    "reconciled_domain",
    "reconciled_start_date",
    "reconciled_end_date",
    "final_intents",
    "final_domain",
    "final_start_date",
    "final_end_date",
    "selected_step",
    "final_primary_ok",
    "final_full_ok",
    "final_intent_recall",
    "final_intent_precision",
    "final_intent_f1",
    "keyword_primary_ok",
    "llm_primary_ok",
    "reconciled_primary_ok",
]


@dataclass
class StepMetrics:
    name: str
    description: str
    evaluated: int
    coverage: float
    primary_accuracy: float
    full_accuracy: float
    intent_recall: float
    intent_precision: float
    intent_f1: float

    def as_row(self) -> str:
        return (
            f"| {self.name} | {self.evaluated} | {self.coverage:.1%} | "
            f"{self.primary_accuracy:.1%} | {self.full_accuracy:.1%} | "
            f"{self.intent_recall:.1%} | {self.intent_precision:.1%} | {self.intent_f1:.1%} |"
        )


def _pending() -> PendingConfirmation:
    return PendingConfirmation(
        action_type="db_write",
        summary="测试确认",
        payload={},
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        domain="nutrition",
    )


def _intent_list(route: RouteResult | None) -> list[str]:
    if route is None:
        return []
    return [item.value for item in route.intents]


def _keyword_effective(trace: IntentClassificationTrace) -> RouteResult:
    if trace.keyword is not None:
        return trace.keyword
    return RouteResult(intents=[Intent.GENERAL])


def _format_intents(intents: list[str]) -> str:
    return "|".join(intents)


def _route_fields(route: RouteResult | None) -> dict[str, str]:
    if route is None:
        return {
            "intents": "",
            "domain": "",
            "start_date": "",
            "end_date": "",
        }
    return {
        "intents": _format_intents([item.value for item in route.intents]),
        "domain": route.domain or "",
        "start_date": route.start_date.isoformat() if route.start_date else "",
        "end_date": route.end_date.isoformat() if route.end_date else "",
    }


def _score(expected: list[str], predicted: list[str]) -> dict[str, float | bool]:
    primary_ok = bool(predicted and expected and predicted[0] == expected[0])
    full_ok = predicted == expected
    exp_set = set(expected)
    pred_set = set(predicted)
    inter = len(exp_set & pred_set)
    recall = inter / len(exp_set) if exp_set else 1.0
    precision = inter / len(pred_set) if pred_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "primary_ok": primary_ok,
        "full_ok": full_ok,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def _aggregate(name: str, description: str, scores: list[dict], *, total: int, covered: int) -> StepMetrics:
    if not scores:
        return StepMetrics(name, description, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(scores)
    return StepMetrics(
        name=name,
        description=description,
        evaluated=n,
        coverage=covered / total if total else 0.0,
        primary_accuracy=sum(bool(s["primary_ok"]) for s in scores) / n,
        full_accuracy=sum(bool(s["full_ok"]) for s in scores) / n,
        intent_recall=sum(float(s["recall"]) for s in scores) / n,
        intent_precision=sum(float(s["precision"]) for s in scores) / n,
        intent_f1=sum(float(s["f1"]) for s in scores) / n,
    )


def evaluate_pipeline(
    samples: list[dict], *, use_llm: bool
) -> tuple[list[StepMetrics], dict[str, dict[str, StepMetrics]], list[dict[str, object]]]:
    pending = _pending()
    total = len(samples)

    buckets: dict[str, list[dict]] = {
        "keyword": [],
        "llm_raw": [],
        "reconciled": [],
        "keyword_fallback_path": [],
        "default_general_path": [],
        "pipeline_final": [],
    }
    covered: dict[str, int] = defaultdict(int)
    by_difficulty: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    records: list[dict[str, object]] = []

    for item in samples:
        text = item["text"]
        expected = item["intents"]
        difficulty = item["difficulty"]
        use_pending = "confirmation_response" in expected

        trace = trace_classify_intent(
            text,
            pending if use_pending else None,
            use_llm=use_llm,
            today=TODAY,
        )
        final = trace.final

        keyword_route = _keyword_effective(trace)
        keyword_fields = _route_fields(keyword_route)
        llm_fields = _route_fields(trace.llm_raw)
        reconciled_fields = _route_fields(trace.reconciled)
        final_fields = _route_fields(final)

        keyword_score = _score(expected, _intent_list(keyword_route))
        buckets["keyword"].append(keyword_score)
        by_difficulty[difficulty]["keyword"].append(keyword_score)

        llm_score: dict[str, float | bool] | None = None
        if trace.llm_raw is not None:
            llm_score = _score(expected, _intent_list(trace.llm_raw))
            buckets["llm_raw"].append(llm_score)
            covered["llm_raw"] += 1
            by_difficulty[difficulty]["llm_raw"].append(llm_score)
        rec_score: dict[str, float | bool] | None = None
        if trace.reconciled is not None:
            rec_score = _score(expected, _intent_list(trace.reconciled))
            buckets["reconciled"].append(rec_score)
            covered["reconciled"] += 1
            by_difficulty[difficulty]["reconciled"].append(rec_score)
        if trace.selected_step == "keyword_fallback":
            fb_score = _score(expected, _intent_list(trace.final))
            buckets["keyword_fallback_path"].append(fb_score)
            covered["keyword_fallback_path"] += 1
            by_difficulty[difficulty]["keyword_fallback_path"].append(fb_score)
        if trace.selected_step == "default_general":
            gen_score = _score(expected, _intent_list(trace.final))
            buckets["default_general_path"].append(gen_score)
            covered["default_general_path"] += 1
            by_difficulty[difficulty]["default_general_path"].append(gen_score)

        final_score = _score(expected, _intent_list(final))
        buckets["pipeline_final"].append(final_score)
        covered["pipeline_final"] += 1
        by_difficulty[difficulty]["pipeline_final"].append(final_score)

        records.append(
            {
                "id": item["id"],
                "text": text,
                "difficulty": difficulty,
                "expected_intents": _format_intents(expected),
                "expected_domain": item.get("domain") or "",
                "keyword_intents": keyword_fields["intents"],
                "keyword_domain": keyword_fields["domain"],
                "keyword_start_date": keyword_fields["start_date"],
                "keyword_end_date": keyword_fields["end_date"],
                "llm_intents": llm_fields["intents"],
                "llm_domain": llm_fields["domain"],
                "llm_start_date": llm_fields["start_date"],
                "llm_end_date": llm_fields["end_date"],
                "reconciled_intents": reconciled_fields["intents"],
                "reconciled_domain": reconciled_fields["domain"],
                "reconciled_start_date": reconciled_fields["start_date"],
                "reconciled_end_date": reconciled_fields["end_date"],
                "final_intents": final_fields["intents"],
                "final_domain": final_fields["domain"],
                "final_start_date": final_fields["start_date"],
                "final_end_date": final_fields["end_date"],
                "selected_step": trace.selected_step,
                "final_primary_ok": final_score["primary_ok"],
                "final_full_ok": final_score["full_ok"],
                "final_intent_recall": final_score["recall"],
                "final_intent_precision": final_score["precision"],
                "final_intent_f1": final_score["f1"],
                "keyword_primary_ok": keyword_score["primary_ok"],
                "llm_primary_ok": llm_score["primary_ok"] if llm_score else "",
                "reconciled_primary_ok": rec_score["primary_ok"] if rec_score else "",
            }
        )

    steps = [
        _aggregate(
            "① 关键词分类",
            "仅 `_keyword_classify`；未命中时视为 general",
            buckets["keyword"],
            total=total,
            covered=total,
        ),
        _aggregate(
            "② LLM 意图 Agent",
            "`run_intent_agent` 原始输出（仅统计 LLM 成功返回的样本）",
            buckets["llm_raw"],
            total=total,
            covered=covered["llm_raw"],
        ),
        _aggregate(
            "③ LLM+关键词调和",
            "`_reconcile(llm, keyword)`（仅 LLM 成功样本）",
            buckets["reconciled"],
            total=total,
            covered=covered["reconciled"],
        ),
        _aggregate(
            "④ 关键词兜底路径",
            "LLM 失败或未启用时走 keyword_fallback 的样本",
            buckets["keyword_fallback_path"],
            total=total,
            covered=covered["keyword_fallback_path"],
        ),
        _aggregate(
            "⑤ 默认 GENERAL",
            "关键词未命中且 LLM 未接管时的样本",
            buckets["default_general_path"],
            total=total,
            covered=covered["default_general_path"],
        ),
        _aggregate(
            "⑥ 流水线最终输出",
            "`classify_intent` 完整路径最终意图",
            buckets["pipeline_final"],
            total=total,
            covered=covered["pipeline_final"],
        ),
    ]

    diff_metrics: dict[str, dict[str, StepMetrics]] = {}
    for difficulty, step_scores in by_difficulty.items():
        tier_total = sum(1 for s in samples if s["difficulty"] == difficulty)
        diff_metrics[difficulty] = {}
        for step_name, scores in step_scores.items():
            diff_metrics[difficulty][step_name] = _aggregate(
                step_name,
                "",
                scores,
                total=tier_total,
                covered=len(scores),
            )

    return steps, diff_metrics, records


def write_csv(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_report(
    steps: list[StepMetrics],
    diff_metrics: dict[str, dict[str, StepMetrics]],
    *,
    use_llm: bool,
    sample_count: int,
) -> str:
    mode = "LLM + 关键词（实际线上配置）" if use_llm else "仅关键词（use_llm=False）"
    lines = [
        "# 意图识别流水线分步评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 测评集：`intent_dataset_300.json`（{sample_count} 条）",
        f"- 评估模式：**{mode}**",
        f"- LLM 已配置：{'是' if is_llm_configured() else '否'}",
        f"- 固定 today：`{TODAY.isoformat()}`",
        f"- 明细 CSV：`intent_pipeline_evaluation_results.csv`",
        "",
        "## 指标说明",
        "",
        "- **覆盖率**：该步骤对多少比例样本产出了有效判定（如 LLM 步仅为成功调用子集）",
        "- **主意图准确率**：`intents[0]` 与标注一致的比例",
        "- **完整序列准确率**：`intents` 列表与标注完全一致",
        "- **意图召回率**：`|预测∩标注| / |标注|`（多意图宏平均）",
        "- **意图精确率**：`|预测∩标注| / |预测|`",
        "",
        "## 分步结果",
        "",
        "| 步骤 | 评测样本数 | 覆盖率 | 主意图准确 | 完整序列准确 | 意图召回 | 意图精确 | F1 |",
        "|------|------------|--------|------------|--------------|----------|----------|-----|",
    ]
    for step in steps:
        lines.append(step.as_row())

    lines.extend(["", "## 按难度（流水线最终输出）", ""])
    for difficulty in ("simple", "medium", "complex"):
        label = {"simple": "简单", "medium": "中等", "complex": "复杂"}[difficulty]
        final = diff_metrics.get(difficulty, {}).get("pipeline_final")
        if not final:
            continue
        lines.append(
            f"- **{label}**：主意图 {final.primary_accuracy:.1%}，"
            f"完整序列 {final.full_accuracy:.1%}，意图召回 {final.intent_recall:.1%}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="意图识别流水线分步评测")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调用 LLM，仅评测关键词与兜底路径（与线上下线 LLM 时一致）",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="推理明细 CSV 输出路径",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm and is_llm_configured()
    if not args.no_llm and not is_llm_configured():
        print("LLM 未配置，将仅运行关键词路径（等同 --no-llm）")

    samples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    steps, diff_metrics, records = evaluate_pipeline(samples, use_llm=use_llm)
    report = write_report(steps, diff_metrics, use_llm=use_llm, sample_count=len(samples))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    write_csv(records, args.csv)

    print(report)
    print()
    print(f"Report saved: {REPORT_PATH}")
    print(f"CSV saved:   {args.csv}")


if __name__ == "__main__":
    main()
