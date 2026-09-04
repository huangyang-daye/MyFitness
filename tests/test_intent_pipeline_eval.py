"""意图识别流水线分步评测（关键词路径，不调用 LLM）。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_intent_pipeline import evaluate_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "intent_dataset_300.json"


def test_intent_pipeline_keyword_steps():
    samples = json.loads(FIXTURE.read_text(encoding="utf-8"))
    steps, _, records = evaluate_pipeline(samples, use_llm=False)
    by_name = {step.name: step for step in steps}

    keyword = by_name["① 关键词分类"]
    final = by_name["⑥ 流水线最终输出"]

    assert keyword.evaluated == 300
    assert keyword.coverage == 1.0
    assert keyword.primary_accuracy >= 0.65
    assert final.primary_accuracy == keyword.primary_accuracy
    assert by_name["② LLM 意图 Agent"].evaluated == 0
    assert by_name["④ 关键词兜底路径"].coverage >= 0.5
    assert len(records) == 300
    assert set(records[0]) >= {"text", "final_intents", "expected_intents", "selected_step"}
