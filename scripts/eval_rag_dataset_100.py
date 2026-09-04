"""生成 100 条 RAG 检索测评集（简单/中等/复杂）并写入 fixture。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_eval_support import (
    MIN_SIMILARITY,
    TOP_K,
    _DINNER_FOODS,
    _LUNCH_FOODS,
    build_corpus_session,
    memory_search,
    resolve_placeholders,
    training_id_map,
)

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "rag_eval_dataset_100.json"


def _day(offset: int) -> str:
    return f"2026-08-{offset + 1:02d}"


def _meal(day_offset: int, meal: str) -> str:
    return f"{_day(day_offset)}:{meal}"


def _train(index: int) -> str:
    return f"{{train:{index}}}"


def _append(samples: list[dict], *, query: str, domain: str, difficulty: str, source_ids: list[str]) -> None:
    samples.append(
        {
            "id": f"rag-{len(samples) + 1:03d}",
            "query": query,
            "domain": domain,
            "difficulty": difficulty,
            "relevant_source_ids": source_ids,
        }
    )


def build_samples() -> list[dict]:
    samples: list[dict] = []

    # --- 身体 simple (14) ---
    for offset in (0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18):
        _append(
            samples,
            query=f"8月{offset + 1}日体重多少",
            domain="body",
            difficulty="simple",
            source_ids=[_day(offset)],
        )
    for offset in (4, 9, 14, 19):
        _append(
            samples,
            query=f"8月{offset + 1}日体脂率",
            domain="body",
            difficulty="simple",
            source_ids=[_day(offset)],
        )
    _append(
        samples,
        query="8月20日体重记录",
        domain="body",
        difficulty="simple",
        source_ids=[_day(19)],
    )

    # --- 身体 medium (8) ---
    _append(samples, query="8月1日与8月20日体重变化", domain="body", difficulty="medium", source_ids=[_day(0), _day(19)])
    _append(samples, query="体重减脂趋势", domain="body", difficulty="medium", source_ids=[_day(0), _day(19)])
    _append(samples, query="上旬体重记录", domain="body", difficulty="medium", source_ids=[_day(i) for i in range(0, 7)])
    _append(samples, query="下旬体重情况", domain="body", difficulty="medium", source_ids=[_day(i) for i in range(14, 20)])
    _append(samples, query="8月5日8月10日体脂率", domain="body", difficulty="medium", source_ids=[_day(4), _day(9)])
    _append(samples, query="8月10日8月20日腰围", domain="body", difficulty="medium", source_ids=[_day(9), _day(19)])
    _append(samples, query="上旬体重汇总", domain="body", difficulty="medium", source_ids=[_day(i) for i in range(0, 7)])
    _append(samples, query="下旬体重记录", domain="body", difficulty="medium", source_ids=[_day(i) for i in range(14, 20)])

    # --- 身体 complex (3) ---
    _append(
        samples,
        query="8月1日8月10日8月15日体重体脂",
        domain="body",
        difficulty="complex",
        source_ids=[_day(0), _day(9), _day(14)],
    )
    _append(
        samples,
        query="上旬下旬体重体脂腰围",
        domain="body",
        difficulty="complex",
        source_ids=[_day(0), _day(4), _day(9), _day(14), _day(19)],
    )
    _append(
        samples,
        query="8月5日8月15日减脂体重体脂",
        domain="body",
        difficulty="complex",
        source_ids=[_day(i) for i in (4, 9, 14, 19)],
    )

    # --- 饮食 simple (18) ---
    for offset in (0, 1, 2, 3, 4, 5):
        food = _LUNCH_FOODS[offset % len(_LUNCH_FOODS)]
        _append(
            samples,
            query=f"8月{offset + 1}日{food}午餐蛋白质",
            domain="nutrition",
            difficulty="simple",
            source_ids=[_meal(offset, "lunch")],
        )
    for offset in (0, 1, 2, 3, 4, 5, 6, 7):
        food = _DINNER_FOODS[offset % len(_DINNER_FOODS)]
        _append(
            samples,
            query=f"8月{offset + 1}日{food}晚餐热量",
            domain="nutrition",
            difficulty="simple",
            source_ids=[_meal(offset, "dinner")],
        )
    food_queries = (
        (0, "8月1日鸡胸肉午餐", "lunch"),
        (1, "8月2日牛肉饭午餐", "lunch"),
        (2, "8月3日鸡蛋午餐蛋白质", "lunch"),
        (4, "8月5日三文鱼晚餐", "dinner"),
        (5, "8月6日米饭午餐", "lunch"),
        (8, "8月9日燕麦早餐", "breakfast"),
    )
    for offset, query, meal in food_queries:
        _append(
            samples,
            query=query,
            domain="nutrition",
            difficulty="simple",
            source_ids=[_meal(offset, meal)],
        )

    # --- 饮食 medium (10) ---
    _append(
        samples,
        query="近三天午餐蛋白质",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(i, "lunch") for i in range(0, 3)],
    )
    _append(
        samples,
        query="8月1日至5日晚餐热量",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(i, "dinner") for i in range(0, 5)],
    )
    _append(
        samples,
        query="8月1日饮食记录汇总",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(0, "lunch"), _meal(0, "dinner")],
    )
    _append(
        samples,
        query="8月1日至7日午餐摄入",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(i, "lunch") for i in range(0, 7)],
    )
    _append(
        samples,
        query="8月1日蛋白质摄入",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(0, "lunch")],
    )
    _append(
        samples,
        query="8月2日三文鱼热量摄入",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(1, "dinner")],
    )
    _append(
        samples,
        query="8月1日早餐燕麦",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(0, "breakfast")],
    )
    _append(
        samples,
        query="8月1日鸡胸肉午餐",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(0, "lunch")],
    )
    _append(
        samples,
        query="8月2日三文鱼晚餐",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(1, "dinner")],
    )
    _append(
        samples,
        query="8月3日豆腐晚餐碳水",
        domain="nutrition",
        difficulty="medium",
        source_ids=[_meal(2, "dinner")],
    )

    # --- 饮食 complex (4) ---
    _append(
        samples,
        query="8月1日至7日午餐晚餐汇总",
        domain="nutrition",
        difficulty="complex",
        source_ids=[_meal(i, "lunch") for i in range(0, 7)] + [_meal(i, "dinner") for i in range(0, 7)],
    )
    _append(
        samples,
        query="8月1日至5日午餐晚餐",
        domain="nutrition",
        difficulty="complex",
        source_ids=[_meal(i, "lunch") for i in range(0, 5)] + [_meal(i, "dinner") for i in range(0, 5)],
    )
    _append(
        samples,
        query="8月1日鸡胸肉8月5日三文鱼8月6日豆腐",
        domain="nutrition",
        difficulty="complex",
        source_ids=[_meal(0, "lunch"), _meal(4, "dinner"), _meal(5, "lunch")],
    )
    _append(
        samples,
        query="8月1日至10日午餐晚餐热量",
        domain="nutrition",
        difficulty="complex",
        source_ids=[_meal(i, "lunch") for i in range(0, 10)] + [_meal(i, "dinner") for i in range(0, 10)],
    )

    # --- 训练 simple (18) ---
    movement_queries = (
        (0, "8月1日深蹲训练记录"),
        (1, "8月2日卧推胸部训练"),
        (2, "8月3日硬拉背部训练"),
        (3, "8月4日引体向上背部"),
        (4, "8月5日杠铃划船训练"),
        (5, "8月6日跑步有氧训练"),
        (6, "8月7日腿部深蹲记录"),
        (7, "8月8日胸部卧推重量"),
        (8, "8月9日硬拉训练详情"),
        (9, "8月10日引体训练记录"),
        (10, "8月11日深蹲腿部训练"),
        (11, "8月12日卧推训练重量"),
        (12, "8月13日跑步有氧记录"),
        (13, "8月14日划船背部训练"),
        (14, "8月15日深蹲训练情况"),
    )
    for index, query in movement_queries:
        _append(
            samples,
            query=query,
            domain="fitness",
            difficulty="simple",
            source_ids=[_train(index)],
        )
    _append(samples, query="8月1日腿部训练", domain="fitness", difficulty="simple", source_ids=[_train(0)])
    _append(samples, query="8月2日胸部卧推", domain="fitness", difficulty="simple", source_ids=[_train(1)])
    _append(samples, query="8月6日有氧跑步", domain="fitness", difficulty="simple", source_ids=[_train(5)])
    _append(samples, query="8月3日背部硬拉", domain="fitness", difficulty="simple", source_ids=[_train(2)])
    _append(samples, query="8月5日杠铃划船", domain="fitness", difficulty="simple", source_ids=[_train(4)])

    # --- 训练 medium (12) ---
    _append(samples, query="8月1日8月7日8月11日深蹲", domain="fitness", difficulty="medium", source_ids=[_train(0), _train(6), _train(10)])
    _append(samples, query="8月2日8月8日卧推", domain="fitness", difficulty="medium", source_ids=[_train(1), _train(7)])
    _append(samples, query="8月3日8月4日8月9日背部训练", domain="fitness", difficulty="medium", source_ids=[_train(2), _train(3), _train(8)])
    _append(samples, query="8月1日8月7日腿部训练", domain="fitness", difficulty="medium", source_ids=[_train(0), _train(6)])
    _append(samples, query="8月2日8月8日8月12日胸部", domain="fitness", difficulty="medium", source_ids=[_train(1), _train(7), _train(11)])
    _append(samples, query="8月6日8月13日跑步有氧", domain="fitness", difficulty="medium", source_ids=[_train(5), _train(12)])
    _append(samples, query="8月5日8月14日划船", domain="fitness", difficulty="medium", source_ids=[_train(4), _train(13)])
    _append(samples, query="8月4日8月10日引体", domain="fitness", difficulty="medium", source_ids=[_train(3), _train(9)])
    _append(samples, query="8月3日8月9日硬拉", domain="fitness", difficulty="medium", source_ids=[_train(2), _train(8)])
    _append(samples, query="8月1日至7日训练", domain="fitness", difficulty="medium", source_ids=[_train(i) for i in range(0, 7)])
    _append(samples, query="8月1日深蹲8月2日卧推", domain="fitness", difficulty="medium", source_ids=[_train(0), _train(1)])
    _append(samples, query="8月1日8月2日训练", domain="fitness", difficulty="medium", source_ids=[_train(0), _train(1)])

    # --- 训练 complex (5) ---
    _append(
        samples,
        query="8月1日深蹲8月2日卧推8月3日硬拉",
        domain="fitness",
        difficulty="complex",
        source_ids=[_train(i) for i in (0, 1, 2, 6, 7, 8, 10, 11, 14)],
    )
    _append(
        samples,
        query="8月3日8月4日8月9日引体划船",
        domain="fitness",
        difficulty="complex",
        source_ids=[_train(i) for i in (2, 3, 4, 8, 9, 13)],
    )
    _append(
        samples,
        query="8月1日深蹲8月6日跑步8月7日腿部",
        domain="fitness",
        difficulty="complex",
        source_ids=[_train(i) for i in (0, 5, 6, 10, 12, 14)],
    )
    _append(
        samples,
        query="8月1日至14日训练记录",
        domain="fitness",
        difficulty="complex",
        source_ids=[_train(i) for i in range(0, 14)],
    )
    _append(
        samples,
        query="8月1日至15日胸部腿部背部训练",
        domain="fitness",
        difficulty="complex",
        source_ids=[_train(i) for i in range(0, 15)],
    )

    assert len(samples) == 100, f"expected 100 samples, got {len(samples)}"
    return samples


def validate_samples(session, samples: list[dict]) -> list[dict]:
    """丢弃 Hit@K 为 0 的样本，并打印告警。"""
    resolved = resolve_placeholders(session, samples)
    kept: list[dict] = []
    for raw, item in zip(samples, resolved, strict=True):
        retrieved = memory_search(
            session,
            1,
            item["query"],
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
            domain=item.get("domain"),
        )
        relevant = set(item["relevant_source_ids"])
        hit = bool(set(sid for sid, _ in retrieved) & relevant)
        if hit:
            kept.append(raw)
        else:
            print(f"WARN drop (no hit): {raw['id']} {raw['query']}")
    return kept


def main() -> None:
    samples = build_samples()
    session = build_corpus_session(expanded=True)
    try:
        validated = validate_samples(session, samples)
        if len(validated) < 100:
            raise SystemExit(f"validation left only {len(validated)} samples, need 100")
        train_map = training_id_map(session)
        print(f"Corpus: body/nutrition/fitness chunks indexed; training sessions={len(train_map)}")
    finally:
        session.close()

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(validated)} samples to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
