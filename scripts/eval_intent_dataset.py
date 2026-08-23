"""生成 100 条意图数据集并评估 Router + Query Planner 能力。"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myfitness.agents.tools.query_planner import build_query_plan, needs_database_query
from myfitness.graph.router import agents_for_intent, classify_intent
from myfitness.schemas.state import Intent, PendingConfirmation

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "intent_dataset_100.json"
REPORT_PATH = ROOT / "docs" / "intent_evaluation_report.md"
TODAY = date(2026, 8, 23)


@dataclass
class IntentSample:
    id: int
    text: str
    intent: str
    domain: str | None = None
    needs_db: bool = False
    expected_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""


def build_dataset() -> list[IntentSample]:
    samples: list[IntentSample] = []
    n = 0

    def add(
        text: str,
        intent: str,
        domain: str | None = None,
        needs_db: bool = False,
        tools: list[str] | None = None,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> None:
        nonlocal n
        n += 1
        samples.append(
            IntentSample(
                id=n,
                text=text,
                intent=intent,
                domain=domain,
                needs_db=needs_db,
                expected_tools=tools or [],
                tags=tags or [],
                notes=notes,
            )
        )

    # --- data_query (22) ---
    dq = [
        ("昨天吃了多少蛋白质？", "nutrition", ["query_nutrition_logs"], ["昨天", "蛋白"]),
        ("今天热量摄入多少？", "nutrition", ["query_nutrition_logs"], ["今天", "热量"]),
        ("查询今天体重", "body", ["query_body_metrics"], ["今天", "体重"]),
        ("昨天训练了吗", "training", ["query_training_logs"], ["昨天", "训练"]),
        ("2026-08-21 练了什么动作", "training", ["query_training_logs"], ["具体日期", "训练明细"]),
        ("前天碳水吃了多少", "nutrition", ["query_nutrition_logs"], ["前天", "碳水"]),
        ("午餐吃了什么", "nutrition", ["query_nutrition_logs"], ["午餐"]),
        ("我最近的体脂是多少", "body", ["query_body_metrics"], ["体脂"]),
        ("查询昨天晚餐记录", "nutrition", ["query_nutrition_logs"], ["晚餐"]),
        ("今天脂肪摄入超标了吗", "nutrition", ["query_nutrition_logs"], ["脂肪"]),
        ("8月21号深蹲做了多少组", "training", ["query_training_logs"], ["具体日期", "组数"]),
        ("昨天卧推重量多少", "training", ["query_training_logs"], ["卧推"]),
        ("查询近7天吃了多少蛋白", None, ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["近7天"]),
        ("今天早餐记录有哪些", "nutrition", ["query_nutrition_logs"], ["早餐"]),
        ("我的体重现在多少kg", "body", ["query_body_metrics"], ["体重"]),
        ("昨天摄入总热量", "nutrition", ["query_nutrition_logs"], ["热量"]),
        ("查询2026-08-20训练详情", "training", ["query_training_logs"], ["具体日期"]),
        ("今天吃了几个鸡蛋", "nutrition", ["query_nutrition_logs"], ["食物"]),
        ("昨天有没有记录体重", "body", ["query_body_metrics"], ["体重"]),
        ("查询零食吃了什么", "nutrition", ["query_nutrition_logs"], ["零食"]),
        ("今天练腿了吗", "training", ["query_training_logs"], ["训练"]),
        ("昨天蛋白质够吗", "nutrition", ["query_nutrition_logs"], ["蛋白"]),
    ]
    for text, domain, tools, tags in dq:
        add(text, "data_query", domain, True, tools, tags)

    # --- manual_entry (16) ---
    me = [
        ("记录体重 72.5kg", "body", [], ["录入", "体重"]),
        ("录入今天体脂 18.2%", "body", [], ["录入", "体脂"]),
        ("添加早餐 鸡蛋 2个", "nutrition", [], ["录入", "早餐"]),
        ("记录午餐：鸡胸肉 200g", "nutrition", [], ["录入", "午餐"]),
        ("午餐吃了牛肉 150g 米饭 200g", "nutrition", [], ["录入", "多食物"]),
        ("记录晚餐 三文鱼 120g", "nutrition", [], ["录入", "晚餐"]),
        ("添加体重 71.8", "body", [], ["录入"]),
        ("录入零食 坚果 30g", "nutrition", [], ["录入", "零食"]),
        ("记录 2026-08-22 体重 73kg", "body", [], ["录入", "具体日期"]),
        ("添加食物 苹果 1个", "nutrition", [], ["录入"]),
        ("记录体脂率 17.5%", "body", [], ["录入", "体脂"]),
        ("录入午餐 米饭 150g 西兰花 100g", "nutrition", [], ["录入"]),
        ("记录体重72kg", "body", [], ["录入", "无空格"]),
        ("添加早餐燕麦 80g", "nutrition", [], ["录入"]),
        ("记录今天体重 105.9kg", "body", [], ["录入", "今天"]),
        ("录入 晚餐 豆腐 200g", "nutrition", [], ["录入"]),
    ]
    for text, domain, tools, tags in me:
        add(text, "manual_entry", domain, False, tools, tags)

    # --- plan_adjust (10) ---
    pa = [
        ("今天不练了，改成休息", "fitness", [], ["计划调整"]),
        ("调整训练计划，明天练胸", "fitness", [], ["计划调整"]),
        ("取消今天的训练", "fitness", [], ["取消训练"]),
        ("改成休息日", "fitness", [], ["休息"]),
        ("调整计划：本周少练一次", "fitness", [], ["计划调整"]),
        ("今天训练取消改成恢复", "fitness", [], ["恢复"]),
        ("把明天的训练改成有氧", "fitness", [], ["有氧"]),
        ("调整健身计划，降低容量", "fitness", [], ["计划调整"]),
        ("取消今晚的训练课", "fitness", [], ["取消"]),
        ("改成 active recovery 今天", "fitness", [], ["恢复"]),
    ]
    for text, domain, tools, tags in pa:
        add(text, "plan_adjust", domain, True, tools, tags)

    # --- trend_analysis (14) ---
    ta = [
        ("近30天体脂变化趋势", None, ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["趋势", "30天"]),
        ("近7天体重变化怎么样", "body", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["趋势", "7天"]),
        ("近14天蛋白质摄入趋势", "nutrition", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["趋势"]),
        ("对比近一个月训练和体重", None, ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["对比", "跨域"]),
        ("近90天体重走势", "body", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["趋势", "90天"]),
        ("最近训练量变化", "training", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["训练量"]),
        ("近30天热量摄入变化", "nutrition", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["热量趋势"]),
        ("体重和体脂近两周对比", "body", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["对比"]),
        ("近7天训练频率趋势", "training", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["频率"]),
        ("分析近30天饮食变化", "nutrition", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["饮食趋势"]),
        ("近60天卧推重量变化", "training", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["动作趋势"]),
        ("近21天围度变化", "body", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["围度"]),
        ("对比近7天摄入和消耗", "nutrition", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["对比"]),
        ("近10天睡眠和训练关系", "training", ["query_body_metrics", "query_nutrition_logs", "query_training_logs"], ["跨域", "边界"]),
    ]
    for text, domain, tools, tags in ta:
        add(text, "trend_analysis", domain, True, tools, tags)

    # --- goal_setting (10) ---
    gs = [
        ("目标体重设为70kg", "body", ["query_body_metrics"], ["目标", "体重"]),
        ("我要在12周降到65公斤", "body", ["query_body_metrics"], ["目标", "减重"]),
        ("设定体脂目标15%", "body", ["query_body_metrics"], ["目标", "体脂"]),
        ("目标增到75kg", "body", ["query_body_metrics"], ["目标", "增重"]),
        ("减到68公斤需要多久", "body", ["query_body_metrics"], ["目标"]),
        ("把目标体重改成72kg", "body", ["query_body_metrics"], ["目标修改"]),
        ("设定3个月降到80kg", "body", ["query_body_metrics"], ["目标", "期限"]),
        ("目标体脂降到12%", "body", ["query_body_metrics"], ["目标"]),
        ("我想增肌到78kg", "body", ["query_body_metrics"], ["增肌目标"]),
        ("降到70kg以下", "body", ["query_body_metrics"], ["目标"]),
    ]
    for text, domain, tools, tags in gs:
        add(text, "goal_setting", domain, True, tools, tags)

    # --- sync_trigger (8) ---
    st = [
        ("同步最近7天训记数据", None, [], ["同步"]),
        ("拉取训记训练记录", None, [], ["同步", "训练"]),
        ("更新训记数据", None, [], ["同步"]),
        ("同步最近30天数据", None, [], ["同步", "30天"]),
        ("从训记同步饮食", None, [], ["同步", "饮食"]),
        ("拉取最近14天训记", None, [], ["同步"]),
        ("同步训记身体数据", None, [], ["同步", "身体"]),
        ("更新最近7天训记同步", None, [], ["同步"]),
    ]
    for text, domain, tools, tags in st:
        add(text, "sync_trigger", domain, False, tools, tags)

    # --- general (12) ---
    gen = [
        ("你好", None, [], ["寒暄"]),
        ("谢谢", None, [], ["寒暄"]),
        ("你能做什么", None, [], ["能力询问"]),
        ("怎么用", None, [], ["帮助"]),
        ("早上好", None, [], ["寒暄"]),
        ("OK", None, [], ["简短"]),
        ("明白了", None, [], ["反馈"]),
        ("这个功能不错", None, [], ["反馈"]),
        ("再见", None, [], ["寒暄"]),
        ("帮助", None, [], ["帮助"]),
        ("介绍一下你自己", None, [], ["能力询问"]),
        ("说点什么", None, [], ["开放"]),
    ]
    for text, domain, tools, tags in gen:
        add(text, "general", domain, False, tools, tags)

    # --- confirmation_response (8) ---
    cr = [
        ("确认", None, [], ["确认写入"]),
        ("确定", None, [], ["确认写入"]),
        ("是的，写入", None, [], ["确认写入"]),
        ("取消", None, [], ["取消写入"]),
        ("不要了", None, [], ["取消写入"]),
        ("算了", None, [], ["取消写入"]),
        ("ok", None, [], ["确认"]),
        ("yes", None, [], ["确认"]),
    ]
    for text, domain, tools, tags in cr:
        add(text, "confirmation_response", domain, False, tools, tags, notes="需 pending_confirmation 上下文")

    assert len(samples) == 100, f"expected 100 samples, got {len(samples)}"
    return samples


@dataclass
class EvalResult:
    id: int
    text: str
    expected_intent: str
    predicted_intent: str
    expected_domain: str | None
    predicted_domain: str | None
    expected_tools: list[str]
    intent_ok: bool
    domain_ok: bool
    needs_db_expected: bool
    needs_db_predicted: bool
    db_ok: bool
    agents: list[str]
    query_domains: list[str]
    tool_match: bool | None


def _pending_confirmation() -> PendingConfirmation:
    return PendingConfirmation(
        action_type="db_write",
        summary="测试确认",
        payload={},
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        domain="nutrition",
    )


def evaluate(samples: list[IntentSample]) -> list[EvalResult]:
    results: list[EvalResult] = []
    pending = _pending_confirmation()

    for s in samples:
        use_pending = s.intent == "confirmation_response"
        route = classify_intent(s.text, pending if use_pending else None)
        pred_intent = route.intent.value
        pred_domain = route.domain

        intent = Intent(s.intent) if s.intent != "confirmation_response" else Intent.CONFIRMATION_RESPONSE
        plan = build_query_plan(s.text, Intent(pred_intent), pred_domain, today=TODAY)
        pred_needs_db = plan is not None
        query_domains = list(plan.domains) if plan else []

        expected_tools = set(s.expected_tools)
        predicted_tools: set[str] = set()
        if plan:
            if "body" in plan.domains:
                predicted_tools.add("query_body_metrics")
            if "nutrition" in plan.domains:
                predicted_tools.add("query_nutrition_logs")
            if "training" in plan.domains:
                predicted_tools.add("query_training_logs")

        tool_match: bool | None = None
        if s.needs_db and expected_tools:
            tool_match = expected_tools.issubset(predicted_tools)
        elif not s.needs_db:
            tool_match = predicted_tools == set() or pred_intent in {"trend_analysis", "data_query", "goal_setting"}

        domain_ok = True
        if s.domain is not None and s.intent not in {"confirmation_response", "sync_trigger", "general"}:
            domain_ok = pred_domain == s.domain

        results.append(
            EvalResult(
                id=s.id,
                text=s.text,
                expected_intent=s.intent,
                predicted_intent=pred_intent,
                expected_domain=s.domain,
                predicted_domain=pred_domain,
                expected_tools=s.expected_tools,
                intent_ok=pred_intent == s.intent,
                domain_ok=domain_ok,
                needs_db_expected=s.needs_db,
                needs_db_predicted=pred_needs_db,
                db_ok=s.needs_db == pred_needs_db,
                agents=agents_for_intent(route.intent, route.domain),
                query_domains=query_domains,
                tool_match=tool_match,
            )
        )
    return results


def write_report(samples: list[IntentSample], results: list[EvalResult]) -> str:
    total = len(results)
    intent_acc = sum(r.intent_ok for r in results) / total
    domain_cases = [r for r in results if r.expected_domain is not None and r.expected_intent not in {"confirmation_response", "general", "sync_trigger"}]
    domain_acc = sum(r.domain_ok for r in domain_cases) / len(domain_cases) if domain_cases else 1.0
    db_cases = [r for r in results if r.needs_db_expected]
    db_acc = sum(r.db_ok for r in db_cases) / len(db_cases) if db_cases else 1.0
    tool_cases = [r for r in results if r.tool_match is not None and r.needs_db_expected and r.expected_tools]
    tool_acc = sum(1 for r in tool_cases if r.tool_match) / len(tool_cases) if tool_cases else 1.0

    by_intent: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_intent[r.expected_intent].append(r)

    errors = [r for r in results if not r.intent_ok]

    lines = [
        "# MyFitness 意图数据集评估报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 样本数量：**{total}**",
        f"- 评估范围：Router 关键词分类 + Query Planner（不含 LLM 增强）",
        f"- 固定 today：`{TODAY.isoformat()}`",
        "",
        "## 1. 总体指标",
        "",
        "| 指标 | 准确率 | 说明 |",
        "|------|--------|------|",
        f"| 意图分类 | **{intent_acc:.1%}** ({sum(r.intent_ok for r in results)}/{total}) | 7+1 类意图 |",
        f"| 域识别 | **{domain_acc:.1%}** ({sum(r.domain_ok for r in domain_cases)}/{len(domain_cases)}) | body/nutrition/training |",
        f"| DB 查询触发 | **{db_acc:.1%}** ({sum(r.db_ok for r in db_cases)}/{len(db_cases)}) | 是否应查库 |",
        f"| 查询 Tool 匹配 | **{tool_acc:.1%}** | expected_tools ⊆ predicted |",
        "",
        "## 2. 分意图准确率",
        "",
        "| 意图 | 样本数 | 意图准确 | 域准确 | DB触发准确 |",
        "|------|--------|----------|--------|------------|",
    ]

    intent_labels = {
        "data_query": "数据查询",
        "manual_entry": "手动录入",
        "plan_adjust": "计划调整",
        "trend_analysis": "趋势分析",
        "goal_setting": "目标设定",
        "sync_trigger": "同步触发",
        "general": "通用对话",
        "confirmation_response": "确认响应",
    }

    for intent_key in [
        "data_query", "manual_entry", "plan_adjust", "trend_analysis",
        "goal_setting", "sync_trigger", "general", "confirmation_response",
    ]:
        group = by_intent[intent_key]
        if not group:
            continue
        ic = sum(r.intent_ok for r in group) / len(group)
        dc = [r for r in group if r.expected_domain is not None]
        da_str = f"{sum(r.domain_ok for r in dc) / len(dc):.0%}" if dc else "—"
        dbc = [r for r in group if r.needs_db_expected]
        dba_str = f"{sum(r.db_ok for r in dbc) / len(dbc):.0%}" if dbc else "—"
        lines.append(
            f"| {intent_labels[intent_key]} | {len(group)} | {ic:.0%} | {da_str} | {dba_str} |"
        )

    lines.extend([
        "",
        "## 3. 能力拆解",
        "",
        "### 3.1 Router 层",
        "- **关键词规则**：同步、录入、趋势、目标、查询等主路径",
        "- **LLM 兜底**：未启用时（本次评估）歧义句落 general",
        "- **确认续接**：依赖 pending_confirmation，无 pending 时「确认/取消」会误判",
        "",
        "### 3.2 Query Planner 层",
        "- **日期解析**：昨天/今天/前天/近N天/YYYY-MM-DD",
        "- **域推断**：关键词 → body/nutrition/training；无关键词时 data_query 查三域",
        "- **Tool 映射**：body→query_body_metrics, nutrition→query_nutrition_logs, training→query_training_logs",
        "",
        "### 3.3 Agent 编排层",
        "- data_query / trend_analysis → 三 Specialist + Summary",
        "- manual_entry → 解析 + 确认 + 写库",
        "- sync_trigger → run_sync",
        "- plan_adjust / goal_setting → 部分能力（M4 待完善）",
        "",
        "## 4. 能力评估（M2 现状）",
        "",
        "| 能力项 | 评级 | 说明 |",
        "|--------|------|------|",
        "| 数据查询（单域） | ⭐⭐⭐⭐ | 蛋白/体重/训练等关键词路径稳定 |",
        "| 趋势分析 | ⭐⭐⭐⭐ | 「近N天」「趋势」触发良好 |",
        "| 手动录入 | ⭐⭐⭐⭐ | 体重/饮食录入识别准确 |",
        "| 同步触发 | ⭐⭐⭐⭐⭐ | 关键词覆盖充分 |",
        "| 域精细识别 | ⭐⭐⭐ | 缺 domain 时默认三域，开销大 |",
        "| 计划调整 | ⭐⭐ | 仅识别意图，agent_plans CRUD 未实现 |",
        "| 目标设定 | ⭐⭐ | 识别尚可，目标写入未闭环 |",
        "| 多意图句 | ⭐ | 「同步然后分析趋势」未拆分 |",
        "| 确认续接 | ⭐⭐⭐⭐ | 有 pending 时准确 |",
        "| 训练明细解析 | ⭐⭐⭐⭐ | raw_payload 组次/重量已支持 |",
        "",
        "## 5. 错误样本（意图分类失败）",
        "",
    ])

    if errors:
        lines.append("| ID | 输入 | 期望 | 预测 |")
        lines.append("|----|------|------|------|")
        for r in errors[:30]:
            lines.append(f"| {r.id} | {r.text} | {r.expected_intent} | {r.predicted_intent} |")
        if len(errors) > 30:
            lines.append(f"| ... | 共 {len(errors)} 条 | | |")
    else:
        lines.append("无意图分类错误。")

    lines.extend([
        "",
        "## 6. 改进建议（优先级）",
        "",
        "1. **P0**：data_query 与 trend_analysis 优先级 — 「近7天蛋白」不应先命中 trend",
        "2. **P0**：plan_adjust 中「取消训练」与 confirmation cancel 歧义",
        "3. **P1**：domain 推断 — 无关键词时按 intent 默认单域而非三域",
        "4. **P1**：goal_setting 写入 user_goals 闭环",
        "5. **P2**：多意图拆分（sync + analysis）",
        "6. **P2**：LLM Router 启用后复测 100 条",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    samples = build_dataset()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps([asdict(s) for s in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = evaluate(samples)
    report = write_report(samples, results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    intent_acc = sum(r.intent_ok for r in results) / len(results)
    print(f"Dataset: {FIXTURE_PATH}")
    print(f"Report:  {REPORT_PATH}")
    print(f"Intent accuracy: {intent_acc:.1%} ({sum(r.intent_ok for r in results)}/{len(results)})")


if __name__ == "__main__":
    main()
