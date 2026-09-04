"""生成 300 条意图测评集（简单/中等/复杂各 100）并评估 Router + Query Planner。"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myfitness.agents.tools.query_planner import build_query_plan
from myfitness.graph.router import agents_for_intent, classify_intent
from myfitness.schemas.state import Intent, PendingConfirmation

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "intent_dataset_300.json"
REPORT_PATH = ROOT / "docs" / "intent_evaluation_report_300.md"
TODAY = date(2026, 8, 23)


@dataclass
class IntentSample:
    id: int
    text: str
    intents: list[str]
    difficulty: str
    domain: str | None = None
    needs_db: bool = False
    expected_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def intent(self) -> str:
        return self.intents[0]


def _sample(
    samples: list[IntentSample],
    *,
    text: str,
    intents: list[str] | str,
    difficulty: str,
    domain: str | None = None,
    needs_db: bool = False,
    tools: list[str] | None = None,
    tags: list[str] | None = None,
    notes: str = "",
) -> None:
    if isinstance(intents, str):
        intents = [intents]
    samples.append(
        IntentSample(
            id=len(samples) + 1,
            text=text,
            intents=intents,
            difficulty=difficulty,
            domain=domain,
            needs_db=needs_db,
            expected_tools=tools or [],
            tags=tags or [],
            notes=notes,
        )
    )


def _simple_samples(samples: list[IntentSample]) -> None:
    """100 条：单一意图、短句、歧义少。"""
    data_queries = [
        ("昨天吃了多少蛋白质？", "nutrition", ["query_nutrition_logs"]),
        ("今天热量摄入多少？", "nutrition", ["query_nutrition_logs"]),
        ("查询今天体重", "body", ["query_body_metrics"]),
        ("昨天训练了吗", "fitness", ["query_training_logs"]),
        ("前天碳水吃了多少", "nutrition", ["query_nutrition_logs"]),
        ("午餐吃了什么", "nutrition", ["query_nutrition_logs"]),
        ("我最近的体脂是多少", "body", ["query_body_metrics"]),
        ("今天脂肪摄入超标了吗", "nutrition", ["query_nutrition_logs"]),
        ("我的体重现在多少kg", "body", ["query_body_metrics"]),
        ("昨天摄入总热量", "nutrition", ["query_nutrition_logs"]),
        ("今天吃了几个鸡蛋", "nutrition", ["query_nutrition_logs"]),
        ("昨天有没有记录体重", "body", ["query_body_metrics"]),
        ("今天练腿了吗", "fitness", ["query_training_logs"]),
        ("8月21号深蹲做了多少组", "fitness", ["query_training_logs"]),
        ("查询零食吃了什么", "nutrition", ["query_nutrition_logs"]),
        ("昨天卧推重量多少", "fitness", ["query_training_logs"]),
        ("今天早餐记录有哪些", "nutrition", ["query_nutrition_logs"]),
        ("查询2026-08-20训练详情", "fitness", ["query_training_logs"]),
        ("昨天蛋白质够吗", "nutrition", ["query_nutrition_logs"]),
        ("查询昨天晚餐记录", "nutrition", ["query_nutrition_logs"]),
    ]
    for text, domain, tools in data_queries:
        _sample(samples, text=text, intents="data_query", difficulty="simple", domain=domain, needs_db=True, tools=tools)

    manual = [
        ("记录体重 72.5kg", "body"),
        ("录入今天体脂 18.2%", "body"),
        ("添加早餐 鸡蛋 2个", "nutrition"),
        ("记录午餐：鸡胸肉 200g", "nutrition"),
        ("午餐吃了牛肉 150g 米饭 200g", "nutrition"),
        ("记录晚餐 三文鱼 120g", "nutrition"),
        ("添加体重 71.8", "body"),
        ("录入零食 坚果 30g", "nutrition"),
        ("记录 2026-08-22 体重 73kg", "body"),
        ("记录体脂率 17.5%", "body"),
        ("录入午餐 米饭 150g 西兰花 100g", "nutrition"),
        ("记录体重72kg", "body"),
        ("添加早餐燕麦 80g", "nutrition"),
        ("记录今天体重 105.9kg", "body"),
        ("录入 晚餐 豆腐 200g", "nutrition"),
    ]
    for text, domain in manual:
        _sample(samples, text=text, intents="manual_entry", difficulty="simple", domain=domain)

    trends = [
        ("近30天体脂变化趋势", None),
        ("近7天体重变化怎么样", "body"),
        ("近14天蛋白质摄入趋势", "nutrition"),
        ("近90天体重走势", "body"),
        ("最近训练量变化", "fitness"),
        ("近30天热量摄入变化", "nutrition"),
        ("体重和体脂近两周对比", "body"),
        ("近7天训练频率趋势", "fitness"),
        ("分析近30天饮食变化", "nutrition"),
        ("近60天卧推重量变化", "fitness"),
        ("近21天围度变化", "body"),
        ("对比近7天摄入和消耗", "nutrition"),
        ("给我一个近7天的体重变化报告", "body"),
        ("近7天饮食分析报告", "nutrition"),
    ]
    tools3 = ["query_body_metrics", "query_nutrition_logs", "query_training_logs"]
    for text, domain in trends:
        _sample(samples, text=text, intents="trend_analysis", difficulty="simple", domain=domain, needs_db=True, tools=tools3)

    goals = [
        "目标体重设为70kg",
        "我要在12周降到65公斤",
        "设定体脂目标15%",
        "目标增到75kg",
        "减到68公斤需要多久",
        "把目标体重改成72kg",
        "设定3个月降到80kg",
        "目标体脂降到12%",
        "我想增肌到78kg",
        "降到70kg以下",
    ]
    for text in goals:
        _sample(samples, text=text, intents="goal_setting", difficulty="simple", domain="body", needs_db=True, tools=["query_body_metrics"])

    syncs = [
        "同步最近7天训记数据",
        "拉取训记训练记录",
        "更新训记数据",
        "同步最近30天数据",
        "从训记同步饮食",
        "拉取最近14天训记",
        "同步训记身体数据",
        "更新最近7天训记同步",
    ]
    for text in syncs:
        _sample(samples, text=text, intents="sync_trigger", difficulty="simple")

    charts = [
        "生成最近7天体重折线图",
        "画近30天体脂趋势图",
        "生成热量柱状图保存成文档",
        "近14天训练量折线图",
        "把近7天蛋白摄入画成图",
        "生成体重折线图",
        "近30天摄入热量统计图",
        "画一个体脂变化折线图",
    ]
    for text in charts:
        _sample(samples, text=text, intents="chart_trigger", difficulty="simple", domain="body", needs_db=True)

    reports = [
        "生成昨天的日报",
        "出一份8月22日的报告",
        "生成8月20日到8月25日的报告",
        "生成今日晨报",
        "帮我生成综合健康报告",
        "生成8.21的报告",
        "生成昨天的健康日报",
        "出一份完整健康报告",
    ]
    for text in reports:
        _sample(samples, text=text, intents="report_trigger", difficulty="simple")

    web = [
        "搜一下HIIT一周练几次比较好",
        "蛋白质推荐摄入量有什么科学依据",
        "联网查减脂期碳水怎么分配",
        "网上搜一下增肌训练容量建议",
        "查资料：间歇跑和匀速跑哪个减脂更好",
        "搜一下深蹲膝盖内扣怎么纠正",
    ]
    for text in web:
        _sample(samples, text=text, intents="web_search", difficulty="simple", domain="fitness")

    schedules = [
        "每天早上7点生成日报",
        "查看定时任务",
        "取消每天同步",
        "设置每天21点同步训记",
        "停用定时日报任务",
        "每天8点自动同步数据",
    ]
    for text in schedules:
        _sample(samples, text=text, intents="schedule_manage", difficulty="simple")

    plans = [
        "今天不练了，改成休息",
        "调整训练计划，明天练胸",
        "取消今天的训练",
        "改成休息日",
        "把明天的训练改成有氧",
    ]
    for text in plans:
        _sample(samples, text=text, intents="plan_adjust", difficulty="simple", domain="fitness", needs_db=True)


def _medium_samples(samples: list[IntentSample]) -> None:
    """100 条：1~3 个意图或较强歧义/日期/域组合。"""
    combos = [
        ("同步8月24日数据并生成日报", ["sync_trigger", "report_trigger"], None),
        ("同步今日数据然后生成日报", ["sync_trigger", "report_trigger"], None),
        ("拉取昨天数据并出报告", ["sync_trigger", "report_trigger"], None),
        ("同步最近7天数据并生成周期报告", ["sync_trigger", "report_trigger"], None),
        ("更新训记后生成8月22日日报", ["sync_trigger", "report_trigger"], None),
        ("同步前天和昨天的数据并生成日报", ["sync_trigger", "report_trigger"], None),
        ("同步8月20号和8月25号的数据再生成报告", ["sync_trigger", "report_trigger"], None),
        ("先同步训记再生成昨天的晨报", ["sync_trigger", "report_trigger"], None),
        ("同步今天数据，然后生成日报", ["sync_trigger", "report_trigger"], None),
        ("拉取近3天训记并生成报告", ["sync_trigger", "report_trigger"], None),
        ("生成最近7天的报告并附上体重折线图", ["report_trigger", "chart_trigger"], "body"),
        ("同步数据后画近7天体重折线图", ["sync_trigger", "chart_trigger"], "body"),
        ("生成日报并把体脂趋势图插进去", ["report_trigger", "chart_trigger"], "body"),
        ("出报告同时生成热量柱状图", ["report_trigger", "chart_trigger"], "nutrition"),
        ("同步并生成报告，再附训练量折线图", ["sync_trigger", "report_trigger", "chart_trigger"], "fitness"),
        ("拉取数据、生成日报、画体重图", ["sync_trigger", "report_trigger", "chart_trigger"], "body"),
        ("记录体重72kg并设定目标70kg", ["manual_entry", "goal_setting"], "body"),
        ("录入体脂18%然后设目标15%", ["manual_entry", "goal_setting"], "body"),
        ("记录初始体重130kg，目标减到85kg", ["manual_entry", "goal_setting"], "body"),
        ("添加今天体重71kg并修改目标为68kg", ["manual_entry", "goal_setting"], "body"),
        (
            "以2025年9月1日为起点记录初始体重130kg，目标减到85kg，评价减肥进度",
            ["manual_entry", "goal_setting", "trend_analysis"],
            "body",
        ),
        ("记录体重72.5kg，帮我看看减脂进度怎么样", ["manual_entry", "trend_analysis"], "body"),
        ("录入今天体脂后分析近30天变化", ["manual_entry", "trend_analysis"], "body"),
        ("记录午餐后看看今天蛋白够不够", ["manual_entry", "data_query"], "nutrition"),
        ("我昨天蛋白质摄入和推荐量差多少", ["data_query", "web_search"], "nutrition"),
        ("查昨天训练记录并搜一下背部训练动作推荐", ["data_query", "web_search"], "fitness"),
        ("对比近7天摄入并查减脂期蛋白建议", ["trend_analysis", "web_search"], "nutrition"),
        ("生成饮食规划文档", ["trend_analysis"], "nutrition"),
        ("写一份背部训练计划文档", ["trend_analysis"], "fitness"),
        ("导出近30天训练总结文档", ["trend_analysis"], "fitness"),
        ("把体重折线图插入到8月24日的日报", ["chart_trigger"], "body"),
        ("生成8月20日到8月27日周期报表", ["report_trigger"], None),
        ("近30天体重和训练对比分析", ["trend_analysis"], "body"),
        ("分析近14天饮食并给减脂建议", ["trend_analysis"], "nutrition"),
        ("根据我过往训练记录安排今天练背", ["trend_analysis"], "fitness"),
        ("结合历史数据给我个性化饮食建议", ["trend_analysis"], "nutrition"),
        ("今天练什么，参考我最近的腿部训练", ["trend_analysis"], "fitness"),
        ("查询今天体重并分析近7天趋势", ["data_query", "trend_analysis"], "body"),
        ("昨天吃了多少蛋白，顺便看近一周变化", ["data_query", "trend_analysis"], "nutrition"),
        ("同步后分析近7天体重变化", ["sync_trigger", "trend_analysis"], "body"),
        ("每天早上7点同步并生成日报", ["schedule_manage"], None),
        ("设置每晚21点同步最近3天数据", ["schedule_manage"], None),
        ("查看定时任务并修改日报时间为8点", ["schedule_manage"], None),
        ("今天不练了，顺便查一下昨天训练量", ["plan_adjust", "data_query"], "fitness"),
        ("取消今天训练，分析本周训练频率", ["plan_adjust", "trend_analysis"], "fitness"),
        ("调整计划：明天练背，参考上次背部训练", ["plan_adjust", "trend_analysis"], "fitness"),
        ("目标降到70kg，分析当前进度", ["goal_setting", "trend_analysis"], "body"),
        ("设定体脂15%并看近30天变化", ["goal_setting", "trend_analysis"], "body"),
        ("生成昨天日报，不要插入其他内容", ["report_trigger"], None),
        ("同步昨天和今天的数据", ["sync_trigger"], None),
        ("同步最近7天和今天的数据", ["sync_trigger"], None),
        ("生成最近7天体重折线图并保存", ["chart_trigger"], "body"),
        ("画近30天训练次数柱状图", ["chart_trigger"], "fitness"),
        ("搜一下减脂期每天蛋白吃多少，并对照我昨天摄入", ["web_search", "data_query"], "nutrition"),
        ("联网查HIIT训练频率，再结合我上周训练记录给建议", ["web_search", "trend_analysis"], "fitness"),
        ("记录体重71.2kg", ["manual_entry"], "body"),
        ("添加晚餐牛肉200g", ["manual_entry"], "nutrition"),
        ("查询8月21日练了什么", ["data_query"], "fitness"),
        ("近7天碳水趋势", ["trend_analysis"], "nutrition"),
        ("近10天睡眠和训练关系", ["trend_analysis"], "fitness"),
        ("对比近一个月训练和体重", ["trend_analysis"], None),
        ("生成8.21的报告", ["report_trigger"], None),
        ("同步训记后出一份完整健康报告", ["sync_trigger", "report_trigger"], None),
        ("更新数据、分析趋势、生成报告", ["sync_trigger", "trend_analysis", "report_trigger"], None),
        ("录入体重后同步训记再分析趋势", ["manual_entry", "sync_trigger", "trend_analysis"], "body"),
        ("拉取近7天数据，画体重图，写进日报", ["sync_trigger", "chart_trigger", "report_trigger"], "body"),
        ("设定目标68kg，生成饮食规划文档", ["goal_setting", "trend_analysis"], "nutrition"),
        ("记录体脂并生成近30天体脂报告", ["manual_entry", "trend_analysis"], "body"),
        ("查今天热量并搜减脂热量缺口建议", ["data_query", "web_search"], "nutrition"),
        ("分析近7天训练并搜背部恢复方法", ["trend_analysis", "web_search"], "fitness"),
        ("生成训练计划文档，结合我最近的卧推记录", ["trend_analysis"], "fitness"),
        ("同步、出报告、插入折线图一条龙", ["sync_trigger", "report_trigger", "chart_trigger"], "body"),
        ("每天6点同步并7点出日报", ["schedule_manage"], None),
        ("取消定时同步任务", ["schedule_manage"], None),
        ("把明天训练改成休息并更新计划", ["plan_adjust"], "fitness"),
        ("调整本周训练容量，参考近30天训练量", ["plan_adjust", "trend_analysis"], "fitness"),
        ("你好，顺便查一下今天体重", ["general", "data_query"], "body"),
        ("谢谢，再帮我生成昨天日报", ["general", "report_trigger"], None),
        ("明白了，那同步一下今天数据", ["general", "sync_trigger"], None),
        ("近7天体重变化报告", ["trend_analysis"], "body"),
        ("体重变化报告", ["trend_analysis"], "body"),
        ("生成减脂饮食规划文档，结合我的目标", ["trend_analysis"], "nutrition"),
        ("写训练计划并分析近14天训练频率", ["trend_analysis"], "fitness"),
        ("同步8月23日数据", ["sync_trigger"], None),
        ("生成8月23日日报", ["report_trigger"], None),
        ("记录午餐鸡胸肉并查蛋白", ["manual_entry", "data_query"], "nutrition"),
        ("设定目标72kg并同步训记", ["goal_setting", "sync_trigger"], "body"),
        ("查昨天体脂并画近14天折线图", ["data_query", "chart_trigger"], "body"),
        ("搜深蹲技巧并对照昨天训练", ["web_search", "data_query"], "fitness"),
        ("生成近7天训练报告", ["trend_analysis"], "fitness"),
        ("同步近7天后分析饮食趋势", ["sync_trigger", "trend_analysis"], "nutrition"),
        ("取消训练并录入今天体重", ["plan_adjust", "manual_entry"], "body"),
        ("每天早上同步并每周一生成周报", ["schedule_manage"], None),
        ("生成热量折线图插入8月22日报", ["chart_trigger", "report_trigger"], "nutrition"),
        ("记录晚餐后分析近3天热量", ["manual_entry", "trend_analysis"], "nutrition"),
        ("查今天训练并调整明天计划", ["data_query", "plan_adjust"], "fitness"),
        ("目标增肌到78kg并写训练文档", ["goal_setting", "trend_analysis"], "fitness"),
        ("同步、分析、出图、写报告", ["sync_trigger", "trend_analysis", "chart_trigger", "report_trigger"], "body"),
        ("你好，帮我同步今天数据", ["general", "sync_trigger"], None),
        ("分析近7天背部训练并安排今天练背", ["trend_analysis"], "fitness"),
    ]
    tools3 = ["query_body_metrics", "query_nutrition_logs", "query_training_logs"]
    assert len(combos) >= 100, f"medium combos only {len(combos)}"
    for text, intents, domain in combos[:100]:
        needs_db = any(
            i in intents
            for i in ("data_query", "trend_analysis", "goal_setting", "plan_adjust", "chart_trigger")
        )
        tools = tools3 if needs_db and domain is None else (
            ["query_body_metrics"] if domain == "body" and needs_db else
            ["query_nutrition_logs"] if domain == "nutrition" and needs_db else
            ["query_training_logs"] if domain == "fitness" and needs_db else
            []
        )
        _sample(
            samples,
            text=text,
            intents=intents,
            difficulty="medium",
            domain=domain,
            needs_db=needs_db,
            tools=tools,
            tags=["multi" if len(intents) > 1 else "single", f"n={len(intents)}"],
        )


def _complex_samples(samples: list[IntentSample]) -> None:
    """100 条：长句、多约束、多意图、个性化与跨域组合。"""
    templates = [
        (
            "我是减脂期，今天练背日，请根据我过往一个月背部训练记录和当前最新体重体脂，"
            "安排今天的训练动作、组次和注意事项，并告诉我练后蛋白该怎么吃。",
            ["trend_analysis"],
            "fitness",
        ),
        (
            "先把昨天和今天的训记数据同步下来，再生成这两天的综合健康日报，"
            "并在日报里插入近7天体重和体脂的双折线图。",
            ["sync_trigger", "report_trigger", "chart_trigger"],
            "body",
        ),
        (
            "以2025年9月1日作为起点，帮我记录初始体重130公斤、体脂37%，目标是在年底前减到85公斤，"
            "然后结合从起点到今天所有身体数据和饮食记录，评价我的减肥进度并指出下一步重点。",
            ["manual_entry", "goal_setting", "trend_analysis"],
            "body",
        ),
        (
            "请联网检索减脂期每日蛋白质摄入建议，再对照我昨天、前天和大前天的饮食记录，"
            "判断蛋白是否不足，并给出接下来三天食堂可执行的高蛋白食谱。",
            ["web_search", "data_query", "trend_analysis"],
            "nutrition",
        ),
        (
            "取消今天原本安排的腿部训练，改成主动恢复；同时分析近两周训练频率和体重变化，"
            "告诉我是不是练得太多导致恢复不足。",
            ["plan_adjust", "trend_analysis"],
            "fitness",
        ),
        (
            "设置每天早上6点半自动同步训记、7点生成日报；另外把现有定时同步任务列出来，"
            "如果有重复的就帮我停用。",
            ["schedule_manage"],
            None,
        ),
        (
            "生成8月15日到8月23日的周期健康报告，要求包含每日体重体脂明细、日均热量蛋白、"
            "训练次数汇总，并在报告里附上体重趋势折线图。",
            ["report_trigger", "chart_trigger"],
            "body",
        ),
        (
            "记录今天午餐：鸡胸肉200g、米饭150g、西兰花100g，然后查询今天总热量和蛋白是否达标，"
            "并结合我近7天饮食趋势判断减脂节奏是否合适。",
            ["manual_entry", "data_query", "trend_analysis"],
            "nutrition",
        ),
        (
            "搜一下硬拉圆背怎么纠正，再结合我最近一次硬拉训练记录里的重量和组次，"
            "判断我是否该降重量练技术。",
            ["web_search", "data_query", "trend_analysis"],
            "fitness",
        ),
        (
            "帮我写一份为期四周的背部增肌训练计划文档，要求参考我近30天背部相关训练历史、"
            "当前卧拉划船重量水平，并考虑我每周只能练四次。",
            ["trend_analysis"],
            "fitness",
        ),
    ]
    tools3 = ["query_body_metrics", "query_nutrition_logs", "query_training_logs"]
    for text, intents, domain in templates:
        _sample(
            samples,
            text=text,
            intents=intents,
            difficulty="complex",
            domain=domain,
            needs_db=True,
            tools=tools3,
            tags=["long", f"n={len(intents)}"],
        )

    variants = [
        ("同步{sync_range}训记数据，生成{report_day}日报，并附{metric}折线图", ["sync_trigger", "report_trigger", "chart_trigger"], "body"),
        ("记录体重{weight}kg、体脂{bf}%，设定目标{target}kg，再分析{window}减脂进度", ["manual_entry", "goal_setting", "trend_analysis"], "body"),
        ("查{day}训练明细，联网搜{topic}，结合我的记录给改进建议", ["data_query", "web_search", "trend_analysis"], "fitness"),
        ("生成{window}{metric}变化报告，并导出为文档保存", ["trend_analysis"], "body"),
        ("{schedule}自动同步并{schedule2}生成日报，查看当前定时任务状态", ["schedule_manage"], None),
        ("取消{day}训练改休息，同时分析{window}训练总量和恢复情况", ["plan_adjust", "trend_analysis"], "fitness"),
        ("录入{meal}：{food}，查询今日{nutrient}摄入并分析{window}趋势", ["manual_entry", "data_query", "trend_analysis"], "nutrition"),
        ("先同步{sync_range}，再基于新数据分析{domain_topic}并生成{report_day}报告", ["sync_trigger", "trend_analysis", "report_trigger"], None),
        ("结合历史{muscle}训练记录，安排今天{muscle}训练计划并说明注意事项", ["trend_analysis"], "fitness"),
        ("搜{topic}科学依据，对照我{day}饮食与训练记录评估是否执行到位", ["web_search", "trend_analysis"], "nutrition"),
    ]
    sync_ranges = ["最近7天", "昨天和今天", "8月20日到8月25日", "近14天", "今天"]
    report_days = ["昨天", "8月22日", "8月20日到8月25日", "今天", "8.21"]
    metrics = ["体重", "体脂", "热量", "训练量", "蛋白摄入"]
    weights = ["72.5", "71.8", "130", "105.9", "68.5"]
    bfs = ["18.2", "17.5", "37", "22", "15"]
    targets = ["70", "85", "68", "75", "65"]
    windows = ["近7天", "近30天", "近14天", "近60天", "近21天"]
    days = ["昨天", "前天", "8月21日", "今天", "上周三"]
    topics = ["HIIT频率", "蛋白摄入", "深蹲技术", "减脂碳水", "背部恢复"]
    meals = ["午餐", "晚餐", "早餐", "加餐", "午餐"]
    foods = ["鸡胸肉200g", "牛肉饭1份", "燕麦80g", "鸡蛋2个", "三文鱼120g"]
    nutrients = ["蛋白", "热量", "碳水", "脂肪", "蛋白"]
    muscle = ["背部", "腿部", "胸部", "肩部", "手臂"]
    schedules = ["每天早上7点", "每天21点", "每周一7点", "每天6点", "每晚22点"]
    schedule2 = ["7点", "8点", "6点半", "7点半", "9点"]
    domain_topics = ["体重体脂", "饮食摄入", "训练恢复", "力量进步", "减脂节奏"]

    idx = 0
    while len([s for s in samples if s.difficulty == "complex"]) < 100:
        template, intents, domain = variants[idx % len(variants)]
        i = idx // len(variants)
        text = template.format(
            sync_range=sync_ranges[i % len(sync_ranges)],
            report_day=report_days[i % len(report_days)],
            metric=metrics[i % len(metrics)],
            weight=weights[i % len(weights)],
            bf=bfs[i % len(bfs)],
            target=targets[i % len(targets)],
            window=windows[i % len(windows)],
            day=days[i % len(days)],
            topic=topics[i % len(topics)],
            meal=meals[i % len(meals)],
            food=foods[i % len(foods)],
            nutrient=nutrients[i % len(nutrients)],
            muscle=muscle[i % len(muscle)],
            schedule=schedules[i % len(schedules)],
            schedule2=schedule2[i % len(schedule2)],
            domain_topic=domain_topics[i % len(domain_topics)],
        )
        _sample(
            samples,
            text=text,
            intents=intents,
            difficulty="complex",
            domain=domain,
            needs_db=any(i in intents for i in ("data_query", "trend_analysis", "goal_setting", "plan_adjust", "chart_trigger")),
            tools=tools3 if domain else [],
            tags=["generated", f"n={len(intents)}"],
        )
        idx += 1


def build_dataset_300() -> list[IntentSample]:
    samples: list[IntentSample] = []
    _simple_samples(samples)
    _medium_samples(samples)
    _complex_samples(samples)

    by_diff = Counter(s.difficulty for s in samples)
    assert by_diff["simple"] == 100, f"simple={by_diff['simple']}"
    assert by_diff["medium"] == 100, f"medium={by_diff['medium']}"
    assert by_diff["complex"] == 100, f"complex={by_diff['complex']}"
    assert len(samples) == 300, f"total={len(samples)}"
    for index, sample in enumerate(samples, start=1):
        sample.id = index
    return samples


@dataclass
class EvalResult:
    id: int
    text: str
    difficulty: str
    expected_intents: list[str]
    predicted_intents: list[str]
    expected_domain: str | None
    predicted_domain: str | None
    intent_ok: bool
    intents_ok: bool
    domain_ok: bool
    needs_db_expected: bool
    needs_db_predicted: bool
    db_ok: bool
    query_domains: list[str]


def _pending_confirmation() -> PendingConfirmation:
    return PendingConfirmation(
        action_type="db_write",
        summary="测试确认",
        payload={},
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        domain="nutrition",
    )


def evaluate(samples: list[IntentSample], *, use_llm: bool = False) -> list[EvalResult]:
    results: list[EvalResult] = []
    pending = _pending_confirmation()

    for sample in samples:
        use_pending = "confirmation_response" in sample.intents
        route = classify_intent(
            sample.text,
            pending if use_pending else None,
            use_llm=use_llm,
            today=TODAY,
        )
        predicted_intents = [item.value for item in route.intents]
        plan = build_query_plan(sample.text, route.intent, route.domain, today=TODAY)
        pred_needs_db = plan is not None

        domain_ok = True
        if sample.domain is not None and not any(
            i in sample.intents for i in ("confirmation_response", "sync_trigger", "general", "schedule_manage")
        ):
            domain_ok = route.domain == sample.domain

        results.append(
            EvalResult(
                id=sample.id,
                text=sample.text,
                difficulty=sample.difficulty,
                expected_intents=sample.intents,
                predicted_intents=predicted_intents,
                expected_domain=sample.domain,
                predicted_domain=route.domain,
                intent_ok=predicted_intents[0] == sample.intents[0],
                intents_ok=predicted_intents == sample.intents,
                domain_ok=domain_ok,
                needs_db_expected=sample.needs_db,
                needs_db_predicted=pred_needs_db,
                db_ok=sample.needs_db == pred_needs_db,
                query_domains=list(plan.domains) if plan else [],
            )
        )
    return results


def write_report(samples: list[IntentSample], results: list[EvalResult], *, use_llm: bool) -> str:
    total = len(results)
    primary_acc = sum(r.intent_ok for r in results) / total
    multi_acc = sum(r.intents_ok for r in results) / total

    def tier_acc(key: str, attr: str) -> tuple[float, int, int]:
        tier = [r for r in results if r.difficulty == key]
        ok = sum(getattr(r, attr) for r in tier)
        return ok / len(tier), ok, len(tier)

    lines = [
        "# MyFitness 意图测评集（300 条）评估报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 样本数量：**{total}**（简单/中等/复杂 各 100）",
        f"- 评估模式：**{'LLM + 关键词' if use_llm else '关键词规则（use_llm=False）'}**",
        f"- 固定 today：`{TODAY.isoformat()}`",
        "",
        "## 1. 总体指标",
        "",
        "| 指标 | 准确率 |",
        "|------|--------|",
        f"| 主意图（intents[0]） | **{primary_acc:.1%}** ({sum(r.intent_ok for r in results)}/{total}) |",
        f"| 完整意图序列 | **{multi_acc:.1%}** ({sum(r.intents_ok for r in results)}/{total}) |",
        "",
        "## 2. 分难度",
        "",
        "| 难度 | 样本 | 主意图准确 | 完整序列准确 |",
        "|------|------|------------|--------------|",
    ]

    for key, label in [("simple", "简单"), ("medium", "中等"), ("complex", "复杂")]:
        p_acc, p_ok, p_n = tier_acc(key, "intent_ok")
        m_acc, m_ok, m_n = tier_acc(key, "intents_ok")
        lines.append(f"| {label} | {p_n} | {p_acc:.1%} ({p_ok}/{p_n}) | {m_acc:.1%} ({m_ok}/{m_n}) |")

    by_intent: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        by_intent[result.expected_intents[0]].append(result)

    lines.extend(["", "## 3. 分主意图（Top）", "", "| 主意图 | 样本 | 主意图准确 | 完整序列准确 |", "|--------|------|------------|--------------|"])
    for intent_key, group in sorted(by_intent.items(), key=lambda item: -len(item[1])):
        ic = sum(r.intent_ok for r in group) / len(group)
        mc = sum(r.intents_ok for r in group) / len(group)
        lines.append(f"| {intent_key} | {len(group)} | {ic:.0%} | {mc:.0%} |")

    errors = [r for r in results if not r.intents_ok]
    lines.extend(["", "## 4. 错误样本（完整意图序列不匹配，前 40 条）", ""])
    if errors:
        lines.append("| ID | 难度 | 输入 | 期望 | 预测 |")
        lines.append("|----|------|------|------|------|")
        for row in errors[:40]:
            text = row.text if len(row.text) <= 48 else row.text[:45] + "…"
            lines.append(
                f"| {row.id} | {row.difficulty} | {text} | {row.expected_intents} | {row.predicted_intents} |"
            )
        if len(errors) > 40:
            lines.append(f"| … | 共 {len(errors)} 条 | | | |")
    else:
        lines.append("无错误。")

    return "\n".join(lines)


def main() -> None:
    samples = build_dataset_300()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps([asdict(s) for s in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = evaluate(samples, use_llm=False)
    report = write_report(samples, results, use_llm=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    primary = sum(r.intent_ok for r in results) / len(results)
    multi = sum(r.intents_ok for r in results) / len(results)
    print(f"Dataset: {FIXTURE_PATH}")
    print(f"Report:  {REPORT_PATH}")
    print(f"Primary intent accuracy: {primary:.1%}")
    print(f"Full intents accuracy:   {multi:.1%}")
    for key in ("simple", "medium", "complex"):
        tier = [r for r in results if r.difficulty == key]
        print(
            f"  {key:7s} primary={sum(r.intent_ok for r in tier)/len(tier):.1%} "
            f"full={sum(r.intents_ok for r in tier)/len(tier):.1%}"
        )


if __name__ == "__main__":
    main()
