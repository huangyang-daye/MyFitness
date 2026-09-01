"""意图识别 Agent — LLM 结构化意图识别优先，失败回退关键词规则（见 graph/router.py）。

职责：
1. 用完善的系统提示词让 LLM 输出结构化意图（支持一条消息多个意图，如「同步X并生成日报」）；
2. 提取消息中的日期范围（今天/昨天/N月N日/最近N天等），供同步与日报直接使用；
3. 严格校验 LLM 输出，任何解析/校验失败都返回 None，由 Router 走关键词兜底。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from myfitness.debug import trace_agent
from myfitness.llm.factory import chat_completion, is_llm_configured
from myfitness.schemas.state import Intent, RouteResult

logger = logging.getLogger(__name__)

_WEEKDAY_NAMES = ("一", "二", "三", "四", "五", "六", "日")

# 单条消息最多识别的动作数，防止 LLM 幻觉出过长的执行链
_MAX_INTENTS = 3

_INTENT_VALUES = {i.value for i in Intent}
_VALID_DOMAINS = {"body", "nutrition", "fitness"}
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
# 域的常见别名归一
_DOMAIN_ALIASES = {
    "training": "fitness",
    "diet": "nutrition",
    "food": "nutrition",
    "exercise": "fitness",
}


def build_system_prompt(today: date) -> str:
    """构造意图识别系统提示词（注入当前日期，用于相对日期换算与示例）。"""
    yesterday = today - timedelta(days=1)
    last_week_start = today - timedelta(days=6)
    month_start = today - timedelta(days=29)
    example_cn = f"{yesterday.month}月{yesterday.day}日"
    return f"""# 角色
你是 MyFitness 健身助手的意图识别 Agent。用户会发送一条自然语言消息，你需要判断：
1. 用户想执行的操作（意图）——一条消息可能包含**多个按顺序执行**的操作；
2. 操作涉及的日期或日期范围；
3. 涉及的数据域。

你只输出一个 JSON 对象，禁止输出任何其他文字、解释或 Markdown 代码块。

# 当前日期
今天是 {today.isoformat()}（星期{_WEEKDAY_NAMES[today.weekday()]}）。
所有相对日期（今天/昨天/最近N天等）都以它为基准换算。

# 意图类别（intents 数组元素只能是以下英文值）

| intent | 定义 | 典型例句 |
|---|---|---|
| sync_trigger | 从训记 App 同步/拉取/更新数据到本地 | 同步今日数据 / 拉取训记 / 更新最近7天数据 |
| report_trigger | 生成**完整**日报/晨报/综合健康报告（未限定单一主题） | 生成昨天的日报 / 出一份8月24日的报告 / 生成8月20日到8月25日的报告 |
| chart_trigger | 画统计图（折线图、柱状图等），可生成文档或插入现有文档 | 生成最近7天体重折线图 / 把近30天体脂画成图插入昨天的日报 |
| web_search | 联网检索公开资料（指南、研究、推荐摄入、训练方法等），不是查用户自己的记录 | 搜一下HIIT一周练几次 / 蛋白质推荐摄入量有什么科学依据 |
| schedule_manage | 创建/查看/修改/取消**定时/每天/每日**重复任务 | 每天早上7点生成日报 / 查看定时任务 / 取消每天同步 |
| data_query | 查询某天/某段时间**已记录**的数据 | 昨天吃了多少蛋白质 / 查询今天体重 / 8月21日练了什么 |
| trend_analysis | 分析/报告某一主题的**变化/趋势/对比**（含领域专项报告） | 近30天体脂变化 / 体重变化报告 / 对比近7天摄入和消耗 / 近7天饮食分析报告 |
| manual_entry | 手动录入体重/体脂/饮食等数据 | 记录体重72.5kg / 午餐吃了鸡胸肉200g / 添加早餐 |
| plan_adjust | 调整训练计划（改休息/改内容/取消当天训练） | 今天不练了改成休息 / 把明天的训练改成有氧 |
| goal_setting | 设定/修改身体目标 | 目标体重70kg / 设定体脂目标15% |
| general | 寒暄、能力询问、与健身数据无关的闲聊 | 你好 / 你能做什么 / 谢谢 |
| confirmation_response | 对上一个操作的确认/取消 | 确认 / 取消 / 是的，写入 |

# 判定规则（务必遵守）

## 多意图
1. 用户用「并/然后/再/之后/同时」等连接多个**不同动作**时，按执行顺序返回多个意图。
   例：「同步{example_cn}数据并生成日报」→ intents = ["sync_trigger", "report_trigger"]（先同步、再出报告）。
2. 同一个动作的修饰语不算新意图。「帮我同步一下训记数据」只有 sync_trigger。
3. 最多 {_MAX_INTENTS} 个意图；无法确定时宁可只给最有把握的一个。

## 日期提取（date_range）
1. 仅当消息中出现明确时间信息时才填 date_range，否则填 null。
   例：「同步数据」（未提日期）→ date_range = null，表示使用系统默认范围。
2. 支持的表达式（以今天 {today.isoformat()} 为基准换算）：
   - 今天/今日 → 当天；昨天/昨日 → 前一天；前天 → 前两天
   - N月N日 / N月N号 / YYYY-MM-DD / M.D / M/D → 对应日期
   - 最近N天 / 近N天 / 过去N天 / 前N天 / 最近一周 → 今天往前推N天（**含今天**）
   - A到B / A至B / A~B（如「8月20日到8月25日」）→ 连续区间 start=A, end=B
3. 输出格式：{{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}；单日时 start 与 end 相同。
   **report_trigger 与 chart_trigger 都允许 start < end（区间报表 / 区间趋势图）。**
4. 换算出的日期不得晚于今天；若消息中的日期在未来，date_range 置 null。
5. **一个消息提及多个日期点（用「和/与/、/到/至」连接，如「昨天和今天」
   「前天和昨天」「8月20号和8月25号」「上周三和今天」）时，date_range 取这些
   日期的**最早到最晚**的连续范围**：start = 最早一天，end = 最晚一天。
   例：「同步昨天和今天的数据」→ start = 昨天，end = 今天（覆盖两天）。
   注意：不要把「昨天和今天」只收敛成今天；必须包含提到的每一个日期。

## 报告 vs 分析（易混淆，务必区分）
1. **report_trigger** 仅用于完整日报/晨报/综合健康报告：
   - 含「日报/晨报/综合报告/完整报告/健康报告」；
   - 或「生成XX日的报告/XX到XX的报告」且**未限定单一主题**（如体重、饮食、训练）。
2. **trend_analysis** 用于领域/主题专项报告或趋势分析：
   - 含「变化/趋势/对比/分析」+「报告/分析」→ trend_analysis，并填对应 domain；
   - 例：「体重变化报告」「近7天饮食分析报告」「近30天体脂变化」→ trend_analysis，**不是** report_trigger。
3. 出现「折线图/趋势图/柱状图/统计图/画个图/可视化」等**明确要图**的措辞 → chart_trigger，
   而不是 trend_analysis（后者只要文字分析）。
   例：「近7天体重折线图」→ chart_trigger；「近7天体重变化」→ trend_analysis。
4. 「把图插入到/加到…（已有日报/文档）」只做插入 → 只给 chart_trigger，
   **不要**同时给 report_trigger（不重新生成报告）。

## 易混淆情况
1. 出现「每天/每日/定时/固定」+ 任何动作 → schedule_manage，而**不是** report_trigger / sync_trigger。
   例：「每天8点同步数据」→ schedule_manage。
2. 「生成日报」未指明日期时 date_range 填 null，由对话层向用户追问具体日期；指明日期则用该日期。
3. data_query 与 trend_analysis 的区别：问「某天/某段是多少/有没有」是 data_query；问「变化/趋势/对比/XX报告（有主题）」是 trend_analysis。
4. 带数量描述的食物语句（吃了鸡胸肉200g、鸡蛋2个）→ manual_entry，不是 data_query。
5. 「目标/降到/增到/减到 + 数值单位」→ goal_setting。
6. 同一句含「记录初始体重/体脂」+「评价/进度/怎么样」→ 必须返回 manual_entry、goal_setting（若有目标）、trend_analysis 多个意图；不要把年份当成体重数值。
7. 与健身数据无关的问候/闲聊/帮助请求 → general。
8. 「搜一下/联网/网上查/查资料」或询问公开知识（什么是、如何练、推荐摄入、科学依据、指南、最新研究）→ web_search，**不是** data_query。
   问用户自己某天吃了/练了/体重是多少仍是 data_query。
   若同时要对照自己的数据和公开推荐（如「我昨天蛋白质对照推荐量够不够」），intents 可含 data_query 与 web_search。

## 域推断（domain）
- body：体重、体脂、围度等身体指标
- nutrition：热量、蛋白、碳水、脂肪、饮食、餐、吃了
- fitness：训练、动作、组数、卧推、深蹲、计划
- 无法判断或与数据域无关 → null
- 多意图且各操作域不同时，取**第一个操作**的域。

# 输出格式（严格遵守）
只输出一个 JSON 对象，禁止任何额外文字或代码块标记：
{{"intents": ["<intent>", ...], "domain": "<body|nutrition|fitness|null>", "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} 或 null, "reasoning": "<不超过30字的判断理由>"}}

# 示例

用户：同步今日数据
输出：{{"intents": ["sync_trigger"], "domain": null, "date_range": {{"start": "{today.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "同步当天数据"}}

用户：同步昨天和今天的数据
输出：{{"intents": ["sync_trigger"], "domain": null, "date_range": {{"start": "{yesterday.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "同步昨天与今天两天"}}

用户：同步前天和昨天的数据
输出：{{"intents": ["sync_trigger"], "domain": null, "date_range": {{"start": "{(yesterday - timedelta(days=1)).isoformat()}", "end": "{yesterday.isoformat()}"}}, "reasoning": "同步前天与昨天两天"}}

用户：同步{example_cn}数据并生成日报
输出：{{"intents": ["sync_trigger", "report_trigger"], "domain": null, "date_range": {{"start": "{yesterday.isoformat()}", "end": "{yesterday.isoformat()}"}}, "reasoning": "先同步该日数据再生成日报"}}

用户：生成昨天的日报
输出：{{"intents": ["report_trigger"], "domain": null, "date_range": {{"start": "{yesterday.isoformat()}", "end": "{yesterday.isoformat()}"}}, "reasoning": "生成昨日日报"}}

用户：生成8月20日到8月25日的报告
输出：{{"intents": ["report_trigger"], "domain": null, "date_range": {{"start": "{month_start.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "生成区间周期报表"}}

用户：生成最近7天体重折线图
输出：{{"intents": ["chart_trigger"], "domain": "body", "date_range": {{"start": "{last_week_start.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "绘制体重折线图"}}

用户：把近30天摄入热量画成柱状图保存到文档
输出：{{"intents": ["chart_trigger"], "domain": "nutrition", "date_range": {{"start": "{month_start.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "生成热量柱状图文档"}}

用户：拉取最近7天训记数据
输出：{{"intents": ["sync_trigger"], "domain": null, "date_range": {{"start": "{last_week_start.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "同步最近7天含今天"}}

用户：昨天吃了多少蛋白质
输出：{{"intents": ["data_query"], "domain": "nutrition", "date_range": {{"start": "{yesterday.isoformat()}", "end": "{yesterday.isoformat()}"}}, "reasoning": "查询昨日饮食"}}

用户：近30天体脂变化趋势
输出：{{"intents": ["trend_analysis"], "domain": "body", "date_range": {{"start": "{month_start.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "体脂趋势分析"}}

用户：给我一个近7天的体重变化报告
输出：{{"intents": ["trend_analysis"], "domain": "body", "date_range": {{"start": "{last_week_start.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "体重变化专项报告"}}

用户：近7天饮食分析报告
输出：{{"intents": ["trend_analysis"], "domain": "nutrition", "date_range": {{"start": "{last_week_start.isoformat()}", "end": "{today.isoformat()}"}}, "reasoning": "饮食专项分析报告"}}

用户：记录体重 72.5kg
输出：{{"intents": ["manual_entry"], "domain": "body", "date_range": null, "reasoning": "手动录入体重"}}

用户：以2025年9月1日为起点，记录初始体重130kg、体脂37%，目标减到85kg，评价减肥进度
输出：{{"intents": ["manual_entry", "goal_setting", "trend_analysis"], "domain": "body", "date_range": {{"start": "2025-09-01", "end": "{today.isoformat()}"}}, "reasoning": "录入基准、设目标、评进度"}}

用户：每天早上7点生成日报
输出：{{"intents": ["schedule_manage"], "domain": null, "date_range": null, "reasoning": "定时重复任务"}}

用户：搜一下HIIT一周练几次比较好
输出：{{"intents": ["web_search"], "domain": "fitness", "date_range": null, "reasoning": "联网检索训练知识"}}

用户：蛋白质推荐摄入量有什么科学依据
输出：{{"intents": ["web_search"], "domain": "nutrition", "date_range": null, "reasoning": "检索公开营养资料"}}

用户：你好
输出：{{"intents": ["general"], "domain": null, "date_range": null, "reasoning": "寒暄"}}
"""


@trace_agent("IntentAgent")
def run_intent_agent(message: str, today: date | None = None) -> RouteResult | None:
    """LLM 意图识别。未配置 LLM 或调用/解析失败时返回 None（由关键词兜底）。"""
    if not is_llm_configured():
        return None

    today = today or datetime.now(_LOCAL_TZ).date()
    try:
        content = chat_completion(
            [
                {"role": "system", "content": build_system_prompt(today)},
                {"role": "user", "content": message},
            ],
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - LLM failure must fall back to rules
        logger.warning("意图 Agent LLM 调用失败，回退关键词匹配: %s", exc)
        return None

    return parse_agent_response(content, today=today)


def parse_agent_response(content: str, today: date | None = None) -> RouteResult | None:
    """解析并校验 LLM 输出；任何不合法之处都尽量降级而不是整体失败。"""
    today = today or datetime.now(_LOCAL_TZ).date()
    data = _extract_json(content)
    if data is None or not isinstance(data, dict):
        logger.warning("意图 Agent 输出非 JSON: %r", content[:200])
        return None

    intents = _parse_intents(data)
    if not intents:
        return None

    domain = _parse_domain(data.get("domain"))
    start_date, end_date = _parse_date_range(data.get("date_range"), today)

    return RouteResult(
        intents=intents,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
    )


def _extract_json(content: str) -> dict | None:
    """从 LLM 输出提取 JSON 对象（容忍 Markdown 代码块和前后缀文字）。"""
    if not content:
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_intents(data: dict) -> list[Intent]:
    """解析 intents 字段（容忍单数 intent、字符串、未知值、confirmation 误报）。"""
    raw = data.get("intents")
    if raw is None:
        raw = data.get("intent")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    intents: list[Intent] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        value = item.strip().lower()
        if value not in _INTENT_VALUES:
            logger.warning("意图 Agent 返回未知意图: %r", item)
            continue
        # 确认/取消由 Router 在 pending_confirmation 上下文中优先处理，LLM 无法感知该上下文
        if value == Intent.CONFIRMATION_RESPONSE.value:
            continue
        intent = Intent(value)
        if intent not in intents:
            intents.append(intent)
        if len(intents) >= _MAX_INTENTS:
            break
    return intents


def _parse_domain(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in ("null", "none", ""):
        return None
    value = _DOMAIN_ALIASES.get(value, value)
    return value if value in _VALID_DOMAINS else None


def _parse_date_range(raw: object, today: date) -> tuple[date | None, date | None]:
    """解析 date_range 字段；无日期、格式错误、顺序颠倒或晚于今天时返回 (None, None)。"""
    if not isinstance(raw, dict):
        return None, None

    start = _parse_iso_date(raw.get("start"))
    end = _parse_iso_date(raw.get("end"))
    if start is None and end is None:
        return None, None
    if start is None:
        start = end
    if end is None:
        end = start
    if start > end:
        start, end = end, start
    if end > today:
        # 未来日期视为无效（LLM 幻觉或用户口误），交由默认逻辑处理
        logger.warning("意图 Agent 返回未来日期范围: %s ~ %s", start, end)
        return None, None
    return start, end


def _parse_iso_date(raw: object) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None
