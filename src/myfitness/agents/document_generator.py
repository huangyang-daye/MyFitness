"""主题文档生成 — 按目标格式生成可交付正文。"""

from __future__ import annotations

import json
import logging
import re

from myfitness.agents.document_blocks import blocks_to_markdown, parse_document_blocks
from myfitness.agents.summary import build_rule_based_summary
from myfitness.agents.tools.query_format import format_query_results
from myfitness.llm.factory import chat_completion, is_llm_configured
from myfitness.rag.format import format_retrieved_chunks
from myfitness.rag.schemas import RetrievedChunk
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import ContextSnapshot, Intent

logger = logging.getLogger(__name__)

_DOCX_JSON_SCHEMA = """{
  "blocks": [
    {"type": "title", "text": "文档主标题"},
    {"type": "heading", "level": 2, "text": "小节标题"},
    {"type": "paragraph", "text": "正文段落，纯文本，不要使用 Markdown 或 HTML。"},
    {"type": "bullet_list", "items": ["要点一", "要点二"]},
    {"type": "numbered_list", "items": ["步骤一", "步骤二"]},
    {"type": "table", "rows": [["列A", "列B"], ["值1", "值2"]]}
  ]
}"""


def build_document_messages(
    user_message: str,
    agent_outputs: AgentOutputs | None,
    context: ContextSnapshot | None,
    *,
    output_format: str = "md",
) -> list[dict[str, str]]:
    parts = [f"用户诉求：{user_message.strip()}"]

    outputs = agent_outputs or AgentOutputs()
    if outputs.body and outputs.body.narrative:
        parts.append(f"【身体分析要点】\n{outputs.body.narrative}")
        if outputs.body.recommendations:
            parts.append("身体建议：" + "；".join(outputs.body.recommendations))
    if outputs.nutrition and outputs.nutrition.narrative:
        parts.append(f"【饮食分析要点】\n{outputs.nutrition.narrative}")
    if outputs.fitness and outputs.fitness.narrative:
        parts.append(f"【训练分析要点】\n{outputs.fitness.narrative}")

    if context and context.query_results:
        parts.append(
            "【参考数据 — 把具体数字融入文档正文，不要原样粘贴此段标题或列表】\n"
            + format_query_results(context.query_results)
        )
    if context and context.retrieved_chunks:
        chunks = _to_retrieved_chunks(context.retrieved_chunks)
        if chunks:
            parts.append(
                "【历史参考 — 可引用，勿整段复制】\n" + format_retrieved_chunks(chunks)
            )
    if context and context.memory_long_term.strip():
        parts.append("【用户画像】\n" + context.memory_long_term.strip())
    if context and context.reflection_notes.strip():
        parts.append(
            "【上下文反思 — 已从数据库确认的个体事实】\n" + context.reflection_notes.strip()
        )
    if context and context.memory_short_term.strip():
        parts.append("【本轮会话要点】\n" + context.memory_short_term.strip())
    if context and context.data_gaps:
        parts.append("【数据缺口】\n" + "\n".join(f"- {g}" for g in context.data_gaps))

    if output_format == "docx":
        system = (
            "你是 MyFitness 健康助手的 Word 文档撰写模块。"
            "请根据用户诉求与参考数据，输出**结构化 JSON**，用于直接生成 .docx 文件。\n\n"
            "要求：\n"
            "1. 只输出一个 JSON 对象，格式如下（不要 Markdown 代码围栏外的说明）：\n"
            f"{_DOCX_JSON_SCHEMA}\n"
            "2. 允许的类型：title / heading / paragraph / bullet_list / numbered_list / table。\n"
            "3. paragraph 与 list 项必须是纯文本，禁止 #、**、|、` 等 Markdown 标记。\n"
            "4. 文首用 title 作为主标题；小节用 heading（level 2 或 3）。\n"
            "5. 将体重、热量、蛋白等数字自然写入段落、列表或 table，禁止编造。\n"
            "6. 不做医疗诊断；不要输出文件路径或「已保存」类元信息。"
        )
    else:
        system = (
            "你是 MyFitness 健康助手的文档撰写模块。请根据用户诉求与参考数据，撰写**可独立阅读、可直接交付**"
            "的 Markdown 文档正文。\n\n"
            "要求：\n"
            "1. 这是文档而非聊天回复：禁止「好的」「如下」「数据库查询结果」等会话或系统用语。\n"
            "2. 紧扣用户主题与全部约束（例如不喝蛋白粉、只吃食堂、宏量比例原则等）。\n"
            "3. 将体重、热量、蛋白、碳水、脂肪等数字自然写入表格或列表，禁止编造未提供的数据。\n"
            "4. 饮食规划类文档建议包含：目标与计算依据、每日营养素目标、三餐/加餐构成、"
            "食堂可选食物示例、注意事项。\n"
            "5. 结构用 Markdown 标题组织；文首用单个 # 作为主标题。\n"
            "6. 不做医疗诊断；不要输出文件路径或「已保存」类元信息。\n"
            "7. 只输出文档正文 Markdown。"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def generate_document_body(
    user_message: str,
    agent_outputs: AgentOutputs | None = None,
    context: ContextSnapshot | None = None,
    *,
    fallback: str = "",
    output_format: str = "md",
) -> str:
    """生成文档正文；docx 返回 JSON，md/pdf 返回 Markdown。"""
    fmt = output_format.strip().lower()
    if fmt == "markdown":
        fmt = "md"

    if is_llm_configured():
        try:
            messages = build_document_messages(
                user_message,
                agent_outputs,
                context,
                output_format=fmt,
            )
            content = chat_completion(messages, temperature=0.3)
            if content.strip():
                if fmt == "docx":
                    blocks = parse_document_blocks(content, prefer_json=True)
                    if blocks:
                        return content.strip()
                    logger.warning("DOCX JSON 解析失败，回退规则模板")
                else:
                    return content.strip()
            logger.warning("LLM 文档生成为空，使用规则兜底")
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 文档生成失败，使用规则兜底: %s", exc)

    return _rule_based_document_body(user_message, agent_outputs, context, fallback, output_format=fmt)


def infer_document_title(user_message: str) -> str:
    text = user_message.strip()
    if re.search(r"饮食|营养|餐|碳水|蛋白|脂肪|食堂", text):
        return "饮食规划"
    if re.search(r"训练|健身|锻炼|计划", text):
        return "训练计划"
    if re.search(r"减肥|减脂|增肌|体重", text):
        return "身体管理规划"
    return "健康规划"


def _rule_based_document_body(
    user_message: str,
    agent_outputs: AgentOutputs | None,
    context: ContextSnapshot | None,
    fallback: str,
    *,
    output_format: str = "md",
) -> str:
    title = infer_document_title(user_message)
    if output_format == "docx":
        blocks: list[dict] = [
            {"type": "title", "text": title},
            {"type": "heading", "level": 2, "text": "诉求摘要"},
            {"type": "paragraph", "text": user_message.strip()},
        ]
        outputs = agent_outputs or AgentOutputs()
        if outputs.body and outputs.body.narrative:
            blocks.extend(
                [
                    {"type": "heading", "level": 2, "text": "身体数据与建议"},
                    {"type": "paragraph", "text": outputs.body.narrative},
                ]
            )
        if outputs.nutrition and outputs.nutrition.narrative:
            blocks.extend(
                [
                    {"type": "heading", "level": 2, "text": "饮食建议"},
                    {"type": "paragraph", "text": outputs.nutrition.narrative},
                ]
            )
        if outputs.fitness and outputs.fitness.narrative:
            blocks.extend(
                [
                    {"type": "heading", "level": 2, "text": "训练建议"},
                    {"type": "paragraph", "text": outputs.fitness.narrative},
                ]
            )
        if not (outputs.body or outputs.nutrition or outputs.fitness) and fallback.strip():
            cleaned = _strip_chat_artifacts(fallback)
            if cleaned:
                blocks.extend(
                    [
                        {"type": "heading", "level": 2, "text": "补充说明"},
                        {"type": "paragraph", "text": cleaned},
                    ]
                )
        return json.dumps({"blocks": blocks}, ensure_ascii=False)

    sections = [f"# {title}", "", f"## 诉求摘要\n\n{user_message.strip()}"]

    outputs = agent_outputs or AgentOutputs()
    if outputs.body and outputs.body.narrative:
        sections.extend(["", "## 身体数据与建议", "", outputs.body.narrative])
    if outputs.nutrition and outputs.nutrition.narrative:
        sections.extend(["", "## 饮食建议", "", outputs.nutrition.narrative])
    if outputs.fitness and outputs.fitness.narrative:
        sections.extend(["", "## 训练建议", "", outputs.fitness.narrative])

    if not (outputs.body or outputs.nutrition or outputs.fitness) and fallback.strip():
        cleaned = _strip_chat_artifacts(fallback)
        if cleaned:
            sections.extend(["", cleaned])

    if context and not (outputs.body or outputs.nutrition or outputs.fitness):
        draft = build_rule_based_summary(
            outputs,
            context,
            Intent.TREND_ANALYSIS,
            include_query_results=False,
        )
        cleaned = _strip_chat_artifacts(draft)
        if cleaned:
            sections.extend(["", cleaned])

    return "\n".join(sections).strip() + "\n"


def _strip_chat_artifacts(text: str) -> str:
    """去掉对话摘要里不适合写入交付文档的段落。"""
    drop_prefixes = (
        "**数据库查询结果**",
        "**语义检索",
        "**用户画像",
        "**本轮会话",
        "**数据提示**",
        "**参考资料**",
        "【数据库查询",
        "【语义检索",
        "【用户画像",
        "【本轮会话",
    )
    kept: list[str] = []
    for block in re.split(r"\n\n+", text.strip()):
        if any(block.startswith(prefix) for prefix in drop_prefixes):
            continue
        kept.append(block)
    return "\n\n".join(kept).strip()


def _to_retrieved_chunks(items: list[dict]) -> list[RetrievedChunk]:
    from datetime import date as date_cls

    chunks: list[RetrievedChunk] = []
    for item in items:
        record_date = item.get("record_date")
        parsed_date = date_cls.fromisoformat(record_date) if isinstance(record_date, str) else None
        chunks.append(
            RetrievedChunk(
                id=int(item.get("id", 0)),
                source_type=str(item.get("source_type", "")),
                source_id=str(item.get("source_id", "")),
                domain=str(item.get("domain", "")),
                title=str(item.get("title", "")),
                content=str(item.get("content", "")),
                record_date=parsed_date,
                similarity=float(item.get("similarity", 0.0)),
                metadata=item.get("metadata"),
            )
        )
    return chunks
