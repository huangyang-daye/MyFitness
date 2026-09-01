"""文档读写 Tool — 支持 md / doc / docx / pdf 的读取与生成。

读取范围：data_dir 下的 documents、reports 等子目录。
写入范围：仅 document_output_dir（默认 <data_dir>/documents）。
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool
from sqlalchemy.orm import Session

from myfitness.agents.document_blocks import (
    blocks_to_markdown,
    markdown_to_blocks,
    parse_document_blocks,
    write_docx_blocks,
    write_pdf_blocks,
)
from myfitness.config import Settings, get_settings
from myfitness.llm.factory import chat_completion, is_llm_configured
from myfitness.rag.document_parser import (
    DocumentParseError,
    SUPPORTED_EXTENSIONS,
    parse_document,
)

logger = logging.getLogger(__name__)

MAX_WRITE_CHARS = 200_000
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WRITE_HINTS = (
    "生成文档",
    "写成文档",
    "保存成文档",
    "保存为",
    "导出为",
    "导出成",
    "输出文档",
    "整理成文档",
    "写成md",
    "写成 word",
    "写成word",
    "写成 pdf",
    "写成pdf",
    "word文档",
    "pdf文档",
    "markdown文档",
    "md文档",
    "的文档",
)
_DOCUMENT_GEN_RE = re.compile(
    r"(?:生成|写|做|整理|输出|导出|给我).{0,20}(?:的)?文档"
    r"|(?:饮食|训练|健身|减肥|减脂|营养).{0,12}(?:规划|计划).{0,12}文档"
    r"|文档.{0,12}(?:生成|保存|导出|写入)"
)
_REPORT_DOC_WORDS = ("日报", "晨报", "健康日报", "综合报告", "完整报告", "健康报告", "周期报表")
_MINIMAL_CHAT_HINTS = (
    "不要输出其他内容",
    "不要其他内容",
    "只输出文档",
    "仅输出文档",
    "输出为文档",
    "然后输出为文档",
    "生成规划然后输出为文档",
    "文档里也不要",
    "不要输出其他",
)
_READ_HINTS = (
    "读取文档",
    "打开文档",
    "看看文档",
    "阅读文档",
    "读一下文档",
    "文档内容",
)
_DOC_PATH_RE = re.compile(
    r"(?:documents/|reports/)?[\w\u4e00-\u9fff.\-_ ]+\.(?:md|markdown|docx?|pdf)",
    re.IGNORECASE,
)
_FORMAT_HINTS: dict[str, tuple[str, ...]] = {
    "pdf": (".pdf", "pdf", "PDF", "pd文档", "pd文件", "pd 文档", "pd 文件"),
    "docx": (".docx", "word", "Word", "WORD", "docx"),
    "doc": (".doc",),
    "md": (".md", ".markdown", "markdown", "Markdown", "MD"),
}
_PD_FORMAT_RE = re.compile(r"(?:保存|导出|输出|写成|整理成|产出).{0,6}pd(?:文档|文件)?|pd\s*文档|pd\s*文件", re.I)


class DocumentToolError(ValueError):
    pass


def is_document_generation_request(message: str) -> bool:
    """用户要求生成/导出主题文档（非日报、非统计图文档）。"""
    text = message.strip()
    if not text:
        return False
    if any(word in text for word in _REPORT_DOC_WORDS):
        return False
    if _DOCUMENT_GEN_RE.search(text):
        return True
    return any(token in text for token in _WRITE_HINTS)


def needs_document_write(message: str) -> bool:
    return is_document_generation_request(message)


def wants_minimal_chat_for_document(message: str) -> bool:
    """用户希望交付物是文档，对话里只保留简短确认。"""
    text = message.strip()
    if not is_document_generation_request(text):
        return False
    return any(token in text for token in _MINIMAL_CHAT_HINTS) or "输出为文档" in text


def format_document_saved_reply(result: dict[str, Any]) -> str:
    exports = result.get("exports")
    if exports:
        lines = ["已根据你的要求生成文档："]
        for item in exports:
            if item.get("error"):
                lines.append(f"- 保存失败（{item.get('format', '?')}）：{item['error']}")
                continue
            title = Path(str(item.get("filename", "文档"))).stem
            lines.append(f"- **{title}**（{item.get('format', '')}）：`{item.get('path', '')}`")
        return "\n".join(lines)

    title = Path(str(result.get("filename", "文档"))).stem
    path = result.get("path", "")
    return f"已根据你的要求生成文档：**{title}**\n\n文件：`{path}`"


def needs_document_read(message: str) -> bool:
    text = message.strip()
    if not text:
        return False
    return any(token in text for token in _READ_HINTS)


def extract_document_path(message: str) -> str | None:
    match = _DOC_PATH_RE.search(message)
    if not match:
        return None
    return match.group(0).strip().replace("\\", "/")


def infer_document_format(message: str, explicit: str | None = None) -> str:
    if explicit:
        fmt = explicit.strip().lower().lstrip(".")
        if fmt in {"pd", "pdf"}:
            return "pdf"
        if fmt == "markdown":
            return "md"
        if fmt in {"md", "docx", "pdf", "doc"}:
            return fmt
    text = message.lower()
    if _PD_FORMAT_RE.search(message):
        return "pdf"
    for fmt, hints in _FORMAT_HINTS.items():
        if any(h.lower() in text for h in hints):
            return fmt
    return "md"


def infer_document_formats(message: str, explicit: str | None = None) -> list[str]:
    """从用户消息中识别需要导出的全部文档格式。"""
    if explicit:
        return [infer_document_format(message, explicit)]
    text = message.strip()
    lower = text.lower()
    formats: list[str] = []

    def add(fmt: str) -> None:
        if fmt not in formats:
            formats.append(fmt)

    if _PD_FORMAT_RE.search(text) or "pdf" in lower:
        add("pdf")
    if "docx" in lower or "word" in lower:
        add("docx")
    if (
        "md文档" in lower
        or "markdown" in lower
        or ".md" in lower
        or re.search(r"(?:^|[\s、,，/])md(?:文档|[\s、,，/]|$)", lower)
    ):
        add("md")

    if formats:
        return formats
    return [infer_document_format(message)]


def document_output_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    path = Path(settings.document_output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def data_root(settings: Settings | None = None) -> Path:
    return Path((settings or get_settings()).data_dir).resolve()


def sanitize_filename(name: str) -> str:
    cleaned = Path(str(name).replace("\\", "/")).name.strip()
    cleaned = _INVALID_FILENAME.sub("-", cleaned).strip(". ")
    if not cleaned:
        cleaned = "document.md"
    return cleaned


def resolve_readable_path(path: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    raw = str(path or "").strip()
    if not raw:
        raise DocumentToolError("缺少文档路径")
    candidate = Path(raw).expanduser()
    target = candidate.resolve() if candidate.is_absolute() else (data_root(settings) / candidate).resolve()
    allowed_roots = _readable_roots(settings)
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise DocumentToolError("只能读取数据目录内的文档")
    if not target.is_file():
        raise DocumentToolError(f"文档不存在：{target}")
    if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentToolError(f"不支持的文档类型：{target.suffix}")
    return target


def _readable_roots(settings: Settings) -> list[Path]:
    roots = {
        data_root(settings),
        Path(settings.document_output_dir).expanduser().resolve(),
        Path(settings.daily_report_output_dir).expanduser().resolve(),
        Path(settings.chart_output_dir).expanduser().resolve(),
    }
    return list(roots)


def resolve_writable_path(filename: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    out_dir = document_output_dir(settings)
    safe_name = sanitize_filename(filename)
    target = (out_dir / safe_name).resolve()
    if not target.is_relative_to(out_dir):
        raise DocumentToolError("文件名无效")
    return target


def read_document_file(path: str, settings: Settings | None = None) -> dict[str, Any]:
    target = resolve_readable_path(path, settings)
    data = target.read_bytes()
    parsed = parse_document(target.name, data)
    return {
        "tool": "read_document",
        "path": str(target),
        "filename": parsed.filename,
        "format": parsed.format,
        "title": parsed.title,
        "content": parsed.content,
        "char_count": parsed.char_count,
        "truncated": parsed.truncated,
    }


def write_document_file(
    filename: str,
    content: str,
    *,
    format: str = "md",
    settings: Settings | None = None,
) -> dict[str, Any]:
    if not content.strip():
        raise DocumentToolError("文档内容不能为空")
    if len(content) > MAX_WRITE_CHARS:
        raise DocumentToolError(f"文档内容不能超过 {MAX_WRITE_CHARS // 1000}K 字符")

    fmt = infer_document_format("", format)
    if fmt == "doc":
        raise DocumentToolError("不支持写入旧版 .doc，请使用 docx、md 或 pdf")

    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()
    expected_ext = {
        "md": {".md", ".markdown", ""},
        "docx": {".docx", ""},
        "pdf": {".pdf", ""},
    }.get(fmt, {".md", ""})
    if ext not in expected_ext and ext:
        safe_name = f"{Path(safe_name).stem}.{fmt if fmt != 'md' else 'md'}"

    target = resolve_writable_path(safe_name, settings)
    target.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "md":
        blocks = parse_document_blocks(content, prefer_json=False)
        text = blocks_to_markdown(blocks) if blocks else content.rstrip() + "\n"
        target.write_text(text, encoding="utf-8")
    elif fmt == "docx":
        blocks = parse_document_blocks(content, prefer_json=True)
        if not blocks:
            raise DocumentToolError("DOCX 内容无法解析为有效文档结构")
        write_docx_blocks(target, blocks)
    elif fmt == "pdf":
        blocks = parse_document_blocks(content, prefer_json=False)
        if not blocks:
            raise DocumentToolError("PDF 内容为空")
        write_pdf_blocks(target, blocks)
    else:
        raise DocumentToolError(f"不支持的写入格式：{fmt}")

    stat = target.stat()
    return {
        "tool": "write_document",
        "path": str(target),
        "filename": target.name,
        "format": fmt,
        "size": stat.st_size,
        "char_count": len(content),
        "created_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def list_document_files(
    *,
    subdir: str | None = None,
    limit: int = 50,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    root = document_output_dir(settings)
    scan_root = root
    if subdir:
        scan_root = (root / subdir).resolve()
        if not scan_root.is_relative_to(root):
            raise DocumentToolError("子目录必须在 documents 目录内")

    files: list[dict[str, Any]] = []
    if scan_root.exists():
        for path in sorted(scan_root.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            stat = path.stat()
            files.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "format": path.suffix.lstrip(".").lower(),
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                }
            )
            if len(files) >= limit:
                break

    return {"tool": "list_documents", "count": len(files), "documents": files}


def suggest_document_metadata(user_message: str, content: str) -> dict[str, str]:
    fmt = infer_document_format(user_message)
    if is_llm_configured():
        try:
            raw = chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是文档命名助手。根据用户诉求与文档摘要，输出 JSON："
                            '{"filename":"文件名含扩展名","format":"md|docx|pdf"}。'
                            "文件名不要含路径，可用中文，扩展名须与 format 一致。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户诉求：{user_message}\n\n"
                            f"文档开头：\n{content[:1200]}"
                        ),
                    },
                ],
                temperature=0.2,
            )
            data = json.loads(_extract_json(raw))
            filename = sanitize_filename(str(data.get("filename", "")))
            chosen = infer_document_format(user_message, str(data.get("format", fmt)))
            if not Path(filename).suffix:
                filename = f"{Path(filename).stem}.{chosen if chosen != 'md' else 'md'}"
            return {"filename": filename, "format": chosen}
        except Exception as exc:  # noqa: BLE001
            logger.warning("文档命名 LLM 失败，使用规则兜底: %s", exc)
    return _fallback_metadata(user_message, fmt)


def apply_document_export(
    session: Session,
    user_id: int,
    user_message: str,
    content_md: str,
    *,
    agent_outputs=None,
    context=None,
) -> dict[str, Any] | None:
    """用户要求导出文档时，生成独立文档正文并写入文件（支持多格式）。"""
    if not needs_document_write(user_message):
        return None

    from myfitness.agents.document_generator import generate_document_body
    from myfitness.agents.tools.base import invoke_tool

    formats = infer_document_formats(user_message)
    docx_body = ""
    md_body = ""

    if "docx" in formats:
        docx_body = generate_document_body(
            user_message,
            agent_outputs,
            context,
            fallback=content_md,
            output_format="docx",
        )
    if "md" in formats or "pdf" in formats:
        if docx_body.strip():
            blocks = parse_document_blocks(docx_body, prefer_json=True)
            md_body = blocks_to_markdown(blocks) if blocks else ""
        if not md_body.strip():
            md_body = generate_document_body(
                user_message,
                agent_outputs,
                context,
                fallback=content_md,
                output_format="md",
            )

    if not any(body.strip() for body in (docx_body, md_body)):
        return None

    snippet = _metadata_snippet(md_body or docx_body, formats[0])
    meta = suggest_document_metadata(user_message, snippet)
    stem = Path(meta["filename"]).stem

    exports: list[dict[str, Any]] = []
    for fmt in formats:
        body = docx_body if fmt == "docx" else md_body
        if not body.strip():
            continue
        ext = "md" if fmt == "md" else fmt
        filename = sanitize_filename(f"{stem}.{ext}")
        result = invoke_tool(
            write_document,
            session,
            user_id,
            filename=filename,
            content=body,
            format=fmt,
        )
        if isinstance(result, dict):
            result.setdefault("format", fmt)
            exports.append(result)

    if not exports:
        return None

    document_only = wants_minimal_chat_for_document(user_message)
    payload: dict[str, Any] = {
        "tool": "write_document",
        "exports": exports,
        "paths": [item["path"] for item in exports if item.get("path")],
        "document_only": document_only,
    }
    if len(exports) == 1:
        payload.update(exports[0])
    else:
        payload["filename"] = exports[0].get("filename", "")
        payload["path"] = exports[0].get("path", "")
    return payload


@tool
def read_document(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    path: str,
) -> dict[str, Any]:
    """读取 md / doc / docx / pdf 文档并提取纯文本。

    Args:
        path: 文档路径。可为 data_dir 下的绝对路径，或相对路径（如 documents/计划.md）。
    """
    _ = session, user_id
    try:
        return read_document_file(path)
    except (DocumentParseError, DocumentToolError) as exc:
        return {"tool": "read_document", "path": path, "error": str(exc)}


@tool
def write_document(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    filename: str,
    content: str,
    format: str = "md",
) -> dict[str, Any]:
    """根据用户诉求生成并保存文档，由你自行命名文件。

    Args:
        filename: 文件名（含扩展名），不要包含路径，例如「减肥进度总结-2026-09.md」。
        content: 文档正文。docx 为结构化 JSON（blocks）；md/pdf 为 Markdown。
        format: 输出格式 md / docx / pdf，默认 md。
    """
    _ = session, user_id
    try:
        return write_document_file(filename, content, format=format)
    except DocumentToolError as exc:
        return {"tool": "write_document", "filename": filename, "error": str(exc)}


@tool
def list_documents(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    subdir: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """列出 documents 目录下已保存的文档。

    Args:
        subdir: 可选子目录。
        limit: 最多返回条数。
    """
    _ = session, user_id
    try:
        return list_document_files(subdir=subdir, limit=limit)
    except DocumentToolError as exc:
        return {"tool": "list_documents", "error": str(exc), "documents": []}


def _metadata_snippet(body: str, fmt: str) -> str:
    if fmt == "docx":
        blocks = parse_document_blocks(body, prefer_json=True)
        if blocks:
            return blocks_to_markdown(blocks)[:1200]
    return body[:1200]


def _fallback_metadata(user_message: str, fmt: str) -> dict[str, str]:
    slug = _slugify(user_message)
    ext = "md" if fmt == "md" else fmt
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return {"filename": f"{slug}-{stamp}.{ext}", "format": fmt}


def _slugify(text: str, *, max_len: int = 24) -> str:
    cleaned = re.sub(r"\s+", "-", text.strip())
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "", cleaned, flags=re.UNICODE)
    cleaned = cleaned.strip("-")
    if not cleaned:
        cleaned = "document"
    return cleaned[:max_len]


def _extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text
