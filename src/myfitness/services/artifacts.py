"""会话产物读取 — 只允许读取 data_dir 之内的文件。

产物（报告 / 统计图文档）由对话过程生成，前端需要按路径取回内容展示。
路径由前端传入，因此必须校验落在数据目录之内，避免退化成任意文件读取。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from myfitness.agents.document_blocks import docx_to_preview_html
from myfitness.config import Settings, get_settings
from myfitness.rag.document_parser import DocumentParseError, parse_document

MAX_CHARS = 400_000  # 约 400 KB，超出截断以免前端渲染卡死


class ArtifactError(ValueError):
    """产物不可读：路径越界 / 不存在 / 不是文件。"""


def artifact_root(settings: Settings | None = None) -> Path:
    return Path((settings or get_settings()).data_dir).resolve()


def resolve_artifact(path: str, settings: Settings | None = None) -> Path:
    """校验产物路径落在 data_dir 之内，返回解析后的绝对路径。"""
    raw = str(path or "").strip().strip('"')
    if not raw:
        raise ArtifactError("缺少产物路径")
    root = artifact_root(settings)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        target = candidate.expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:  # Windows 非法盘符 / 过长路径
        raise ArtifactError(f"产物路径无效：{raw}") from exc
    if not target.is_relative_to(root):
        raise ArtifactError("产物必须位于数据目录之内")
    if not target.is_file():
        raise ArtifactError("产物不存在或不是文件")
    return target


def _artifact_kind(posix: str) -> str:
    if "/charts/" in f"/{posix}":
        return "chart"
    if "/documents/" in f"/{posix}":
        return "document"
    return "report"


def read_artifact(path: str, settings: Settings | None = None) -> dict[str, Any]:
    """读取产物元数据与可预览内容，返回可直接下发给前端的结构。"""
    target = resolve_artifact(path, settings)
    stat = target.stat()
    posix = target.as_posix()
    ext = target.suffix.lower()
    base: dict[str, Any] = {
        "path": posix,
        "name": target.name,
        "title": target.stem,
        "kind": _artifact_kind(posix),
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "format": ext.lstrip(".") or "txt",
        "truncated": False,
    }

    if ext == ".pdf":
        return {
            **base,
            "preview_type": "pdf",
            "content": "",
            "preview_html": "",
        }

    if ext in {".docx", ".doc"}:
        data = target.read_bytes()
        try:
            preview_html = docx_to_preview_html(data)
        except DocumentParseError as exc:
            raise ArtifactError(str(exc)) from exc
        return {
            **base,
            "preview_type": "docx_html",
            "preview_html": preview_html,
            "content": "",
        }

    raw = target.read_bytes()
    content = raw.decode("utf-8", errors="replace")
    truncated = len(content) > MAX_CHARS
    if truncated:
        content = content[:MAX_CHARS] + "\n\n…（内容过长，已截断显示）"
    return {
        **base,
        "preview_type": "markdown",
        "content": content,
        "preview_html": "",
        "truncated": truncated,
    }


def read_artifact_bytes(path: str, settings: Settings | None = None) -> tuple[Path, bytes, str]:
    """读取产物原始字节，供浏览器内嵌预览 PDF 等二进制文件。"""
    target = resolve_artifact(path, settings)
    ext = target.suffix.lower()
    content_type = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
    }.get(ext, "application/octet-stream")
    return target, target.read_bytes(), content_type
