"""知识库文档解析 — 将上传文件转为可索引文本。"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from myfitness.rag.knowledge_service import MAX_CONTENT_LEN, MAX_TITLE_LEN

MAX_FILE_BYTES = 8 * 1024 * 1024
SUPPORTED_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".pdf",
    ".doc",
    ".docx",
    ".html",
    ".htm",
    ".rtf",
}

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_MAGIC = b"PK"
_PDF_MAGIC = b"%PDF"
_RTF_MAGIC = b"{\\rtf"


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    content: str
    filename: str
    format: str
    truncated: bool
    char_count: int


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    """按扩展名与文件头识别类型，提取纯文本。"""
    safe_name = Path(str(filename).replace("\\", "/")).name or "untitled"
    if not data:
        raise DocumentParseError("文件为空")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentParseError(f"文件不能超过 {MAX_FILE_BYTES // (1024 * 1024)}MB")

    fmt = detect_format(safe_name, data)
    if fmt == "pdf":
        text = _parse_pdf(data)
    elif fmt == "docx":
        text = _parse_docx(data)
    elif fmt == "doc":
        text = _parse_doc(data)
    elif fmt == "html":
        text = _parse_html(data)
    elif fmt == "rtf":
        text = _parse_rtf(data)
    else:
        text = _decode_text(data)

    content, truncated = _normalize_content(text)
    if not content:
        raise DocumentParseError("未能从文件中提取到文本，请检查文件是否损坏或为扫描件")
    return ParsedDocument(
        title=_title_from_filename(safe_name),
        content=content,
        filename=safe_name,
        format=fmt,
        truncated=truncated,
        char_count=len(content),
    )


def detect_format(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".xlsx", ".xls", ".pptx", ".ppt", ".zip", ".png", ".jpg", ".jpeg", ".gif"}:
        raise DocumentParseError(
            f"不支持的文件类型：{ext}。请上传 md / pdf / doc / docx / txt / html"
        )
    head = data[:16]
    stripped = data.lstrip()[:24].lower()
    if head.startswith(_PDF_MAGIC):
        return "pdf"
    if head.startswith(_OLE_MAGIC):
        return "doc"
    if head.startswith(_ZIP_MAGIC):
        return "docx"
    if stripped.startswith(_RTF_MAGIC):
        return "rtf"
    if stripped.startswith((b"<html", b"<!doctype")):
        return "html"
    mapping = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "doc",
        ".md": "md",
        ".markdown": "md",
        ".txt": "txt",
        ".html": "html",
        ".htm": "html",
        ".rtf": "rtf",
    }
    if ext in mapping:
        return mapping[ext]
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(
            f"不支持的文件类型：{ext}。请上传 md / pdf / doc / docx / txt / html"
        )
    return "txt"


def _title_from_filename(filename: str) -> str:
    name = Path(filename).stem.strip() or "未命名文档"
    return name[:MAX_TITLE_LEN]


def _normalize_content(text: str) -> tuple[str, bool]:
    cleaned = text.replace("\x00", "")
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    truncated = len(cleaned) > MAX_CONTENT_LEN
    if truncated:
        cleaned = cleaned[:MAX_CONTENT_LEN].rstrip()
    return cleaned, truncated


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _parse_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentParseError("解析 PDF 需要安装 pypdf") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise DocumentParseError("无法读取 PDF 文件") from exc
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - 个别 PDF 页提取失败时跳过
            extracted = ""
        text = extracted.strip()
        if text:
            pages.append(f"## 第 {index} 页\n\n{text}")
    return "\n\n".join(pages)


def _parse_docx(data: bytes) -> str:
    try:
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise DocumentParseError("解析 Word 需要安装 python-docx") from exc
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentParseError("无法读取 Word 文档，请另存为 .docx 后再试") from exc

    parts: list[str] = []
    for block in document.element.body:
        if isinstance(block, CT_P):
            paragraph = Paragraph(block, document)
            line = _docx_paragraph_text(paragraph)
            if line:
                parts.append(line)
        elif isinstance(block, CT_Tbl):
            table_text = _docx_table_text(Table(block, document))
            if table_text:
                parts.append(table_text)
    return "\n\n".join(parts)


def _docx_paragraph_text(paragraph: object) -> str:
    text = str(getattr(paragraph, "text", "") or "").strip()
    if not text:
        return ""
    style = getattr(paragraph, "style", None)
    style_name = str(getattr(style, "name", "") or "")
    if style_name in {"Title", "标题"}:
        return f"# {text}"
    match = re.match(r"(?:Heading|标题)\s*(\d+)", style_name, re.IGNORECASE)
    if match:
        level = max(1, min(int(match.group(1)), 6))
        return f"{'#' * level} {text}"
    return text


def _docx_table_text(table: object) -> str:
    rows: list[list[str]] = []
    for row in getattr(table, "rows", []):
        cells = [" ".join(str(cell.text).split()) for cell in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(normalized[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _parse_doc(data: bytes) -> str:
    """尽力从旧版 .doc（OLE）提取文本；失败时提示另存为 docx。"""
    text = _extract_utf16le_runs(data)
    if len(text) < 20:
        ascii_text = _extract_ascii_runs(data)
        text = text + "\n" + ascii_text if ascii_text else text
    text = _normalize_content(text)[0]
    if len(text) < 12:
        raise DocumentParseError("无法解析该 .doc 文件，请另存为 .docx 或 PDF 后再上传")
    return text


def _extract_utf16le_runs(data: bytes) -> str:
    even = _utf16_runs_aligned(data, 0)
    odd = _utf16_runs_aligned(data, 1)
    return even if len(even) >= len(odd) else odd


def _utf16_runs_aligned(data: bytes, offset: int) -> str:
    parts: list[str] = []
    buffer: list[str] = []
    i = offset
    end = len(data) - 1
    while i < end:
        code = data[i] | (data[i + 1] << 8)
        char = chr(code) if 0 < code < 0xFFFE else ""
        if _is_extractable_char(char):
            buffer.append("\n" if char in "\r\n" else char)
        else:
            if len(buffer) >= 4:
                parts.append("".join(buffer))
            buffer = []
        i += 2
    if len(buffer) >= 4:
        parts.append("".join(buffer))
    return "\n".join(part.strip() for part in parts if part.strip())


def _extract_ascii_runs(data: bytes) -> str:
    parts: list[str] = []
    buffer: list[str] = []
    for byte in data:
        if 32 <= byte < 127 or byte in {9, 10, 13}:
            buffer.append("\n" if byte in {10, 13} else chr(byte))
            continue
        if len(buffer) >= 8:
            parts.append("".join(buffer))
        buffer = []
    if len(buffer) >= 8:
        parts.append("".join(buffer))
    return "\n".join(part.strip() for part in parts if part.strip())


def _is_extractable_char(char: str) -> bool:
    if not char:
        return False
    if char in "\n\r\t":
        return True
    code = ord(char)
    if 32 <= code < 127:
        return True
    if code < 128:
        return False
    category = unicodedata.category(char)
    return category[0] in {"L", "N", "P", "S", "Z"}


def _parse_html(data: bytes) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(_decode_text(data))
    parser.close()
    return parser.text()


def _parse_rtf(data: bytes) -> str:
    raw = _decode_text(data)
    raw = re.sub(r"\\'[0-9a-fA-F]{2}", lambda m: bytes.fromhex(m.group(0)[2:]).decode("latin-1"), raw)
    raw = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) & 0xFFFF), raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    raw = raw.replace("{", "").replace("}", "")
    raw = raw.replace("\\~", " ").replace("\\-", "-").replace("\\_", "_")
    return raw


class _HTMLTextExtractor(HTMLParser):
    _SKIP: ClassVar[set[str]] = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"}:
            self._parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._skip_depth == 0:
            level = int(tag[1])
            self._parts.append("#" * level + " ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)
