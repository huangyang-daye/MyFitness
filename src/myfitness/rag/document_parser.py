"""知识库文档解析 — 将上传文件转为可索引文本。"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from myfitness.rag.knowledge_service import MAX_CONTENT_LEN, MAX_TITLE_LEN

logger = logging.getLogger(__name__)

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
        text = _parse_pdf(data, filename=safe_name)
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
    # 笔记类 PDF 常把康熙/部首字形嵌进正文（⻣+骨），先去掉再 NFKC
    radical_count = len(re.findall(r"[\u2e80-\u2fdf]", cleaned))
    cleaned = re.sub(r"[\u2e80-\u2fdf]", "", cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    # 假粗体叠字；部首很多时折叠所有连续相同汉字（足足→足）
    cleaned = _collapse_doubled_cjk(cleaned, aggressive=radical_count >= 50)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    truncated = len(cleaned) > MAX_CONTENT_LEN
    if truncated:
        cleaned = cleaned[:MAX_CONTENT_LEN].rstrip()
    return cleaned, truncated


def _collapse_doubled_cjk(text: str, *, aggressive: bool = False) -> str:
    """折叠假粗体造成的连续相同汉字。

    默认要求至少 3 组「字字」才折叠，保留「明明 / 慢慢 / 浩浩荡荡」；
    aggressive 时折叠所有连续相同汉字（扫描笔记假粗体）。
    """
    if aggressive:
        return re.sub(r"([\u4e00-\u9fff])\1+", r"\1", text)

    pattern = re.compile(r"(?:([\u4e00-\u9fff])\1){3,}")

    def _repl(match: re.Match[str]) -> str:
        return match.group(0)[::2]

    return pattern.sub(_repl, text)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def parse_pdf_with_coordinates(data: bytes) -> str:
    """使用 PyMuPDF 的坐标信息进行段落合并，返回纯文本。"""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise DocumentParseError("坐标解析 PDF 需要安装 pymupdf") from exc
    doc = fitz.open(stream=data, filetype="pdf")
    full_text = []
    for page_num, page in enumerate(doc, start=1):
        # 获取页面所有文本块
        blocks = page.get_text("dict")["blocks"]
        page_lines = []

        # 1. 从 spans 中提取行（按 y0 坐标聚类）
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                # 同一行内所有 span 按 x0 排序后拼接
                spans.sort(key=lambda s: s["origin"][0])
                # 注意：span 自带文本，可能包含空格，需要合理拼接
                line_text = ""
                for index, span in enumerate(spans):
                    # 若前一个 span 末尾与当前 span 开头之间距离过大，补空格
                    if (
                        line_text
                        and index > 0
                        and span["bbox"][0] - spans[index - 1]["bbox"][2] > 2
                    ):
                        line_text += " "
                    line_text += span["text"]
                y0 = line["bbox"][1]  # 行的纵坐标
                page_lines.append((y0, line_text))

        # 2. 按 y0 排序（从上到下）
        page_lines.sort(key=lambda x: x[0])

        # 3. 将行合并为段落（依据垂直间距）
        paragraphs = []
        if page_lines:
            current_para = [page_lines[0][1]]
            prev_y = page_lines[0][0]
            for y, text in page_lines[1:]:
                gap = y - prev_y
                # 若行间距超过阈值（经验值，单位点为磅），则视为新段落
                if gap > 8:  # 8 points，可根据实际情况调整
                    paragraphs.append(" ".join(current_para))
                    current_para = [text]
                else:
                    # 同一段落：检查是否需要空格连接（按你的 _join_pdf_soft_wrap 逻辑）
                    last = current_para[-1]
                    if last and text:
                        # 可复用你的 _join_pdf_soft_wrap 函数
                        joined = _join_pdf_soft_wrap(last, text)
                        current_para[-1] = joined
                    else:
                        current_para.append(text)
                prev_y = y
            if current_para:
                paragraphs.append(" ".join(current_para))

        full_text.append("\n\n".join(paragraphs))
    doc.close()
    return "\n\n".join(full_text)

def _parse_pdf(data: bytes, *, filename: str = "document.pdf") -> str:
    """PDF 默认走 MinerU；失败时可按配置回退 pypdf。"""
    from myfitness.config import get_settings

    settings = get_settings()
    parser = (settings.pdf_parser or "mineru").strip().lower()
    if parser not in {"mineru", "pypdf"}:
        raise DocumentParseError("PDF_PARSER 须为 mineru 或 pypdf")

    if parser == "mineru":
        try:
            from myfitness.rag.mineru_pdf import MinerUParseError, parse_pdf_with_mineru

            text = parse_pdf_with_mineru(data, filename=filename, settings=settings)
            if text.strip():
                # MinerU 已输出段落化 Markdown，不再做版面折行重排，避免叠词
                return text
            raise MinerUParseError("MinerU 返回空文本")
        except Exception as exc:  # noqa: BLE001
            if not settings.mineru_fallback_pypdf:
                message = str(exc).strip() or "MinerU 解析失败"
                raise DocumentParseError(message) from exc
            logger.warning("MinerU 解析失败，回退 pypdf：%s", exc)

    try:
        return _reflow_pdf_text(parse_pdf_with_coordinates(data))
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyMuPDF 坐标解析失败，回退 pypdf：%s", exc)
        return _reflow_pdf_text(_parse_pdf_pypdf(data))

def _parse_pdf_pypdf(data: bytes) -> str:
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


_PDF_SENTENCE_END = re.compile(r"[。！？；….!?;:：」』”’）)\]】]$")
_PDF_STRUCTURAL_LINE = re.compile(
    r"^(?:"
    r"#{1,6}\s+"  # markdown 标题
    r"|第\s*\d+\s*页\s*$"
    r"|[-*●•]\s+"  # 无序列表
    r"|\d+[.)、．]\s+"  # 有序列表
    r"|[（(]?[一二三四五六七八九十百千]+[）)]、\s*"
    r"|第[一二三四五六七八九十百千0-9]+[章节篇部分条款]\s*"
    r"|!\[|"  # 图片
    r"\|.+\|"  # 表格行
    r")"
)
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
_LATIN_WORD = re.compile(r"[A-Za-z0-9]")


def _reflow_pdf_text(text: str) -> str:
    """把 PDF 版面折行拼回段落，保留空行分段与标题/列表结构。"""
    if not text or "\n" not in text:
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", normalized)
    reflowed = [_reflow_pdf_block(block) for block in blocks]
    return "\n\n".join(part for part in reflowed if part.strip())


def _reflow_pdf_block(block: str) -> str:
    lines = [line.rstrip() for line in block.split("\n")]
    lines = [line for line in lines if line.strip()]
    if len(lines) <= 1:
        return "\n".join(lines)

    merged: list[str] = [lines[0]]
    for nxt in lines[1:]:
        prev = merged[-1]
        if _should_keep_pdf_linebreak(prev, nxt):
            merged.append(nxt)
        else:
            merged[-1] = _join_pdf_soft_wrap(prev, nxt)
    return "\n".join(merged)


def _should_keep_pdf_linebreak(prev: str, nxt: str) -> bool:
    prev_s = prev.strip()
    nxt_s = nxt.strip()
    if not prev_s or not nxt_s:
        return True
    if _PDF_STRUCTURAL_LINE.match(prev_s) or _PDF_STRUCTURAL_LINE.match(nxt_s):
        return True
    # 上一行已收束，视为段落边界，保留换行
    if _PDF_SENTENCE_END.search(prev_s):
        return True
    return False


def _join_pdf_soft_wrap(prev: str, nxt: str) -> str:
    left = prev.rstrip()
    right = nxt.lstrip()
    # Unicode 软连字符去掉；普通 "-" 保留（anti-inflammatory）
    if left.endswith("­"):
        left = left[:-1]
    # 折行重叠：上一行末尾与下一行开头重复（需要提升 + 提升体重 → 需要提升体重）
    overlap = _soft_wrap_overlap(left, right)
    if overlap:
        right = right[overlap:]
        if not right:
            return left
    left_tail = left[-1:] if left else ""
    right_head = right[:1] if right else ""
    # 两侧都是拉丁字符时补空格；中文或标点相邻直接拼接
    need_space = bool(
        _LATIN_WORD.match(left_tail)
        and _LATIN_WORD.match(right_head)
        and not left.endswith(("(", "[", "{", "/", "\\", "-"))
    )
    if need_space:
        return f"{left} {right}"
    # 避免中文折行中间插入空格；CJK 互拼不加空格
    if _CJK_CHAR.match(left_tail) or _CJK_CHAR.match(right_head):
        return left + right
    if left_tail.isspace() or right_head.isspace():
        return left + right
    if left_tail in '-([{（【「『“"\'' or right_head in ')]}）】」』”"\'，,、。.!?;:：；':
        return left + right
    if _LATIN_WORD.match(left_tail) or _LATIN_WORD.match(right_head):
        return f"{left} {right}"
    return left + right


def _soft_wrap_overlap(left: str, right: str) -> int:
    """下一行若重复上一行末尾片段，返回应去掉的前缀长度。"""
    if not left or not right:
        return 0
    max_k = min(len(left), len(right), 16)
    for k in range(max_k, 0, -1):
        if left.endswith(right[:k]):
            # 单字重叠仅在两侧都是 CJK 时采纳，降低误伤
            if k == 1 and not (
                _CJK_CHAR.match(left[-1]) and _CJK_CHAR.match(right[0])
            ):
                continue
            return k
    return 0


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
