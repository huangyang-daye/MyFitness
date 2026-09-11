"""知识库文档解析测试。"""

from io import BytesIO
from types import SimpleNamespace

import pytest

from myfitness.api.web import parse_multipart_file
from myfitness.rag.document_parser import DocumentParseError, _reflow_pdf_text, parse_document
from myfitness.rag.knowledge_service import MAX_CONTENT_LEN
from myfitness.rag import mineru_pdf


def test_parse_markdown():
    parsed = parse_document("减脂原则.md", "# 原则\n\n蛋白质每公斤 1.6g".encode())
    assert parsed.format == "md"
    assert parsed.title == "减脂原则"
    assert "1.6g" in parsed.content
    assert not parsed.truncated


def test_parse_gb18030_txt():
    parsed = parse_document("note.txt", "少油少盐多蔬菜".encode("gb18030"))
    assert parsed.format == "txt"
    assert "少油少盐" in parsed.content


def test_parse_html_strips_script():
    html = "<html><body><h1>饮食</h1><p>高蛋白</p><script>alert(1)</script></body></html>".encode()
    parsed = parse_document("tips.html", html)
    assert parsed.format == "html"
    assert "高蛋白" in parsed.content
    assert "alert" not in parsed.content
    assert "# 饮食" in parsed.content


def test_parse_rtf():
    parsed = parse_document("note.rtf", rb"{\rtf1\ansi Protein target 1.6g per kg}")
    assert parsed.format == "rtf"
    assert "Protein target" in parsed.content


def test_parse_docx():
    pytest.importorskip("docx")
    from docx import Document

    document = Document()
    document.add_heading("训练原则", level=1)
    document.add_paragraph("卧推时肩胛骨收紧。")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "动作"
    table.cell(0, 1).text = "组数"
    table.cell(1, 0).text = "深蹲"
    table.cell(1, 1).text = "4"
    buffer = BytesIO()
    document.save(buffer)
    parsed = parse_document("训练计划.docx", buffer.getvalue())
    assert parsed.format == "docx"
    assert parsed.title == "训练计划"
    assert "肩胛骨" in parsed.content
    assert "深蹲" in parsed.content


def test_parse_pdf_uses_mineru(monkeypatch):
    monkeypatch.setattr(
        "myfitness.rag.mineru_pdf.parse_pdf_with_mineru",
        lambda data, filename="document.pdf", settings=None: "## 第 1 页\n\n蛋白质每公斤体重 1.6g",
    )
    monkeypatch.setattr(
        "myfitness.config.get_settings",
        lambda: SimpleNamespace(
            pdf_parser="mineru",
            mineru_fallback_pypdf=False,
        ),
    )
    parsed = parse_document("plan.pdf", b"%PDF-1.4 fake-body")
    assert parsed.format == "pdf"
    assert "1.6g" in parsed.content
    assert "第 1 页" in parsed.content


def test_parse_pdf_falls_back_to_pypdf(monkeypatch):
    class FakePage:
        def extract_text(self) -> str:
            return (
                "蛋白质摄入建议为每公斤\n"
                "体重 1.6 到 2.2 克，并在训练后\n"
                "及时补充。\n"
                "力量训练每周至少三次。"
            )

    class FakeReader:
        def __init__(self, _stream) -> None:
            self.pages = [FakePage()]

    def boom(*_args, **_kwargs):
        raise mineru_pdf.MinerUParseError("mineru unavailable")

    monkeypatch.setattr("myfitness.rag.mineru_pdf.parse_pdf_with_mineru", boom)
    monkeypatch.setattr("pypdf.PdfReader", FakeReader)
    monkeypatch.setattr(
        "myfitness.config.get_settings",
        lambda: SimpleNamespace(
            pdf_parser="mineru",
            mineru_fallback_pypdf=True,
        ),
    )
    parsed = parse_document("plan.pdf", b"%PDF-1.4 fake-body")
    assert parsed.format == "pdf"
    assert "第 1 页" in parsed.content
    assert "每公斤体重 1.6 到 2.2 克" in parsed.content
    assert "训练后及时补充。" in parsed.content
    # 句号后保留分段，不把下一段硬拼上
    assert "补充。\n力量训练每周至少三次。" in parsed.content


def test_reflow_pdf_text_joins_soft_wraps_keeps_structure():
    raw = (
        "## 第 1 页\n\n"
        "减脂期应保证充足的\n"
        "蛋白质摄入，避免肌肉\n"
        "流失。\n\n"
        "- 早餐鸡蛋\n"
        "- 午餐鸡胸\n\n"
        "Weekly protein target is 1.6\n"
        "to 2.2 grams per kilogram."
    )
    text = _reflow_pdf_text(raw)
    assert "充足的蛋白质摄入，避免肌肉流失。" in text
    assert "- 早餐鸡蛋\n- 午餐鸡胸" in text
    assert "1.6 to 2.2 grams" in text
    assert "## 第 1 页" in text


def test_collapse_doubled_cjk_from_fake_bold():
    from myfitness.rag.document_parser import _collapse_doubled_cjk, _normalize_content

    # 假粗体叠字：至少 3 组才折叠
    assert _collapse_doubled_cjk("疼疼痛痛是是常见") == "疼痛是常见"
    # 保留正常叠词
    assert _collapse_doubled_cjk("明明可以慢慢来") == "明明可以慢慢来"
    assert _collapse_doubled_cjk("浩浩荡荡") == "浩浩荡荡"
    assert _collapse_doubled_cjk("小小腿前侧", aggressive=True) == "小腿前侧"
    # 含大量部首的笔记：去部首 + 积极折叠
    raw = ("⻣" * 60) + "足足踝关节小小腿胫⻣骨舟舟⻣骨"
    cleaned, _ = _normalize_content(raw)
    assert "足足" not in cleaned
    assert "小小" not in cleaned
    assert "⻣" not in cleaned
    assert "足踝关节" in cleaned
    assert "小腿" in cleaned
    assert "胫骨" in cleaned
    assert "舟骨" in cleaned
    # 普通文本仍保留「明明」
    plain, _ = _normalize_content("明明可以慢慢来")
    assert "明明可以慢慢来" in plain


def test_soft_wrap_overlap_strips_repeated_tail():
    from myfitness.rag.document_parser import _join_pdf_soft_wrap

    assert _join_pdf_soft_wrap("需要提升", "提升体重") == "需要提升体重"


def test_parse_pdf_mineru_required_raises(monkeypatch):
    def boom(*_args, **_kwargs):
        raise mineru_pdf.MinerUParseError("未找到 MinerU")

    monkeypatch.setattr("myfitness.rag.mineru_pdf.parse_pdf_with_mineru", boom)
    monkeypatch.setattr(
        "myfitness.config.get_settings",
        lambda: SimpleNamespace(
            pdf_parser="mineru",
            mineru_fallback_pypdf=False,
        ),
    )
    with pytest.raises(DocumentParseError, match="MinerU"):
        parse_document("plan.pdf", b"%PDF-1.4 fake-body")


def test_collect_markdown_prefers_main_doc(tmp_path):
    main = tmp_path / "doc" / "auto"
    main.mkdir(parents=True)
    (main / "doc.md").write_text("# 主文\n\n蛋白质", encoding="utf-8")
    (main / "doc_content_list.md").write_text("[]", encoding="utf-8")
    text = mineru_pdf._collect_markdown(tmp_path)
    assert "蛋白质" in text


def test_parse_legacy_doc_utf16():
    body = "减脂期蛋白质要达到每公斤 1.6 克以上。".encode("utf-16le")
    data = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 80 + body + b"\x00\x00"
    parsed = parse_document("原则.doc", data)
    assert parsed.format == "doc"
    assert "蛋白质" in parsed.content


def test_parse_rejects_empty():
    with pytest.raises(DocumentParseError, match="空"):
        parse_document("empty.md", b"")


def test_parse_rejects_unsupported():
    with pytest.raises(DocumentParseError, match="不支持"):
        parse_document("photo.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)


def test_parse_truncates_long_text():
    payload = ("原则\n" + ("蛋白质 " * 20_000)).encode("utf-8")
    parsed = parse_document("long.md", payload)
    assert parsed.truncated
    assert len(parsed.content) <= MAX_CONTENT_LEN


def test_parse_multipart_file_extracts_payload():
    boundary = "----TestBoundary"
    filename = "原则.md"
    content = "# hello\n"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/markdown\r\n"
        "\r\n"
        f"{content}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    name, data = parse_multipart_file(body, f"multipart/form-data; boundary={boundary}")
    assert name == filename
    assert data == content.encode("utf-8")


def test_parse_multipart_requires_file_field():
    boundary = "----TestBoundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="title"\r\n'
        "\r\n"
        "only-title\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    with pytest.raises(ValueError, match="未找到"):
        parse_multipart_file(body, f"multipart/form-data; boundary={boundary}")
