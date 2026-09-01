"""文档读写 Tool 测试。"""

from pathlib import Path

import pytest

from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.document_tools import (
    DocumentToolError,
    extract_document_path,
    infer_document_format,
    is_document_generation_request,
    list_document_files,
    needs_document_write,
    read_document,
    read_document_file,
    sanitize_filename,
    write_document,
    write_document_file,
)
from myfitness.config import get_settings


@pytest.fixture
def document_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data))
    get_settings.cache_clear()
    yield Path(get_settings().document_output_dir)
    get_settings.cache_clear()


def test_sanitize_filename():
    assert sanitize_filename("减肥总结.md") == "减肥总结.md"
    assert sanitize_filename(r"bad\name.pdf").endswith(".pdf")


def test_needs_document_write():
    assert needs_document_write("请把分析保存为 pdf 文档")
    assert needs_document_write("给我生成饮食规划的文档")
    assert not needs_document_write("今天体重多少")
    assert not needs_document_write("生成昨天的日报")


def test_is_document_generation_request():
    assert is_document_generation_request("根据会话写一份训练计划文档")
    assert not is_document_generation_request("生成8月20日到8月25日的报告")


def test_infer_document_format():
    assert infer_document_format("导出成 word 文档") == "docx"
    assert infer_document_format("保存为 pdf") == "pdf"
    assert infer_document_format("产出保存为pd文档") == "pdf"
    assert infer_document_format("写成 markdown") == "md"


def test_infer_document_formats_multi():
    from myfitness.agents.tools.document_tools import infer_document_formats

    message = (
        "根据最近的训练记录，给我一些训练建议，产出保存为pdf、docx、md文档各一份"
    )
    assert infer_document_formats(message) == ["pdf", "docx", "md"]


def test_extract_document_path():
    assert extract_document_path("请读取 documents/训练计划.docx") == "documents/训练计划.docx"


def test_write_and_read_markdown(document_dir):
    result = write_document_file("测试文档.md", "# 标题\n\n正文内容", format="md")
    assert result["format"] == "md"
    path = Path(result["path"])
    assert path.is_file()
    assert "正文内容" in path.read_text(encoding="utf-8")

    loaded = read_document_file(result["path"])
    assert loaded["format"] == "md"
    assert "正文内容" in loaded["content"]


def test_write_docx(document_dir):
    pytest.importorskip("docx")
    result = write_document_file(
        "计划.docx",
        "# 训练计划\n\n今天练胸。",
        format="docx",
    )
    assert Path(result["path"]).is_file()
    loaded = read_document_file(result["path"])
    assert "训练计划" in loaded["content"]


def test_write_docx_with_table(document_dir):
    pytest.importorskip("docx")
    content = (
        "# 饮食规划\n\n"
        "| 餐次 | 食物 |\n"
        "| --- | --- |\n"
        "| 早餐 | 鸡蛋 |\n\n"
        "备注：适量饮水。"
    )
    result = write_document_file("饮食规划.docx", content, format="docx")
    assert Path(result["path"]).is_file()
    loaded = read_document_file(result["path"])
    assert "饮食规划" in loaded["content"]
    assert "早餐" in loaded["content"]


def test_write_pdf_with_table(document_dir):
    pytest.importorskip("fpdf")
    content = (
        "# 饮食规划\n\n"
        "| 餐次 | 食物 |\n"
        "| --- | --- |\n"
        "| 早餐 | 鸡蛋 |\n"
    )
    result = write_document_file("饮食规划.pdf", content, format="pdf")
    assert Path(result["path"]).is_file()
    assert Path(result["path"]).stat().st_size > 0


def test_list_documents(document_dir):
    write_document_file("a.md", "hello", format="md")
    write_document_file("b.md", "world", format="md")
    listing = list_document_files()
    assert listing["count"] == 2
    names = {item["name"] for item in listing["documents"]}
    assert names == {"a.md", "b.md"}


def test_write_doc_not_supported(document_dir):
    with pytest.raises(DocumentToolError, match="不支持写入旧版"):
        write_document_file("legacy.doc", "text", format="doc")


def test_read_outside_data_dir(document_dir, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(DocumentToolError, match="数据目录"):
        read_document_file(str(outside))


@pytest.fixture
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from myfitness.db.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_tool_invoke_write_and_read(db_session, document_dir):
    written = invoke_tool(
        write_document,
        db_session,
        1,
        filename="工具写入.md",
        content="# 工具\n\n由 tool 写入",
        format="md",
    )
    assert written["filename"] == "工具写入.md"

    loaded = invoke_tool(read_document, db_session, 1, path=written["path"])
    assert "由 tool 写入" in loaded["content"]
