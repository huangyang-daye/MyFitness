"""MinerU PDF 适配层测试。"""

from pathlib import Path

import pytest

from myfitness.rag.mineru_pdf import MinerUParseError, _collect_markdown, parse_pdf_with_mineru


def test_parse_pdf_with_mineru_requires_backend(monkeypatch):
    class Cfg:
        mineru_api_url = ""
        mineru_api_token = ""
        mineru_backend = "pipeline"
        mineru_parse_method = "auto"
        mineru_lang = "ch"
        mineru_formula_enable = True
        mineru_table_enable = True
        mineru_timeout = 60

    monkeypatch.setattr("myfitness.rag.mineru_pdf._mineru_package_available", lambda: False)
    monkeypatch.setattr("myfitness.rag.mineru_pdf.shutil.which", lambda _name: None)
    with pytest.raises(MinerUParseError, match="未找到 MinerU"):
        parse_pdf_with_mineru(b"%PDF-1.4", filename="a.pdf", settings=Cfg())


def test_parse_via_do_parse_reads_markdown(monkeypatch, tmp_path):
    class Cfg:
        mineru_api_url = ""
        mineru_api_token = ""
        mineru_backend = "pipeline"
        mineru_parse_method = "auto"
        mineru_lang = "ch"
        mineru_formula_enable = True
        mineru_table_enable = True
        mineru_timeout = 60

    def fake_do_parse(output_dir, names, _bytes_list, _langs, **_kwargs):
        target = Path(output_dir) / names[0] / "auto"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{names[0]}.md").write_text("# 训练\n\n深蹲 4 组", encoding="utf-8")

    import types
    import sys

    fake_common = types.ModuleType("mineru.cli.common")
    fake_common.do_parse = fake_do_parse
    fake_cli = types.ModuleType("mineru.cli")
    fake_cli.common = fake_common
    fake_root = types.ModuleType("mineru")
    fake_root.cli = fake_cli
    monkeypatch.setitem(sys.modules, "mineru", fake_root)
    monkeypatch.setitem(sys.modules, "mineru.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "mineru.cli.common", fake_common)
    monkeypatch.setattr("myfitness.rag.mineru_pdf._mineru_package_available", lambda: True)

    text = parse_pdf_with_mineru(b"%PDF-1.4 body", filename="plan.pdf", settings=Cfg())
    assert "深蹲" in text


def test_collect_markdown_skips_readme(tmp_path):
    (tmp_path / "README.md").write_text("ignore", encoding="utf-8")
    doc_dir = tmp_path / "doc" / "auto"
    doc_dir.mkdir(parents=True)
    (doc_dir / "doc.md").write_text("正文内容", encoding="utf-8")
    assert _collect_markdown(tmp_path) == "正文内容"
