"""结构化文档块测试。"""

import json
from pathlib import Path

import pytest

from myfitness.agents.document_blocks import (
    blocks_to_markdown,
    parse_document_blocks,
    write_docx_blocks,
)


def test_parse_docx_json_blocks():
    payload = json.dumps(
        {
            "blocks": [
                {"type": "title", "text": "训练建议"},
                {"type": "paragraph", "text": "保持力量训练频率。"},
                {"type": "bullet_list", "items": ["每周 3 练", "注意恢复"]},
            ]
        },
        ensure_ascii=False,
    )
    blocks = parse_document_blocks(payload, prefer_json=True)
    assert len(blocks) == 3
    assert blocks[0]["type"] == "title"


def test_write_docx_from_json_blocks(tmp_path):
    pytest.importorskip("docx")
    blocks = [
        {"type": "title", "text": "饮食规划"},
        {"type": "heading", "level": 2, "text": "目标"},
        {"type": "paragraph", "text": "减脂同时保肌。"},
        {"type": "bullet_list", "items": ["高蛋白", "适量碳水"]},
    ]
    target = tmp_path / "plan.docx"
    write_docx_blocks(target, blocks)
    assert target.is_file()
    assert target.stat().st_size > 0


def test_strip_inline_markdown():
    from myfitness.agents.document_blocks import strip_inline_markdown

    assert strip_inline_markdown("**重点**与`代码`") == "重点与代码"


def test_write_pdf_blocks_complex(tmp_path):
    pytest.importorskip("fpdf")
    from myfitness.agents.document_blocks import write_pdf_blocks

    blocks = [
        {"type": "title", "text": "减重期训练建议"},
        {"type": "heading", "level": 2, "text": "核心原则"},
        {"type": "paragraph", "text": "在**减脂**期间保持力量训练频率。"},
        {"type": "bullet_list", "items": ["每周 3-4 练", "控制有氧时长"]},
        {"type": "table", "rows": [["项目", "建议"], ["力量训练", "每周 3 次"], ["有氧", "每周 2 次"]]},
    ]
    target = tmp_path / "plan.pdf"
    write_pdf_blocks(target, blocks)
    assert target.stat().st_size > 500


def test_blocks_to_markdown_roundtrip():
    md = "# 标题\n\n正文\n\n- 要点"
    blocks = parse_document_blocks(md, prefer_json=False)
    rendered = blocks_to_markdown(blocks)
    assert "# 标题" in rendered
    assert "- 要点" in rendered
