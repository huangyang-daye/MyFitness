"""知识库 / 报告切块：按句子边界，不从句中切断。"""

from types import SimpleNamespace

from myfitness.rag.chunking import _split_report_sections, entry_to_chunks


def test_does_not_split_inside_chinese_sentence():
    prefix = "训练注意点。" * 40  # 240 chars
    sentence = "如何判断自己是否肥胖？滚查自己的肚子，若肚子还在可控范围内，可继续增肌。若明显肚子变胖，增肌应停止。"
    filler = "补充说明。" * 80
    text = prefix + sentence + filler
    chunks = _split_report_sections(text, max_chars=400, min_chars=80)
    blob = "".join(chunks)
    assert "若明显肚子变胖" in blob
    assert not any(chunk.endswith("若明显肚") for chunk in chunks)
    assert not any(chunk.startswith("子变胖") for chunk in chunks)


def test_packs_small_heading_sections():
    text = (
        "# 用户画像\n"
        "## 目标\n- 减脂\n- 减重\n"
        "## 饮食\n- 只在学校食堂吃饭\n"
        "## 训练\n- 练背\n- 练腿\n"
    )
    chunks = _split_report_sections(text, max_chars=1800, min_chars=200)
    assert len(chunks) == 1
    assert "## 目标" in chunks[0]
    assert "## 训练" in chunks[0]


def test_keeps_heading_when_section_overflows():
    body = "这是完整的一句说明。" * 80
    text = f"## 训练注意点\n\n{body}"
    chunks = _split_report_sections(text, max_chars=400, min_chars=80)
    assert len(chunks) > 1
    assert all(chunk.startswith("## 训练注意点") for chunk in chunks)
    for chunk in chunks:
        assert not chunk.endswith("这是完整的一句说")
        assert not chunk.startswith("明。")


def test_splits_on_sentence_not_fixed_width():
    text = ("第一句到此结束。" * 30) + ("第二句也到此结束。" * 30)
    chunks = _split_report_sections(text, max_chars=200, min_chars=40)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.endswith("。")


def test_does_not_split_on_pdf_page_headings_inside_sentence():
    text = (
        "所以在高碳饮食时，蛋白摄入要\n"
        "## 第 3 页\n\n"
        "适量。通过公式计算没有必要。每个人情况不同。"
    )
    chunks = _split_report_sections(text, max_chars=1800, min_chars=50)
    assert len(chunks) == 1
    assert "蛋白摄入要" in chunks[0]
    assert "适量。" in chunks[0]


def test_stitches_sentence_broken_across_chunks():
    left = ("补充说明。" * 20) + "蛋白摄入要"
    right = "适量。后面还有一句完整的话。" + ("继续说明。" * 20)
    text = left + "\n## 第 3 页\n\n" + right
    chunks = _split_report_sections(text, max_chars=120, min_chars=20)
    blob = "\n".join(chunks)
    assert "蛋白摄入要" in blob
    assert "适量。" in blob
    assert not any(chunk.endswith("蛋白摄入要") for chunk in chunks)


def test_empty_and_short_passthrough():
    assert _split_report_sections("") == []
    assert _split_report_sections("短文本") == ["短文本"]


def test_knowledge_entry_uses_sentence_chunks():
    content = "准则一：食物多样。\n\n" + ("核心推荐是每天均衡搭配。" * 50)
    entry = SimpleNamespace(
        id=9,
        title="膳食指南",
        content=content,
        kind="user",
        updated_at=None,
        created_at=None,
    )
    docs = entry_to_chunks(entry)
    assert docs
    assert docs[0].source_type == "knowledge"
    assert docs[0].content.startswith("膳食指南")
    joined = "".join(doc.content for doc in docs)
    assert "核心推荐是每天均衡搭配。" in joined
    for doc in docs:
        assert not doc.content.endswith("核心推荐是每天均衡搭")
        assert not doc.content.startswith("配。")
