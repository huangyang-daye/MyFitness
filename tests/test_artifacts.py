import pytest

from myfitness.services.artifacts import ArtifactError, read_artifact, resolve_artifact


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "reports" / "charts").mkdir(parents=True)
    (tmp_path / "reports" / "2026-08-29.md").write_text("# 日报\n\n内容\n", encoding="utf-8")
    (tmp_path / "reports" / "charts" / "weight.md").write_text("```mermaid\nline\n```", encoding="utf-8")
    (tmp_path / "secret.env").write_text("KEY=1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(data_dir, monkeypatch):
    from myfitness.config import Settings

    value = Settings(data_dir=str(data_dir))
    monkeypatch.setattr("myfitness.services.artifacts.get_settings", lambda: value)
    return value


def test_reads_report_artifact(settings, data_dir):
    payload = read_artifact(str(data_dir / "reports" / "2026-08-29.md"))
    assert payload["name"] == "2026-08-29.md"
    assert payload["kind"] == "report"
    assert "# 日报" in payload["content"]
    assert payload["truncated"] is False


def test_recognizes_chart_directory(settings, data_dir):
    payload = read_artifact(str(data_dir / "reports" / "charts" / "weight.md"))
    assert payload["kind"] == "chart"


def test_blocks_paths_outside_data_dir(settings, data_dir, tmp_path):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactError, match="数据目录"):
        read_artifact(str(outside))
    with pytest.raises(ArtifactError, match="数据目录"):
        read_artifact(str(data_dir.parent))


def test_rejects_missing_and_empty_paths(settings, data_dir):
    with pytest.raises(ArtifactError, match="缺少产物路径"):
        read_artifact("")
    with pytest.raises(ArtifactError, match="不是文件"):
        read_artifact(str(data_dir / "reports"))
    with pytest.raises(ArtifactError, match="不存在"):
        read_artifact(str(data_dir / "reports" / "nope.md"))


def test_truncates_very_long_content(settings, data_dir):
    target = data_dir / "reports" / "huge.md"
    target.write_text("一" * 500_000, encoding="utf-8")
    payload = read_artifact(str(target))
    assert payload["truncated"] is True
    assert len(payload["content"]) < 500_000


def test_resolve_returns_absolute_path(settings, data_dir):
    resolved = resolve_artifact(str(data_dir / "reports" / "2026-08-29.md"))
    assert resolved.is_absolute()
    assert resolved.name == "2026-08-29.md"


def test_pdf_artifact_preview_metadata(settings, data_dir):
    pdf = data_dir / "documents" / "plan.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    payload = read_artifact(str(pdf))
    assert payload["preview_type"] == "pdf"
    assert payload["kind"] == "document"
    assert payload["content"] == ""


def test_docx_artifact_preview_html(settings, data_dir):
    pytest.importorskip("docx")
    from myfitness.agents.document_blocks import write_docx_blocks

    docx = data_dir / "documents" / "plan.docx"
    docx.parent.mkdir(parents=True, exist_ok=True)
    write_docx_blocks(
        docx,
        [
            {"type": "title", "text": "训练计划"},
            {"type": "paragraph", "text": "每周训练 3 次。"},
        ],
    )
    payload = read_artifact(str(docx))
    assert payload["preview_type"] == "docx_html"
    assert "训练计划" in payload["preview_html"]
