import pytest

from myfitness.workspace_browser import WorkspaceBrowser, WorkspacePathError


def test_lists_reads_and_searches_project_text(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "sample.py").write_text("answer = 'needle'\n", encoding="utf-8")
    browser = WorkspaceBrowser(tmp_path)

    listing = browser.list_files("")
    assert listing["entries"][0]["name"] == "src"
    content = browser.read_file("src/sample.py")
    assert content["language"] == "python"
    assert "needle" in content["content"]
    result = browser.search("needle")
    assert result["results"][0]["path"] == "src/sample.py"
    assert result["results"][0]["line"] == 1


def test_blocks_escape_sensitive_and_history_paths(tmp_path):
    (tmp_path / ".env").write_text("SECRET=yes", encoding="utf-8")
    (tmp_path / ".chatHistory").mkdir()
    (tmp_path / "skills").mkdir()
    browser = WorkspaceBrowser(tmp_path)

    names = {item["name"] for item in browser.list_files()["entries"]}
    assert ".env" not in names
    assert ".chatHistory" not in names
    assert "skills" not in names
    with pytest.raises(WorkspacePathError):
        browser.read_file("../outside.txt")
    with pytest.raises(WorkspacePathError):
        browser.read_file(".env")
