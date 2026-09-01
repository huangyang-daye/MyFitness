import pytest

from myfitness.api.web import AgentWebApplication, inline_content_disposition
from myfitness.llm.factory import LlmConfig
from myfitness.llm.registry import ModelRegistryError
from myfitness.services.artifacts import ArtifactError


@pytest.fixture
def web_app(tmp_path):
    return AgentWebApplication(tmp_path)


def test_list_models_exposes_env_preset_and_providers(web_app, monkeypatch):
    monkeypatch.setattr("myfitness.llm.registry.get_settings", lambda: _settings())
    payload = web_app.list_models()
    assert payload["active_id"] == "env"
    assert payload["models"][0]["source"] == "env"
    assert payload["providers"], "前端需要服务商模板做快速填入"


def test_saved_model_becomes_active_and_masks_key(web_app, monkeypatch):
    monkeypatch.setattr("myfitness.llm.registry.get_settings", lambda: _settings())
    saved = web_app.save_model({
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key": "sk-abcdef1234",
    })
    model_id = saved["model"]["id"]
    assert saved["model"]["key_hint"] == "****1234"
    assert "api_key" not in saved["model"]

    activated = web_app.activate_model(model_id)
    assert activated["active_id"] == model_id
    assert any(row["id"] == model_id and row["active"] for row in activated["models"])


def test_save_model_rejects_invalid_payload(web_app, monkeypatch):
    monkeypatch.setattr("myfitness.llm.registry.get_settings", lambda: _settings())
    with pytest.raises(ModelRegistryError, match="Base URL"):
        web_app.save_model({"name": "x", "base_url": "not-a-url", "model": "m"})


def test_delete_model_falls_back_to_env(web_app, monkeypatch):
    monkeypatch.setattr("myfitness.llm.registry.get_settings", lambda: _settings())
    saved = web_app.save_model({
        "name": "M", "base_url": "https://x/v1", "model": "m", "api_key": "sk-1",
    })
    model_id = saved["model"]["id"]
    after = web_app.delete_model(model_id)
    assert not [row for row in after["models"] if row["source"] == "user"]
    assert after["active_id"] == "env"


def test_model_probe_reports_success(web_app, monkeypatch):
    captured = {}

    def fake_probe(config: LlmConfig, prompt: str = "回复 OK"):
        captured["config"] = config
        captured["prompt"] = prompt
        return {"model": config.model, "base_url": config.base_url, "reply": "OK",
                "usage": {}, "latency_ms": 12}

    monkeypatch.setattr("myfitness.api.web.probe_llm_config", fake_probe)
    result = web_app.test_model({
        "base_url": "https://api.example.com/v1", "model": "m1", "api_key": "sk-1",
    })
    assert result["ok"] is True
    assert captured["config"].base_url == "https://api.example.com/v1"
    assert captured["config"].timeout <= 30, "探测超时要收敛，避免前端长时间等待"


def test_model_probe_reports_failure_without_raising(web_app, monkeypatch):
    def fake_probe(config: LlmConfig, prompt: str = "回复 OK"):
        raise RuntimeError("HTTP 401")

    monkeypatch.setattr("myfitness.api.web.probe_llm_config", fake_probe)
    result = web_app.test_model({
        "base_url": "https://api.example.com/v1", "model": "m1", "api_key": "sk-1",
    })
    assert result == {"ok": False, "error": "HTTP 401"}


def test_model_probe_requires_credentials(web_app):
    with pytest.raises(ValueError, match="API Key"):
        web_app.test_model({"base_url": "https://x/v1", "model": "m"})


def test_inline_content_disposition_supports_unicode_filename():
    header = inline_content_disposition("减重期训练建议.pdf")
    header.encode("latin-1")  # http.server 仅允许 latin-1 响应头
    assert "filename*=" in header
    assert "%E5%87%8F%E9%87%8D" in header


def test_read_artifact_scoped_to_data_dir(web_app, tmp_path, monkeypatch):
    from myfitness.config import Settings

    data_dir = tmp_path / "data"
    (data_dir / "reports").mkdir(parents=True)
    report = data_dir / "reports" / "2026-08-29.md"
    report.write_text("# 日报\n", encoding="utf-8")
    monkeypatch.setattr(
        "myfitness.services.artifacts.get_settings",
        lambda: Settings(data_dir=str(data_dir)),
    )

    payload = web_app.read_artifact_file(str(report))
    assert payload["name"] == "2026-08-29.md"
    assert "# 日报" in payload["content"]

    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ArtifactError, match="数据目录"):
        web_app.read_artifact_file(str(outside))


def _settings():
    from myfitness.config import Settings

    return Settings(llm_api_key="sk-env", llm_model="env-model", data_dir="/tmp/mf")
