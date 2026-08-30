import json
from pathlib import Path

import pytest

from myfitness.config import Settings
from myfitness.llm.factory import get_llm_config, is_llm_configured, preset_llm_config
from myfitness.llm.registry import (
    ENV_PRESET_ID,
    REGISTRY_ENV_VAR,
    ModelRegistry,
    ModelRegistryError,
    get_registry,
    registry_path,
)


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "llm-models.json")


def test_empty_registry_falls_back_to_env(registry, monkeypatch):
    monkeypatch.setattr(
        "myfitness.llm.registry.get_settings",
        lambda: Settings(llm_api_key="sk-env", llm_model="env-model", data_dir="/tmp/mf"),
    )
    assert registry.active_id() == ENV_PRESET_ID
    assert registry.active().model == "env-model"
    # 未保存任何预设时，生效配置来自 .env
    assert get_llm_config().model == "env-model"


def test_registry_without_any_source_reports_unset(registry, monkeypatch):
    monkeypatch.setattr(
        "myfitness.llm.registry.get_settings",
        lambda: Settings(llm_api_key="", llm_model="", data_dir="/tmp/mf"),
    )
    assert registry.active_id() is None
    assert registry.active() is None


def test_add_activate_and_persist(registry, monkeypatch):
    monkeypatch.setattr(
        "myfitness.llm.registry.get_settings",
        lambda: Settings(llm_api_key="sk-env", llm_model="env-model", data_dir="/tmp/mf"),
    )
    preset = registry.upsert({
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1/",
        "model": "deepseek-chat",
        "api_key": "sk-abcdef1234",
    })
    assert preset.base_url == "https://api.deepseek.com/v1"  # 末尾斜杠被规范化

    registry.set_active(preset.id)
    assert registry.active_id() == preset.id

    # 重新加载后仍在，说明已落盘
    reloaded = ModelRegistry(registry.path)
    assert [item.id for item in reloaded.models()] == [preset.id]
    assert reloaded.active_id() == preset.id
    assert json.loads(registry.path.read_text(encoding="utf-8"))["active_id"] == preset.id


def test_public_payload_masks_api_key(registry):
    registry.upsert({
        "name": "M", "base_url": "https://example.com/v1", "model": "m1", "api_key": "sk-secret-9876",
    })
    payload = registry.public_payload()
    item = next(row for row in payload["models"] if row["source"] == "user")
    assert "api_key" not in item
    assert item["key_hint"] == "****9876"
    assert item["has_key"] is True
    assert payload["providers"], "应返回服务商模板供前端快速填入"


def test_update_keeps_key_when_not_provided(registry):
    preset = registry.upsert({
        "name": "M", "base_url": "https://example.com/v1", "model": "m1", "api_key": "sk-keep-4321",
    })
    updated = registry.upsert({"id": preset.id, "name": "M2", "base_url": "https://example.com/v1", "model": "m2"})
    assert updated.api_key == "sk-keep-4321"
    assert updated.model == "m2"


def test_update_can_clear_key_explicitly(registry):
    preset = registry.upsert({
        "name": "M", "base_url": "https://example.com/v1", "model": "m1", "api_key": "sk-keep-4321",
    })
    registry.upsert({"id": preset.id, "name": "M", "base_url": "https://example.com/v1", "model": "m1", "api_key": ""})
    assert registry.get(preset.id).api_key == ""
    assert registry.get(preset.id).public_dict()["has_key"] is False


def test_rejects_invalid_payloads(registry):
    with pytest.raises(ModelRegistryError, match="Base URL"):
        registry.upsert({"name": "M", "base_url": "ftp://x", "model": "m"})
    with pytest.raises(ModelRegistryError, match="模型名称"):
        registry.upsert({"name": "", "base_url": "https://x/v1", "model": "m"})
    with pytest.raises(ModelRegistryError, match="温度"):
        registry.upsert({"name": "M", "base_url": "https://x/v1", "model": "m", "temperature": 5})
    with pytest.raises(ModelRegistryError, match="超时"):
        registry.upsert({"name": "M", "base_url": "https://x/v1", "model": "m", "timeout": 1})


def test_env_preset_is_not_editable(registry, monkeypatch):
    monkeypatch.setattr(
        "myfitness.llm.registry.get_settings",
        lambda: Settings(llm_api_key="sk-env", llm_model="env-model", data_dir="/tmp/mf"),
    )
    with pytest.raises(ModelRegistryError, match="不可编辑"):
        registry.upsert({"id": ENV_PRESET_ID, "name": "x", "base_url": "https://x/v1", "model": "y"})


def test_delete_falls_back_to_env(registry, monkeypatch):
    monkeypatch.setattr(
        "myfitness.llm.registry.get_settings",
        lambda: Settings(llm_api_key="sk-env", llm_model="env-model", data_dir="/tmp/mf"),
    )
    preset = registry.upsert({"name": "M", "base_url": "https://x/v1", "model": "m", "api_key": "sk-x"})
    registry.set_active(preset.id)
    registry.delete(preset.id)
    assert registry.models() == []
    assert registry.active_id() == ENV_PRESET_ID


def test_delete_unknown_id(registry):
    with pytest.raises(ModelRegistryError, match="未找到模型"):
        registry.delete("nope")


# --------------------------------------------------------------------- 运行时

def test_active_preset_drives_runtime_config(monkeypatch):
    """用户在前端选中的模型应立刻成为生效配置。"""
    settings = Settings(llm_api_key="sk-env", llm_model="env-model", data_dir="/tmp/mf")
    monkeypatch.setattr("myfitness.llm.registry.get_settings", lambda: settings)
    registry = get_registry()
    preset = registry.upsert({
        "name": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat", "api_key": "sk-ui-5678", "temperature": 0.2, "timeout": 42,
    })
    registry.set_active(preset.id)

    config = get_llm_config()
    assert config.model == "deepseek-chat"
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.api_key == "sk-ui-5678"
    assert config.temperature == 0.2
    assert config.timeout == 42
    assert is_llm_configured() is True


def test_preset_missing_key_falls_back_to_env(monkeypatch):
    settings = Settings(llm_api_key="sk-env", llm_model="env-model", data_dir="/tmp/mf")
    monkeypatch.setattr("myfitness.llm.registry.get_settings", lambda: settings)
    monkeypatch.setattr("myfitness.llm.factory.get_settings", lambda: settings)
    registry = get_registry()
    preset = registry.upsert({"name": "M", "base_url": "https://x/v1", "model": "m"})
    registry.set_active(preset.id)

    assert preset_llm_config(preset) is None
    assert get_llm_config().model == "env-model"


def test_explicit_settings_still_win(monkeypatch):
    """显式传入 settings 时不套用前端选择，保持既有语义。"""
    monkeypatch.setattr(
        "myfitness.llm.registry.get_settings",
        lambda: Settings(llm_api_key="sk-env", llm_model="env-model", data_dir="/tmp/mf"),
    )
    registry = get_registry()
    preset = registry.upsert({
        "name": "M", "base_url": "https://x/v1", "model": "ui-model", "api_key": "sk-ui",
    })
    registry.set_active(preset.id)

    explicit = Settings(llm_api_key="sk-explicit", llm_model="explicit-model", data_dir="/tmp/mf")
    assert get_llm_config(explicit).model == "explicit-model"
    assert is_llm_configured(explicit) is True
    assert is_llm_configured(Settings(llm_api_key="", llm_model="", data_dir="/tmp/mf")) is False


def test_registry_default_path_under_data_dir(monkeypatch):
    """未设置环境变量覆盖时，注册表文件落在 data_dir 之下（与项目本体分离）。"""
    monkeypatch.delenv(REGISTRY_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "myfitness.llm.registry.get_settings", lambda: Settings(data_dir="/tmp/mf-data")
    )
    assert registry_path() == Path("/tmp/mf-data") / "llm-models.json"
