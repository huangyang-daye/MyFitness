"""用户可编辑的 LLM 模型清单（OpenAI 兼容协议）。

模型预设持久化在 ``<data_dir>/llm-models.json``，与项目本体分离；
未保存任何预设时回落到 ``.env`` 的 ``LLM_*`` 配置，行为与改造前一致。
API Key 只在进程内保留明文，对外一律脱敏（``key_hint`` + ``has_key``）。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from myfitness.config import get_settings

ENV_PRESET_ID = "env"
REGISTRY_FILENAME = "llm-models.json"
# 测试用：把注册表文件指向临时目录，避免污染真实数据目录
REGISTRY_ENV_VAR = "MYFITNESS_LLM_REGISTRY_FILE"

_lock = threading.Lock()
_cached: "ModelRegistry | None" = None


class ModelRegistryError(ValueError):
    """模型配置不合法。"""


# 常见 OpenAI 兼容服务商模板：用户通常只需补 API Key
PROVIDER_PRESETS: list[dict[str, str]] = [
    {"name": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    {
        "name": "通义千问（兼容模式）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    {"name": "月之暗面 Kimi", "base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    {"name": "智谱 GLM", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    {"name": "本地 Ollama", "base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
    {"name": "自定义（OpenAI 兼容）", "base_url": "", "model": ""},
]


class ModelPreset(BaseModel):
    id: str = ""
    name: str = ""
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    temperature: float = 0.7
    timeout: int = 120
    source: str = "user"  # user | env
    created_at: str = ""

    def masked_key(self) -> str:
        key = self.api_key
        if not key:
            return ""
        return f"****{key[-4:]}" if len(key) > 4 else "****"

    def public_dict(self, *, active: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "has_key": bool(self.api_key),
            "key_hint": self.masked_key(),
            "temperature": self.temperature,
            "timeout": self.timeout,
            "source": self.source,
            "active": active,
        }


class ModelRegistry:
    """模型清单存储；读写带锁，进程内缓存，落盘即失效 LLM 客户端缓存。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._models: list[ModelPreset] = []
        self._active_id: str | None = None
        self._loaded = False
        # 可变方法内部会再调用查询方法，必须可重入
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ load
    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._load_locked()

    def _load_locked(self) -> None:
        self._models = []
        self._active_id = None
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            if isinstance(raw, dict):
                for item in raw.get("models") or []:
                    if isinstance(item, dict):
                        try:
                            self._models.append(ModelPreset.model_validate(item))
                        except Exception:  # noqa: BLE001 - 跳过损坏条目
                            continue
                active = raw.get("active_id")
                self._active_id = str(active) if active else None
        self._loaded = True

    def reload(self) -> None:
        with self._lock:
            self._load_locked()

    def _flush_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "active_id": self._active_id,
            "models": [item.model_dump() for item in self._models],
        }
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.path)
        _invalidate_llm_cache()

    # ----------------------------------------------------------------- query
    def models(self) -> list[ModelPreset]:
        self.ensure_loaded()
        return list(self._models)

    def get(self, model_id: str) -> ModelPreset | None:
        self.ensure_loaded()
        for item in self._models:
            if item.id == model_id:
                return item
        return None

    def env_preset(self) -> ModelPreset | None:
        """.env 派生出来的虚拟预设；未配置时返回 None。"""
        settings = get_settings()
        key = settings.resolved_llm_api_key()
        if not key or not settings.llm_model:
            return None
        return ModelPreset(
            id=ENV_PRESET_ID,
            name="环境配置（.env）",
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
            source="env",
        )

    def all_models(self) -> list[ModelPreset]:
        env = self.env_preset()
        return ([env] if env else []) + self.models()

    def active_id(self) -> str | None:
        self.ensure_loaded()
        if self._active_id:
            if self._active_id == ENV_PRESET_ID:
                return ENV_PRESET_ID if self.env_preset() else None
            if self.get(self._active_id):
                return self._active_id
        env = self.env_preset()
        if env:
            return ENV_PRESET_ID
        models = self.models()
        return models[0].id if models else None

    def active(self) -> ModelPreset | None:
        model_id = self.active_id()
        if not model_id:
            return None
        if model_id == ENV_PRESET_ID:
            return self.env_preset()
        return self.get(model_id)

    # ---------------------------------------------------------------- mutate
    def upsert(self, payload: dict[str, Any]) -> ModelPreset:
        data = dict(payload or {})
        model_id = str(data.get("id") or "").strip()
        existing = self.get(model_id) if model_id else None
        if model_id and model_id != ENV_PRESET_ID and not existing:
            raise ModelRegistryError(f"未找到模型：{model_id}")
        if model_id == ENV_PRESET_ID:
            raise ModelRegistryError("环境配置不可编辑，请复制为新模型后再修改")

        name = str(data.get("name") or "").strip()
        base_url = str(data.get("base_url") or "").strip().rstrip("/")
        model_name = str(data.get("model") or "").strip()
        if not name or len(name) > 64:
            raise ModelRegistryError("模型名称长度必须为 1～64 个字符")
        if not base_url or not base_url.lower().startswith(("http://", "https://")):
            raise ModelRegistryError("Base URL 必须以 http:// 或 https:// 开头")
        if not model_name or len(model_name) > 128:
            raise ModelRegistryError("模型标识长度必须为 1～128 个字符")

        fallback_temperature = existing.temperature if existing else 0.7
        raw_temperature = data.get("temperature")
        if raw_temperature in (None, ""):
            temperature = fallback_temperature
        else:
            try:
                temperature = float(raw_temperature)
            except (TypeError, ValueError) as exc:
                raise ModelRegistryError("温度必须是数字") from exc
        if not 0.0 <= temperature <= 2.0:
            raise ModelRegistryError("温度须在 0.0 ~ 2.0 之间")

        fallback_timeout = existing.timeout if existing else 120
        raw_timeout = data.get("timeout")
        if raw_timeout in (None, ""):
            timeout = fallback_timeout
        else:
            try:
                timeout = int(raw_timeout)
            except (TypeError, ValueError) as exc:
                raise ModelRegistryError("超时必须是整数秒") from exc
        if not 10 <= timeout <= 600:
            raise ModelRegistryError("超时须在 10 ~ 600 秒之间")

        # api_key 缺省（未传）表示保留原值；显式传空串表示清空
        if "api_key" in data:
            api_key = str(data.get("api_key") or "").strip()
        else:
            api_key = existing.api_key if existing else ""

        if existing:
            existing.name = name
            existing.base_url = base_url
            existing.model = model_name
            existing.api_key = api_key
            existing.temperature = temperature
            existing.timeout = timeout
            preset = existing
        else:
            preset = ModelPreset(
                id=uuid.uuid4().hex[:12],
                name=name,
                base_url=base_url,
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                timeout=timeout,
                created_at=datetime.now(UTC).isoformat(),
            )

        with self._lock:
            self.ensure_loaded()
            if existing is None:
                self._models.append(preset)
            self._flush_locked()
        return preset

    def delete(self, model_id: str) -> None:
        with self._lock:
            self.ensure_loaded()
            target = self.get(model_id)
            if target is None:
                raise ModelRegistryError(f"未找到模型：{model_id}")
            self._models = [item for item in self._models if item.id != model_id]
            if self._active_id == model_id:
                self._active_id = None
            self._flush_locked()

    def set_active(self, model_id: str) -> None:
        self.ensure_loaded()
        known = {item.id for item in self.all_models()}
        if model_id not in known:
            raise ModelRegistryError(f"未找到模型：{model_id}")
        with self._lock:
            self._active_id = model_id
            self._flush_locked()

    def public_payload(self) -> dict[str, Any]:
        active = self.active_id()
        models = [item.public_dict(active=item.id == active) for item in self.all_models()]
        return {
            "models": models,
            "active_id": active,
            "providers": PROVIDER_PRESETS,
        }


def _invalidate_llm_cache() -> None:
    """模型改动后丢弃已缓存的 LangChain 客户端，下次调用重建。"""
    try:
        from myfitness.llm.factory import get_llm

        get_llm.cache_clear()
    except Exception:  # noqa: BLE001 - langchain 可选依赖，未装时无需清理
        pass


def registry_path() -> Path:
    override = os.environ.get(REGISTRY_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(get_settings().data_dir).expanduser() / REGISTRY_FILENAME


def get_registry() -> ModelRegistry:
    global _cached
    with _lock:
        if _cached is None:
            _cached = ModelRegistry(registry_path())
        return _cached


def reset_registry() -> None:
    """清空进程内缓存（测试用；下次访问按当前路径重新加载）。"""
    global _cached
    with _lock:
        _cached = None
    _invalidate_llm_cache()


def get_active_preset() -> ModelPreset | None:
    return get_registry().active()
