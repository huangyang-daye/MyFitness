"""LLM 通用 OpenAI 兼容 API 集成。"""

from myfitness.llm.factory import (
    LlmConfig,
    LlmUnavailableError,
    LlmWarmupResult,
    chat_completion,
    get_llm,
    get_llm_config,
    is_llm_configured,
    preset_llm_config,
    probe_llm_config,
    probe_llm_connection,
    warmup_llm,
)
from myfitness.llm.guard import (
    LlmCallStats,
    LlmCircuitOpenError,
    LlmGuard,
    get_llm_guard,
)
from myfitness.llm.registry import (
    ENV_PRESET_ID,
    ModelPreset,
    ModelRegistry,
    ModelRegistryError,
    get_active_preset,
    get_registry,
    reset_registry,
)

__all__ = [
    "ENV_PRESET_ID",
    "LlmCallStats",
    "LlmCircuitOpenError",
    "LlmConfig",
    "LlmGuard",
    "LlmUnavailableError",
    "LlmWarmupResult",
    "ModelPreset",
    "ModelRegistry",
    "ModelRegistryError",
    "chat_completion",
    "get_active_preset",
    "get_llm",
    "get_llm_config",
    "get_llm_guard",
    "get_registry",
    "is_llm_configured",
    "preset_llm_config",
    "probe_llm_config",
    "probe_llm_connection",
    "reset_registry",
    "warmup_llm",
]
