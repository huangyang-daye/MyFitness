"""全局测试夹具：把运行时数据隔离到临时目录，避免污染真实数据目录。"""

from __future__ import annotations

import pytest

from myfitness.llm.registry import REGISTRY_ENV_VAR, reset_registry


@pytest.fixture(autouse=True)
def isolated_llm_registry(tmp_path, monkeypatch):
    """每个用例使用独立的模型注册表文件，并在结束后清掉进程内缓存。"""
    monkeypatch.setenv(REGISTRY_ENV_VAR, str(tmp_path / "llm-models.json"))
    reset_registry()
    yield
    reset_registry()
