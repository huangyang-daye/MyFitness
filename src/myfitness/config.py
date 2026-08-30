import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from myfitness.paths import PROJECT_ROOT


def default_data_dir() -> str:
    """运行时数据的平台默认根目录（与项目代码分离）。"""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return str(Path(base) / "MyFitness")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 绝对与相对两条都给：从任意工作目录启动（比如开始菜单里的 myfitness ui）
        # 都能加载到项目配置，否则 DATA_DIR 会静默回落到平台默认目录、数据分裂。
        # 列表后者覆盖前者，因此当前目录下的 .env 优先级更高。
        env_file=(str(PROJECT_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://postgres:123456@127.0.0.1:5432/myfitness"
    db_connect_timeout: int = 5

    xunji_body_api_key: str = ""
    xunji_food_api_key: str = ""
    xunji_food_search_key: str = ""
    xunji_training_api_key: str = ""

    # LLM — OpenAI 兼容通用接口
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_max_tokens: int | None = None
    llm_timeout: int = 120

    # 兼容旧变量名
    openai_api_key: str = ""
    llm_provider: str = "openai_compat"

    # 使用记录（报告、对话历史等）与项目本体分离，统一存放在 data_dir 之下
    data_dir: str = ""

    daily_report_time: str = "07:00"
    daily_report_mode: str = "full"
    # 留空则回落到 <data_dir>/reports
    daily_report_output_dir: str = ""
    # 统计图（mermaid）独立文档输出目录；留空则回落到 <data_dir>/reports/charts
    chart_output_dir: str = ""
    # 对话记录目录；留空则回落到 <data_dir>/chat-history
    chat_history_dir: str = ""

    log_level: str = "INFO"
    default_user_id: int = 1

    sync_default_days: int = 90
    xunji_cache_ttl_seconds: int = 300

    @model_validator(mode="after")
    def derive_data_paths(self) -> "Settings":
        """把留空的派生路径回填为 data_dir 之下的子目录。

        data_dir 缺省时落到平台用户数据目录，保证运行时产物永远不会写回项目目录。
        """
        root = Path(self.data_dir.strip() or default_data_dir()).expanduser()
        self.data_dir = str(root)
        if not self.daily_report_output_dir.strip():
            self.daily_report_output_dir = str(root / "reports")
        if not self.chart_output_dir.strip():
            self.chart_output_dir = str(root / "reports" / "charts")
        if not self.chat_history_dir.strip():
            self.chat_history_dir = str(root / "chat-history")
        return self

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        value = value.strip()
        # Windows：localhost 常先解析 IPv6(::1)，而本地 PG 多仅监听 127.0.0.1，导致 2min+ 连接等待
        if sys.platform == "win32":
            value = re.sub(r"@localhost(?=[:/])", "@127.0.0.1", value)
        return value

    @field_validator("db_connect_timeout", mode="before")
    @classmethod
    def parse_db_connect_timeout(cls, value: object) -> object:
        if value == "" or value is None:
            return 5
        return value

    @field_validator("llm_base_url")
    @classmethod
    def normalize_llm_base_url(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def parse_temperature(cls, value: object) -> object:
        if value == "" or value is None:
            return 0.7
        return value

    @field_validator("llm_max_tokens", mode="before")
    @classmethod
    def parse_max_tokens(cls, value: object) -> object:
        if value == "" or value is None:
            return None
        return value

    @field_validator("llm_timeout", mode="before")
    @classmethod
    def parse_timeout(cls, value: object) -> object:
        if value == "" or value is None:
            return 120
        return value

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("LLM_TEMPERATURE 须在 0.0 ~ 2.0 之间")
        return value

    def resolved_llm_api_key(self) -> str:
        return self.llm_api_key or self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
