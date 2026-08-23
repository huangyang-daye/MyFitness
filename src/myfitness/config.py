from functools import lru_cache
import re
import sys

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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

    daily_report_time: str = "07:00"
    daily_report_mode: str = "full"
    daily_report_output_dir: str = "./reports"

    log_level: str = "INFO"
    default_user_id: int = 1

    sync_default_days: int = 90
    xunji_cache_ttl_seconds: int = 300

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
