"""SQL 日志开关测试。"""

from myfitness.config import get_settings
from myfitness.db.sql_logging import is_sql_echo_enabled


def test_sql_echo_follows_settings(monkeypatch):
    monkeypatch.setenv("SQL_ECHO", "false")
    monkeypatch.setenv("DEBUG_MODE", "false")
    get_settings.cache_clear()
    assert is_sql_echo_enabled() is False

    monkeypatch.setenv("SQL_ECHO", "true")
    get_settings.cache_clear()
    assert is_sql_echo_enabled() is True

    monkeypatch.setenv("SQL_ECHO", "false")
    monkeypatch.setenv("DEBUG_MODE", "true")
    get_settings.cache_clear()
    assert is_sql_echo_enabled() is True
