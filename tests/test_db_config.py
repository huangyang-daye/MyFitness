"""数据库连接配置测试。"""

import sys
from unittest.mock import patch

from myfitness.config import Settings


def test_database_url_localhost_normalized_on_windows():
    with patch.object(sys, "platform", "win32"):
        s = Settings(database_url="postgresql+psycopg://u:p@localhost:5432/myfitness")
    assert "@127.0.0.1:" in s.database_url


def test_database_url_localhost_unchanged_on_linux():
    with patch.object(sys, "platform", "linux"):
        s = Settings(database_url="postgresql+psycopg://u:p@localhost:5432/myfitness")
    assert "@localhost:" in s.database_url
