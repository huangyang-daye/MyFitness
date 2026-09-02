"""CLI 对话输入框测试。"""

import logging

from myfitness.api.cli import _chat_input_hint, _configure_interactive_chat_logging
from myfitness.debug import set_debug_mode


def test_chat_input_hint_home_vs_conversation():
    assert "空输入" in _chat_input_hint(on_home=True)
    assert "/model" in _chat_input_hint(on_home=False)


def test_interactive_chat_suppresses_console_logs():
    set_debug_mode(False)
    _configure_interactive_chat_logging()
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger("myfitness").level == logging.WARNING
    assert logging.getLogger("sqlalchemy.engine.Engine").level == logging.WARNING


def test_interactive_chat_keeps_logs_in_debug_mode():
    set_debug_mode(True)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("myfitness").setLevel(logging.INFO)
    _configure_interactive_chat_logging()
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("myfitness").level == logging.INFO
