"""会话标题摘要测试。"""

from myfitness.chat_history import summarize_session_title


def test_summarize_session_title_strips_export_tail():
    message = (
        "根据最近的训练记录，给我一些训练建议，我的目标是减重，"
        "同时保持肌肉不流失，产出保存为pdf、docx、md文档各一份"
    )
    assert summarize_session_title(message) == "减重期训练建议"


def test_summarize_session_title_body_metric():
    assert summarize_session_title("今天体重多少") == "身体数据"
