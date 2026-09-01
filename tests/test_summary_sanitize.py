"""Summary 回复清洗测试。"""

from myfitness.agents.summary import sanitize_user_facing_reply


def test_sanitize_removes_summary_agent_intro():
    raw = "好的，我作为 Summary Agent，已经整合了您的训练记录。"
    cleaned = sanitize_user_facing_reply(raw)
    assert "Summary Agent" not in cleaned
    assert "已经整合了您的训练记录" in cleaned
