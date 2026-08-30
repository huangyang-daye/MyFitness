"""使用记录（报告 / 图表 / 对话）必须与项目本体处在不同目录。"""

from pathlib import Path

from myfitness.config import Settings, get_settings
from myfitness.paths import PROJECT_ROOT


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_derived_paths_live_under_data_dir(tmp_path):
    settings = _settings(data_dir=str(tmp_path / "runtime"))

    assert settings.daily_report_output_dir == str(tmp_path / "runtime" / "reports")
    assert settings.chart_output_dir == str(tmp_path / "runtime" / "reports" / "charts")
    assert settings.chat_history_dir == str(tmp_path / "runtime" / "chat-history")


def test_derived_paths_never_fall_inside_project():
    """即使不显式配置，运行时产物也不得回落到项目目录。"""
    settings = _settings(data_dir="")

    for value in (
        settings.daily_report_output_dir,
        settings.chart_output_dir,
        settings.chat_history_dir,
    ):
        resolved = Path(value).expanduser().resolve()
        assert resolved != PROJECT_ROOT
        assert PROJECT_ROOT not in resolved.parents, f"{resolved} 仍位于项目目录内"


def test_explicit_overrides_win(tmp_path):
    settings = _settings(
        data_dir=str(tmp_path / "runtime"),
        daily_report_output_dir=str(tmp_path / "custom-reports"),
        chart_output_dir=str(tmp_path / "custom-charts"),
        chat_history_dir=str(tmp_path / "custom-chats"),
    )

    assert settings.daily_report_output_dir == str(tmp_path / "custom-reports")
    assert settings.chart_output_dir == str(tmp_path / "custom-charts")
    assert settings.chat_history_dir == str(tmp_path / "custom-chats")


def test_data_dir_expands_user_home():
    settings = _settings(data_dir="~/myfitness-runtime")

    assert not settings.data_dir.startswith("~")
    assert settings.data_dir.endswith("myfitness-runtime")
    assert Path(settings.data_dir).is_absolute()


def test_cached_settings_expose_new_fields():
    settings = get_settings()

    assert settings.data_dir
    assert settings.daily_report_output_dir
    assert settings.chart_output_dir
    assert settings.chat_history_dir
