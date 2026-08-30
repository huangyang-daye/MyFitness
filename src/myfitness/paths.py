"""Path constants that keep project sources apart from runtime data.

项目本体（源码 / 测试 / 文档）只有一份，就是 ``PROJECT_ROOT``；
使用记录（生成的报告、对话历史、日志等）一律写在 ``Settings.data_dir`` 之下，
两者互不污染。详见 README 的「目录约定」一节。
"""

from __future__ import annotations

from pathlib import Path

# 项目本体根目录：src/myfitness/paths.py -> 上溯两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = PROJECT_ROOT / "skills"
