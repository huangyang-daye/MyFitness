"""CLI 进度展示测试。"""

from myfitness.api.cli_progress import CliTurnProgress


def test_cli_turn_progress_task_plan_and_status():
    tracker = CliTurnProgress()
    tracker.handle(
        {
            "type": "task_plan",
            "user_requirements": "安排练背计划",
            "tasks": [
                {
                    "id": "t1",
                    "description": "检索训练历史",
                    "domain": "fitness",
                    "status": "pending",
                },
                {
                    "id": "t2",
                    "description": "生成练背计划",
                    "domain": "fitness",
                    "status": "pending",
                },
            ],
        }
    )
    tracker.handle({"type": "task_status", "task_id": "t1", "status": "running"})
    tracker.handle({"type": "task_status", "task_id": "t1", "status": "success"})
    tracker.handle("Summary 生成回复中…")

    assert tracker.tasks["t1"]["status"] == "success"
    assert "检索训练历史" in tracker.summary_lines()[0]
    renderable = tracker.renderable()
    assert renderable is not None
