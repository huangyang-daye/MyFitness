"""用户目标 Repository。"""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from myfitness.db.models import UserGoal


class UserGoalRepository:
    def __init__(self, session: Session, user_id: int):
        self.session = session
        self.user_id = user_id

    def list_active(self) -> list[UserGoal]:
        stmt = (
            select(UserGoal)
            .where(UserGoal.user_id == self.user_id, UserGoal.status == "active")
            .order_by(UserGoal.created_at.desc())
        )
        return list(self.session.scalars(stmt).all())

    def create(
        self,
        goal_type: str,
        target_value: float,
        start_date: date,
        start_value: float | None = None,
        target_date: date | None = None,
    ) -> UserGoal:
        goal = UserGoal(
            user_id=self.user_id,
            goal_type=goal_type,
            target_value=target_value,
            start_value=start_value,
            start_date=start_date,
            target_date=target_date,
            status="active",
        )
        self.session.add(goal)
        self.session.flush()
        return goal
