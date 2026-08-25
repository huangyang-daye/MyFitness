from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from myfitness.db.models import BodyMetric, Food, NutritionLog, SyncJob, TrainingExercise, TrainingLog


SOURCE_MANUAL = "manual"
SOURCE_XUNJI = "xunji_sync"


class BodyMetricRepository:
    def __init__(self, session: Session, user_id: int):
        self.session = session
        self.user_id = user_id

    def has_manual_record(self, record_date: date, metric_type: str) -> bool:
        stmt = select(BodyMetric.id).where(
            BodyMetric.user_id == self.user_id,
            BodyMetric.record_date == record_date,
            BodyMetric.metric_type == metric_type,
            BodyMetric.source == SOURCE_MANUAL,
        )
        return self.session.scalar(stmt) is not None

    def upsert_from_sync(
        self,
        record_date: date,
        metric_type: str,
        value: float,
        unit: str,
        xunji_ref: str,
        synced_at: datetime | None = None,
    ) -> bool:
        if self.has_manual_record(record_date, metric_type):
            return False

        synced_at = synced_at or datetime.now(UTC)
        existing = self.session.scalar(
            select(BodyMetric).where(
                BodyMetric.user_id == self.user_id,
                BodyMetric.record_date == record_date,
                BodyMetric.metric_type == metric_type,
                BodyMetric.source == SOURCE_XUNJI,
            )
        )
        if existing:
            existing.value = value
            existing.unit = unit
            existing.xunji_ref = xunji_ref
            existing.synced_at = synced_at
        else:
            self.session.add(
                BodyMetric(
                    user_id=self.user_id,
                    record_date=record_date,
                    metric_type=metric_type,
                    value=value,
                    unit=unit,
                    source=SOURCE_XUNJI,
                    xunji_ref=xunji_ref,
                    synced_at=synced_at,
                )
            )
        self.session.flush()
        return True

    def query_range(self, start: date, end: date, metric_type: str | None = None) -> list[BodyMetric]:
        stmt = select(BodyMetric).where(
            BodyMetric.user_id == self.user_id,
            BodyMetric.record_date >= start,
            BodyMetric.record_date <= end,
        )
        if metric_type:
            stmt = stmt.where(BodyMetric.metric_type == metric_type)
        stmt = stmt.order_by(BodyMetric.record_date.desc())
        return list(self.session.scalars(stmt).all())

    def get_effective_value(self, record_date: date, metric_type: str) -> BodyMetric | None:
        """manual 优先于 xunji_sync。"""
        manual = self.session.scalar(
            select(BodyMetric).where(
                BodyMetric.user_id == self.user_id,
                BodyMetric.record_date == record_date,
                BodyMetric.metric_type == metric_type,
                BodyMetric.source == SOURCE_MANUAL,
            )
        )
        if manual:
            return manual
        return self.session.scalar(
            select(BodyMetric).where(
                BodyMetric.user_id == self.user_id,
                BodyMetric.record_date == record_date,
                BodyMetric.metric_type == metric_type,
                BodyMetric.source == SOURCE_XUNJI,
            )
        )

    def upsert_manual(
        self,
        record_date: date,
        metric_type: str,
        value: float,
        unit: str,
    ) -> BodyMetric:
        existing = self.session.scalar(
            select(BodyMetric).where(
                BodyMetric.user_id == self.user_id,
                BodyMetric.record_date == record_date,
                BodyMetric.metric_type == metric_type,
                BodyMetric.source == SOURCE_MANUAL,
            )
        )
        if existing:
            existing.value = value
            existing.unit = unit
            metric = existing
        else:
            metric = BodyMetric(
                user_id=self.user_id,
                record_date=record_date,
                metric_type=metric_type,
                value=value,
                unit=unit,
                source=SOURCE_MANUAL,
            )
            self.session.add(metric)
        self.session.flush()
        return metric


class FoodRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create(
        self,
        name: str,
        uniquekey: str | None,
        ntr: dict,
        units: dict | list | None = None,
        source: str = "xunji",
    ) -> Food:
        if uniquekey:
            existing = self.session.scalar(select(Food).where(Food.uniquekey == uniquekey))
            if existing:
                return existing

        food = Food(name=name, uniquekey=uniquekey, ntr=ntr, units=units, source=source)
        self.session.add(food)
        self.session.flush()
        return food


class NutritionLogRepository:
    def __init__(self, session: Session, user_id: int):
        self.session = session
        self.user_id = user_id

    def upsert_from_sync(
        self,
        record_date: date,
        meal_type: str,
        food_name: str,
        amount: float,
        unit: str,
        nutrients_snapshot: dict,
        food_id: int | None,
        xunji_record_id: str | None,
    ) -> None:
        if xunji_record_id:
            existing = self.session.scalar(
                select(NutritionLog).where(
                    NutritionLog.user_id == self.user_id,
                    NutritionLog.xunji_record_id == xunji_record_id,
                    NutritionLog.source == SOURCE_XUNJI,
                )
            )
            if existing:
                existing.record_date = record_date
                existing.meal_type = meal_type
                existing.food_name = food_name
                existing.amount = amount
                existing.unit = unit
                existing.nutrients_snapshot = nutrients_snapshot
                existing.food_id = food_id
                return

        self.session.add(
            NutritionLog(
                user_id=self.user_id,
                record_date=record_date,
                meal_type=meal_type,
                food_id=food_id,
                food_name=food_name,
                amount=amount,
                unit=unit,
                nutrients_snapshot=nutrients_snapshot,
                source=SOURCE_XUNJI,
                xunji_record_id=xunji_record_id,
            )
        )

    def query_range(self, start: date, end: date) -> list[NutritionLog]:
        stmt = (
            select(NutritionLog)
            .where(
                NutritionLog.user_id == self.user_id,
                NutritionLog.record_date >= start,
                NutritionLog.record_date <= end,
            )
            .order_by(NutritionLog.record_date.desc())
        )
        return list(self.session.scalars(stmt).all())

    def add_manual(
        self,
        record_date: date,
        meal_type: str,
        food_name: str,
        amount: float,
        unit: str,
        nutrients_snapshot: dict,
    ) -> NutritionLog:
        log = NutritionLog(
            user_id=self.user_id,
            record_date=record_date,
            meal_type=meal_type,
            food_name=food_name,
            amount=amount,
            unit=unit,
            nutrients_snapshot=nutrients_snapshot,
            source=SOURCE_MANUAL,
        )
        self.session.add(log)
        self.session.flush()
        return log

    def daily_totals(self, record_date: date) -> dict[str, float]:
        logs = self.query_range(record_date, record_date)
        totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
        for log in logs:
            ntr = log.nutrients_snapshot or {}
            totals["calories"] += float(ntr.get("cal", 0) or 0)
            totals["protein_g"] += float(ntr.get("protein", 0) or 0)
            totals["carbs_g"] += float(ntr.get("carb", 0) or 0)
            totals["fat_g"] += float(ntr.get("fat", 0) or 0)
        return totals


class TrainingLogRepository:
    def __init__(self, session: Session, user_id: int):
        self.session = session
        self.user_id = user_id

    def upsert_from_sync(
        self,
        record_date: date,
        title: str | None,
        raw_payload: dict,
        xunji_localid: str,
        exercises: list[dict],
    ) -> None:
        existing = self.session.scalar(
            select(TrainingLog).where(
                TrainingLog.user_id == self.user_id,
                TrainingLog.xunji_localid == xunji_localid,
            )
        )
        if existing:
            existing.record_date = record_date
            existing.title = title
            existing.raw_payload = raw_payload
            existing.exercises.clear()
            log = existing
        else:
            log = TrainingLog(
                user_id=self.user_id,
                record_date=record_date,
                title=title,
                raw_payload=raw_payload,
                source=SOURCE_XUNJI,
                xunji_localid=xunji_localid,
            )
            self.session.add(log)
            self.session.flush()

        for ex in exercises:
            log.exercises.append(
                TrainingExercise(
                    movement_name=ex["movement_name"],
                    set_count=ex["set_count"],
                    sets_detail=ex.get("sets_detail"),
                )
            )

    def query_range(self, start: date, end: date) -> list[TrainingLog]:
        stmt = (
            select(TrainingLog)
            .where(
                TrainingLog.user_id == self.user_id,
                TrainingLog.record_date >= start,
                TrainingLog.record_date <= end,
            )
            .order_by(TrainingLog.record_date.desc())
        )
        return list(self.session.scalars(stmt).all())


class SyncJobRepository:
    def __init__(self, session: Session, user_id: int):
        self.session = session
        self.user_id = user_id

    def create(
        self,
        sync_type: str,
        sync_start_date: date | None,
        sync_end_date: date | None,
    ) -> SyncJob:
        job = SyncJob(
            user_id=self.user_id,
            sync_type=sync_type,
            sync_start_date=sync_start_date,
            sync_end_date=sync_end_date,
            status="running",
            last_run_at=datetime.now(UTC),
        )
        self.session.add(job)
        self.session.flush()
        return job

    def finish(
        self,
        job: SyncJob,
        status: str,
        stats: dict | None = None,
        error_log: str | None = None,
    ) -> None:
        job.status = status
        job.stats = stats
        job.error_log = error_log
        job.last_run_at = datetime.now(UTC)

    def last_success_end_date(self, sync_type: str) -> date | None:
        stmt = (
            select(SyncJob)
            .where(
                SyncJob.user_id == self.user_id,
                SyncJob.sync_type == sync_type,
                SyncJob.status.in_(["success", "partial"]),
            )
            .order_by(SyncJob.last_run_at.desc())
            .limit(1)
        )
        job = self.session.scalar(stmt)
        return job.sync_end_date if job else None
