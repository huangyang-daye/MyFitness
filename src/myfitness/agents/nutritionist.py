"""NutritionistAgent — 营养分析与建议。"""

from __future__ import annotations

from datetime import date

from myfitness.schemas.agent_outputs import (
    DailyTotals,
    MealAnalysisItem,
    NutritionAgentOutput,
    NutritionBalance,
    TdeeEstimate,
)
from myfitness.schemas.state import ContextSnapshot


def run_nutrition_agent(
    context: ContextSnapshot, analysis_date: date | None = None
) -> NutritionAgentOutput:
    analysis_date = analysis_date or context.date_range.end
    summary = context.nutrition_summary
    daily = _daily_totals_from_context(context, summary)

    weight = context.body_metrics_summary.get("latest_weight_kg") or 70
    tdee = TdeeEstimate(
        bmr=round(10 * weight + 6.25 * 170 - 5 * 30 + 5, 0),  # 默认假设，缺 profile
        activity_factor=1.55,
        tdee=round((10 * weight + 995) * 1.55, 0),
        target_calories=round((10 * weight + 995) * 1.55 - 300, 0),
        goal_mode="cut",
    )

    balance = NutritionBalance(
        calorie_delta=daily.calories - (tdee.target_calories or 0) if tdee.target_calories else None,
        protein_per_kg=round(daily.protein_g / weight, 2) if weight else None,
        assessment=_assess_balance(daily, tdee),
    )

    recommendations: list[str] = []
    if balance.protein_per_kg and balance.protein_per_kg < 1.6:
        recommendations.append("蛋白质摄入偏低，建议提升至 1.6–2.0 g/kg。")

    narrative = _build_narrative(daily, context)

    return NutritionAgentOutput(
        analysis_date=analysis_date,
        daily_totals=daily,
        tdee_estimate=tdee,
        balance=balance,
        recommendations=recommendations,
        narrative=narrative,
    )


def _daily_totals_from_context(context: ContextSnapshot, summary: dict) -> DailyTotals:
    nutrition_query = (context.query_results or {}).get("nutrition")
    if nutrition_query and nutrition_query.get("daily_totals"):
        totals_map = nutrition_query["daily_totals"]
        target_day = max(totals_map.keys())
        day = totals_map[target_day]
        return DailyTotals(
            calories=float(day.get("calories", 0)),
            protein_g=float(day.get("protein_g", 0)),
            carbs_g=float(day.get("carbs_g", 0)),
            fat_g=float(day.get("fat_g", 0)),
        )

    today = summary.get("today_totals") or {}
    return DailyTotals(
        calories=float(today.get("calories", 0)),
        protein_g=float(today.get("protein_g", 0)),
        carbs_g=float(today.get("carbs_g", 0)),
        fat_g=float(today.get("fat_g", 0)),
    )


def _build_narrative(daily: DailyTotals, context: ContextSnapshot) -> str:
    nutrition_query = (context.query_results or {}).get("nutrition")
    if nutrition_query:
        start = nutrition_query.get("start_date")
        end = nutrition_query.get("end_date")
        if daily.calories:
            if start == end:
                return f"{start} 摄入约 {daily.calories:.0f} kcal，蛋白质 {daily.protein_g:.0f} g。"
            return (
                f"{start} ~ {end} 查询范围内最近一天摄入约 "
                f"{daily.calories:.0f} kcal，蛋白质 {daily.protein_g:.0f} g。"
            )
        return f"{start} ~ {end} 查询范围内暂无饮食记录。"

    if daily.calories:
        return f"今日摄入约 {daily.calories:.0f} kcal，蛋白质 {daily.protein_g:.0f} g。"
    return "今日暂无饮食记录。"


def _assess_balance(daily: DailyTotals, tdee: TdeeEstimate) -> str:
    if not tdee.target_calories:
        return "unknown"
    delta = daily.calories - tdee.target_calories
    if abs(delta) <= 100:
        return "on_target"
    return "surplus" if delta > 0 else "deficit"
