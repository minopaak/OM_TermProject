"""Forecast Apply — deterministic compound, no LLM.

Apply selected_insights to baseline by per-day compound multiplication:

    yhat_final[d] = baseline[d] × ∏ (multipliers of insights matching d)

Cap each multiplier and the daily compound to [_FACTOR_MIN, _FACTOR_MAX]
as a safety net.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.decision_maker import SelectedInsight
from src.forecasting.package import PredictionPackage


class AdjustedDay(BaseModel):
    date: str = Field(description="YYYY-MM-DD")
    yhat_baseline: float
    yhat_final: float
    applied: str | None = Field(default=None)


class FinalForecast(BaseModel):
    final_forecast: list[AdjustedDay]
    summary: str


_FACTOR_MIN = 0.50  # daily compound floor (after multiple multipliers stack)
_FACTOR_MAX = 2.00  # daily compound ceiling


def apply_insights(
    package: PredictionPackage,
    selected_insights: list[SelectedInsight],
) -> FinalForecast:
    """package.forecast_window 와 selected_insights compound 곱."""
    rows: list[AdjustedDay] = []
    n_adjusted = 0

    for d, base in zip(package.forecast_window.dates, package.forecast_window.values):
        ds = str(d)
        baseline_v = float(base)
        # 이 일자에 매칭되는 insight 들 찾기
        matching = [si for si in selected_insights if ds in si.dates]
        if not matching:
            rows.append(
                AdjustedDay(
                    date=ds,
                    yhat_baseline=baseline_v,
                    yhat_final=baseline_v,
                    applied=None,
                )
            )
            continue

        factor = 1.0
        labels: list[str] = []
        for si in matching:
            m = max(_FACTOR_MIN, min(_FACTOR_MAX, float(si.multiplier)))
            factor *= m
            sign = "+" if m >= 1.0 else "-"
            pct = abs(m - 1.0) * 100
            labels.append(f"{si.type} {sign}{pct:.0f}%")
        # 최종 factor 도 cap
        factor = max(_FACTOR_MIN, min(_FACTOR_MAX, factor))

        if factor != 1.0:
            n_adjusted += 1
            applied = " × ".join(labels)
            if len(matching) > 1:
                applied += f" → 통합 ×{factor:.2f}"
        else:
            applied = None
        final_v = max(0.0, baseline_v * factor)
        rows.append(
            AdjustedDay(
                date=ds,
                yhat_baseline=baseline_v,
                yhat_final=final_v,
                applied=applied,
            )
        )

    summary = f"28일 중 {n_adjusted}일 보정 적용 (compound)"
    return FinalForecast(final_forecast=rows, summary=summary)
