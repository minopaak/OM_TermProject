"""Agent 3: Decision Maker — deterministic merge, no LLM call.

Each specialist (time-series / event / data-quality) already proposed
multipliers based on its own analysis. This layer:
  1. filters proposals whose dates fall inside the forecast window,
  2. drops proposals with multiplier == 1.0 (no-ops),
  3. returns the unified list as `selected_insights`.

No LLM cross-checks at this layer; specialists are trusted to make domain
calls. If we observe systematic specialist mistakes later we can add an
LLM cross-check here.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.agents.pattern_analyst import ProposedSignal, SpecialistOutput
from src.forecasting.package import PredictionPackage


class SelectedInsight(BaseModel):
    type: str = Field(description="signal type from specialist")
    dates: list[str] = Field(description="forecast dates (YYYY-MM-DD)")
    multiplier: float = Field(description="× multiplier (1.0 = identity)")
    reason: str = Field(description="short reason citing source specialist")


class DecisionOutput(BaseModel):
    selected_insights: list[SelectedInsight] = Field(default_factory=list)


def _signal_to_insight(
    signal: ProposedSignal,
    forecast_dates: set[str],
    source: str,
) -> SelectedInsight | None:
    # Filter dates to those inside the forecast window.
    valid_dates = [d for d in signal.dates if d in forecast_dates]
    if not valid_dates:
        return None
    if abs(signal.multiplier - 1.0) < 1e-6:
        return None
    return SelectedInsight(
        type=signal.type,
        dates=valid_dates,
        multiplier=round(float(signal.multiplier), 4),
        reason=f"{source} ({signal.confidence}): {signal.rationale}",
    )


def compile_decision(
    package: PredictionPackage,
    context: dict[str, Any],
    ts_output: SpecialistOutput,
    event_output: SpecialistOutput,
    quality_output: SpecialistOutput,
) -> DecisionOutput:
    """Merge specialist proposals into selected_insights."""
    forecast_dates = {str(d) for d in package.forecast_window.dates}
    selected: list[SelectedInsight] = []

    for s in ts_output.proposed_signals:
        ins = _signal_to_insight(s, forecast_dates, "time-series specialist")
        if ins:
            selected.append(ins)

    for s in event_output.proposed_signals:
        ins = _signal_to_insight(s, forecast_dates, "event specialist")
        if ins:
            selected.append(ins)

    for s in quality_output.proposed_signals:
        ins = _signal_to_insight(s, forecast_dates, "data quality specialist")
        if ins:
            selected.append(ins)

    return DecisionOutput(selected_insights=selected)
