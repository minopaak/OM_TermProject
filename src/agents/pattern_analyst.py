"""Agent 2: Multi-perspective specialists.

Three specialists (gpt-4o-mini), each in its own domain:
- Time Series   : weekday/weekend rhythm, level bias
- Event & Causal: event-driven perturbations
- Data Quality  : supply outage, sparsity caveats

Each specialist:
- Receives the relevant tool output (ratio matrices, weekday patterns,
  history tables — already structured for analysis) plus the SKU's input
  window and baseline forecast.
- Returns:
    * `narrative`        — English analytical narrative for the report writer
    * `proposed_signals` — multiplier proposals based on the specialist's
                           judgment. The LLM decides which signals to apply
                           and with what multiplier. No code-side formula.

Code's role is to forward tool data and to execute LLM-decided multipliers.
The LLM is the analyst.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents._llm import get_llm
from src.agents.prompts import (
    DATA_QUALITY_ANALYST_SYSTEM,
    EVENT_ANALYST_SYSTEM,
    TIME_SERIES_ANALYST_SYSTEM,
)
from src.forecasting.analysis import (
    format_event_windows,
    get_same_period_history_md,
    get_snap_effect_md,
    get_weekday_pattern_md,
)
from src.forecasting.package import PredictionPackage


_SPECIALIST_MODEL = "gpt-4o-mini"


def _to_date(d) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _split_weekday_weekend(dates: list, values: list[float]) -> tuple[float, float, int, int]:
    wd_vals: list[float] = []
    we_vals: list[float] = []
    for d, v in zip(dates, values):
        wd = _to_date(d).weekday()
        if wd >= 5:
            we_vals.append(v)
        else:
            wd_vals.append(v)
    wd_avg = sum(wd_vals) / len(wd_vals) if wd_vals else 0.0
    we_avg = sum(we_vals) / len(we_vals) if we_vals else 0.0
    return wd_avg, we_avg, len(wd_vals), len(we_vals)


# ============================================================
# Output schemas
# ============================================================


class ProposedSignal(BaseModel):
    type: str = Field(
        description=(
            "Signal type: level_weekday | level_weekend | "
            "event_peak | event_buildup | event_antispike | event_postlift | "
            "supply_outage_caveat"
        )
    )
    dates: list[str] = Field(
        description=(
            "Forecast-window dates (YYYY-MM-DD) the multiplier should apply to. "
            "level_weekday → all weekday dates; level_weekend → all weekend dates; "
            "event_peak → the event date; event_buildup → days BEFORE the event; "
            "event_postlift → days AFTER the event; supply_outage_caveat → all 28 days."
        )
    )
    multiplier: float = Field(
        description=(
            "× multiplier to apply to baseline (1.0 = no change, 1.5 = +50%, "
            "0.7 = -30%). Decide based on your full analysis of the data."
        )
    )
    confidence: str = Field(description='"high" | "medium" | "low"')
    rationale: str = Field(
        description="1-2 sentences citing specific evidence (numbers, year-by-year ratios, etc)."
    )


class SpecialistOutput(BaseModel):
    narrative: str = Field(
        description=(
            "3-6 short English paragraphs analyzing the SKU's pattern in this "
            "specialist's domain. Quote specific numbers from the input."
        )
    )
    proposed_signals: list[ProposedSignal] = Field(
        default_factory=list,
        description=(
            "The signals you propose to apply, with multipliers based on your judgment. "
            "If no signal is reliable, output an empty list."
        ),
    )


# ============================================================
# Common input formatting
# ============================================================


def _common_input(package: PredictionPackage, context: dict[str, Any]) -> str:
    in_dates = [str(d) for d in package.input_window.dates]
    in_values = [round(float(v), 2) for v in package.input_window.values]
    fc_dates = [str(d) for d in package.forecast_window.dates]
    fc_values = [round(float(v), 2) for v in package.forecast_window.values]

    in_wd, in_we, _, _ = _split_weekday_weekend(
        package.input_window.dates, package.input_window.values
    )
    fc_wd, fc_we, _, _ = _split_weekday_weekend(
        package.forecast_window.dates, package.forecast_window.values
    )

    return f"""\
## Target SKU
- sku_id: `{package.sku_id}` (cat={package.cat_id}, dept={package.dept_id}, store={package.store_id}, state={package.state_id})

## Input window (last 28 days actual sales)
- Period: {in_dates[0]} ~ {in_dates[-1]}
- Total {sum(in_values):.0f}, avg {sum(in_values)/28:.2f}
- Weekday avg: {in_wd:.2f}
- Weekend avg: {in_we:.2f}
- Daily: {list(zip(in_dates, in_values))}

## Baseline forecast (next 28 days)
- Period: {fc_dates[0]} ~ {fc_dates[-1]}
- Total {sum(fc_values):.2f}, avg {sum(fc_values)/28:.2f}
- Weekday avg: {fc_wd:.2f}
- Weekend avg: {fc_we:.2f}
- Daily: {list(zip(fc_dates, fc_values))}

## Calendar context (events, weekdays, SNAP)
```json
{json.dumps(context, ensure_ascii=False, indent=2)}
```
"""


# ============================================================
# Specialist runners
# ============================================================


def _llm_analyze_structured(
    system_prompt: str,
    user_msg: str,
    callbacks: list | None,
) -> SpecialistOutput:
    llm = get_llm(model=_SPECIALIST_MODEL)
    structured = llm.with_structured_output(SpecialistOutput)
    return structured.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)],
        config={"callbacks": callbacks or []},
    )


def run_time_series_analyst(
    package: PredictionPackage,
    context: dict[str, Any],
    callbacks: list | None = None,
) -> SpecialistOutput:
    """Time series specialist."""
    weekday_md = get_weekday_pattern_md(package.sku_id)
    snap_md = get_snap_effect_md(package.sku_id, package.state_id)
    history_md = get_same_period_history_md(
        package.sku_id,
        str(package.forecast_window.dates[0]),
        str(package.forecast_window.dates[-1]),
        n_years=3,
    )

    user_msg = _common_input(package, context) + f"""
---

## Reference data (for analysis)

### 365-day weekday pattern
{weekday_md}

### SNAP effect (365-day)
{snap_md}

### Same-period history (past years)
{history_md}

---

Analyze the time-series characteristics of this SKU and propose multipliers
where you see a real signal. Use your judgment — there are no formulas.
"""
    return _llm_analyze_structured(TIME_SERIES_ANALYST_SYSTEM, user_msg, callbacks)


def run_event_analyst(
    package: PredictionPackage,
    context: dict[str, Any],
    callbacks: list | None = None,
) -> SpecialistOutput:
    """Event specialist."""
    events = context.get("events", []) or []
    event_blocks: list[str] = []
    for ev in events:
        name = ev.get("name", "")
        if not name:
            continue
        try:
            md = format_event_windows(package.sku_id, name, days=5)
        except Exception as exc:  # noqa: BLE001
            md = f"## {name}\n\n(data fetch error: {exc})"
        event_blocks.append(md)

    events_section = "\n\n---\n\n".join(event_blocks) if event_blocks else "(no events)"

    user_msg = _common_input(package, context) + f"""
---

## Past event windows (year-by-year ratio matrices)

Each event section shows past occurrences with the year-by-year ratio
(sales / same-year same-weekday average). Inspect across years to judge
consistency and signal strength.

{events_section}

---

For each event in the forecast window, decide whether to apply a multiplier
and what value, based on your analysis of the year-by-year ratios. Use your
judgment — there are no formulas.
"""
    return _llm_analyze_structured(EVENT_ANALYST_SYSTEM, user_msg, callbacks)


def run_data_quality_analyst(
    package: PredictionPackage,
    context: dict[str, Any],
    callbacks: list | None = None,
) -> SpecialistOutput:
    """Data quality specialist."""
    weekday_md = get_weekday_pattern_md(package.sku_id)
    in_dates = [str(d) for d in package.input_window.dates]
    in_values = [round(float(v), 2) for v in package.input_window.values]
    zero_days = [(d, v) for d, v in zip(in_dates, in_values) if v == 0]

    user_msg = _common_input(package, context) + f"""
---

## Input window zero analysis
- Total zero days: {len(zero_days)}
- Zero dates: {zero_days}

## 365-day weekday pattern (for sparsity context)
{weekday_md}

---

Assess data quality: are input window zeros consecutive (supply outage) or
scattered (normal low demand)? Is the SKU sparse overall? Note caveats but
generally do not propose multipliers (data quality is for context, not
adjustment) unless you have strong reason to.
"""
    return _llm_analyze_structured(DATA_QUALITY_ANALYST_SYSTEM, user_msg, callbacks)
