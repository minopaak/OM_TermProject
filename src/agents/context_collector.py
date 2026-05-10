"""Agent 1: Context Collector.

Pulls calendar context (events, weekday, SNAP) for the forecast window from
`calendar.parquet`. Deterministic — no LLM call.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.config import CALENDAR_PARQUET
from src.forecasting.package import PredictionPackage

_calendar_df: pd.DataFrame | None = None


def _calendar() -> pd.DataFrame:
    global _calendar_df
    if _calendar_df is None:
        df = pd.read_parquet(CALENDAR_PARQUET)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        _calendar_df = df
    return _calendar_df


def _serialize_window(start_date: str, end_date: str) -> dict[str, Any]:
    """Extract events / weekdays / SNAP rows for [start_date, end_date]."""
    cal = _calendar()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    win = cal[(cal["date"] >= start) & (cal["date"] <= end)]

    events: list[dict[str, str]] = []
    for _, row in win.iterrows():
        for n_col, t_col in (("event_name_1", "event_type_1"), ("event_name_2", "event_type_2")):
            name = row.get(n_col)
            if pd.notna(name) and name != "":
                events.append(
                    {
                        "date": str(row["date"]),
                        "name": str(name),
                        "type": str(row.get(t_col, "")),
                    }
                )

    weekdays = {str(row["date"]): str(row["weekday"]) for _, row in win.iterrows()}

    snap = {state: [] for state in ("CA", "TX", "WI")}
    for _, row in win.iterrows():
        for state in ("CA", "TX", "WI"):
            if int(row.get(f"snap_{state}", 0)) == 1:
                snap[state].append(str(row["date"]))

    return {"events": events, "weekdays": weekdays, "snap": snap}


def collect_context(package: PredictionPackage) -> dict[str, Any]:
    """Forecast window의 events·weekdays·SNAP을 추출한다."""
    fc = package.forecast_window
    start = str(fc.dates[0])
    end = str(fc.dates[-1])
    raw = _serialize_window(start, end)
    snap_for_state = raw["snap"].get(package.state_id, [])
    return {
        "sku_id": package.sku_id,
        "state_id": package.state_id,
        "forecast_period": {"start": start, "end": end},
        "events": raw["events"],
        "weekdays": raw["weekdays"],
        "snap_days": snap_for_state,
        "all_state_snap": raw["snap"],
    }


def format_context_for_prompt(context: dict[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=False, indent=2)
