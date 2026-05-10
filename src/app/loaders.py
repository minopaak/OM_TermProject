"""Run-dir loaders.

Reads the artifacts produced by `src.agents.batch.run_batch` and packages
them into a single `RunBundle` consumed by the Streamlit app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from src.config import DATA_DIR, TEST_PARQUET


@dataclass
class Insight:
    id: str
    type: str
    dates: list[str]
    multiplier: float
    reason: str


@dataclass
class RunBundle:
    run_id: str
    run_dir: Path
    sku_id: str
    metadata: dict
    baseline_model: str
    input_dates: list[str]
    input_values: list[float]
    baseline_dates: list[str]
    baseline_values: list[float]
    agent_final_values: list[float]
    insights: list[Insight]
    context: dict
    report_md: str
    intermediates_md: str = ""
    extras: dict = field(default_factory=dict)


def list_run_ids(runs_root: Path | None = None) -> list[str]:
    runs_root = runs_root or (DATA_DIR / "runs")
    if not runs_root.exists():
        return []
    return sorted(
        [p.name for p in runs_root.iterdir() if (p / "state").exists()],
        reverse=True,
    )


def list_skus(run_id: str, runs_root: Path | None = None) -> list[str]:
    runs_root = runs_root or (DATA_DIR / "runs")
    state_dir = runs_root / run_id / "state"
    if not state_dir.exists():
        return []
    return sorted(p.stem for p in state_dir.glob("*.json"))


def load_bundle(
    run_id: str,
    sku_id: str,
    runs_root: Path | None = None,
) -> RunBundle:
    runs_root = runs_root or (DATA_DIR / "runs")
    run_dir = runs_root / run_id

    state_path = run_dir / "state" / f"{sku_id}.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"state file missing: {state_path}. Re-run the batch with the "
            "current code to regenerate it."
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))

    report_path = run_dir / "reports" / f"{sku_id}.md"
    report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    inter_path = run_dir / "intermediates" / f"{sku_id}.md"
    inter_md = inter_path.read_text(encoding="utf-8") if inter_path.exists() else ""

    insights = [
        Insight(
            id=ins["id"],
            type=ins["type"],
            dates=list(ins["dates"]),
            multiplier=float(ins["multiplier"]),
            reason=ins.get("reason", ""),
        )
        for ins in state.get("selected_insights", [])
    ]

    return RunBundle(
        run_id=run_id,
        run_dir=run_dir,
        sku_id=state["sku_id"],
        metadata=state.get("metadata", {}),
        baseline_model=state.get("baseline_model", ""),
        input_dates=state["input_window"]["dates"],
        input_values=state["input_window"]["values"],
        baseline_dates=state["forecast_baseline"]["dates"],
        baseline_values=state["forecast_baseline"]["values"],
        agent_final_values=state["forecast_agent_final"]["values"],
        insights=insights,
        context=state.get("context", {}),
        report_md=report_md,
        intermediates_md=inter_md,
    )


def load_actuals(sku_id: str, dates: list[str]) -> list[float] | None:
    """Pull test-set actuals for the given dates.

    Returns the values aligned to `dates`, or `None` if the test set has
    no row for any of the requested dates (i.e. forecast horizon goes
    beyond the held-out window).
    """
    if not dates:
        return None
    if not TEST_PARQUET.exists():
        return None
    con = duckdb.connect(":memory:")
    try:
        df = con.execute(
            "SELECT CAST(date AS VARCHAR) AS date, CAST(sales AS DOUBLE) AS y "
            "FROM read_parquet(?) "
            "WHERE sku_id = ? AND date BETWEEN ? AND ? "
            "ORDER BY date",
            [TEST_PARQUET.as_posix(), sku_id, dates[0], dates[-1]],
        ).df()
    finally:
        con.close()
    by_date = dict(zip(df["date"].astype(str), df["y"].astype(float)))
    aligned = [by_date.get(str(d)) for d in dates]
    if any(v is None for v in aligned):
        return None
    return [float(v) for v in aligned]
