"""Multi-SKU batch runner.

Run the workflow on each SKU sequentially. Persist per-SKU artifacts under
`data/runs/<run_id>/`:
  summary.parquet         — SKU rollup (baseline/final totals, cost, etc.)
  summary.txt             — human-readable summary
  reports/<sku>.md        — Agent 4 manager report (Korean)
  intermediates/<sku>.md  — Agent 1 context + 3 specialists' narratives +
                            proposed signals + compiled selected_insights
  forecasts/<sku>.parquet — daily baseline / final / applied label
  state/<sku>.json        — structured payload consumed by the review app
                            (input_window, baseline, selected_insights,
                            context, metadata)
  traces/<sku>.jsonl      — LLM call traces

Usage:
    from datetime import date
    from src.agents.batch import BatchCase, run_batch

    cases = [
        BatchCase("FOODS_3_090_CA_3", date(2016, 1, 28)),
        ...
    ]
    result = run_batch(cases, run_id="my_run_id")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.agents.workflow import run_workflow
from src.config import DATA_DIR
from src.forecasting.model import get_backend_name
from src.forecasting.package import PredictionPackage, build_prediction_package

RUNS_DIR = DATA_DIR / "runs"


@dataclass
class BatchCase:
    sku_id: str
    input_end_date: date


def _build_state_payload(
    case: "BatchCase",
    package: PredictionPackage,
    context: dict,
    decision,
    forecast_df: pd.DataFrame,
) -> dict:
    """Compose the JSON payload consumed by the review app."""
    insights = []
    for idx, si in enumerate(decision.selected_insights):
        insights.append(
            {
                "id": f"i{idx}",
                "type": si.type,
                "dates": list(si.dates),
                "multiplier": float(si.multiplier),
                "reason": si.reason,
            }
        )

    return {
        "sku_id": case.sku_id,
        "metadata": {
            "item_id": package.item_id,
            "cat_id": package.cat_id,
            "dept_id": package.dept_id,
            "store_id": package.store_id,
            "state_id": package.state_id,
        },
        "input_end_date": str(case.input_end_date),
        "baseline_model": get_backend_name(),
        "input_window": {
            "dates": [str(d) for d in package.input_window.dates],
            "values": [float(v) for v in package.input_window.values],
        },
        "forecast_baseline": {
            "dates": [str(d) for d in forecast_df["date"].tolist()],
            "values": [float(v) for v in forecast_df["yhat_baseline"].tolist()],
        },
        "forecast_agent_final": {
            "dates": [str(d) for d in forecast_df["date"].tolist()],
            "values": [float(v) for v in forecast_df["yhat_final"].tolist()],
        },
        "selected_insights": insights,
        "context": context,
    }


@dataclass
class BatchResult:
    run_id: str
    out_dir: Path
    n_cases: int
    n_success: int
    n_failed: int
    summary_path: Path
    total_cost_usd: float
    total_elapsed_s: float


def run_batch(
    cases: list[BatchCase],
    run_id: str,
    output_root: Path | None = None,
    verbose: bool = True,
) -> BatchResult:
    """SKU 리스트 순차 실행."""
    output_root = output_root or RUNS_DIR
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces").mkdir(exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)
    (out_dir / "forecasts").mkdir(exist_ok=True)
    (out_dir / "intermediates").mkdir(exist_ok=True)
    (out_dir / "state").mkdir(exist_ok=True)

    rows: list[dict] = []
    failed = 0

    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"[{i}/{len(cases)}] {case.sku_id} (input_end={case.input_end_date})")
        t0 = time.time()
        try:
            pkg = build_prediction_package(case.sku_id, case.input_end_date)
            result, tracer = run_workflow(
                pkg,
                run_id=f"{run_id}__{case.sku_id}",
            )
            elapsed = time.time() - t0

            tracer.write(out_dir / "traces" / f"{case.sku_id}.jsonl")
            decision = result["decision"]
            report_md = result["report_md"]
            (out_dir / "reports" / f"{case.sku_id}.md").write_text(
                report_md, encoding="utf-8"
            )

            import json as _json
            insights_md = "\n".join(
                f"- **{si.type}** (×{si.multiplier:.3f}): {si.dates} — {si.reason}"
                for si in decision.selected_insights
            )

            def _render_specialist(out) -> str:
                lines = [out.narrative if out else "(no output)", ""]
                if out and out.proposed_signals:
                    lines.append("### Proposed signals")
                    for s in out.proposed_signals:
                        lines.append(
                            f"- `{s.type}` ×{s.multiplier:.3f} (conf={s.confidence}): "
                            f"dates={s.dates}"
                        )
                        lines.append(f"  - {s.rationale}")
                return "\n".join(lines)

            ts_out = result.get("timeseries_output")
            ev_out = result.get("event_output")
            q_out = result.get("quality_output")

            (out_dir / "intermediates" / f"{case.sku_id}.md").write_text(
                f"# Agent 1 Context\n\n```json\n{_json.dumps(result.get('context'), ensure_ascii=False, indent=2)}\n```\n\n"
                f"# Agent 2a 시계열 분석\n\n{_render_specialist(ts_out)}\n\n"
                f"---\n\n# Agent 2b 이벤트·인과 분석\n\n{_render_specialist(ev_out)}\n\n"
                f"---\n\n# Agent 2c 데이터 품질 분석\n\n{_render_specialist(q_out)}\n\n"
                f"---\n\n# Agent 3 Selected Insights (compiled)\n\n{insights_md}\n",
                encoding="utf-8",
            )

            fc = result["final_forecast"]
            forecast_df = pd.DataFrame(
                [
                    {
                        "date": d.date,
                        "yhat_baseline": d.yhat_baseline,
                        "yhat_final": d.yhat_final,
                        "applied": d.applied,
                    }
                    for d in fc.final_forecast
                ]
            )
            forecast_df.to_parquet(
                out_dir / "forecasts" / f"{case.sku_id}.parquet", index=False
            )

            state_payload = _build_state_payload(
                case=case,
                package=pkg,
                context=result.get("context") or {},
                decision=decision,
                forecast_df=forecast_df,
            )
            (out_dir / "state" / f"{case.sku_id}.json").write_text(
                json.dumps(state_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            summary = tracer.summary()
            row = {
                "sku_id": case.sku_id,
                "input_end_date": str(case.input_end_date),
                "elapsed_s": round(elapsed, 2),
                "baseline_total": float(forecast_df["yhat_baseline"].sum()),
                "final_total": float(forecast_df["yhat_final"].sum()),
                "n_adjusted_days": int(forecast_df["applied"].notna().sum()),
                "adjustment_summary": fc.summary,
                "tokens_in": summary["tokens_in"],
                "tokens_out": summary["tokens_out"],
                "n_llm_calls": summary["n_llm_calls"],
                "cost_usd": summary["estimated_cost_usd"],
                "status": "ok",
                "error": None,
            }
            rows.append(row)
            if verbose:
                print(
                    f"  완료 ({elapsed:.1f}s, "
                    f"${summary['estimated_cost_usd']:.4f}, "
                    f"{summary['n_llm_calls']} LLM 호출)"
                )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            rows.append(
                {
                    "sku_id": case.sku_id,
                    "input_end_date": str(case.input_end_date),
                    "elapsed_s": round(time.time() - t0, 2),
                    "baseline_total": None,
                    "final_total": None,
                    "n_adjusted_days": None,
                    "adjustment_summary": None,
                    "tokens_in": None,
                    "tokens_out": None,
                    "n_llm_calls": None,
                    "cost_usd": None,
                    "status": "failed",
                    "error": str(exc)[:500],
                }
            )
            if verbose:
                print(f"  실패: {exc}")

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "summary.parquet"
    summary_df.to_parquet(summary_path, index=False)

    total_cost = float(summary_df["cost_usd"].fillna(0).sum())
    total_time = float(summary_df["elapsed_s"].fillna(0).sum())

    text_summary = (
        f"run_id: {run_id}\n"
        f"cases: {len(cases)} (success {len(cases)-failed}, failed {failed})\n"
        f"total elapsed: {total_time:.1f}s\n"
        f"total cost: ${total_cost:.4f}\n"
        f"avg cost/sku: ${total_cost/max(1,len(cases)-failed):.4f}\n"
    )
    (out_dir / "summary.txt").write_text(text_summary, encoding="utf-8")
    if verbose:
        print(f"\n=== Batch 완료 ===\n{text_summary}")

    return BatchResult(
        run_id=run_id,
        out_dir=out_dir,
        n_cases=len(cases),
        n_success=len(cases) - failed,
        n_failed=failed,
        summary_path=summary_path,
        total_cost_usd=total_cost,
        total_elapsed_s=total_time,
    )
