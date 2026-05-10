"""평가 지표: baseline forecast vs final(보정 후) forecast vs 실제값.

지표:
    - MAE (mean absolute error): 0이 많은 retail count에 robust
    - sMAPE (symmetric MAPE): 0% ~ 200%, actual=0 대응
    - RMSE: 큰 오차 강조

FVA (Forecast Value Added) = baseline_error - final_error
    양수면 에이전트 보정이 정확도를 개선했다는 뜻.

이벤트일 vs 비-이벤트일 분해:
    calendar.event_name_1/2 이 있는 날을 이벤트일로 표시, 별도 집계.

사용:
    from src.evaluation.metrics import evaluate_batch_run

    eval_df = evaluate_batch_run(run_dir)
    # SKU별 metrics + aggregate 요약 자동 저장.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import DUCKDB_PATH, TEST_PARQUET


def _load_actuals(sku_id: str, start: date, end: date) -> pd.DataFrame:
    """test_set.parquet에서 [start, end] 사이 actual sales 로드."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT date, CAST(sales AS DOUBLE) AS actual
            FROM read_parquet('{TEST_PARQUET.as_posix()}')
            WHERE sku_id = '{sku_id}'
              AND date BETWEEN DATE '{start}' AND DATE '{end}'
            ORDER BY date
            """
        ).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _load_event_dates() -> set[date]:
    """calendar에서 이벤트가 있는 날 set 반환."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute(
            """
            SELECT date FROM calendar
            WHERE (event_name_1 IS NOT NULL AND event_name_1 <> '')
               OR (event_name_2 IS NOT NULL AND event_name_2 <> '')
            """
        ).df()
    finally:
        con.close()
    return set(pd.to_datetime(df["date"]).dt.date)


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - forecast)))


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE (%). 0 actuals를 안전하게 처리."""
    if len(actual) == 0:
        return float("nan")
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(actual[mask] - forecast[mask]) / denom[mask]) * 100)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def evaluate_sku(
    sku_id: str,
    forecast_path: Path,
    event_dates: set[date] | None = None,
    save_comparison_dir: Path | None = None,
) -> dict:
    """한 SKU의 baseline·final forecast를 actual과 비교.

    save_comparison_dir 주어지면 baseline·final·actual을 병합한 일자별 표를
    `<sku_id>.parquet`로 저장 (사람·다른 도구에서 보기 쉽게).
    """
    fc = pd.read_parquet(forecast_path)
    fc["date"] = pd.to_datetime(fc["date"]).dt.date
    actuals = _load_actuals(sku_id, fc["date"].min(), fc["date"].max())

    df = fc.merge(actuals, on="date", how="inner")  # actual 있는 일자만

    if save_comparison_dir is not None:
        save_comparison_dir.mkdir(parents=True, exist_ok=True)
        comp = df.copy()
        comp["err_baseline"] = (comp["actual"] - comp["yhat_baseline"]).abs()
        comp["err_final"] = (comp["actual"] - comp["yhat_final"]).abs()
        comp = comp[
            [
                "date",
                "actual",
                "yhat_baseline",
                "yhat_final",
                "err_baseline",
                "err_final",
                "applied",
            ]
        ]
        comp.to_parquet(save_comparison_dir / f"{sku_id}.parquet", index=False)

    a = df["actual"].to_numpy(dtype=float)
    b = df["yhat_baseline"].to_numpy(dtype=float)
    f = df["yhat_final"].to_numpy(dtype=float)

    metrics: dict[str, float | int | str | None] = {
        "sku_id": sku_id,
        "n_days_evaluated": len(df),
        "actual_total": float(a.sum()),
        "baseline_total": float(b.sum()),
        "final_total": float(f.sum()),
        "baseline_mae": mae(a, b),
        "final_mae": mae(a, f),
        "baseline_smape": smape(a, b),
        "final_smape": smape(a, f),
        "baseline_rmse": rmse(a, b),
        "final_rmse": rmse(a, f),
        "fva_mae": mae(a, b) - mae(a, f),  # 양수=개선
        "fva_smape": smape(a, b) - smape(a, f),
    }

    if event_dates:
        is_event = df["date"].isin(event_dates).to_numpy()
        if is_event.any():
            metrics["event_n_days"] = int(is_event.sum())
            metrics["event_baseline_mae"] = mae(a[is_event], b[is_event])
            metrics["event_final_mae"] = mae(a[is_event], f[is_event])
            metrics["event_fva_mae"] = (
                metrics["event_baseline_mae"] - metrics["event_final_mae"]
            )
        else:
            metrics["event_n_days"] = 0
        if (~is_event).any():
            metrics["nonevent_n_days"] = int((~is_event).sum())
            metrics["nonevent_baseline_mae"] = mae(a[~is_event], b[~is_event])
            metrics["nonevent_final_mae"] = mae(a[~is_event], f[~is_event])
            metrics["nonevent_fva_mae"] = (
                metrics["nonevent_baseline_mae"] - metrics["nonevent_final_mae"]
            )

    return metrics


def evaluate_batch_run(run_dir: Path) -> pd.DataFrame:
    """data/runs/<run_id>/ 디렉토리의 모든 SKU 결과 평가.

    저장:
        run_dir/evaluation.parquet  (SKU별 metrics)
        run_dir/evaluation.txt      (aggregate 요약)
    """
    summary = pd.read_parquet(run_dir / "summary.parquet")
    ok_skus = summary[summary["status"] == "ok"]["sku_id"].tolist()

    event_dates = _load_event_dates()
    comparison_dir = run_dir / "comparison"

    rows: list[dict] = []
    for sku_id in ok_skus:
        fc_path = run_dir / "forecasts" / f"{sku_id}.parquet"
        if not fc_path.exists():
            continue
        try:
            rows.append(
                evaluate_sku(
                    sku_id,
                    fc_path,
                    event_dates=event_dates,
                    save_comparison_dir=comparison_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"평가 실패 {sku_id}: {exc}")

    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "evaluation.parquet", index=False)

    # Aggregate (간단 텍스트 요약)
    if df.empty:
        text = "(평가 가능 SKU 없음)\n"
    else:
        text = (
            f"=== {run_dir.name} 평가 ===\n"
            f"n_skus: {len(df)}\n"
            f"actual_total:   {df['actual_total'].sum():,.0f}\n"
            f"baseline_total: {df['baseline_total'].sum():,.0f}\n"
            f"final_total:    {df['final_total'].sum():,.0f}\n"
            f"\n"
            f"avg_baseline_mae:  {df['baseline_mae'].mean():.3f}\n"
            f"avg_final_mae:     {df['final_mae'].mean():.3f}\n"
            f"avg_fva_mae:       {df['fva_mae'].mean():+.3f}  (양수=개선)\n"
            f"\n"
            f"avg_baseline_smape: {df['baseline_smape'].mean():.2f}%\n"
            f"avg_final_smape:    {df['final_smape'].mean():.2f}%\n"
            f"avg_fva_smape:      {df['fva_smape'].mean():+.2f}%p  (양수=개선)\n"
            f"\n"
            f"SKU 개선:    {int((df['fva_mae'] > 0).sum())} / {len(df)}\n"
            f"SKU 악화:    {int((df['fva_mae'] < 0).sum())} / {len(df)}\n"
            f"SKU 변화없음: {int((df['fva_mae'] == 0).sum())} / {len(df)}\n"
        )
        if "event_baseline_mae" in df.columns:
            ev = df.dropna(subset=["event_baseline_mae"])
            if not ev.empty:
                text += (
                    f"\n=== 이벤트일 ({ev['event_n_days'].sum():.0f}일 in 평가) ===\n"
                    f"avg_event_baseline_mae: {ev['event_baseline_mae'].mean():.3f}\n"
                    f"avg_event_final_mae:    {ev['event_final_mae'].mean():.3f}\n"
                    f"avg_event_fva_mae:      {ev['event_fva_mae'].mean():+.3f}\n"
                )
        if "nonevent_baseline_mae" in df.columns:
            nev = df.dropna(subset=["nonevent_baseline_mae"])
            if not nev.empty:
                text += (
                    f"\n=== 비-이벤트일 ({nev['nonevent_n_days'].sum():.0f}일 in 평가) ===\n"
                    f"avg_nonevent_baseline_mae: {nev['nonevent_baseline_mae'].mean():.3f}\n"
                    f"avg_nonevent_final_mae:    {nev['nonevent_final_mae'].mean():.3f}\n"
                    f"avg_nonevent_fva_mae:      {nev['nonevent_fva_mae'].mean():+.3f}\n"
                )

    (run_dir / "evaluation.txt").write_text(text, encoding="utf-8")
    print(text)
    return df
