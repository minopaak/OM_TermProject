"""SARIMAX univariate baseline (statsmodels).

Per-SKU SARIMAX(1,1,1)(1,0,1,7) fit on last 730 days of train. State is
reapplied to the input window via `.apply(refit=False)` so that train
parameters stay fixed and only state is re-seeded with the input window.
SKU-level pickle cache under `data/models/arima/{sku_id}.pkl`.
"""
from __future__ import annotations

import pickle
import warnings
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from src.config import DATA_DIR, DUCKDB_PATH, TRAIN_END_DATE

CACHE_DIR = DATA_DIR / "models" / "arima"
LOOKBACK_DAYS = 730
ARIMA_ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 7)


def _load_train_series(con: duckdb.DuckDBPyConnection, sku_id: str) -> pd.Series:
    start_date = pd.Timestamp(TRAIN_END_DATE) - pd.Timedelta(days=LOOKBACK_DAYS - 1)
    df = con.execute(
        """
        SELECT date, CAST(sales AS DOUBLE) AS y
        FROM sales_train
        WHERE sku_id = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        [sku_id, str(start_date.date()), TRAIN_END_DATE],
    ).df()
    if df.empty:
        raise ValueError(f"No train data for SKU: {sku_id}")
    return pd.Series(
        df["y"].values.astype(float),
        index=pd.DatetimeIndex(pd.to_datetime(df["date"]), freq="D"),
        name="sales",
    )


def _fit_one_sku(train_series: pd.Series):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series,
            order=ARIMA_ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=50)


def get_fitted_model(sku_id: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{sku_id}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        series = _load_train_series(con, sku_id)
    finally:
        con.close()
    fitted = _fit_one_sku(series)
    with cache_path.open("wb") as f:
        pickle.dump(fitted, f)
    return fitted


def forecast_for_demo(
    sku_id: str,
    input_window_dates: list,
    input_window_values: list[float],
    horizon: int = 28,
) -> tuple[list[date], np.ndarray]:
    fitted = get_fitted_model(sku_id)
    input_series = pd.Series(
        np.asarray(input_window_values, dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(input_window_dates), freq="D"),
        name="sales",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extended = fitted.apply(input_series, refit=False)
        forecast = extended.forecast(steps=horizon)
    values = np.clip(np.asarray(forecast.values, dtype=float), a_min=0.0, a_max=None)
    dates = [ts.date() for ts in forecast.index]
    return dates, values
