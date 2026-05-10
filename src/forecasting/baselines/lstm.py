"""LSTM baseline (PyTorch) — flat univariate (frozen design).

Per-SKU 1-layer LSTM trained on (up to) the last 1820 days of train, with
known-broken date ranges per SKU excluded. Single feature per timestep:
log1p + standardized value. 56-day window → 1-day prediction; recursive
28-step forecast.

Design rationale: the LSTM intentionally learns *level only*, not weekly
seasonality. The role of injecting weekly / event shape sits with the
LLM specialists in `src.agents.pattern_analyst`. Adding dow / month
features here caused double-counting because specialists already propose
level_weekday / level_weekend multipliers on top of the baseline.

Cache: `data/models/lstm_flat/{sku_id}.pt`.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import DATA_DIR, DUCKDB_PATH, TRAIN_END_DATE

CACHE_DIR = DATA_DIR / "models" / "lstm_flat"
LOOKBACK_DAYS = 1820
LOOKBACK = 56
HIDDEN = 64
LAYERS = 1
EPOCHS = 80
LR = 1e-3
TORCH_SEED = 1234
INPUT_FEATURES = 1  # value_z only — specialists handle weekly / event shape

SKU_EXCLUDE_RANGES: dict[str, list[tuple[str, str]]] = {
    "FOODS_3_295_CA_1": [
        ("2011-01-29", "2011-12-31"),
        ("2015-01-01", "2015-12-31"),
    ],
}


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES, hidden_size=hidden,
            num_layers=layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _load_train_series(sku_id: str) -> pd.Series:
    start_date = pd.Timestamp(TRAIN_END_DATE) - pd.Timedelta(days=LOOKBACK_DAYS - 1)
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute(
            """
            SELECT date, CAST(sales AS DOUBLE) AS y
            FROM sales_train
            WHERE sku_id = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [sku_id, str(start_date.date()), TRAIN_END_DATE],
        ).df()
    finally:
        con.close()
    if df.empty:
        raise ValueError(f"No train data for SKU: {sku_id}")

    df["date"] = pd.to_datetime(df["date"])
    for exc_start, exc_end in SKU_EXCLUDE_RANGES.get(sku_id, []):
        mask = (df["date"] >= pd.Timestamp(exc_start)) & (
            df["date"] <= pd.Timestamp(exc_end)
        )
        df = df.loc[~mask]

    if df.empty:
        raise ValueError(f"No train data left after exclusions for SKU: {sku_id}")

    return pd.Series(
        df["y"].values.astype(float),
        index=pd.DatetimeIndex(df["date"]),
        name="sales",
    )


def _make_windows(z_values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows. Single feature per timestep: value_z. y = next value."""
    n = len(z_values) - lookback
    if n <= 0:
        raise ValueError(f"series too short for lookback={lookback}")
    X = np.zeros((n, lookback, INPUT_FEATURES), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = z_values[i : i + lookback]
        y[i, 0] = z_values[i + lookback]
    return X, y


def _fit_one_sku(series: pd.Series) -> tuple[LSTMForecaster, float, float]:
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)

    raw = series.values.astype(np.float32)
    log = np.log1p(raw)
    mean = float(log.mean())
    std = float(log.std()) if log.std() > 1e-6 else 1.0
    z = (log - mean) / std

    X, y = _make_windows(z, LOOKBACK)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    model.eval()
    return model, mean, std


def _cache_path(sku_id: str) -> Path:
    return CACHE_DIR / f"{sku_id}.pt"


def get_fitted_model(sku_id: str) -> tuple[LSTMForecaster, float, float]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(sku_id)
    if cache.exists():
        ckpt = torch.load(cache, map_location="cpu", weights_only=False)
        model = LSTMForecaster()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, float(ckpt["mean"]), float(ckpt["std"])

    series = _load_train_series(sku_id)
    model, mean, std = _fit_one_sku(series)
    torch.save(
        {"state_dict": model.state_dict(), "mean": mean, "std": std},
        cache,
    )
    return model, mean, std


def _recursive_forecast(
    model: LSTMForecaster,
    seed_z: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Recursive 1-step-ahead forecast over `horizon` steps."""
    if len(seed_z) != LOOKBACK:
        raise ValueError(f"seed length {len(seed_z)} != LOOKBACK {LOOKBACK}")
    out = np.zeros(horizon, dtype=np.float32)
    window = seed_z.astype(np.float32).copy()
    with torch.no_grad():
        for h in range(horizon):
            x = window.reshape(1, LOOKBACK, INPUT_FEATURES)
            x_t = torch.from_numpy(x)
            pred = model(x_t).item()
            out[h] = pred
            window = np.roll(window, -1)
            window[-1] = pred
    return out


def forecast_for_demo(
    sku_id: str,
    input_window_dates: list,
    input_window_values: list[float],
    horizon: int = 28,
) -> tuple[list[date], np.ndarray]:
    model, mean, std = get_fitted_model(sku_id)

    in_dates = pd.to_datetime(input_window_dates).date.tolist()
    in_values = np.asarray(input_window_values, dtype=float)

    if len(in_values) >= LOOKBACK:
        seed_raw = in_values[-LOOKBACK:]
    else:
        n_need = LOOKBACK - len(in_values)
        first_in_date = pd.Timestamp(in_dates[0])
        end_date = first_in_date - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=n_need - 1)
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            pre_df = con.execute(
                """
                SELECT date, CAST(sales AS DOUBLE) AS y
                FROM sales_train
                WHERE sku_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                [sku_id, str(start_date.date()), str(end_date.date())],
            ).df()
        finally:
            con.close()
        pre = pre_df["y"].values.astype(float) if not pre_df.empty else np.zeros(n_need)
        if len(pre) < n_need:
            pre = np.concatenate([np.zeros(n_need - len(pre)), pre])
        seed_raw = np.concatenate([pre, in_values])[-LOOKBACK:]

    seed_z = (np.log1p(seed_raw) - mean) / std
    pred_z = _recursive_forecast(model, seed_z, horizon)
    pred_log = pred_z * std + mean
    values = np.clip(np.expm1(pred_log), a_min=0.0, a_max=None).astype(float)

    last_date = pd.Timestamp(in_dates[-1])
    forecast_dates = [(last_date + pd.Timedelta(days=i + 1)).date() for i in range(horizon)]
    return forecast_dates, values
