"""Baseline forecast facade.

Dispatches `forecast_for_demo` to the configured backend. Backends live
under `src.forecasting.baselines.<name>` and all expose the same signature:

    forecast_for_demo(
        sku_id, input_window_dates, input_window_values, horizon=28
    ) -> (forecast_dates: list[date], forecast_values: np.ndarray)

Selection: env var `BASELINE_MODEL`
    "lstm"  (default) → src.forecasting.baselines.lstm
    "arima"           → src.forecasting.baselines.arima

Per-SKU caches are stored in separate folders so backends never overlap.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np

from src.forecasting.baselines import arima as arima_backend
from src.forecasting.baselines import lstm as lstm_backend

_BACKENDS = {
    "lstm": lstm_backend,
    "arima": arima_backend,
}


def _selected_backend():
    name = os.environ.get("BASELINE_MODEL", "lstm").strip().lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown BASELINE_MODEL '{name}'. Allowed: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]


def forecast_for_demo(
    sku_id: str,
    input_window_dates: list,
    input_window_values: list[float],
    horizon: int = 28,
) -> tuple[list[date], np.ndarray]:
    """Forecast horizon days following the input window.

    Backend is selected by `BASELINE_MODEL` env var (default `"lstm"`).
    """
    return _selected_backend().forecast_for_demo(
        sku_id=sku_id,
        input_window_dates=input_window_dates,
        input_window_values=input_window_values,
        horizon=horizon,
    )


def get_backend_name() -> str:
    """Return the active backend name for logging."""
    return os.environ.get("BASELINE_MODEL", "lstm").strip().lower()
