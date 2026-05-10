"""Baseline forecast backends.

Each backend module exposes a single function:

    forecast_for_demo(
        sku_id: str,
        input_window_dates: list,
        input_window_values: list[float],
        horizon: int = 28,
    ) -> tuple[list[date], np.ndarray]

The dispatcher in `src.forecasting.model` picks a backend at runtime based
on the `BASELINE_MODEL` env var (default `"lstm"`).
"""
