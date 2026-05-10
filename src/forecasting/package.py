"""PredictionPackage 데이터 구조 + builder.

한 SKU에 대한 시연 1건 = (input window 28일 actual, forecast 28일 predicted).
모든 에이전트의 공통 입력.

사용:
    from src.forecasting.package import build_prediction_package

    pkg = build_prediction_package(
        sku_id="FOODS_3_090_CA_3",
        input_end_date=date(2016, 1, 28),  # input window 마지막 날
    )
    # pkg.input_window: 2016-01-01 ~ 01-28 (test, actual)
    # pkg.forecast_window: 2016-01-29 ~ 02-25 (test 안, ARIMA forecast)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import pandas as pd

from src.config import DUCKDB_PATH, TEST_PARQUET, TRAIN_END_DATE
from src.forecasting.model import forecast_for_demo


@dataclass
class TimeWindow:
    """시계열 한 구간. dates와 values 길이는 동일."""
    dates: list[date]
    values: list[float]

    def __len__(self) -> int:
        return len(self.dates)


@dataclass
class PredictionPackage:
    """시연 1건의 입력 단위. 모든 에이전트가 공유."""
    sku_id: str
    item_id: str
    cat_id: str
    dept_id: str
    store_id: str
    state_id: str
    input_window: TimeWindow      # 실제 관측치
    forecast_window: TimeWindow   # ARIMA 예측치


def _load_metadata(con: duckdb.DuckDBPyConnection, sku_id: str) -> dict:
    df = con.execute(
        f"""
        SELECT item_id, cat_id, dept_id, store_id, state_id
        FROM sku_metadata
        WHERE sku_id = '{sku_id}'
        """
    ).df()
    if df.empty:
        raise ValueError(f"SKU not found in sku_metadata: {sku_id}")
    return df.iloc[0].to_dict()


def _load_actual_window(
    con: duckdb.DuckDBPyConnection,
    sku_id: str,
    start_date: date,
    end_date: date,
) -> tuple[list[date], list[float]]:
    """[start_date, end_date] 사이 실제 sales를 train 또는 test에서 로드.

    train_end 이전이면 sales_train, 이후면 test_set.parquet, 걸쳐 있으면 둘 다 union.
    """
    train_end = pd.Timestamp(TRAIN_END_DATE).date()

    if end_date <= train_end:
        df = con.execute(
            f"""
            SELECT date, CAST(sales AS DOUBLE) AS y
            FROM sales_train
            WHERE sku_id = '{sku_id}'
              AND date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            ORDER BY date
            """
        ).df()
    elif start_date > train_end:
        df = con.execute(
            f"""
            SELECT date, CAST(sales AS DOUBLE) AS y
            FROM read_parquet('{TEST_PARQUET.as_posix()}')
            WHERE sku_id = '{sku_id}'
              AND date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            ORDER BY date
            """
        ).df()
    else:
        df_train = con.execute(
            f"""
            SELECT date, CAST(sales AS DOUBLE) AS y
            FROM sales_train
            WHERE sku_id = '{sku_id}'
              AND date BETWEEN DATE '{start_date}' AND DATE '{train_end}'
            """
        ).df()
        df_test = con.execute(
            f"""
            SELECT date, CAST(sales AS DOUBLE) AS y
            FROM read_parquet('{TEST_PARQUET.as_posix()}')
            WHERE sku_id = '{sku_id}'
              AND date BETWEEN DATE '{train_end + timedelta(days=1)}' AND DATE '{end_date}'
            """
        ).df()
        df = pd.concat([df_train, df_test], ignore_index=True).sort_values("date")

    expected = (end_date - start_date).days + 1
    if len(df) != expected:
        raise ValueError(
            f"actual window length mismatch for {sku_id} "
            f"({start_date}~{end_date}): expected {expected}, got {len(df)}"
        )

    dates = [pd.Timestamp(d).date() for d in df["date"].tolist()]
    values = [float(v) for v in df["y"].tolist()]
    return dates, values


def build_prediction_package(
    sku_id: str,
    input_end_date: date,
    input_length: int = 28,
    horizon: int = 28,
) -> PredictionPackage:
    """단일 SKU의 PredictionPackage 생성.

    Args:
        sku_id: 대상 SKU.
        input_end_date: input window 마지막 날 (inclusive).
                        forecast는 input_end_date + 1일부터 horizon일.
        input_length: input window 길이 (default 28).
        horizon: forecast 길이 (default 28).
    """
    input_start = input_end_date - timedelta(days=input_length - 1)

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        meta = _load_metadata(con, sku_id)
        in_dates, in_values = _load_actual_window(
            con, sku_id, input_start, input_end_date
        )
    finally:
        con.close()

    fc_dates, fc_values = forecast_for_demo(
        sku_id=sku_id,
        input_window_dates=in_dates,
        input_window_values=in_values,
        horizon=horizon,
    )

    return PredictionPackage(
        sku_id=sku_id,
        item_id=meta["item_id"],
        cat_id=meta["cat_id"],
        dept_id=meta["dept_id"],
        store_id=meta["store_id"],
        state_id=meta["state_id"],
        input_window=TimeWindow(dates=in_dates, values=in_values),
        forecast_window=TimeWindow(dates=fc_dates, values=[float(v) for v in fc_values]),
    )
