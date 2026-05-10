"""M5 raw CSV → DuckDB + parquet 정제 파이프라인.

실행:
    python -m src.data.prepare

산출물:
    data/knowledge_base.duckdb  (sales_train, prices, sku_metadata)
    data/test_set.parquet       (2016-01-01 ~ 2016-06-19 정답지)
    data/calendar.parquet       (이벤트·SNAP·요일)

주의:
    sales_train_evaluation.csv 는 train/test 양 기간을 모두 담고 있어
    여기서 TRAIN_END_DATE 기준으로 분리한다. test 기간 sales는 DuckDB에
    적재하지 않고 parquet으로만 보관해 에이전트의 누수를 막는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.config import (
    CALENDAR_PARQUET,
    DUCKDB_PATH,
    RAW_DIR,
    TEST_END_DATE,
    TEST_PARQUET,
    TEST_START_DATE,
    TRAIN_END_DATE,
)

SCHEMA_SQL = Path(__file__).parent / "schema.sql"
REQUIRED_FILES = ("sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv")


def _check_raw_files() -> None:
    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"data/raw/에 누락된 파일: {missing}. Kaggle M5 데이터를 받아 두어야 한다."
        )


def _init_db(con: duckdb.DuckDBPyConnection) -> None:
    for tbl in ("sales_train", "prices", "sku_metadata", "calendar"):
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    con.execute(SCHEMA_SQL.read_text(encoding="utf-8"))


def _prepare_calendar(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """calendar.csv → parquet + DuckDB calendar 테이블 + d→date 매핑 반환."""
    cal_csv = (RAW_DIR / "calendar.csv").as_posix()
    cal = con.execute(f"SELECT * FROM read_csv_auto('{cal_csv}')").df()
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    cal.to_parquet(CALENDAR_PARQUET, index=False)
    print(f"  calendar.parquet 저장 ({len(cal):,}행)")

    con.register("_calendar_src", cal)
    con.execute(
        """
        INSERT INTO calendar
        SELECT
            date, wm_yr_wk, weekday, wday, month, year, d,
            event_name_1, event_type_1, event_name_2, event_type_2,
            snap_CA, snap_TX, snap_WI
        FROM _calendar_src
        """
    )
    con.unregister("_calendar_src")
    n_cal = con.execute("SELECT COUNT(*) FROM calendar").fetchone()[0]
    print(f"  calendar 테이블 적재 ({n_cal:,}행)")
    return cal[["d", "date"]].copy()


def _prepare_sales(con: duckdb.DuckDBPyConnection, day_to_date: pd.DataFrame) -> None:
    sales_csv = (RAW_DIR / "sales_train_evaluation.csv").as_posix()

    print("  sales wide CSV 로드 중...")
    con.execute(
        f"CREATE OR REPLACE TEMPORARY TABLE sales_wide AS "
        f"SELECT * FROM read_csv_auto('{sales_csv}')"
    )

    day_cols = [
        c for c in con.execute("SELECT * FROM sales_wide LIMIT 0").df().columns
        if c.startswith("d_")
    ]
    print(f"  일자 컬럼 {len(day_cols)}개 인식")

    con.register("day_to_date", day_to_date)

    print("  wide → long UNPIVOT + 날짜 매핑 중...")
    day_list = ", ".join(day_cols)
    con.execute(
        f"""
        CREATE OR REPLACE TEMPORARY TABLE sales_long AS
        SELECT
            (sw.item_id || '_' || sw.store_id) AS sku_id,
            sw.item_id, sw.dept_id, sw.cat_id, sw.store_id, sw.state_id,
            CAST(dt.date AS DATE) AS date,
            CAST(sw.sales AS INTEGER) AS sales
        FROM (
            UNPIVOT sales_wide ON {day_list} INTO NAME d VALUE sales
        ) sw
        JOIN day_to_date dt ON sw.d = dt.d
        """
    )

    n_long = con.execute("SELECT COUNT(*) FROM sales_long").fetchone()[0]
    print(f"  long format {n_long:,}행 생성")

    print(f"  train (≤ {TRAIN_END_DATE}) 적재 중...")
    con.execute(
        f"""
        INSERT INTO sales_train
        SELECT sku_id, date, sales
        FROM sales_long
        WHERE date <= DATE '{TRAIN_END_DATE}'
        """
    )
    n_train = con.execute("SELECT COUNT(*) FROM sales_train").fetchone()[0]
    print(f"  sales_train {n_train:,}행")

    print(f"  test ({TEST_START_DATE} ~ {TEST_END_DATE}) parquet 저장 중...")
    con.execute(
        f"""
        COPY (
            SELECT sku_id, date, sales
            FROM sales_long
            WHERE date >= DATE '{TEST_START_DATE}'
              AND date <= DATE '{TEST_END_DATE}'
        ) TO '{TEST_PARQUET.as_posix()}' (FORMAT 'parquet')
        """
    )
    n_test = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{TEST_PARQUET.as_posix()}')"
    ).fetchone()[0]
    print(f"  test_set.parquet {n_test:,}행")

    print("  sku_metadata 적재 중...")
    con.execute(
        """
        INSERT INTO sku_metadata
        SELECT DISTINCT sku_id, item_id, cat_id, dept_id, store_id, state_id
        FROM sales_long
        """
    )
    n_meta = con.execute("SELECT COUNT(*) FROM sku_metadata").fetchone()[0]
    print(f"  sku_metadata {n_meta:,}행")


def _prepare_prices(con: duckdb.DuckDBPyConnection) -> None:
    prices_csv = (RAW_DIR / "sell_prices.csv").as_posix()
    print("  sell_prices.csv 적재 중...")
    con.execute(
        f"""
        INSERT INTO prices
        SELECT
            (item_id || '_' || store_id) AS sku_id,
            CAST(wm_yr_wk AS INTEGER) AS week,
            CAST(sell_price AS FLOAT) AS sell_price
        FROM read_csv_auto('{prices_csv}')
        """
    )
    n = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"  prices {n:,}행")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    _check_raw_files()
    print(f"DuckDB 초기화: {DUCKDB_PATH}")
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        _init_db(con)
        day_to_date = _prepare_calendar(con)
        _prepare_sales(con, day_to_date)
        _prepare_prices(con)
        print("완료.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
