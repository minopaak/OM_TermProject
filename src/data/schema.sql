-- DuckDB schema for M5 train data.
-- Test 기간(2016-01 ~ 2016-06)은 이 DB에 적재하지 않는다.

CREATE TABLE IF NOT EXISTS sales_train (
    sku_id  VARCHAR,
    date    DATE,
    sales   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sales_sku_date ON sales_train(sku_id, date);
CREATE INDEX IF NOT EXISTS idx_sales_date     ON sales_train(date);

CREATE TABLE IF NOT EXISTS prices (
    sku_id     VARCHAR,
    week       INTEGER,
    sell_price FLOAT
);
CREATE INDEX IF NOT EXISTS idx_prices_sku ON prices(sku_id);

CREATE TABLE IF NOT EXISTS sku_metadata (
    sku_id   VARCHAR PRIMARY KEY,
    item_id  VARCHAR,
    cat_id   VARCHAR,
    dept_id  VARCHAR,
    store_id VARCHAR,
    state_id VARCHAR
);

CREATE TABLE IF NOT EXISTS calendar (
    date         DATE,
    wm_yr_wk     INTEGER,
    weekday      VARCHAR,
    wday         INTEGER,
    month        INTEGER,
    year         INTEGER,
    d            VARCHAR,
    event_name_1 VARCHAR,
    event_type_1 VARCHAR,
    event_name_2 VARCHAR,
    event_type_2 VARCHAR,
    snap_CA      INTEGER,
    snap_TX      INTEGER,
    snap_WI      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_calendar_date ON calendar(date);
