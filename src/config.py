"""중앙 경로/설정 상수."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DUCKDB_PATH = DATA_DIR / "knowledge_base.duckdb"
TEST_PARQUET = DATA_DIR / "test_set.parquet"
CALENDAR_PARQUET = DATA_DIR / "calendar.parquet"

TRAIN_END_DATE = "2015-12-31"
TEST_START_DATE = "2016-01-01"
TEST_END_DATE = "2016-05-22"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
AGENT_MAX_SQL_CALLS = int(os.getenv("AGENT_MAX_SQL_CALLS", "10"))

PATTERN_ANALYST_MODEL = os.getenv("PATTERN_ANALYST_MODEL", "gpt-4o")
