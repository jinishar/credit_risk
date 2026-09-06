"""Loads the application dataset (real if present, else synthetic) into DuckDB.

Priority order for data/application_train.csv / application_test.csv:
  1. Real files already sitting in DATA_DIR (e.g. dropped in from Kaggle) - used as-is.
  2. Otherwise, src.data.synthetic generates schema-accurate synthetic files.

Nothing downstream (preprocessing, training, the chatbot) needs to know or
care which source produced the CSVs.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.data import synthetic
from src.utils.config import (
    APPLICATIONS_TABLE,
    DUCKDB_PATH,
    TEST_CSV,
    TRAIN_CSV,
)
from src.utils.docker_utils import ensure_runtime_dirs
from src.utils.logger import get_logger

log = get_logger(__name__)


def ensure_data_available() -> None:
    """Generate synthetic CSVs if no dataset (real or previously-generated) exists yet."""
    ensure_runtime_dirs()
    if not TRAIN_CSV.exists():
        synthetic.generate_and_save()
    else:
        log.info("Using existing dataset at %s", TRAIN_CSV)


def load_train_df() -> pd.DataFrame:
    ensure_data_available()
    return pd.read_csv(TRAIN_CSV)


def load_test_df() -> pd.DataFrame:
    ensure_data_available()
    if TEST_CSV.exists():
        return pd.read_csv(TEST_CSV)
    log.warning("%s not found; returning an empty test frame.", TEST_CSV)
    return pd.DataFrame()


def get_duckdb_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with the `applications` table loaded from the train CSV.

    Used by both the ML EDA notebook and the NL-to-SQL chatbot so both query
    the exact same data.
    """
    ensure_data_available()
    con = duckdb.connect(str(DUCKDB_PATH), read_only=False)
    con.execute(
        f"CREATE OR REPLACE TABLE {APPLICATIONS_TABLE} AS "
        f"SELECT * FROM read_csv_auto(?, ALL_VARCHAR=FALSE)",
        [str(TRAIN_CSV)],
    )
    log.info("Loaded %s into DuckDB table '%s'", TRAIN_CSV, APPLICATIONS_TABLE)
    return con


if __name__ == "__main__":
    df = load_train_df()
    print(df.shape)
    con = get_duckdb_connection()
    print(con.execute(f"SELECT COUNT(*), AVG(TARGET) FROM {APPLICATIONS_TABLE}").fetchall())
