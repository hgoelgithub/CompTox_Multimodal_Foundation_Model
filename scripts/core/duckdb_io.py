"""DuckDB access to Parquet mirrors of the normalized CompTox tables.

The one-off MySQL export (01a/01b) and the EPA normalization step (01) write
data/normalized/*.csv. Repeated lookups against those CSVs -- the training
cohort builder and, especially, the per-chemical interactive query -- used to
reload the full file into pandas on every call. This module mirrors each CSV
to Parquet and exposes it as a view in a small on-disk DuckDB database, so
callers can filter/project with SQL instead of loading whole tables.
"""
from pathlib import Path
import duckdb
import pandas as pd
from core.paths import DATA_NORMALIZED

PARQUET_DIR = DATA_NORMALIZED / "parquet"
DUCKDB_PATH = DATA_NORMALIZED / "comptox.duckdb"

TABLES = ("chemicals", "bioactivity", "hazard", "exposure", "chemical_identifiers")


def csv_path(name: str) -> Path:
    return DATA_NORMALIZED / f"{name}.csv"


def parquet_path(name: str) -> Path:
    return PARQUET_DIR / f"{name}.parquet"


def refresh(tables=TABLES) -> dict:
    """Rewrite the Parquet mirrors and DuckDB views from the current normalized CSVs.

    Returns {table_name: row_count} for every CSV that was found.
    """
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in tables:
        src = csv_path(name)
        if not src.exists():
            continue
        df = pd.read_csv(src, low_memory=False)
        df.to_parquet(parquet_path(name), index=False)
        written[name] = len(df)
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        for name in written:
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{parquet_path(name).as_posix()}')"
            )
    finally:
        con.close()
    return written


def connect(read_only=True):
    """Open the DuckDB database of views over the normalized Parquet tables.

    Builds it on first use if it doesn't exist yet.
    """
    if not DUCKDB_PATH.exists():
        refresh()
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def query_df(sql: str, params=None) -> pd.DataFrame:
    """Run a SQL query against the normalized views and return a DataFrame."""
    con = connect()
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def table_available(name: str) -> bool:
    return parquet_path(name).exists()
