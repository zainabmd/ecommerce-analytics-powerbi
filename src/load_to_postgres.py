"""
Load generated Parquet files into PostgreSQL.

Pipeline
--------
1. Validate the batch against the data contract (fail before touching the DB).
2. (Re)create the schema                                  sql/01_schema.sql
3. COPY each table in foreign-key dependency order.
4. Build indexes after the load (much faster than during)  sql/02_indexes.sql
5. Create the semantic layer / materialised views          sql/03_analytics_views.sql
6. Reconcile row counts source vs. target.

Usage
-----
    export DATABASE_URL=postgresql+psycopg2://ecom:ecom@localhost:5432/ecommerce
    python -m src.load_to_postgres
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import DB_SCHEMA, RAW_DIR, SQL_DIR, get_logger
from src.data_quality import DataQualityError, validate_tables
from src.db import DatabaseUnavailable, copy_dataframe, get_engine, read_sql, run_sql_file

log = get_logger("load_to_postgres")

# Foreign-key safe order.
LOAD_ORDER = ["dim_country", "dim_date", "products", "customers", "orders", "order_items", "web_sessions"]


def read_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [t for t in LOAD_ORDER if not (raw_dir / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing Parquet files in {raw_dir}: {missing}. Run `python -m src.generate_data` first.")
    return {t: pd.read_parquet(raw_dir / f"{t}.parquet") for t in LOAD_ORDER}


def reconcile(engine, tables: dict[str, pd.DataFrame]) -> None:
    """Compare source and target row counts; raise if anything was dropped."""
    problems = []
    for name, df in tables.items():
        n_db = int(read_sql(f"SELECT COUNT(*) AS n FROM {DB_SCHEMA}.{name}", engine)["n"].iloc[0])
        status = "OK" if n_db == len(df) else "MISMATCH"
        log.info("  %-14s source=%10s  db=%10s  %s", name, f"{len(df):,}", f"{n_db:,}", status)
        if n_db != len(df):
            problems.append(name)
    if problems:
        raise RuntimeError(f"Row-count reconciliation failed for: {problems}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Load Parquet extracts into PostgreSQL.")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--skip-validation", action="store_true", help="skip the pre-load data contract checks")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    try:
        tables = read_raw(args.raw_dir)
        if not args.skip_validation:
            validate_tables(tables)
        engine = get_engine()
    except (FileNotFoundError, DataQualityError, DatabaseUnavailable) as exc:
        log.error("%s", exc)
        return 1

    try:
        run_sql_file(engine, SQL_DIR / "01_schema.sql")
        for name in LOAD_ORDER:
            t = time.perf_counter()
            n = copy_dataframe(engine, tables[name], name)
            log.info("COPY %-14s %10s rows in %5.1fs", name, f"{n:,}", time.perf_counter() - t)
        run_sql_file(engine, SQL_DIR / "02_indexes.sql")
        run_sql_file(engine, SQL_DIR / "03_analytics_views.sql")
        reconcile(engine, tables)
    except Exception as exc:  # noqa: BLE001 – surface any DB error with context, then exit non-zero
        log.exception("Load failed: %s", exc)
        return 1

    log.info("Warehouse load complete in %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
