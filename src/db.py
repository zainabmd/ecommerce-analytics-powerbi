"""
Thin database access layer: engine creation with retry, SQL-file execution and
high-throughput bulk loading through PostgreSQL's COPY protocol.
"""
from __future__ import annotations

import io
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from src.config import DATABASE_URL, DB_SCHEMA, get_logger

log = get_logger("db")


class DatabaseUnavailable(RuntimeError):
    """Raised when DATABASE_URL is unset or the server cannot be reached."""


def get_engine(url: str | None = None, retries: int = 5, backoff_s: float = 2.0) -> Engine:
    """Create an engine and prove connectivity, retrying with exponential backoff.

    Retrying matters in containerised setups where the app may start before
    PostgreSQL finishes initialising.
    """
    url = url or DATABASE_URL
    if not url:
        raise DatabaseUnavailable("DATABASE_URL is not set")
    engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except OperationalError as exc:
            if attempt == retries:
                raise DatabaseUnavailable(f"Could not connect after {retries} attempts: {exc}") from exc
            wait = backoff_s * 2 ** (attempt - 1)
            log.warning("DB not ready (attempt %d/%d); retrying in %.0fs", attempt, retries, wait)
            time.sleep(wait)
    raise DatabaseUnavailable("unreachable")  # pragma: no cover


@contextmanager
def raw_cursor(engine: Engine) -> Iterator:
    """Yield a DB-API cursor inside a transaction (commit on success, rollback on error)."""
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_sql_file(engine: Engine, path: Path) -> None:
    """Execute a multi-statement SQL file (psycopg2 handles `;`-separated batches and $$ bodies)."""
    sql = path.read_text(encoding="utf-8")
    log.info("Executing %s", path.name)
    with raw_cursor(engine) as cur:
        cur.execute(sql)


def copy_dataframe(engine: Engine, df: pd.DataFrame, table: str, chunk_rows: int = 250_000) -> int:
    """Bulk-load a DataFrame with COPY ... FROM STDIN (10-50x faster than INSERTs).

    The frame is streamed in chunks so peak memory stays bounded regardless of
    table size. Column order is taken from the DataFrame, so it must match
    existing table columns by name.
    """
    cols = ", ".join(f'"{c}"' for c in df.columns)
    copy_sql = f'COPY {DB_SCHEMA}.{table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL \'\\N\')'
    loaded = 0
    with raw_cursor(engine) as cur:
        for start in range(0, len(df), chunk_rows):
            chunk = df.iloc[start:start + chunk_rows]
            buf = io.StringIO()
            chunk.to_csv(buf, index=False, header=False, na_rep="\\N", date_format="%Y-%m-%d %H:%M:%S")
            buf.seek(0)
            cur.copy_expert(copy_sql, buf)
            loaded += len(chunk)
    return loaded


def read_sql(query: str, engine: Engine | None = None, params: dict | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
