"""
Data-quality contract enforced on every generated (or ingested) batch.

The same checks mirror the PostgreSQL constraints in `sql/01_schema.sql`, so a
batch that passes here will also load cleanly into the warehouse. Failing fast
in Python gives a readable error instead of a half-loaded database.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_logger

log = get_logger("data_quality")

PRIMARY_KEYS = {
    "dim_country": "country_code",
    "dim_date": "date_key",
    "customers": "customer_id",
    "products": "product_id",
    "orders": "order_id",
    "order_items": "order_item_id",
    "web_sessions": "session_id",
}

# (child_table, child_col, parent_table, parent_col, nullable)
FOREIGN_KEYS = [
    ("customers", "country_code", "dim_country", "country_code", False),
    ("orders", "customer_id", "customers", "customer_id", False),
    ("orders", "country_code", "dim_country", "country_code", False),
    ("order_items", "order_id", "orders", "order_id", False),
    ("order_items", "product_id", "products", "product_id", False),
    ("web_sessions", "customer_id", "customers", "customer_id", True),
    ("web_sessions", "order_id", "orders", "order_id", True),
]

NON_NEGATIVE = {
    "orders": ["gross_amount", "discount_amount", "shipping_amount", "net_amount", "cogs_amount"],
    "order_items": ["unit_price", "discount_amount", "line_revenue", "line_cogs"],
    "web_sessions": ["duration_sec"],
}

VALID_STATUSES = {"completed", "returned", "cancelled"}


class DataQualityError(Exception):
    """Raised when a batch violates the data contract."""


def _check_pk(name: str, df: pd.DataFrame, col: str, errors: list[str]) -> None:
    if df[col].isna().any():
        errors.append(f"{name}.{col}: NULL primary key values")
    dupes = int(df[col].duplicated().sum())
    if dupes:
        errors.append(f"{name}.{col}: {dupes:,} duplicate primary keys")


def _check_fk(tables: dict[str, pd.DataFrame], spec: tuple, errors: list[str]) -> None:
    child, ccol, parent, pcol, nullable = spec
    if child not in tables or parent not in tables:
        return
    values = tables[child][ccol]
    if not nullable and values.isna().any():
        errors.append(f"{child}.{ccol}: NULLs in non-nullable foreign key")
    values = values.dropna()
    orphans = int((~values.isin(tables[parent][pcol])).sum())
    if orphans:
        errors.append(f"{child}.{ccol} -> {parent}.{pcol}: {orphans:,} orphan rows")


def validate_tables(tables: dict[str, pd.DataFrame], tolerance: float = 0.011) -> None:
    """Run all checks; raise DataQualityError listing *every* violation found."""
    errors: list[str] = []

    for name, col in PRIMARY_KEYS.items():
        if name in tables:
            _check_pk(name, tables[name], col, errors)

    for spec in FOREIGN_KEYS:
        _check_fk(tables, spec, errors)

    for name, cols in NON_NEGATIVE.items():
        if name not in tables:
            continue
        for col in cols:
            n_bad = int((tables[name][col] < 0).sum())
            if n_bad:
                errors.append(f"{name}.{col}: {n_bad:,} negative values")

    if "orders" in tables:
        o = tables["orders"]
        recomputed = o["gross_amount"] - o["discount_amount"] + o["shipping_amount"]
        n_bad = int((np.abs(recomputed - o["net_amount"]) > tolerance).sum())
        if n_bad:
            errors.append(f"orders.net_amount: {n_bad:,} rows != gross - discount + shipping")
        if (o["discount_amount"] > o["gross_amount"] + tolerance).any():
            errors.append("orders.discount_amount exceeds gross_amount")
        bad_status = set(o["order_status"].unique()) - VALID_STATUSES
        if bad_status:
            errors.append(f"orders.order_status: unexpected values {sorted(bad_status)}")

    if "orders" in tables and "order_items" in tables:
        line_sum = tables["order_items"].groupby("order_id")["line_revenue"].sum()
        o = tables["orders"].set_index("order_id")
        expected = (o["gross_amount"] - o["discount_amount"]).reindex(line_sum.index)
        n_bad = int((np.abs(expected - line_sum) > 0.05).sum())
        if n_bad:
            errors.append(f"order_items: {n_bad:,} orders whose line revenue doesn't reconcile to header")

    if "web_sessions" in tables:
        s = tables["web_sessions"]
        mismatch = int((s["converted"] != s["order_id"].notna()).sum())
        if mismatch:
            errors.append(f"web_sessions: {mismatch:,} rows where converted != (order_id IS NOT NULL)")
        if (s["pageviews"] < 1).any():
            errors.append("web_sessions.pageviews < 1")

    if errors:
        raise DataQualityError("; ".join(errors))
    log.info("Data quality: all checks passed across %d tables", len(tables))
