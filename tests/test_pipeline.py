"""
Unit + integration tests. Run with:  pytest -q

The integration fixture generates a small (2k-customer) dataset once per
session, so the whole suite runs in seconds.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import SimulationConfig
from src.data_quality import DataQualityError, validate_tables
from src.generate_data import _sample_days_in_windows, generate
from src.rfm_segmentation import (compute_rfm, estimate_clv, name_clusters, persona_for, score_rfm, select_k,
                                  build_pipeline)


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    cfg = replace(SimulationConfig(), n_customers=2_000, n_products=200, seed=7)
    return generate(cfg)


@pytest.fixture(scope="session")
def rfm(tables) -> pd.DataFrame:
    o = tables["orders"]
    o = o[o["order_status"] == "completed"][["customer_id", "order_date", "net_amount"]]
    return estimate_clv(score_rfm(compute_rfm(o, snapshot=date(2024, 1, 1))))


# ---------------------------------------------------------------- generator ---
def test_generator_is_deterministic():
    cfg = replace(SimulationConfig(), n_customers=300, n_products=50, seed=1)
    a, b = generate(cfg), generate(cfg)
    pd.testing.assert_frame_equal(a["orders"], b["orders"])


def test_generated_data_passes_contract(tables):
    validate_tables({k: v for k, v in tables.items() if not k.startswith("_")})


def test_orders_within_window_and_after_signup(tables):
    o = tables["orders"].merge(tables["customers"][["customer_id", "signup_date"]], on="customer_id")
    assert o["order_date"].min() >= date(2021, 1, 1)
    assert o["order_date"].max() <= date(2023, 12, 31)
    assert (pd.to_datetime(o["order_date"]) >= pd.to_datetime(o["signup_date"])).all()


def test_products_not_sold_before_launch(tables):
    li = (tables["order_items"].merge(tables["orders"][["order_id", "order_date"]], on="order_id")
          .merge(tables["products"][["product_id", "launch_date"]], on="product_id"))
    assert (pd.to_datetime(li["order_date"]) >= pd.to_datetime(li["launch_date"])).all()


def test_holiday_seasonality_present(tables):
    o = tables["orders"]
    month = pd.to_datetime(o["order_date"]).dt.month
    assert (month == 11).sum() > (month == 2).sum(), "November should out-sell February"


def test_conversion_rate_close_to_target(tables):
    s = tables["web_sessions"]
    rate = (s["converted"] * s["sample_weight"]).sum() / s["sample_weight"].sum()
    assert 0.025 < rate < 0.04


def test_window_sampler_respects_bounds():
    rng = np.random.default_rng(0)
    cdf = np.concatenate([[0.0], np.cumsum(np.ones(100))])
    lo = rng.integers(0, 50, 10_000)
    hi = lo + rng.integers(0, 50, 10_000)
    d = _sample_days_in_windows(rng, cdf, lo, hi)
    assert ((d >= lo) & (d <= hi)).all()


# -------------------------------------------------------------- data quality ---
def test_contract_catches_orphans_and_bad_totals(tables):
    bad = {k: v.copy() for k, v in tables.items() if k in ("customers", "orders", "dim_country")}
    bad["orders"].loc[bad["orders"].index[0], "customer_id"] = -1
    bad["orders"].loc[bad["orders"].index[1], "net_amount"] += 100
    with pytest.raises(DataQualityError) as exc:
        validate_tables(bad)
    assert "orphan" in str(exc.value) and "net_amount" in str(exc.value)


# ---------------------------------------------------------------------- RFM ---
def test_rfm_basic_math():
    orders = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "order_date": [date(2023, 12, 1), date(2023, 12, 21), date(2023, 6, 1)],
        "net_amount": [10.0, 30.0, 5.0],
    })
    r = compute_rfm(orders, snapshot=date(2024, 1, 1)).set_index("customer_id")
    assert r.loc[1, "recency_days"] == 11 and r.loc[1, "frequency"] == 2 and r.loc[1, "monetary"] == 40
    assert r.loc[2, "frequency"] == 1


def test_rfm_rejects_snapshot_before_data():
    orders = pd.DataFrame({"customer_id": [1], "order_date": [date(2024, 2, 1)], "net_amount": [1.0]})
    with pytest.raises(ValueError):
        compute_rfm(orders, snapshot=date(2024, 1, 1))


def test_scores_are_quintiles_and_segments_complete(rfm):
    for col in ("r_score", "f_score", "m_score"):
        assert set(rfm[col].unique()) <= {1, 2, 3, 4, 5}
        counts = rfm[col].value_counts()
        assert counts.max() - counts.min() <= 1  # rank-based qcut gives equal bins
    assert rfm["rfm_segment"].notna().all()
    # Lower recency should mean higher R score
    assert rfm.groupby("r_score")["recency_days"].median().is_monotonic_decreasing


def test_clv_bounds(rfm):
    assert rfm["p_alive"].between(0, 1).all()
    assert (rfm["predicted_clv_12m"] >= 0).all()


def test_kmeans_and_personas(rfm):
    X = rfm[["recency_days", "frequency", "monetary"]]
    best_k, diag = select_k(X, k_range=range(3, 6), sample=1000)
    assert best_k >= 4 and set(diag.columns) >= {"k", "inertia", "silhouette"}
    labels = build_pipeline(best_k).fit(X)["kmeans"].labels_
    mapping, prof = name_clusters(rfm, labels)
    assert len(mapping) == best_k and len(set(mapping.values())) == best_k  # names are unique
    # The highest value-score cluster must spend more (median) than the lowest
    assert prof.iloc[0]["monetary_median"] > prof.iloc[-1]["monetary_median"]


@pytest.mark.parametrize("r,f,m,expected", [
    (20, 8, 5000, "VIP Champions"),
    (20, 3, 300, "Loyal Regulars"),
    (400, 6, 5000, "Can't-Lose High-Value"),
    (400, 3, 200, "At-Risk Repeat Buyers"),
    (30, 1, 100, "New & Promising"),
    (500, 1, 900, "Lapsed Big-Ticket"),
    (500, 1, 50, "Dormant One-Timers"),
])
def test_persona_rules(r, f, m, expected):
    assert persona_for(r, f, m, pop_recency_med=200, pop_monetary_med=400, pop_monetary_p75=1500) == expected
