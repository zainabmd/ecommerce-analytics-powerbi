"""
Synthetic global e-commerce data generator.

Produces a realistic, internally consistent dataset (~3M rows by default):

    dim_country   20 rows      ISO-3 countries with region, currency, price index
    dim_date      1,095 rows   calendar dimension
    products      2,000 rows   8 categories, Zipf-like popularity, staggered launches
    customers     100,000 rows latent behavioural types drive purchasing & churn
    orders        ~330,000     seasonality (weekly, Black Friday, Christmas), statuses
    order_items   ~930,000     category affinity, discounts, local price levels
    web_sessions  ~1,300,000   converting sessions + 1:N down-sampled non-converting

Design principles
-----------------
* **Fully vectorised NumPy** – no per-row Python loops over orders or sessions,
  so the default scale generates in well under a minute on a laptop.
* **Ground truth is behavioural, not random noise.** Each customer has a hidden
  type (whale / loyal / regular / occasional / one-time / never) that controls
  purchase rate and churn, so RFM + K-Means has real structure to recover.
* **Deterministic** – a single seed reproduces the exact dataset.
* **Validated** – every run passes `data_quality.validate_tables` before writing.

Usage
-----
    python -m src.generate_data                     # default scale
    python -m src.generate_data --customers 20000   # quick dev run
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ACQUISITION_CHANNELS,
    CATEGORIES,
    DEVICES,
    LANDING_PAGES,
    PAYMENT_METHODS,
    RAW_DIR,
    SimulationConfig,
    get_logger,
)
from src.data_quality import DataQualityError, validate_tables

log = get_logger("generate_data")

# Hidden behavioural archetypes: (share, orders/year, mean lifetime in months)
CUSTOMER_TYPES: dict[str, tuple[float, float, float]] = {
    "whale": (0.025, 18.0, 40.0),
    "loyal": (0.100, 9.0, 30.0),
    "regular": (0.250, 4.0, 18.0),
    "occasional": (0.280, 1.5, 10.0),
    "one_time": (0.220, 0.0, 1.0),
    "never": (0.125, 0.0, 0.0),
}
CHANNEL_RATE_MULTIPLIER = {"Email": 1.15, "Referral": 1.15, "Direct": 1.05, "Organic Search": 1.0,
                           "Paid Search": 0.95, "Social": 0.85, "Affiliate": 0.85}
BRANDS = ["Apex", "Nova", "Lumen", "Vertex", "Orbit", "Kairo", "Halden", "Brisk", "Sora", "Mistral",
          "Pinecrest", "Atlas", "Ember", "Quill", "Zephyr", "Cobalt", "Juniper", "Tundra", "Velo", "Ridge"]
AGE_BANDS = {"18-24": 0.16, "25-34": 0.31, "35-44": 0.24, "45-54": 0.15, "55-64": 0.09, "65+": 0.05}
GENDERS = {"Female": 0.51, "Male": 0.46, "Non-binary": 0.015, "Undisclosed": 0.015}
# Hour-of-day traffic shape (local time): low overnight, lunch bump, evening peak.
HOUR_WEIGHTS = np.array([0.6, 0.4, 0.3, 0.25, 0.25, 0.35, 0.6, 1.0, 1.4, 1.6, 1.7, 1.8,
                         2.1, 2.0, 1.8, 1.7, 1.8, 2.0, 2.4, 2.8, 3.0, 2.7, 1.9, 1.1])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _weighted_choice(rng: np.random.Generator, options: dict[str, float], size: int) -> np.ndarray:
    """Vectorised categorical draw from a {label: weight} mapping (weights auto-normalised)."""
    labels = np.array(list(options.keys()), dtype=object)
    p = np.array(list(options.values()), dtype=float)
    return rng.choice(labels, size=size, p=p / p.sum())


def _day_weights(days: pd.DatetimeIndex, trend: float, season_strength: float = 1.0) -> np.ndarray:
    """Relative daily demand: linear growth trend x weekly cycle x retail calendar.

    Black-Friday week and mid-December carry the largest peaks; January and
    late summer dip. `season_strength` < 1 damps seasonality (used for sign-ups).
    """
    t = (days - days[0]).days.to_numpy() / 365.25
    w = 1.0 + trend * t
    weekday = np.array([0.95, 0.93, 0.95, 1.00, 1.08, 1.12, 1.05])[days.day_of_week]
    month, dom = days.month.to_numpy(), days.day.to_numpy()
    season = np.ones(len(days))
    season[month == 1] = 0.82
    season[(month == 8)] = 0.92
    season[(month == 11) & (dom >= 22) & (dom <= 30)] = 2.3   # Black Friday / Cyber Monday window
    season[(month == 12) & (dom <= 20)] = 1.55                # Christmas shopping
    season[(month == 12) & (dom > 26)] = 1.20                 # post-Christmas sales
    season = 1.0 + season_strength * (season - 1.0)
    return w * weekday * season


def _sample_days_in_windows(rng: np.random.Generator, cdf: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Draw one day index per (lo, hi) inclusive window, proportional to daily weights.

    Uses inverse-CDF sampling on the cumulative weight curve so that millions of
    variable-width windows are sampled in a single vectorised call.
    `cdf` has length n_days + 1 with cdf[0] == 0.
    """
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    if np.any(hi < lo):
        raise ValueError("Window upper bound precedes lower bound")
    u = cdf[lo] + rng.random(len(lo)) * (cdf[hi + 1] - cdf[lo])
    idx = np.searchsorted(cdf, u, side="right") - 1
    return np.clip(idx, lo, hi)


def _random_times(rng: np.random.Generator, n: int) -> np.ndarray:
    """Seconds-into-day following the hour-of-day traffic profile."""
    hours = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    return hours * 3600 + rng.integers(0, 3600, size=n)


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def build_dim_country(cfg: SimulationConfig) -> pd.DataFrame:
    rows = [{"country_code": k, "country_name": v["name"], "region": v["region"],
             "currency": v["currency"], "price_index": v["price_index"]} for k, v in cfg.countries.items()]
    return pd.DataFrame(rows)


def build_dim_date(cfg: SimulationConfig) -> pd.DataFrame:
    days = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    holiday = ((days.month == 11) & (days.day >= 22)) | (days.month == 12) | ((days.month == 1) & (days.day == 1))
    return pd.DataFrame({
        "date_key": days.date,
        "year": days.year.astype("int16"),
        "quarter": days.quarter.astype("int16"),
        "month": days.month.astype("int16"),
        "month_name": days.month_name(),
        "week_of_year": days.isocalendar().week.to_numpy().astype("int16"),
        "day_of_week": days.day_of_week.astype("int16"),
        "is_weekend": days.day_of_week >= 5,
        "is_holiday_season": holiday,
    })


def build_products(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_products
    cat_names = list(CATEGORIES.keys())
    cat_w = np.array([CATEGORIES[c]["weight"] for c in cat_names])
    category = rng.choice(cat_names, size=n, p=cat_w / cat_w.sum())
    # Guarantee every category is represented even at tiny scales.
    category[: len(cat_names)] = cat_names

    subcategory = np.empty(n, dtype=object)
    list_price = np.empty(n)
    margin = np.empty(n)
    for c in cat_names:
        m = category == c
        k = int(m.sum())
        spec = CATEGORIES[c]
        subcategory[m] = rng.choice(spec["subcats"], size=k)
        lo, hi = spec["price"]
        # Log-uniform: many cheap items, a long tail of premium ones.
        list_price[m] = np.exp(rng.uniform(np.log(lo), np.log(hi), size=k))
        margin[m] = np.clip(rng.normal(spec["margin"], 0.05, size=k), 0.08, 0.80)

    list_price = np.round(list_price, 2)
    unit_cost = np.round(list_price * (1 - margin), 2)
    unit_cost = np.minimum(unit_cost, list_price - 0.01)  # enforce cost < price

    brand = rng.choice(BRANDS, size=n)
    model_no = rng.integers(100, 999, size=n)
    names = [f"{b} {s} {chr(65 + (i % 26))}-{m}" for i, (b, s, m) in enumerate(zip(brand, subcategory, model_no))]

    # 85% "core" catalogue launched before the window; 15% launched during it.
    pre = pd.Timestamp(cfg.start_date) - pd.to_timedelta(rng.integers(30, 900, size=n), unit="D")
    span = (pd.Timestamp(cfg.end_date) - pd.Timestamp(cfg.start_date)).days - 60
    during = pd.Timestamp(cfg.start_date) + pd.to_timedelta(rng.integers(0, span, size=n), unit="D")
    is_new = rng.random(n) < 0.15
    is_new[: len(cat_names)] = False  # first product per category is always core
    launch = np.where(is_new, during, pre)

    # Popularity: Zipf-like within category, randomly assigned to products.
    popularity = np.empty(n)
    for c in cat_names:
        m = np.flatnonzero(category == c)
        ranks = rng.permutation(len(m)) + 1
        popularity[m] = 1.0 / ranks ** 0.9

    return pd.DataFrame({
        "product_id": np.arange(1, n + 1, dtype=np.int32),
        "sku": [f"SKU{i:07d}" for i in range(1, n + 1)],
        "product_name": names,
        "category": category,
        "subcategory": subcategory,
        "brand": brand,
        "list_price": list_price,
        "unit_cost": unit_cost,
        "launch_date": pd.to_datetime(launch).date,
        "is_active": rng.random(n) > 0.04,
        "_popularity": popularity,  # internal, dropped before persisting
    })


def build_customers(cfg: SimulationConfig, rng: np.random.Generator, days: pd.DatetimeIndex) -> pd.DataFrame:
    n = cfg.n_customers
    # Sign-ups grow ~80%/yr with damped seasonality.
    w = _day_weights(days, trend=0.8, season_strength=0.4)
    signup_idx = rng.choice(len(days), size=n, p=w / w.sum())

    countries = {k: v["weight"] for k, v in cfg.countries.items()}
    type_names = list(CUSTOMER_TYPES.keys())
    type_p = np.array([CUSTOMER_TYPES[t][0] for t in type_names])
    latent = rng.choice(type_names, size=n, p=type_p / type_p.sum())
    cat_names = list(CATEGORIES.keys())
    cat_w = np.array([CATEGORIES[c]["weight"] for c in cat_names])

    ids = np.arange(1, n + 1, dtype=np.int64) + 1_000_000
    # Store only a hash of the email: the warehouse never holds raw PII.
    email_hash = [hashlib.sha256(f"customer{i}@example.com".encode()).hexdigest() for i in ids]

    loyalty_p = np.where(np.isin(latent, ["whale", "loyal"]), 0.55, 0.15)
    return pd.DataFrame({
        "customer_id": ids,
        "email_hash": email_hash,
        "signup_date": days[signup_idx].date,
        "country_code": _weighted_choice(rng, countries, n),
        "age_band": _weighted_choice(rng, AGE_BANDS, n),
        "gender": _weighted_choice(rng, GENDERS, n),
        "acquisition_channel": _weighted_choice(rng, ACQUISITION_CHANNELS, n),
        "preferred_device": _weighted_choice(rng, DEVICES, n),
        "is_loyalty_member": rng.random(n) < loyalty_p,
        "marketing_opt_in": rng.random(n) < 0.55,
        # Internal ground-truth columns (dropped before persisting, kept for model evaluation)
        "_signup_idx": signup_idx,
        "_latent_type": latent,
        "_pref_category": rng.choice(cat_names, size=n, p=cat_w / cat_w.sum()),
    })


def build_orders(cfg: SimulationConfig, rng: np.random.Generator, days: pd.DatetimeIndex,
                 customers: pd.DataFrame) -> pd.DataFrame:
    """Simulate each customer's purchase history from their latent type."""
    n_days = len(days)
    latent = customers["_latent_type"].to_numpy()
    rate = np.array([CUSTOMER_TYPES[t][1] for t in latent])
    life_mean = np.array([CUSTOMER_TYPES[t][2] for t in latent])
    rate = rate * customers["acquisition_channel"].map(CHANNEL_RATE_MULTIPLIER).to_numpy()
    rate = rate * np.where(customers["is_loyalty_member"].to_numpy(), 1.3, 1.0)

    signup = customers["_signup_idx"].to_numpy()
    lifetime_days = (rng.exponential(life_mean) * 30.4).astype(np.int64)
    active_end = np.minimum(signup + np.maximum(lifetime_days, 1), n_days - 1)
    active_years = (active_end - signup + 1) / 365.25

    n_orders = rng.poisson(rate * active_years)
    purchaser = latent != "never"
    n_orders = np.where(purchaser, np.maximum(n_orders, 1), 0)
    n_orders = np.where(latent == "one_time", 1, n_orders)
    n_orders = np.minimum(n_orders, 250)  # guard against pathological tails

    total = int(n_orders.sum())
    log.info("Simulating %s orders for %s purchasing customers", f"{total:,}", f"{int(purchaser.sum()):,}")

    cust_idx = np.repeat(np.arange(len(customers)), n_orders)
    # Position of each order within its customer's history (0 = first order).
    starts = np.repeat(np.cumsum(n_orders) - n_orders, n_orders)
    seq = np.arange(total) - starts

    # First order lands within days of sign-up; the rest spread over the active window
    # following the seasonal demand curve.
    first_offset = np.minimum(rng.exponential(6.0, size=total).astype(np.int64), 60)
    first_day = np.minimum(signup[cust_idx] + first_offset, n_days - 1)
    cdf = np.concatenate([[0.0], np.cumsum(_day_weights(days, trend=0.0))])
    lo = first_day
    hi = np.maximum(active_end[cust_idx], lo)
    later_day = _sample_days_in_windows(rng, cdf, lo, hi)
    day_idx = np.where(seq == 0, first_day, later_day)

    ts = (days[day_idx].to_numpy()
          + pd.to_timedelta(_random_times(rng, total), unit="s").to_numpy())

    pref_device = customers["preferred_device"].to_numpy()[cust_idx]
    device = np.where(rng.random(total) < 0.7, pref_device, _weighted_choice(rng, DEVICES, total))

    orders = pd.DataFrame({
        "_cust_idx": cust_idx,
        "customer_id": customers["customer_id"].to_numpy()[cust_idx],
        "order_ts": ts,
        "country_code": customers["country_code"].to_numpy()[cust_idx],
        "device": device,
        "payment_method": _weighted_choice(rng, PAYMENT_METHODS, total),
        "_day_idx": day_idx,
    })
    orders = orders.sort_values("order_ts", kind="mergesort").reset_index(drop=True)
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1, dtype=np.int64) + 10_000_000)
    orders["order_date"] = orders["order_ts"].dt.date
    orders["is_first_order"] = ~orders.duplicated("customer_id", keep="first")
    return orders


def build_order_items(cfg: SimulationConfig, rng: np.random.Generator, days: pd.DatetimeIndex,
                      orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    n_orders = len(orders)
    n_items = np.minimum(1 + rng.poisson(1.8, size=n_orders), 8)
    total = int(n_items.sum())
    log.info("Simulating %s order lines", f"{total:,}")

    order_pos = np.repeat(np.arange(n_orders), n_items)
    cust_idx = orders["_cust_idx"].to_numpy()[order_pos]
    day_idx = orders["_day_idx"].to_numpy()[order_pos]

    # Category affinity: 55% of lines come from the customer's favourite category.
    cat_names = list(CATEGORIES.keys())
    cat_w = np.array([CATEGORIES[c]["weight"] for c in cat_names])
    pref = customers["_pref_category"].to_numpy()[cust_idx]
    random_cat = rng.choice(cat_names, size=total, p=cat_w / cat_w.sum())
    category = np.where(rng.random(total) < 0.55, pref, random_cat)

    launch_idx = (pd.to_datetime(products["launch_date"]) - days[0]).dt.days.to_numpy()
    product_pos = np.empty(total, dtype=np.int64)
    for c in cat_names:
        m = np.flatnonzero(category == c)
        if len(m) == 0:
            continue
        cand = np.flatnonzero(products["category"].to_numpy() == c)
        pop = products["_popularity"].to_numpy()[cand]
        product_pos[m] = rng.choice(cand, size=len(m), p=pop / pop.sum())
        # A product cannot be sold before it launches: re-draw from the core catalogue.
        bad = m[launch_idx[product_pos[m]] > day_idx[m]]
        if len(bad):
            core = cand[launch_idx[cand] <= 0]
            core_pop = products["_popularity"].to_numpy()[core]
            product_pos[bad] = rng.choice(core, size=len(bad), p=core_pop / core_pop.sum())

    quantity = rng.choice([1, 2, 3, 4], size=total, p=[0.80, 0.14, 0.045, 0.015]).astype(np.int16)
    country = orders["country_code"].to_numpy()[order_pos]
    price_index = pd.Series(country).map({k: v["price_index"] for k, v in cfg.countries.items()}).to_numpy()
    years_in = day_idx / 365.25
    inflation = 1.0 + 0.04 * years_in  # ~4% annual price inflation

    list_price = products["list_price"].to_numpy()[product_pos]
    unit_cost = products["unit_cost"].to_numpy()[product_pos]
    unit_price = np.round(list_price * price_index * inflation * rng.normal(1.0, 0.03, size=total), 2)
    unit_price = np.maximum(unit_price, 0.5)

    # Order-level promo depth, heavier during November promotions.
    month = days[orders["_day_idx"].to_numpy()].month.to_numpy()
    promo_p = np.where(month == 11, 0.65, 0.28)
    depth = rng.choice([0.05, 0.10, 0.15, 0.20, 0.30], size=n_orders, p=[0.25, 0.30, 0.20, 0.15, 0.10])
    order_discount_pct = np.where(rng.random(n_orders) < promo_p, depth, 0.0)
    pct = order_discount_pct[order_pos]

    line_gross = np.round(quantity * unit_price, 2)
    discount = np.round(line_gross * pct, 2)
    line_revenue = np.round(line_gross - discount, 2)
    line_cogs = np.round(quantity * unit_cost * price_index * inflation, 2)

    return pd.DataFrame({
        "order_item_id": np.arange(1, total + 1, dtype=np.int64) + 50_000_000,
        "order_id": orders["order_id"].to_numpy()[order_pos],
        "product_id": products["product_id"].to_numpy()[product_pos],
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_amount": discount,
        "line_revenue": line_revenue,
        "line_cogs": line_cogs,
        "_category": category,
        "_line_gross": line_gross,
    })


def finalise_orders(cfg: SimulationConfig, rng: np.random.Generator, orders: pd.DataFrame,
                    items: pd.DataFrame) -> pd.DataFrame:
    """Roll order lines up into order totals, shipping and order status."""
    items = items.assign(_is_fashion=items["_category"] == "Fashion")
    agg = items.groupby("order_id", sort=False).agg(
        item_count=("quantity", "sum"),
        gross_amount=("_line_gross", "sum"),
        discount_amount=("discount_amount", "sum"),
        cogs_amount=("line_cogs", "sum"),
        has_fashion=("_is_fashion", "max"),
    )
    orders = orders.merge(agg, left_on="order_id", right_index=True, how="left", validate="1:1")
    if orders["gross_amount"].isna().any():
        raise DataQualityError("Orders without any order lines detected")

    price_index = orders["country_code"].map({k: v["price_index"] for k, v in cfg.countries.items()})
    free_ship = orders["gross_amount"] - orders["discount_amount"] >= 75 * price_index
    shipping = np.round(rng.uniform(4.99, 9.99, size=len(orders)) * price_index, 2)
    orders["shipping_amount"] = np.where(free_ship, 0.0, shipping)

    for col in ("gross_amount", "discount_amount", "cogs_amount"):
        orders[col] = orders[col].round(2)
    orders["net_amount"] = (orders["gross_amount"] - orders["discount_amount"] + orders["shipping_amount"]).round(2)
    orders["item_count"] = orders["item_count"].astype("int16")

    # Fashion carries the highest return rate in real retail.
    p_return = np.where(orders["has_fashion"], cfg.p_returned * 1.8, cfg.p_returned * 0.7)
    u = rng.random(len(orders))
    orders["order_status"] = np.select(
        [u < cfg.p_cancelled, u < cfg.p_cancelled + p_return],
        ["cancelled", "returned"], default="completed")
    return orders.drop(columns=["has_fashion"])


def build_web_sessions(cfg: SimulationConfig, rng: np.random.Generator, days: pd.DatetimeIndex,
                       orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Converting sessions (1 per order) + down-sampled non-converting sessions."""
    n_days = len(days)

    # ---- Converting sessions ------------------------------------------------
    n_conv = len(orders)
    acq = customers["acquisition_channel"].to_numpy()[orders["_cust_idx"].to_numpy()]
    returning_mix = {"Direct": 0.32, "Email": 0.22, "Organic Search": 0.20, "Paid Search": 0.12,
                     "Social": 0.06, "Referral": 0.05, "Affiliate": 0.03}
    source_conv = np.where(orders["is_first_order"].to_numpy(), acq, _weighted_choice(rng, returning_mix, n_conv))
    dur_conv = np.clip(rng.lognormal(np.log(540), 0.6, size=n_conv), 60, 7200).astype(np.int32)
    conv = pd.DataFrame({
        "customer_id": orders["customer_id"].to_numpy(),
        "session_start": orders["order_ts"].to_numpy() - pd.to_timedelta(dur_conv, unit="s").to_numpy(),
        "country_code": orders["country_code"].to_numpy(),
        "device": orders["device"].to_numpy(),
        "traffic_source": source_conv,
        "landing_page": _weighted_choice(rng, {"/": 0.25, "/category": 0.25, "/product": 0.35, "/search": 0.10, "/promo": 0.05}, n_conv),
        "pageviews": (4 + rng.poisson(6, size=n_conv)).astype(np.int16),
        "duration_sec": dur_conv,
        "added_to_cart": True,
        "reached_checkout": True,
        "bounced": False,
        "converted": True,
        "order_id": orders["order_id"].to_numpy(),
        "sample_weight": np.int16(1),
    })

    # ---- Non-converting sessions (down-sampled) -----------------------------
    population_neg = n_conv / cfg.target_conversion_rate - n_conv
    n_neg = int(population_neg / cfg.negative_session_sample_rate)
    log.info("Simulating %s converting + %s sampled non-converting sessions (weight=%d)",
             f"{n_conv:,}", f"{n_neg:,}", cfg.negative_session_sample_rate)

    known = rng.random(n_neg) < 0.45
    cust_pick = rng.integers(0, len(customers), size=n_neg)
    signup = customers["_signup_idx"].to_numpy()[cust_pick]
    traffic_cdf = np.concatenate([[0.0], np.cumsum(_day_weights(days, trend=0.6))])
    lo = np.where(known, signup, 0)
    day_idx = _sample_days_in_windows(rng, traffic_cdf, lo, np.full(n_neg, n_days - 1))
    start = days[day_idx].to_numpy() + pd.to_timedelta(_random_times(rng, n_neg), unit="s").to_numpy()

    country_w = {k: v["weight"] for k, v in cfg.countries.items()}
    country = np.where(known, customers["country_code"].to_numpy()[cust_pick], _weighted_choice(rng, country_w, n_neg))
    device = _weighted_choice(rng, DEVICES, n_neg)
    bounce_p = np.where(device == "Mobile", 0.48, 0.38)
    bounced = rng.random(n_neg) < bounce_p
    pageviews = np.where(bounced, 1, 2 + rng.poisson(3, size=n_neg)).astype(np.int16)
    duration = np.where(bounced, rng.integers(0, 30, size=n_neg),
                        np.clip(rng.lognormal(np.log(180), 0.8, size=n_neg), 20, 5400)).astype(np.int32)
    carted = ~bounced & (rng.random(n_neg) < 0.18)
    checkout = carted & (rng.random(n_neg) < 0.35)

    neg = pd.DataFrame({
        "customer_id": pd.array(np.where(known, customers["customer_id"].to_numpy()[cust_pick], 0), dtype="Int64"),
        "session_start": start,
        "country_code": country,
        "device": device,
        "traffic_source": _weighted_choice(rng, ACQUISITION_CHANNELS, n_neg),
        "landing_page": _weighted_choice(rng, LANDING_PAGES, n_neg),
        "pageviews": pageviews,
        "duration_sec": duration,
        "added_to_cart": carted,
        "reached_checkout": checkout,
        "bounced": bounced,
        "converted": False,
        "order_id": pd.array([pd.NA] * n_neg, dtype="Int64"),
        "sample_weight": np.int16(cfg.negative_session_sample_rate),
    })
    neg.loc[~known, "customer_id"] = pd.NA

    conv["customer_id"] = pd.array(conv["customer_id"], dtype="Int64")
    conv["order_id"] = pd.array(conv["order_id"], dtype="Int64")
    sessions = pd.concat([conv, neg], ignore_index=True)
    sessions = sessions.sort_values("session_start", kind="mergesort").reset_index(drop=True)
    sessions.insert(0, "session_id", np.arange(1, len(sessions) + 1, dtype=np.int64) + 900_000_000)
    sessions["session_date"] = sessions["session_start"].dt.date
    # Converting sessions that started before midnight on day 0 would precede the window; clamp.
    first = pd.Timestamp(cfg.start_date)
    sessions.loc[sessions["session_start"] < first, "session_start"] = first
    sessions["session_date"] = sessions["session_start"].dt.date
    return sessions


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate(cfg: SimulationConfig) -> dict[str, pd.DataFrame]:
    """Generate every table in memory and return them keyed by table name."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    days = pd.date_range(cfg.start_date, cfg.end_date, freq="D")

    dim_country = build_dim_country(cfg)
    dim_date = build_dim_date(cfg)
    products = build_products(cfg, rng)
    customers = build_customers(cfg, rng, days)
    orders = build_orders(cfg, rng, days, customers)
    items = build_order_items(cfg, rng, days, orders, customers, products)
    orders = finalise_orders(cfg, rng, orders, items)
    sessions = build_web_sessions(cfg, rng, days, orders, customers)

    # Ground truth for offline evaluation of the segmentation model.
    ground_truth = customers[["customer_id", "_latent_type"]].rename(columns={"_latent_type": "latent_type"})

    tables = {
        "dim_country": dim_country,
        "dim_date": dim_date,
        "products": products.drop(columns=["_popularity"]),
        "customers": customers.drop(columns=[c for c in customers.columns if c.startswith("_")]),
        "orders": orders[[
            "order_id", "customer_id", "order_ts", "order_date", "country_code", "device", "payment_method",
            "order_status", "item_count", "gross_amount", "discount_amount", "shipping_amount", "net_amount",
            "cogs_amount", "is_first_order"]],
        "order_items": items.drop(columns=["_category", "_line_gross"]),
        "web_sessions": sessions[[
            "session_id", "customer_id", "session_start", "session_date", "country_code", "device",
            "traffic_source", "landing_page", "pageviews", "duration_sec", "added_to_cart", "reached_checkout",
            "bounced", "converted", "order_id", "sample_weight"]],
        "_ground_truth": ground_truth,
    }
    log.info("Generation finished in %.1fs", time.perf_counter() - t0)
    return tables


def write_parquet(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Persist each table as Snappy-compressed Parquet (columnar, typed, ~10x smaller than CSV)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        log.info("  wrote %-14s %10s rows -> %s", name, f"{len(df):,}", path.name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic global e-commerce data.")
    p.add_argument("--customers", type=int, default=SimulationConfig.n_customers, help="number of customers")
    p.add_argument("--products", type=int, default=SimulationConfig.n_products, help="catalogue size")
    p.add_argument("--seed", type=int, default=SimulationConfig.seed, help="random seed")
    p.add_argument("--out", type=Path, default=RAW_DIR, help="output directory for Parquet files")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.customers < 100 or args.products < len(CATEGORIES):
        log.error("Need at least 100 customers and %d products", len(CATEGORIES))
        return 2
    cfg = replace(SimulationConfig(), n_customers=args.customers, n_products=args.products, seed=args.seed)
    try:
        tables = generate(cfg)
        validate_tables({k: v for k, v in tables.items() if not k.startswith("_")})
    except DataQualityError as exc:
        log.error("Data quality validation failed: %s", exc)
        return 1
    write_parquet(tables, args.out)
    total = sum(len(v) for k, v in tables.items() if not k.startswith("_"))
    log.info("Done. %s rows across %d tables.", f"{total:,}", len(tables) - 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
