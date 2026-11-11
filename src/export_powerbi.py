"""
Export a Power BI-ready star schema (Parquet) from the pipeline outputs.

    fact_order_lines ──┬── dim_date
                       ├── dim_customer   (incl. RFM scores, K-Means persona, CLV)
                       ├── dim_product
                       └── dim_geography
    fact_web_funnel ───┬── dim_date
                       └── dim_geography
    fact_cohort        (pre-aggregated: cohort month × months since × country × channel)

Why a star schema: Power BI's VertiPaq engine compresses and filters star schemas
far better than a wide flat table, and single-direction 1:* relationships keep
DAX predictable.

Why pre-aggregate cohorts: a cohort matrix computed live in DAX over 830k rows is
slow. A 30k-row aggregate keeps the heatmap instant while still slicing by
country and acquisition channel.

    python -m src.export_powerbi
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MART_DIR, PROJECT_ROOT, RAW_DIR, get_logger

log = get_logger("export_powerbi")
OUT = PROJECT_ROOT / "powerbi" / "data"


def run() -> dict[str, int]:
    OUT.mkdir(parents=True, exist_ok=True)
    rd = lambda n: pd.read_parquet(RAW_DIR / f"{n}.parquet")  # noqa: E731
    orders, items, products, customers = rd("orders"), rd("order_items"), rd("products"), rd("customers")
    countries, sessions = rd("dim_country"), rd("web_sessions")
    seg = pd.read_parquet(MART_DIR / "customer_segments.parquet")

    # ---- dim_date: contiguous calendar covering all facts (Power BI time intelligence needs no gaps) ----
    start = min(pd.to_datetime(orders["order_date"]).min(), pd.to_datetime(customers["signup_date"]).min())
    days = pd.date_range(start.normalize(), pd.Timestamp("2023-12-31"), freq="D")
    dim_date = pd.DataFrame({"date": days})
    dim_date["year"] = days.year
    dim_date["quarter"] = "Q" + days.quarter.astype(str)
    dim_date["year_quarter"] = days.year.astype(str) + " " + dim_date["quarter"]
    dim_date["month_number"] = days.month
    dim_date["month_name"] = days.strftime("%b")
    dim_date["month_start"] = days.to_period("M").to_timestamp()
    dim_date["year_month"] = days.strftime("%Y-%m")
    dim_date["weekday_number"] = days.day_of_week + 1
    dim_date["weekday"] = days.strftime("%a")
    dim_date["is_weekend"] = days.day_of_week >= 5
    dim_date["is_holiday_season"] = ((days.month == 11) & (days.day >= 22)) | (days.month == 12)

    # ---- dim_customer (+ segmentation & CLV) ----
    dim_customer = customers.merge(
        seg[["customer_id", "recency_days", "frequency", "monetary", "r_score", "f_score", "m_score", "rfm_code",
             "rfm_segment", "cluster_label", "p_alive", "historical_clv", "predicted_clv_12m", "first_order_date"]],
        on="customer_id", how="left")
    dim_customer["persona"] = dim_customer["cluster_label"].fillna("No purchase")
    dim_customer["rfm_segment"] = dim_customer["rfm_segment"].fillna("No purchase")
    dim_customer["signup_month"] = pd.to_datetime(dim_customer["signup_date"]).dt.to_period("M").dt.to_timestamp()
    dim_customer["cohort_month"] = pd.to_datetime(dim_customer["first_order_date"]).dt.to_period("M").dt.to_timestamp()
    bands = [-0.01, 0, 100, 250, 500, 1000, 2500, 5000, np.inf]
    labels = ["$0", "$1–100", "$100–250", "$250–500", "$500–1K", "$1K–2.5K", "$2.5K–5K", "$5K+"]
    dim_customer["clv_band"] = pd.cut(dim_customer["historical_clv"].fillna(0), bands, labels=labels).astype(str)
    dim_customer["clv_band_order"] = pd.cut(dim_customer["historical_clv"].fillna(0), bands, labels=False).astype(int)
    dim_customer = dim_customer.drop(columns=["email_hash", "cluster_label", "first_order_date"])

    dim_product = products.drop(columns=["unit_cost"])
    dim_geography = countries.rename(columns={"country_name": "country"})

    # ---- fact_order_lines ----
    o = orders[["order_id", "customer_id", "order_date", "country_code", "device", "payment_method", "order_status",
                "is_first_order"]]
    fact = items.merge(o, on="order_id", how="inner")
    fact["gross_amount"] = (fact["quantity"] * fact["unit_price"]).round(2)
    fact["order_date"] = pd.to_datetime(fact["order_date"])
    fact = fact[["order_item_id", "order_id", "order_date", "customer_id", "product_id", "country_code", "device",
                 "payment_method", "order_status", "is_first_order", "quantity", "unit_price", "gross_amount",
                 "discount_amount", "line_revenue", "line_cogs"]]

    # ---- fact_web_funnel (weighted, daily) ----
    w = sessions["sample_weight"].astype(int)
    sessions = sessions.assign(sessions=w, engaged=w * (~sessions["bounced"]), carts=w * sessions["added_to_cart"],
                               checkouts=w * sessions["reached_checkout"], purchases=w * sessions["converted"])
    funnel = (sessions.groupby(["session_date", "country_code", "device", "traffic_source"], as_index=False)
              [["sessions", "engaged", "carts", "checkouts", "purchases"]].sum())
    funnel = funnel.rename(columns={"session_date": "date"})
    funnel["date"] = pd.to_datetime(funnel["date"])

    # ---- fact_cohort (pre-aggregated) ----
    done = fact[fact["order_status"] == "completed"]
    act = (done.assign(month=done["order_date"].dt.to_period("M"))
           .groupby(["customer_id", "month"], as_index=False)["line_revenue"].sum())
    first = act.groupby("customer_id")["month"].min().rename("cohort")
    act = act.join(first, on="customer_id")
    act["months_since_first"] = (act["month"] - act["cohort"]).apply(lambda p: p.n)
    act = act.merge(customers[["customer_id", "country_code", "acquisition_channel"]], on="customer_id")
    cohort = (act.groupby(["cohort", "months_since_first", "country_code", "acquisition_channel"], as_index=False)
              .agg(active_customers=("customer_id", "nunique"), revenue=("line_revenue", "sum")))
    cohort["cohort_month"] = cohort["cohort"].dt.to_timestamp()
    cohort = cohort.drop(columns="cohort")
    cohort = cohort[cohort["months_since_first"] <= 24]

    tables = {"dim_date": dim_date, "dim_customer": dim_customer, "dim_product": dim_product,
              "dim_geography": dim_geography, "fact_order_lines": fact, "fact_web_funnel": funnel, "fact_cohort": cohort}
    counts = {}
    for name, df in tables.items():
        # Power BI's Parquet connector prefers microsecond timestamps and plain bools/strings.
        df.to_parquet(OUT / f"{name}.parquet", index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
        counts[name] = len(df)
        log.info("  %-18s %10s rows", name, f"{len(df):,}")
    return counts


if __name__ == "__main__":
    run()
