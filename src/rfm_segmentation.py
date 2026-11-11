"""
RFM + K-Means customer segmentation engine.

Two complementary views of every customer are produced:

1. **Rule-based RFM segments** – quintile scores (1-5) on Recency, Frequency and
   Monetary value mapped to the industry-standard segment grid (Champions,
   At Risk, Hibernating ...). Transparent and easy for marketing to act on.
2. **K-Means personas** – unsupervised clusters on log-transformed, standardised
   R/F/M. k is selected automatically by silhouette score (with the elbow curve
   saved for review). Clusters are named by ranking their centroids on a
   composite value score and named from their R/F/M traits, so labels stay
   meaningful and stable across retrains.

Plus a heuristic 12-month CLV forecast and, when the synthetic ground truth is
available, an Adjusted Rand Index showing how well clusters recover the true
behavioural types.

Usage
-----
    python -m src.rfm_segmentation            # auto-select k
    python -m src.rfm_segmentation --k 5      # force k
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import (
    CLV_HORIZON_MONTHS,
    DATABASE_URL,
    DB_SCHEMA,
    KMEANS_K_RANGE,
    KMEANS_MIN_ACTIONABLE_K,
    KMEANS_SILHOUETTE_SAMPLE,
    MART_DIR,
    MODEL_DIR,
    RAW_DIR,
    RFM_SNAPSHOT_DATE,
    get_logger,
)

log = get_logger("rfm_segmentation")

MODEL_VERSION = "rfm-kmeans-1.0.0"
FEATURES = ["recency_days", "frequency", "monetary"]

# Classic RFM grid keyed on "<R score><F score>" (regex, first match wins).
RFM_SEGMENT_MAP: list[tuple[str, str]] = [
    (r"^5[4-5]$", "Champions"),
    (r"^[3-4][4-5]$", "Loyal Customers"),
    (r"^[4-5][2-3]$", "Potential Loyalists"),
    (r"^51$", "New Customers"),
    (r"^41$", "Promising"),
    (r"^33$", "Need Attention"),
    (r"^3[1-2]$", "About to Sleep"),
    (r"^[1-2]5$", "Can't Lose Them"),
    (r"^[1-2][3-4]$", "At Risk"),
    (r"^[1-2][1-2]$", "Hibernating"),
]

MAX_K = 8


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def load_orders(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load completed orders from PostgreSQL when configured, otherwise from Parquet."""
    if DATABASE_URL:
        from src.db import read_sql  # local import keeps the Parquet path dependency-free

        log.info("Reading orders from PostgreSQL")
        return read_sql(
            f"SELECT customer_id, order_date, net_amount FROM {DB_SCHEMA}.orders WHERE order_status = 'completed'"
        )
    path = raw_dir / "orders.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – run `python -m src.generate_data` first")
    log.info("Reading orders from %s", path)
    orders = pd.read_parquet(path, columns=["customer_id", "order_date", "net_amount", "order_status"])
    return orders.loc[orders["order_status"] == "completed"].drop(columns="order_status")


def compute_rfm(orders: pd.DataFrame, snapshot: date = RFM_SNAPSHOT_DATE) -> pd.DataFrame:
    """Aggregate completed orders into one R/F/M row per customer.

    recency_days : days between the customer's last order and the snapshot date
    frequency    : number of completed orders
    monetary     : total net revenue (USD)
    """
    if orders.empty:
        raise ValueError("No completed orders – cannot compute RFM")
    df = orders.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    snap = pd.Timestamp(snapshot)
    if df["order_date"].max() >= snap:
        raise ValueError(f"Snapshot {snapshot} must be after the last order date {df['order_date'].max().date()}")

    rfm = df.groupby("customer_id").agg(
        last_order_date=("order_date", "max"),
        first_order_date=("order_date", "min"),
        frequency=("order_date", "size"),
        monetary=("net_amount", "sum"),
    )
    rfm["recency_days"] = (snap - rfm["last_order_date"]).dt.days.astype(int)
    rfm["tenure_days"] = (snap - rfm["first_order_date"]).dt.days.astype(int)
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm.reset_index()


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add 1-5 quintile scores, the 3-digit RFM code and the rule-based segment.

    `rank(method="first")` breaks ties before `qcut`, which is essential for
    Frequency where most customers share small integer values.
    """
    out = rfm.copy()
    labels = [1, 2, 3, 4, 5]
    out["r_score"] = pd.qcut(out["recency_days"].rank(method="first"), 5, labels=labels[::-1]).astype(int)
    out["f_score"] = pd.qcut(out["frequency"].rank(method="first"), 5, labels=labels).astype(int)
    out["m_score"] = pd.qcut(out["monetary"].rank(method="first"), 5, labels=labels).astype(int)
    out["rfm_code"] = out["r_score"].astype(str) + out["f_score"].astype(str) + out["m_score"].astype(str)

    rf = out["r_score"].astype(str) + out["f_score"].astype(str)
    segment = pd.Series(pd.NA, index=out.index, dtype="object")
    for pattern, name in RFM_SEGMENT_MAP:
        hit = segment.isna() & rf.str.match(pattern)
        segment[hit] = name
    if segment.isna().any():  # the grid is exhaustive; this guards future edits to the map
        raise RuntimeError(f"Unmapped RF combinations: {sorted(rf[segment.isna()].unique())}")
    out["rfm_segment"] = segment.astype(str)
    return out


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def build_pipeline(k: int, seed: int = 42) -> Pipeline:
    """log1p -> StandardScaler -> KMeans. log1p tames the heavy right tails of F and M."""
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=True, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])


def select_k(X: pd.DataFrame, k_range=KMEANS_K_RANGE, sample: int = KMEANS_SILHOUETTE_SAMPLE,
             seed: int = 42, min_k: int = KMEANS_MIN_ACTIONABLE_K) -> tuple[int, pd.DataFrame]:
    """Fit k in `k_range`, score each and return (best_k, diagnostics table).

    Silhouette (higher = better separated) drives the choice among k >= `min_k`.
    RFM space is a continuum, so silhouette alone tends to favour k=3, which is
    too coarse to act on; the business floor keeps at least `min_k` personas.
    Inertia (elbow) and Davies-Bouldin (lower = better) are recorded for review.
    """
    rows = []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    for k in k_range:
        if k >= len(X):
            break
        pipe = build_pipeline(k, seed).fit(X)
        Z = pipe[:-1].transform(X)
        labels = pipe["kmeans"].labels_
        rows.append({
            "k": k,
            "inertia": float(pipe["kmeans"].inertia_),
            "silhouette": float(silhouette_score(Z[idx], labels[idx])),
            "davies_bouldin": float(davies_bouldin_score(Z[idx], labels[idx])),
        })
        log.info("  k=%d  silhouette=%.4f  davies_bouldin=%.3f", k, rows[-1]["silhouette"], rows[-1]["davies_bouldin"])
    diag = pd.DataFrame(rows)
    if diag.empty:
        raise ValueError("Not enough customers to evaluate any k")
    eligible = diag[diag["k"] >= min_k]
    pool = eligible if not eligible.empty else diag
    best = int(pool.loc[pool["silhouette"].idxmax(), "k"])
    return best, diag


def persona_for(recency_med: float, frequency_med: float, monetary_med: float,
                pop_recency_med: float, pop_monetary_med: float, pop_monetary_p75: float) -> str:
    """Name a cluster from its centroid traits relative to the whole customer base.

    recent      : median recency at or below the population median
    repeat      : median customer has bought at least twice
    high value  : median spend above the population median (or 75th pct for VIPs)
    """
    recent = recency_med <= pop_recency_med
    repeat = frequency_med >= 2
    if repeat and recent:
        return "VIP Champions" if monetary_med >= pop_monetary_p75 else "Loyal Regulars"
    if repeat:
        return "Can't-Lose High-Value" if monetary_med >= pop_monetary_p75 else "At-Risk Repeat Buyers"
    if recent:
        return "New & Promising"
    return "Lapsed Big-Ticket" if monetary_med >= pop_monetary_med else "Dormant One-Timers"


def name_clusters(df: pd.DataFrame, labels: np.ndarray) -> tuple[dict[int, str], pd.DataFrame]:
    """Profile each cluster, rank by a composite value score and give it a persona name.

    value = z(log F) + z(log M) - 0.8 * z(log R)  (on cluster medians) orders the
    clusters; `persona_for` names them from their traits so names stay meaningful
    for any k. Two clusters with the same traits get a tier suffix (" II").
    """
    tmp = df[FEATURES].copy()
    tmp["cluster_id"] = labels
    prof = tmp.groupby("cluster_id").agg(
        customers=("frequency", "size"),
        recency_median=("recency_days", "median"),
        frequency_median=("frequency", "median"),
        monetary_median=("monetary", "median"),
        recency_mean=("recency_days", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
        monetary_total=("monetary", "sum"),
    )
    logs = np.log1p(prof[["recency_median", "frequency_median", "monetary_median"]])
    z = (logs - logs.mean()) / logs.std(ddof=0).replace(0, 1)
    prof["value_score"] = z["frequency_median"] + z["monetary_median"] - 0.8 * z["recency_median"]
    prof = prof.sort_values("value_score", ascending=False)

    pop_r, pop_m = df["recency_days"].median(), df["monetary"].median()
    pop_m75 = df["monetary"].quantile(0.75)
    mapping: dict[int, str] = {}
    seen: dict[str, int] = {}
    for cid, row in prof.iterrows():
        name = persona_for(row["recency_median"], row["frequency_median"], row["monetary_median"], pop_r, pop_m, pop_m75)
        seen[name] = seen.get(name, 0) + 1
        mapping[int(cid)] = name if seen[name] == 1 else f"{name} {'I' * seen[name]}"
    prof["cluster_label"] = prof.index.map(mapping)
    prof["value_rank"] = np.arange(1, len(prof) + 1)
    prof["share_customers"] = prof["customers"] / prof["customers"].sum()
    prof["share_revenue"] = prof["monetary_total"] / prof["monetary_total"].sum()
    return mapping, prof.reset_index()


# --------------------------------------------------------------------------- #
# CLV
# --------------------------------------------------------------------------- #
def estimate_clv(rfm: pd.DataFrame, horizon_months: int = CLV_HORIZON_MONTHS) -> pd.DataFrame:
    """Heuristic, BG/NBD-inspired 12-month revenue CLV.

    predicted = P(alive) x expected orders in horizon x average order value
      * expected orders = annualised historical purchase rate x horizon
      * P(alive)        = exp(-recency / (2 x typical inter-purchase gap)),
                          so customers silent for far longer than their own
                          cadence are discounted towards zero.
    It is deliberately simple and explainable; swap in `lifetimes`/`pymc-marketing`
    for a full probabilistic model.
    """
    out = rfm.copy()
    tenure_years = np.maximum(out["tenure_days"], 30) / 365.25
    annual_rate = out["frequency"] / tenure_years
    repeat = out["frequency"] > 1
    gap = pd.Series(np.nan, index=out.index)
    gap[repeat] = (out.loc[repeat, "tenure_days"] - out.loc[repeat, "recency_days"]) / (out.loc[repeat, "frequency"] - 1)
    population_gap = float(np.nanmedian(gap)) if repeat.any() else 180.0
    gap = gap.fillna(population_gap).clip(lower=14)
    p_alive = np.exp(-out["recency_days"] / (2 * gap))
    out["p_alive"] = p_alive.round(4)
    out["historical_clv"] = out["monetary"].round(2)
    out["predicted_clv_12m"] = (p_alive * annual_rate * (horizon_months / 12) * out["avg_order_value"]).round(2)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(k: int | None = None, raw_dir: Path = RAW_DIR, out_dir: Path = MART_DIR,
        model_dir: Path = MODEL_DIR, write_db: bool = True) -> pd.DataFrame:
    t0 = time.perf_counter()
    orders = load_orders(raw_dir)
    rfm = compute_rfm(orders)
    log.info("Computed RFM for %s customers", f"{len(rfm):,}")
    rfm = score_rfm(rfm)
    rfm = estimate_clv(rfm)

    X = rfm[FEATURES]
    best_k, diag = select_k(X)
    chosen_k = k or best_k
    log.info("Selected k=%d (%s)", chosen_k, "forced" if k else f"max silhouette for k >= {KMEANS_MIN_ACTIONABLE_K}")

    pipe = build_pipeline(chosen_k).fit(X)
    labels = pipe["kmeans"].labels_
    mapping, profiles = name_clusters(rfm, labels)
    rfm["cluster_id"] = labels.astype(int)
    rfm["cluster_label"] = rfm["cluster_id"].map(mapping)
    rfm["snapshot_date"] = RFM_SNAPSHOT_DATE
    rfm["model_version"] = MODEL_VERSION

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_customers": int(len(rfm)),
        "k": int(chosen_k),
        "auto_selected_k": int(best_k),
        "silhouette": float(diag.loc[diag["k"] == chosen_k, "silhouette"].iloc[0]) if chosen_k in diag["k"].values else None,
    }

    # Offline evaluation against synthetic ground truth (only exists for generated data).
    gt_path = raw_dir / "_ground_truth.parquet"
    if gt_path.exists():
        gt = pd.read_parquet(gt_path).merge(rfm[["customer_id", "cluster_id"]], on="customer_id")
        metrics["adjusted_rand_index_vs_latent_type"] = round(float(adjusted_rand_score(gt["latent_type"], gt["cluster_id"])), 4)
        crosstab = pd.crosstab(gt["latent_type"], gt["cluster_id"].map(mapping), normalize="index").round(3)
        crosstab.to_csv(out_dir / "cluster_vs_latent_type.csv")
        log.info("ARI vs hidden behavioural type: %.3f", metrics["adjusted_rand_index_vs_latent_type"])

    # ---- Persist --------------------------------------------------------------
    cols = ["customer_id", "snapshot_date", "recency_days", "frequency", "monetary", "avg_order_value",
            "tenure_days", "first_order_date", "last_order_date", "r_score", "f_score", "m_score", "rfm_code",
            "rfm_segment", "cluster_id", "cluster_label", "p_alive", "historical_clv", "predicted_clv_12m",
            "model_version"]
    rfm[cols].to_parquet(out_dir / "customer_segments.parquet", index=False)
    profiles.to_csv(out_dir / "cluster_profiles.csv", index=False)
    diag.to_csv(out_dir / "k_selection.csv", index=False)
    joblib.dump({"pipeline": pipe, "mapping": mapping, "features": FEATURES, "version": MODEL_VERSION},
                model_dir / "rfm_kmeans.joblib")
    (out_dir / "segmentation_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Wrote segments, profiles, diagnostics and model to %s / %s", out_dir, model_dir)

    if write_db and DATABASE_URL:
        _write_segments_to_db(rfm)

    log.info("Segmentation finished in %.1fs", time.perf_counter() - t0)
    return rfm


def _write_segments_to_db(rfm: pd.DataFrame) -> None:
    """Replace the contents of ecom.customer_segments in one transaction."""
    from src.db import copy_dataframe, get_engine, raw_cursor

    engine = get_engine()
    cols = ["customer_id", "snapshot_date", "recency_days", "frequency", "monetary", "r_score", "f_score",
            "m_score", "rfm_code", "rfm_segment", "cluster_id", "cluster_label", "p_alive", "historical_clv",
            "predicted_clv_12m", "model_version"]
    with raw_cursor(engine) as cur:
        cur.execute(f"TRUNCATE {DB_SCHEMA}.customer_segments")
    n = copy_dataframe(engine, rfm[cols], "customer_segments")
    log.info("Wrote %s rows to %s.customer_segments", f"{n:,}", DB_SCHEMA)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RFM scoring + K-Means segmentation")
    p.add_argument("--k", type=int, default=None, help="force number of clusters (default: auto by silhouette)")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--no-db", action="store_true", help="do not write results back to PostgreSQL")
    args = p.parse_args(argv)
    if args.k is not None and not (2 <= args.k <= MAX_K):
        log.error("--k must be between 2 and %d", MAX_K)
        return 2
    try:
        run(k=args.k, raw_dir=args.raw_dir, write_db=not args.no_db)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
