"""
Static exploratory data analysis report (Matplotlib + Seaborn).

Produces publication-quality PNGs in `reports/figures/` plus a Markdown summary
(`reports/EDA_REPORT.md`) – useful for READMEs, slide decks and code review,
where an interactive dashboard isn't available.

Usage
-----
    python -m src.eda_report
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: safe on servers / CI
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import CATEGORIES, MART_DIR, RAW_DIR, REPORT_DIR, get_logger  # noqa: E402

log = get_logger("eda_report")
FIG_DIR = REPORT_DIR / "figures"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT_COLORS = {c: PALETTE[i] for i, c in enumerate(CATEGORIES)}


def _setup() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "grid.alpha": 0.35,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 110})
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("  saved %s", path.name)
    return path


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name in ("orders", "order_items", "products"):
        if not (RAW_DIR / f"{name}.parquet").exists():
            raise FileNotFoundError(f"Missing {name}.parquet – run `python -m src.generate_data` first")
    orders = pd.read_parquet(RAW_DIR / "orders.parquet")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    items = pd.read_parquet(RAW_DIR / "order_items.parquet").merge(
        pd.read_parquet(RAW_DIR / "products.parquet", columns=["product_id", "category"]), on="product_id")
    items = items.merge(orders[["order_id", "order_date", "order_status", "customer_id"]], on="order_id")
    seg_path = MART_DIR / "customer_segments.parquet"
    seg = pd.read_parquet(seg_path) if seg_path.exists() else pd.DataFrame()
    return orders, items, seg


def build(orders: pd.DataFrame, items: pd.DataFrame, seg: pd.DataFrame) -> list[tuple[str, Path]]:
    figs: list[tuple[str, Path]] = []
    done = items[items["order_status"] == "completed"]

    # 1. Monthly revenue by category (stacked area)
    m = (done.assign(month=done["order_date"].dt.to_period("M").dt.to_timestamp())
         .pivot_table(index="month", columns="category", values="line_revenue", aggfunc="sum").fillna(0))
    m = m[[c for c in CATEGORIES if c in m.columns]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(m.index, m.T.values, labels=m.columns, colors=[CAT_COLORS[c] for c in m.columns],
                 edgecolor="white", linewidth=0.6)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v / 1e6:.1f}M"))
    ax.set_title("Monthly revenue by category")
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=8)
    figs.append(("Monthly revenue by category", _save(fig, "01_revenue_by_category")))

    # 2. Order value distribution (log) by status
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(orders, x="net_amount", hue="order_status", log_scale=True, element="step", fill=False,
                 palette={"completed": PALETTE[0], "returned": PALETTE[3], "cancelled": PALETTE[7]}, ax=ax, bins=60)
    ax.set_title("Order value distribution (log scale)")
    ax.set_xlabel("Net order value (USD)")
    figs.append(("Order value distribution", _save(fig, "02_order_value_distribution")))

    # 3. Weekday x month seasonality heatmap
    o = orders[orders["order_status"] == "completed"]
    hm = o.groupby([o["order_date"].dt.month, o["order_date"].dt.day_of_week]).size().unstack()
    hm = hm.div(hm.values.mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(hm, cmap=sns.blend_palette(SEQ, as_cmap=True), linewidths=1.5, linecolor="white",
                cbar_kws={"label": "index vs mean"}, ax=ax,
                xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                yticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Order seasonality: month × weekday")
    ax.set_xlabel("")
    ax.set_ylabel("")
    figs.append(("Seasonality heatmap", _save(fig, "03_seasonality_heatmap")))

    if not seg.empty:
        # 4. RFM pair plot (log) coloured by persona (sampled)
        s = seg.sample(min(4000, len(seg)), random_state=1).copy()
        for c in ("recency_days", "frequency", "monetary"):
            s[f"log_{c}"] = np.log1p(s[c])
        labels = sorted(s["cluster_label"].unique())
        g = sns.pairplot(s, vars=["log_recency_days", "log_frequency", "log_monetary"], hue="cluster_label",
                         hue_order=labels, palette=PALETTE[: len(labels)], corner=True,
                         plot_kws=dict(s=8, alpha=0.5, linewidth=0), diag_kind="kde")
        g.figure.suptitle("RFM feature space by K-Means persona (log1p)", x=0.02, ha="left", fontweight="bold")
        figs.append(("RFM pair plot", _save(g.figure, "04_rfm_pairplot")))

        # 5. Persona revenue vs headcount share
        p = seg.groupby("cluster_label").agg(customers=("customer_id", "size"), revenue=("monetary", "sum"))
        p = (p / p.sum()).sort_values("revenue")
        fig, ax = plt.subplots(figsize=(8, 4))
        y = np.arange(len(p))
        ax.barh(y - 0.2, p["customers"], height=0.38, color="#8a8985", label="% of customers")
        ax.barh(y + 0.2, p["revenue"], height=0.38, color=PALETTE[0], label="% of revenue")
        ax.set_yticks(y, p.index)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(frameon=False, loc="lower right")
        ax.set_title("Personas: share of customers vs share of revenue")
        figs.append(("Persona value concentration", _save(fig, "05_persona_shares")))

        # 6. Cohort retention heatmap (first 12 months)
        act = done.assign(month=done["order_date"].dt.to_period("M"))[["customer_id", "month"]].drop_duplicates()
        first = act.groupby("customer_id")["month"].min().rename("cohort")
        act = act.join(first, on="customer_id")
        act["period"] = (act["month"] - act["cohort"]).apply(lambda d: d.n)
        tbl = act[act["period"] <= 12].groupby(["cohort", "period"])["customer_id"].nunique().unstack()
        ret = tbl.div(tbl[0], axis=0).iloc[:, 1:]
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.heatmap(ret, cmap=sns.blend_palette(SEQ, as_cmap=True), annot=False, linewidths=0.5, linecolor="white",
                    cbar_kws={"format": mtick.PercentFormatter(1.0)}, ax=ax)
        ax.set_title("Cohort retention (months 1-12 after first purchase)")
        ax.set_xlabel("Months since first purchase")
        ax.set_ylabel("Cohort")
        figs.append(("Cohort retention", _save(fig, "06_cohort_retention")))
    return figs


def write_markdown(figs: list[tuple[str, Path]], orders: pd.DataFrame, items: pd.DataFrame) -> Path:
    done = orders[orders["order_status"] == "completed"]
    lines = [
        "# EDA report",
        "",
        f"* Orders: **{len(orders):,}** ({len(done):,} completed) · order lines: **{len(items):,}**",
        f"* Customers who purchased: **{orders['customer_id'].nunique():,}**",
        f"* Completed net revenue: **${done['net_amount'].sum():,.0f}** · AOV **${done['net_amount'].mean():,.2f}**",
        f"* Return rate: **{(orders['order_status'] == 'returned').mean():.1%}** · "
        f"cancellation rate: **{(orders['order_status'] == 'cancelled').mean():.1%}**",
        "",
    ]
    for title, path in figs:
        lines += [f"## {title}", "", f"![{title}](figures/{path.name})", ""]
    out = REPORT_DIR / "EDA_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    _setup()
    try:
        orders, items, seg = load()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    figs = build(orders, items, seg)
    out = write_markdown(figs, orders, items)
    log.info("EDA report written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
