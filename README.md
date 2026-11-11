# Global E-Commerce Analytics & Customer Segmentation

> **Project 1 of 4 · 2023 · Upper-intermediate ★★★☆☆**

An analytics project for a simulated online retailer. Python generates about 2.3M rows of sales and web data,
RFM scores and K-Means group customers into segments, and a 5-page Power BI report shows the results.
PostgreSQL is optional: everything also runs from local Parquet files.

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![postgres](https://img.shields.io/badge/PostgreSQL-16-336791) ![powerbi](https://img.shields.io/badge/Power%20BI-PBIP-f2c811) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f7931e) ![tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)

## Business problem
The retailer sells in 20 countries and 8 categories. It does not know which customers matter most, so
marketing sends the same emails to everyone. This project asks: how is revenue trending, who are the best
customers, do new customers come back, and which products and channels work best?

## What's inside
| Part | File(s) |
|---|---|
| Data generator and data-quality checks | `src/generate_data.py`, `src/data_quality.py` |
| PostgreSQL schema, indexes and views (optional) | `sql/*.sql`, `src/load_to_postgres.py`, `src/db.py` |
| RFM scores, K-Means segments, simple CLV | `src/rfm_segmentation.py` |
| EDA report with charts | `src/eda_report.py` → `reports/EDA_REPORT.md` |
| Power BI data export and report | `src/export_powerbi.py` → `powerbi/` |
| Pipeline and tests | `run_pipeline.py`, `tests/test_pipeline.py` |
| Docs | [ARCHITECTURE](docs/ARCHITECTURE.md), [DATA_DICTIONARY](docs/DATA_DICTIONARY.md), [POWERBI](docs/POWERBI.md) |

## Quick start
**Parquet only (no database, about 2 minutes):**
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py                                   # add --customers 20000 for a quicker run
pytest                                                   # 20 tests
```
**With PostgreSQL (optional):**
```bash
docker compose up -d postgres
export DATABASE_URL=postgresql+psycopg2://ecom:ecom@localhost:5432/ecommerce
python run_pipeline.py                                   # also loads the tables into PostgreSQL
```

## How segmentation works
1. **RFM** per customer: Recency (days since last order), Frequency (orders), Monetary (spend), plus 1–5 scores.
2. **Scale:** `log1p` (spend and order counts are very skewed), then `StandardScaler`.
3. **K-Means:** try k = 3 to 8; pick the best silhouette score with k ≥ 4 (k = 3 is too coarse to use).
4. **Name the clusters** from their averages, e.g. *VIP Champions* or *Dormant One-Timers*.
5. **Simple CLV:** `P(alive) × orders per year × average order value`; P(alive) drops after a long silence.

## Power BI dashboard
A 5-page Power BI Project with a star schema and 31 DAX measures. To open it: **[docs/POWERBI.md](docs/POWERBI.md)**.

![Executive Overview](powerbi/screenshots/exec.png)

| Page | Shows |
|---|---|
| Executive Overview | KPI cards, monthly revenue trend, revenue by category, region and country |
| Revenue & Products | category → subcategory matrix, discount vs margin, returns, top products |
| Customer Segments | customer and revenue share by segment, RFM treemap, segment profiles |
| Cohort Retention | retention heatmap by sign-up month, retention curves |
| Web Funnel & CLV | conversion funnel, conversion by source and device, predicted value by segment |

| ![Products](powerbi/screenshots/products.png) | ![Segments](powerbi/screenshots/segments.png) |
|---|---|
| ![Cohorts](powerbi/screenshots/cohorts.png) | ![Funnel](powerbi/screenshots/funnel.png) |

## Results (default dataset, seed 42)
| Metric | Value |
|---|---|
| Completed net revenue (2021–2023) | $107.6M |
| Orders / AOV | 285,722 / $421 |
| Weighted web conversion rate | 3.20% |
| Repeat-buyer rate | 36.3% |
| Revenue from the top 20% of customers | 73.9% |

### K-Means segments (k = 4, silhouette 0.348)
| Segment | % customers | % revenue | Median recency | Median orders | Median spend |
|---|---|---|---|---|---|
| VIP Champions | 15.8% | 62.9% | 38 d | 9 | $3,641 |
| Lapsed Big-Ticket | 33.1% | 27.0% | 387 d | 1 | $784 |
| New & Promising | 17.2% | 7.0% | 32 d | 1 | $337 |
| Dormant One-Timers | 34.0% | 3.1% | 422 d | 1 | $104 |

- **About 1 in 6 customers bring almost two thirds of revenue.** Keep them with loyalty perks, not discounts.
- **Lapsed Big-Ticket** customers bought once, spent well, then went quiet: the best win-back target.
- **New & Promising** customers often don't return after month 1, so a second-purchase email matters.

**Validation note.** The data has hidden customer types, so the segments can be checked. The Adjusted Rand
Index is only 0.09, because RFM sees behaviour, not type: a new "whale" looks like any new customer.
Still, 80% of true whales land in VIP Champions and no one-time buyers do.

## Project structure
```
project1_ecommerce_analytics/
├── src/                 # generator, data checks, Postgres loader, segmentation, EDA, Power BI export
├── sql/                 # schema, indexes, analytics views
├── tests/               # pytest suite (20 tests)
├── powerbi/             # .pbip project, data/, theme/, screenshots/
├── reports/             # EDA_REPORT.md + figures/
├── data/, models/       # created by the pipeline (raw, marts, saved model)
├── docs/                # ARCHITECTURE.md, DATA_DICTIONARY.md, POWERBI.md
├── run_pipeline.py      # runs every step in order
└── docker-compose.yml   # optional PostgreSQL 16
```

## Limitations & next steps
- The CLV formula is a simple estimate. A next step is a proper model such as BG/NBD.
- The data is synthetic, so the patterns are cleaner than real data.
- The analysis describes the past; it does not predict churn. SQL logic has no tests. Project 2 adds both.

---
**Series:** **2023 · E-commerce analytics (Power BI)** → [2024 · SaaS revenue & churn (dbt + Tableau)](../project2_saas_churn_analytics)
→ [2025 · Supply-chain digital twin](../project3_supply_chain_twin) → [2026 · Real-time fraud intelligence](../project4_fraud_detection)

*Synthetic data. Any resemblance to real retailers is coincidental.*
