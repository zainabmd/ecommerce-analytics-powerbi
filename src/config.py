"""
Central configuration for the E-Commerce Analytics platform.

All tunables live here so the generator, loader, segmentation engine and
Power BI export share one source of truth. Values can be overridden with environment
variables (see `.env.example`) which keeps secrets such as DATABASE_URL out of
source control.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("ECOM_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # generator output (Parquet, one file per table)
MART_DIR = DATA_DIR / "marts"       # analytical outputs (RFM, cohorts, CLV)
MODEL_DIR = PROJECT_ROOT / "models" # persisted scikit-learn artefacts
REPORT_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"

for _d in (RAW_DIR, MART_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
# Example: postgresql+psycopg2://ecom:ecom@localhost:5432/ecommerce
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
DB_SCHEMA = os.getenv("ECOM_DB_SCHEMA", "ecom")

# --------------------------------------------------------------------------- #
# Simulation parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimulationConfig:
    """Knobs controlling the synthetic data generator.

    The defaults produce ~3M rows in total (well above the 1M requirement):
    100k customers, 2k products, ~400k orders, ~1M order lines and ~1.6M
    (down-sampled) web sessions.
    """

    seed: int = 42
    n_customers: int = 100_000
    n_products: int = 2_000
    start_date: date = date(2021, 1, 1)
    end_date: date = date(2023, 12, 31)
    # Non-converting sessions are down-sampled 1:N and carry sample_weight=N.
    # This keeps the table tractable while preserving true conversion rates.
    negative_session_sample_rate: int = 10
    target_conversion_rate: float = 0.032
    # Order outcome probabilities.
    p_cancelled: float = 0.04
    p_returned: float = 0.06
    countries: dict[str, dict] = field(default_factory=lambda: COUNTRIES)


# ISO-3 codes are used by Plotly choropleths. `weight` = share of customers,
# `price_index` scales basket value, `region` drives regional roll-ups.
COUNTRIES: dict[str, dict] = {
    "USA": {"name": "United States", "region": "North America", "weight": 0.24, "price_index": 1.00, "currency": "USD"},
    "CAN": {"name": "Canada", "region": "North America", "weight": 0.05, "price_index": 0.95, "currency": "CAD"},
    "MEX": {"name": "Mexico", "region": "Latin America", "weight": 0.04, "price_index": 0.70, "currency": "MXN"},
    "BRA": {"name": "Brazil", "region": "Latin America", "weight": 0.05, "price_index": 0.68, "currency": "BRL"},
    "ARG": {"name": "Argentina", "region": "Latin America", "weight": 0.02, "price_index": 0.60, "currency": "ARS"},
    "GBR": {"name": "United Kingdom", "region": "Europe", "weight": 0.09, "price_index": 1.05, "currency": "GBP"},
    "DEU": {"name": "Germany", "region": "Europe", "weight": 0.08, "price_index": 1.02, "currency": "EUR"},
    "FRA": {"name": "France", "region": "Europe", "weight": 0.06, "price_index": 1.00, "currency": "EUR"},
    "ESP": {"name": "Spain", "region": "Europe", "weight": 0.04, "price_index": 0.90, "currency": "EUR"},
    "ITA": {"name": "Italy", "region": "Europe", "weight": 0.04, "price_index": 0.92, "currency": "EUR"},
    "NLD": {"name": "Netherlands", "region": "Europe", "weight": 0.03, "price_index": 1.04, "currency": "EUR"},
    "SWE": {"name": "Sweden", "region": "Europe", "weight": 0.02, "price_index": 1.08, "currency": "SEK"},
    "IND": {"name": "India", "region": "Asia Pacific", "weight": 0.07, "price_index": 0.45, "currency": "INR"},
    "JPN": {"name": "Japan", "region": "Asia Pacific", "weight": 0.05, "price_index": 1.10, "currency": "JPY"},
    "AUS": {"name": "Australia", "region": "Asia Pacific", "weight": 0.04, "price_index": 1.06, "currency": "AUD"},
    "SGP": {"name": "Singapore", "region": "Asia Pacific", "weight": 0.02, "price_index": 1.12, "currency": "SGD"},
    "KOR": {"name": "South Korea", "region": "Asia Pacific", "weight": 0.03, "price_index": 1.00, "currency": "KRW"},
    "ARE": {"name": "United Arab Emirates", "region": "Middle East & Africa", "weight": 0.02, "price_index": 1.15, "currency": "AED"},
    "ZAF": {"name": "South Africa", "region": "Middle East & Africa", "weight": 0.02, "price_index": 0.62, "currency": "ZAR"},
    "NGA": {"name": "Nigeria", "region": "Middle East & Africa", "weight": 0.01, "price_index": 0.50, "currency": "NGN"},
}

# Category -> (share of catalogue, base price range USD, gross margin, popularity weight)
CATEGORIES: dict[str, dict] = {
    "Electronics": {"subcats": ["Smartphones", "Laptops", "Audio", "Wearables", "Accessories"], "price": (25, 1400), "margin": 0.18, "weight": 0.16},
    "Fashion": {"subcats": ["Menswear", "Womenswear", "Footwear", "Bags", "Jewellery"], "price": (12, 260), "margin": 0.55, "weight": 0.20},
    "Home & Kitchen": {"subcats": ["Cookware", "Furniture", "Decor", "Bedding", "Appliances"], "price": (10, 650), "margin": 0.38, "weight": 0.14},
    "Beauty": {"subcats": ["Skincare", "Makeup", "Fragrance", "Haircare"], "price": (6, 180), "margin": 0.62, "weight": 0.12},
    "Sports & Outdoors": {"subcats": ["Fitness", "Cycling", "Camping", "Team Sports"], "price": (12, 900), "margin": 0.35, "weight": 0.09},
    "Books & Media": {"subcats": ["Fiction", "Non-Fiction", "Music", "Games"], "price": (5, 80), "margin": 0.28, "weight": 0.10},
    "Grocery": {"subcats": ["Pantry", "Snacks", "Beverages", "Organic"], "price": (2, 60), "margin": 0.22, "weight": 0.11},
    "Toys & Kids": {"subcats": ["Toys", "Baby", "Learning", "Outdoor Play"], "price": (6, 220), "margin": 0.40, "weight": 0.08},
}

ACQUISITION_CHANNELS = {"Organic Search": 0.30, "Paid Search": 0.20, "Social": 0.17, "Email": 0.08, "Referral": 0.10, "Direct": 0.12, "Affiliate": 0.03}
DEVICES = {"Mobile": 0.58, "Desktop": 0.34, "Tablet": 0.08}
PAYMENT_METHODS = {"Credit Card": 0.46, "PayPal": 0.20, "Debit Card": 0.17, "Apple Pay": 0.09, "Buy Now Pay Later": 0.05, "Gift Card": 0.03}
LANDING_PAGES = {"/": 0.34, "/category": 0.24, "/product": 0.26, "/search": 0.09, "/promo": 0.07}

# --------------------------------------------------------------------------- #
# Segmentation parameters
# --------------------------------------------------------------------------- #
RFM_SNAPSHOT_DATE = date(2024, 1, 1)  # "today" for recency, the day after data ends
KMEANS_K_RANGE = range(3, 9)
KMEANS_MIN_ACTIONABLE_K = 4  # business floor: fewer personas are too coarse for CRM targeting
KMEANS_SILHOUETTE_SAMPLE = 20_000  # silhouette is O(n^2); sample for speed
CLV_HORIZON_MONTHS = 12

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("ECOM_LOG_LEVEL", "INFO")


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent, grep-friendly format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    return logger
