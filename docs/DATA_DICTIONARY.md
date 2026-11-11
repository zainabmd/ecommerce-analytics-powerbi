# Data Dictionary

Tables are saved as Parquet in `data/raw/` and, if PostgreSQL is used, in the `ecom` schema.
Money is in USD. Timestamps have no time zone. Row counts are for the default run (100,000 customers, seed 42).

---

## Dimensions

### `dim_country` — 20 rows
| Column | Type | Description |
|---|---|---|
| `country_code` **PK** | CHAR(3) | ISO 3-letter country code |
| `country_name` | VARCHAR(64) | Display name |
| `region` | VARCHAR(32) | North America · Latin America · Europe · Asia Pacific · Middle East & Africa |
| `currency` | CHAR(3) | Local currency code (for info only; amounts are USD) |
| `price_index` | NUMERIC(4,2) | Price level vs the US (1.00); scales price and cost |

### `dim_date` — 1,095 rows
| Column | Type | Description |
|---|---|---|
| `date_key` **PK** | DATE | Calendar day, 2021-01-01 to 2023-12-31 |
| `year`, `quarter`, `month`, `week_of_year` | SMALLINT | Calendar parts (ISO week) |
| `month_name` | VARCHAR(9) | e.g. `November` |
| `day_of_week` | SMALLINT | 0 = Monday … 6 = Sunday |
| `is_weekend` | BOOLEAN | Saturday or Sunday |
| `is_holiday_season` | BOOLEAN | 22 Nov → 1 Jan (Black Friday to New Year) |

### `products` — 2,000 rows
| Column | Type | Description |
|---|---|---|
| `product_id` **PK** | INTEGER | Surrogate key |
| `sku` | VARCHAR(16) UNIQUE | `SKU0000001` … |
| `product_name` | VARCHAR(128) | `<Brand> <Subcategory> <Model>` |
| `category` | VARCHAR(32) | 8 categories: Electronics, Fashion, Home & Kitchen, Beauty, Sports & Outdoors, Books & Media, Grocery, Toys & Kids |
| `subcategory` | VARCHAR(32) | 4–5 per category |
| `brand` | VARCHAR(32) | 20 synthetic brands |
| `list_price` | NUMERIC(10,2) | US list price |
| `unit_cost` | NUMERIC(10,2) | Cost, always below `list_price` |
| `launch_date` | DATE | Never sold before this date |
| `is_active` | BOOLEAN | ~4% discontinued |

### `customers` — 100,000 rows
| Column | Type | Description |
|---|---|---|
| `customer_id` **PK** | BIGINT | Starts at 1,000,001 |
| `email_hash` | CHAR(64) UNIQUE | SHA-256 of the email address. **No raw PII is stored** |
| `signup_date` | DATE | Account creation; grows ~80%/yr |
| `country_code` **FK** | CHAR(3) | → `dim_country` |
| `age_band` | VARCHAR(8) | 18-24 … 65+ |
| `gender` | VARCHAR(16) | Female · Male · Non-binary · Undisclosed |
| `acquisition_channel` | VARCHAR(32) | Organic Search · Paid Search · Social · Email · Referral · Direct · Affiliate |
| `preferred_device` | VARCHAR(16) | Mobile · Desktop · Tablet |
| `is_loyalty_member` | BOOLEAN | Loyalty programme enrolment |
| `marketing_opt_in` | BOOLEAN | Consent for marketing email (GDPR-style flag) |

---

## Facts

### `orders` — ~295k rows (one row per checkout)
| Column | Type | Description |
|---|---|---|
| `order_id` **PK** | BIGINT | Assigned in time order, starts at 10,000,001 |
| `customer_id` **FK** | BIGINT | → `customers` |
| `order_ts` / `order_date` | TIMESTAMP / DATE | Order time (with weekly, hourly and holiday patterns) |
| `country_code` **FK** | CHAR(3) | Ship-to country |
| `device` | VARCHAR(16) | Device used (70% = customer's preferred device) |
| `payment_method` | VARCHAR(32) | Credit Card, PayPal, Debit Card, Apple Pay, BNPL, Gift Card |
| `order_status` | VARCHAR(16) | `completed` · `returned` · `cancelled` (Fashion baskets return ~2.5× more often) |
| `item_count` | SMALLINT | Total units |
| `gross_amount` | NUMERIC(12,2) | Σ quantity × unit_price |
| `discount_amount` | NUMERIC(12,2) | Promotional discount (deeper and more frequent in November) |
| `shipping_amount` | NUMERIC(10,2) | Free above about $75, otherwise $4.99–9.99 |
| `net_amount` | NUMERIC(12,2) | **gross − discount + shipping** |
| `cogs_amount` | NUMERIC(12,2) | Cost of goods sold |
| `is_first_order` | BOOLEAN | Customer's first order (any status) |

### `order_items` — ~830k rows (one row per product line)
| Column | Type | Description |
|---|---|---|
| `order_item_id` **PK** | BIGINT | |
| `order_id` **FK** | BIGINT | → `orders` (ON DELETE CASCADE) |
| `product_id` **FK** | INTEGER | → `products` |
| `quantity` | SMALLINT | 1–4 |
| `unit_price` | NUMERIC(10,2) | Price paid per unit (local prices, ~4%/yr inflation) |
| `discount_amount` | NUMERIC(10,2) | Line share of the order discount |
| `line_revenue` | NUMERIC(12,2) | quantity × unit_price − discount |
| `line_cogs` | NUMERIC(12,2) | quantity × unit_cost × price index × inflation |

### `web_sessions` — ~1.2M rows
| Column | Type | Description |
|---|---|---|
| `session_id` **PK** | BIGINT | Assigned in time order |
| `customer_id` **FK, nullable** | BIGINT | NULL for anonymous visitors |
| `session_start` / `session_date` | TIMESTAMP / DATE | |
| `country_code` **FK** | CHAR(3) | |
| `device` | VARCHAR(16) | Mobile · Desktop · Tablet |
| `traffic_source` | VARCHAR(32) | Where the visit came from |
| `landing_page` | VARCHAR(32) | `/`, `/category`, `/product`, `/search`, `/promo` |
| `pageviews` | SMALLINT | ≥ 1 |
| `duration_sec` | INTEGER | Session length |
| `added_to_cart`, `reached_checkout`, `bounced`, `converted` | BOOLEAN | Funnel flags |
| `order_id` **FK, nullable** | BIGINT | Set only when `converted` (enforced by CHECK) |
| `sample_weight` | SMALLINT | **1** for buying sessions, **10** for others (only 1 in 10 is stored). Always weight totals by it |

> **Why sample?** A ~3.2% conversion rate would need ~9M non-buying sessions. Storing 1 in 10 keeps the table
> at about 1.2M rows, and weighted metrics stay correct.

---

## Analytics layer

### `customer_segments` — one row per purchasing customer (written by `src/rfm_segmentation.py`)
| Column | Type | Description |
|---|---|---|
| `customer_id` **PK/FK** | BIGINT | |
| `snapshot_date` | DATE | "Today" for recency: 2024-01-01 |
| `recency_days` | INTEGER | Days since last completed order |
| `frequency` | INTEGER | Number of completed orders |
| `monetary` | NUMERIC(14,2) | Lifetime completed net revenue |
| `r_score`, `f_score`, `m_score` | SMALLINT | Quintile scores 1–5 (5 = best; ties broken by rank) |
| `rfm_code` | CHAR(3) | e.g. `545` |
| `rfm_segment` | VARCHAR(32) | Rule-based segment (Champions, At Risk, Hibernating …) |
| `cluster_id` | SMALLINT | Raw K-Means label |
| `cluster_label` | VARCHAR(32) | K-Means segment name (VIP Champions, Lapsed Big-Ticket …) |
| `p_alive` | NUMERIC(6,4) | Estimated chance the customer is still active |
| `historical_clv` | NUMERIC(14,2) | = monetary |
| `predicted_clv_12m` | NUMERIC(14,2) | P(alive) × expected 12-month orders × AOV |
| `model_version` | VARCHAR(32) | e.g. `rfm-kmeans-1.0.0` |
| `scored_at` | TIMESTAMPTZ | Load time |

### Views
| Object | Kind | Grain / purpose |
|---|---|---|
| `vw_fact_sales` | VIEW | Order lines joined to order, product and country |
| `mv_monthly_revenue` | MATERIALIZED VIEW | month × country × category. Completed orders, revenue, units, gross profit |
| `mv_customer_rfm` | MATERIALIZED VIEW | customer. Raw R/F/M inputs plus first and last order dates and AOV |
| `mv_cohort_retention` | MATERIALIZED VIEW | cohort month × months since first purchase. Active customers and retention rate |
| `refresh_marts()` | PROCEDURE | Refreshes the three materialized views |

## Data-quality contract (`src/data_quality.py`)
Every batch is checked before it is saved:
- primary keys are unique and non-null
- every foreign key resolves (nullable FKs are checked when set)
- money columns are non-negative
- `net_amount = gross − discount + shipping` (±$0.01)
- line revenue reconciles to the order header (±$0.05)
- order status is in the allowed set
- `converted ⇔ order_id IS NOT NULL`
- `pageviews ≥ 1`

All problems are reported together in one `DataQualityError`.
