-- =============================================================================
-- 01_schema.sql : Core OLTP-style schema for the e-commerce analytics platform
-- Target       : PostgreSQL 14+
-- Idempotent   : safe to re-run; drops and recreates the `ecom` schema objects.
-- =============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS ecom;
SET search_path TO ecom, public;

-- Drop in dependency order so re-runs are clean.
DROP MATERIALIZED VIEW IF EXISTS mv_customer_rfm      CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_monthly_revenue   CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_cohort_retention  CASCADE;
DROP VIEW  IF EXISTS vw_fact_sales                    CASCADE;
DROP TABLE IF EXISTS customer_segments                CASCADE;
DROP TABLE IF EXISTS web_sessions                     CASCADE;
DROP TABLE IF EXISTS order_items                      CASCADE;
DROP TABLE IF EXISTS orders                           CASCADE;
DROP TABLE IF EXISTS products                         CASCADE;
DROP TABLE IF EXISTS customers                        CASCADE;
DROP TABLE IF EXISTS dim_country                      CASCADE;
DROP TABLE IF EXISTS dim_date                         CASCADE;

-- -----------------------------------------------------------------------------
-- Dimensions
-- -----------------------------------------------------------------------------
CREATE TABLE dim_country (
    country_code   CHAR(3)       PRIMARY KEY,               -- ISO 3166-1 alpha-3
    country_name   VARCHAR(64)   NOT NULL,
    region         VARCHAR(32)   NOT NULL,
    currency       CHAR(3)       NOT NULL,                  -- ISO 4217
    price_index    NUMERIC(4,2)  NOT NULL CHECK (price_index > 0)
);

CREATE TABLE dim_date (
    date_key       DATE          PRIMARY KEY,
    year           SMALLINT      NOT NULL,
    quarter        SMALLINT      NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month          SMALLINT      NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name     VARCHAR(9)    NOT NULL,
    week_of_year   SMALLINT      NOT NULL,
    day_of_week    SMALLINT      NOT NULL CHECK (day_of_week BETWEEN 0 AND 6), -- 0 = Monday
    is_weekend     BOOLEAN       NOT NULL,
    is_holiday_season BOOLEAN    NOT NULL                    -- Black Friday -> New Year
);

CREATE TABLE customers (
    customer_id        BIGINT        PRIMARY KEY,
    email_hash         CHAR(64)      NOT NULL UNIQUE,         -- SHA-256; no raw PII stored
    signup_date        DATE          NOT NULL,
    country_code       CHAR(3)       NOT NULL REFERENCES dim_country(country_code),
    age_band           VARCHAR(8)    NOT NULL,
    gender             VARCHAR(16)   NOT NULL,
    acquisition_channel VARCHAR(32)  NOT NULL,
    preferred_device   VARCHAR(16)   NOT NULL,
    is_loyalty_member  BOOLEAN       NOT NULL DEFAULT FALSE,
    marketing_opt_in   BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE TABLE products (
    product_id     INTEGER        PRIMARY KEY,
    sku            VARCHAR(16)    NOT NULL UNIQUE,
    product_name   VARCHAR(128)   NOT NULL,
    category       VARCHAR(32)    NOT NULL,
    subcategory    VARCHAR(32)    NOT NULL,
    brand          VARCHAR(32)    NOT NULL,
    list_price     NUMERIC(10,2)  NOT NULL CHECK (list_price > 0),
    unit_cost      NUMERIC(10,2)  NOT NULL CHECK (unit_cost > 0),
    launch_date    DATE           NOT NULL,
    is_active      BOOLEAN        NOT NULL DEFAULT TRUE,
    CONSTRAINT chk_cost_below_price CHECK (unit_cost < list_price)
);

-- -----------------------------------------------------------------------------
-- Facts
-- -----------------------------------------------------------------------------
CREATE TABLE orders (
    order_id        BIGINT         PRIMARY KEY,
    customer_id     BIGINT         NOT NULL REFERENCES customers(customer_id),
    order_ts        TIMESTAMP      NOT NULL,
    order_date      DATE           NOT NULL,
    country_code    CHAR(3)        NOT NULL REFERENCES dim_country(country_code),
    device          VARCHAR(16)    NOT NULL,
    payment_method  VARCHAR(32)    NOT NULL,
    order_status    VARCHAR(16)    NOT NULL CHECK (order_status IN ('completed','returned','cancelled')),
    item_count      SMALLINT       NOT NULL CHECK (item_count > 0),
    gross_amount    NUMERIC(12,2)  NOT NULL CHECK (gross_amount >= 0),   -- before discount
    discount_amount NUMERIC(12,2)  NOT NULL DEFAULT 0 CHECK (discount_amount >= 0),
    shipping_amount NUMERIC(10,2)  NOT NULL DEFAULT 0 CHECK (shipping_amount >= 0),
    net_amount      NUMERIC(12,2)  NOT NULL CHECK (net_amount >= 0),     -- gross - discount + shipping
    cogs_amount     NUMERIC(12,2)  NOT NULL CHECK (cogs_amount >= 0),
    is_first_order  BOOLEAN        NOT NULL DEFAULT FALSE,
    CONSTRAINT chk_discount_le_gross CHECK (discount_amount <= gross_amount)
);

CREATE TABLE order_items (
    order_item_id   BIGINT         PRIMARY KEY,
    order_id        BIGINT         NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id      INTEGER        NOT NULL REFERENCES products(product_id),
    quantity        SMALLINT       NOT NULL CHECK (quantity > 0),
    unit_price      NUMERIC(10,2)  NOT NULL CHECK (unit_price > 0),
    discount_amount NUMERIC(10,2)  NOT NULL DEFAULT 0 CHECK (discount_amount >= 0),
    line_revenue    NUMERIC(12,2)  NOT NULL CHECK (line_revenue >= 0),   -- qty*price - discount
    line_cogs       NUMERIC(12,2)  NOT NULL CHECK (line_cogs >= 0)
);

CREATE TABLE web_sessions (
    session_id      BIGINT         PRIMARY KEY,
    customer_id     BIGINT         NULL REFERENCES customers(customer_id), -- NULL = anonymous
    session_start   TIMESTAMP      NOT NULL,
    session_date    DATE           NOT NULL,
    country_code    CHAR(3)        NOT NULL REFERENCES dim_country(country_code),
    device          VARCHAR(16)    NOT NULL,
    traffic_source  VARCHAR(32)    NOT NULL,
    landing_page    VARCHAR(32)    NOT NULL,
    pageviews       SMALLINT       NOT NULL CHECK (pageviews >= 1),
    duration_sec    INTEGER        NOT NULL CHECK (duration_sec >= 0),
    added_to_cart   BOOLEAN        NOT NULL DEFAULT FALSE,
    reached_checkout BOOLEAN       NOT NULL DEFAULT FALSE,
    bounced         BOOLEAN        NOT NULL DEFAULT FALSE,
    converted       BOOLEAN        NOT NULL DEFAULT FALSE,
    order_id        BIGINT         NULL REFERENCES orders(order_id),
    -- Non-converting sessions are down-sampled; weight restores population totals.
    sample_weight   SMALLINT       NOT NULL DEFAULT 1 CHECK (sample_weight >= 1),
    CONSTRAINT chk_converted_has_order CHECK (converted = (order_id IS NOT NULL))
);

-- Written back by the segmentation engine (src/rfm_segmentation.py)
CREATE TABLE customer_segments (
    customer_id      BIGINT         PRIMARY KEY REFERENCES customers(customer_id),
    snapshot_date    DATE           NOT NULL,
    recency_days     INTEGER        NOT NULL,
    frequency        INTEGER        NOT NULL,
    monetary         NUMERIC(14,2)  NOT NULL,
    r_score          SMALLINT       NOT NULL CHECK (r_score BETWEEN 1 AND 5),
    f_score          SMALLINT       NOT NULL CHECK (f_score BETWEEN 1 AND 5),
    m_score          SMALLINT       NOT NULL CHECK (m_score BETWEEN 1 AND 5),
    rfm_code         CHAR(3)        NOT NULL,
    rfm_segment      VARCHAR(32)    NOT NULL,   -- rule-based label
    cluster_id       SMALLINT       NOT NULL,
    cluster_label    VARCHAR(32)    NOT NULL,   -- K-Means persona
    p_alive          NUMERIC(6,4)   NOT NULL CHECK (p_alive BETWEEN 0 AND 1),
    historical_clv   NUMERIC(14,2)  NOT NULL,
    predicted_clv_12m NUMERIC(14,2) NOT NULL,
    model_version    VARCHAR(32)    NOT NULL,
    scored_at        TIMESTAMPTZ    NOT NULL DEFAULT now()
);

COMMENT ON TABLE  web_sessions IS 'Clickstream sessions. Non-converting sessions sampled 1:N with sample_weight = N.';
COMMENT ON COLUMN orders.net_amount IS 'Customer-paid amount in USD: gross - discount + shipping.';
COMMENT ON TABLE  customer_segments IS 'RFM scores + K-Means cluster per customer, refreshed by the segmentation job.';

COMMIT;
