-- =============================================================================
-- 03_analytics_views.sql : Semantic layer consumed by the dashboard and the
-- segmentation engine. Heavy aggregates are MATERIALIZED and refreshed by the
-- pipeline (`REFRESH MATERIALIZED VIEW CONCURRENTLY` needs the unique indexes).
-- =============================================================================
SET search_path TO ecom, public;

-- -----------------------------------------------------------------------------
-- vw_fact_sales: denormalised line-level sales fact (the dashboard's grain)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_fact_sales AS
SELECT
    oi.order_item_id,
    o.order_id,
    o.customer_id,
    o.order_date,
    date_trunc('month', o.order_date)::date        AS order_month,
    o.country_code,
    c.country_name,
    c.region,
    o.device,
    o.payment_method,
    o.order_status,
    o.is_first_order,
    p.product_id,
    p.category,
    p.subcategory,
    p.brand,
    oi.quantity,
    oi.unit_price,
    oi.discount_amount,
    oi.line_revenue,
    oi.line_cogs,
    oi.line_revenue - oi.line_cogs                  AS line_gross_profit
FROM order_items oi
JOIN orders      o ON o.order_id    = oi.order_id
JOIN products    p ON p.product_id  = oi.product_id
JOIN dim_country c ON c.country_code = o.country_code;

-- -----------------------------------------------------------------------------
-- mv_monthly_revenue: monthly KPIs by country x category
-- -----------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_monthly_revenue AS
SELECT
    order_month,
    country_code,
    category,
    COUNT(DISTINCT order_id)                         AS orders,
    COUNT(DISTINCT customer_id)                      AS customers,
    SUM(quantity)                                    AS units,
    SUM(line_revenue)                                AS revenue,
    SUM(line_gross_profit)                           AS gross_profit
FROM vw_fact_sales
WHERE order_status = 'completed'
GROUP BY 1, 2, 3
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_monthly_revenue
    ON mv_monthly_revenue (order_month, country_code, category);

-- -----------------------------------------------------------------------------
-- mv_customer_rfm: raw R/F/M inputs. Scoring + clustering happen in Python,
-- where quantile binning and K-Means are first-class.
-- Snapshot date = day after the last observed order.
-- -----------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_customer_rfm AS
WITH snapshot AS (
    SELECT (MAX(order_date) + INTERVAL '1 day')::date AS snapshot_date FROM orders
)
SELECT
    o.customer_id,
    s.snapshot_date,
    (s.snapshot_date - MAX(o.order_date))            AS recency_days,
    COUNT(*)                                         AS frequency,
    SUM(o.net_amount)                                AS monetary,
    MIN(o.order_date)                                AS first_order_date,
    MAX(o.order_date)                                AS last_order_date,
    AVG(o.net_amount)                                AS avg_order_value
FROM orders o
CROSS JOIN snapshot s
WHERE o.order_status = 'completed'
GROUP BY o.customer_id, s.snapshot_date
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_customer_rfm ON mv_customer_rfm (customer_id);

-- -----------------------------------------------------------------------------
-- mv_cohort_retention: % of a first-purchase cohort active N months later
-- -----------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_cohort_retention AS
WITH first_purchase AS (
    SELECT customer_id, date_trunc('month', MIN(order_date))::date AS cohort_month
    FROM orders WHERE order_status = 'completed'
    GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT o.customer_id, date_trunc('month', o.order_date)::date AS activity_month
    FROM orders o WHERE o.order_status = 'completed'
),
cohort_activity AS (
    SELECT
        f.cohort_month,
        ( (EXTRACT(YEAR FROM a.activity_month) - EXTRACT(YEAR FROM f.cohort_month)) * 12
          + EXTRACT(MONTH FROM a.activity_month) - EXTRACT(MONTH FROM f.cohort_month) )::int AS period,
        COUNT(DISTINCT a.customer_id) AS active_customers
    FROM first_purchase f
    JOIN activity a USING (customer_id)
    GROUP BY 1, 2
),
cohort_size AS (
    SELECT cohort_month, active_customers AS cohort_size
    FROM cohort_activity WHERE period = 0
)
SELECT
    ca.cohort_month,
    ca.period,
    cs.cohort_size,
    ca.active_customers,
    ROUND(ca.active_customers::numeric / NULLIF(cs.cohort_size, 0), 4) AS retention_rate
FROM cohort_activity ca
JOIN cohort_size cs USING (cohort_month)
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mv_cohort_retention ON mv_cohort_retention (cohort_month, period);

-- -----------------------------------------------------------------------------
-- Helper: refresh all marts without blocking dashboard readers
-- -----------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE refresh_marts()
LANGUAGE plpgsql AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_monthly_revenue;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_customer_rfm;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_cohort_retention;
END;
$$;
