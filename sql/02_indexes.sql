-- =============================================================================
-- 02_indexes.sql : Indexes created AFTER bulk load (much faster than indexing
--                  during COPY). Each index is tied to a dashboard access path.
-- =============================================================================
SET search_path TO ecom, public;

-- Orders: time-range filters, customer lookups, country/date roll-ups
CREATE INDEX IF NOT EXISTS ix_orders_order_date        ON orders (order_date);
CREATE INDEX IF NOT EXISTS ix_orders_customer_date     ON orders (customer_id, order_date);
CREATE INDEX IF NOT EXISTS ix_orders_country_date      ON orders (country_code, order_date);
-- Partial index: ~90% of analytical queries only look at completed revenue
CREATE INDEX IF NOT EXISTS ix_orders_completed_date    ON orders (order_date) WHERE order_status = 'completed';

-- Order items: join to orders and product/category roll-ups
CREATE INDEX IF NOT EXISTS ix_order_items_order        ON order_items (order_id);
CREATE INDEX IF NOT EXISTS ix_order_items_product      ON order_items (product_id);

-- Products: category filter
CREATE INDEX IF NOT EXISTS ix_products_category        ON products (category, subcategory);

-- Customers: cohort & geo
CREATE INDEX IF NOT EXISTS ix_customers_signup         ON customers (signup_date);
CREATE INDEX IF NOT EXISTS ix_customers_country        ON customers (country_code);

-- Sessions: BRIN is ideal for append-only, time-ordered clickstream (tiny + fast)
CREATE INDEX IF NOT EXISTS brin_sessions_date          ON web_sessions USING BRIN (session_date);
CREATE INDEX IF NOT EXISTS ix_sessions_customer        ON web_sessions (customer_id) WHERE customer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_sessions_source          ON web_sessions (traffic_source, session_date);

-- Segments: dashboard filters by persona
CREATE INDEX IF NOT EXISTS ix_segments_cluster         ON customer_segments (cluster_label);
CREATE INDEX IF NOT EXISTS ix_segments_rfm_segment     ON customer_segments (rfm_segment);

ANALYZE;
