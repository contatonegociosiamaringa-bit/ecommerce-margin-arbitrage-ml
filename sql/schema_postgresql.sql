-- Schema PostgreSQL para Auditoria de Margem e Arbitragem E-commerce
-- Tabelas: dim_sku_catalog, fact_orders_reconciliation, mart_margin_risk

CREATE TABLE IF NOT EXISTS dim_sku_catalog (
    sku VARCHAR(40) PRIMARY KEY,
    product_title VARCHAR(255) NOT NULL,
    category VARCHAR(80) NOT NULL,
    supplier_name VARCHAR(60) NOT NULL,
    marketplace_name VARCHAR(60) NOT NULL,
    supplier_cost_brl NUMERIC(10, 2) NOT NULL,
    stock_units INTEGER NOT NULL,
    weight_kg NUMERIC(8, 3) NOT NULL,
    volumetric_cm3 NUMERIC(10, 2) NOT NULL,
    monthly_search_volume INTEGER NOT NULL,
    estimated_niche_sales_month INTEGER NOT NULL,
    competitor_price_min_brl NUMERIC(10, 2) NOT NULL,
    competitor_price_avg_brl NUMERIC(10, 2) NOT NULL,
    competitor_price_max_brl NUMERIC(10, 2) NOT NULL,
    cpa_target_brl NUMERIC(10, 2) NOT NULL,
    competition_level VARCHAR(32) NOT NULL,
    estimated_net_margin_cpf_pct NUMERIC(6, 2) NOT NULL,
    estimated_net_margin_cnpj_pct NUMERIC(6, 2) NOT NULL,
    cpa_roi_ratio NUMERIC(8, 3) NOT NULL,
    price_elasticity_coef NUMERIC(8, 3) NOT NULL,
    data_origin VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_orders_reconciliation (
    order_id VARCHAR(40) PRIMARY KEY,
    order_date DATE NOT NULL,
    seller_id VARCHAR(40) NOT NULL,
    supplier_name VARCHAR(60) NOT NULL,
    marketplace_name VARCHAR(60) NOT NULL,
    sku VARCHAR(40) NOT NULL REFERENCES dim_sku_catalog(sku),
    category VARCHAR(80) NOT NULL,
    account_regime VARCHAR(32) NOT NULL,
    traffic_channel VARCHAR(32) NOT NULL,
    order_status VARCHAR(32) NOT NULL,
    quantity INTEGER NOT NULL,
    item_price_brl NUMERIC(10, 2) NOT NULL,
    supplier_cost_brl NUMERIC(10, 2) NOT NULL,
    weight_kg NUMERIC(8, 3) NOT NULL,
    volumetric_cm3 NUMERIC(10, 2) NOT NULL,
    monthly_search_volume INTEGER NOT NULL,
    competition_level VARCHAR(32) NOT NULL,
    cpa_target_brl NUMERIC(10, 2) NOT NULL,
    price_vs_market_avg_ratio NUMERIC(8, 4) NOT NULL,
    markup_ratio NUMERIC(8, 4) NOT NULL,
    gross_spread_pct NUMERIC(6, 2) NOT NULL,
    coupon_brl NUMERIC(10, 2) NOT NULL,
    coupon_pct NUMERIC(6, 2) NOT NULL,
    expected_tax_pct NUMERIC(5, 2) NOT NULL,
    expected_fee_ratio NUMERIC(6, 4) NOT NULL,
    net_shipping_brl NUMERIC(10, 2) NOT NULL,
    expected_fees_brl NUMERIC(10, 2) NOT NULL,
    reported_fees_brl NUMERIC(10, 2),
    fee_delta_brl NUMERIC(10, 2),
    tax_amount_brl NUMERIC(10, 2) NOT NULL,
    ads_spend_brl NUMERIC(10, 2) NOT NULL,
    cash_flow_adjustment_brl NUMERIC(10, 2) NOT NULL,
    observed_payout_brl NUMERIC(10, 2),
    partial_contribution_brl NUMERIC(10, 2),
    net_profit_brl NUMERIC(10, 2),
    net_margin_pct NUMERIC(6, 2),
    audit_classification VARCHAR(32) NOT NULL,
    triggered_rules VARCHAR(160) NOT NULL,
    has_financial_divergence SMALLINT NOT NULL,
    is_healthy_margin SMALLINT NOT NULL,
    audit_sha256 CHAR(64) NOT NULL,
    data_origin VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS mart_margin_risk (
    sku VARCHAR(40) NOT NULL REFERENCES dim_sku_catalog(sku),
    category VARCHAR(80) NOT NULL,
    account_regime VARCHAR(32) NOT NULL,
    orders_count INTEGER NOT NULL,
    gmv_brl NUMERIC(12, 2) NOT NULL,
    avg_net_margin_pct NUMERIC(6, 2),
    divergence_rate_pct NUMERIC(6, 2) NOT NULL,
    freight_loss_rate_pct NUMERIC(6, 2) NOT NULL,
    avg_fee_delta_brl NUMERIC(10, 2),
    healthy_margin_rate_pct NUMERIC(6, 2) NOT NULL,
    risk_tier VARCHAR(24) NOT NULL,
    PRIMARY KEY (sku, account_regime)
);

CREATE INDEX IF NOT EXISTS idx_fact_orders_sku ON fact_orders_reconciliation(sku);
CREATE INDEX IF NOT EXISTS idx_fact_orders_regime ON fact_orders_reconciliation(account_regime);
CREATE INDEX IF NOT EXISTS idx_fact_orders_class ON fact_orders_reconciliation(audit_classification);
CREATE INDEX IF NOT EXISTS idx_fact_orders_origin ON fact_orders_reconciliation(data_origin);
