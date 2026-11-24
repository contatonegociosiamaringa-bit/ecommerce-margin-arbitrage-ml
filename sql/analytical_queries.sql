-- Queries Analíticas de Auditoria Financeira, Margem Líquida e Elasticidade
-- Compatível com PostgreSQL e SQLite (ANSI SQL + Window Functions)

-- QUERY_ID: q1_margin_by_category
-- Margem líquida média, GMV, lucro líquido e taxa de pedidos saudáveis por categoria
SELECT
    category,
    COUNT(*) AS total_orders,
    ROUND(SUM(item_price_brl), 2) AS total_gmv_brl,
    ROUND(SUM(COALESCE(net_profit_brl, 0)), 2) AS total_net_profit_brl,
    ROUND(AVG(net_margin_pct), 2) AS avg_net_margin_pct,
    ROUND(100.0 * SUM(is_healthy_margin) / COUNT(*), 2) AS healthy_order_rate_pct,
    ROUND(100.0 * SUM(has_financial_divergence) / COUNT(*), 2) AS divergence_rate_pct
FROM fact_orders_reconciliation
WHERE audit_classification <> 'SUSPEITA'
GROUP BY category
ORDER BY avg_net_margin_pct DESC;

-- QUERY_ID: q2_freight_weight_impact
-- Impacto das faixas de peso na incidência de frete líquido negativo (cubagem) e erosão de margem
SELECT
    CASE
        WHEN weight_kg <= 0.200 THEN '1_LEVE_ATE_200G'
        WHEN weight_kg <= 0.350 THEN '2_MEDIO_201G_A_350G'
        ELSE '3_PESADO_ACIMA_350G'
    END AS weight_bracket,
    COUNT(*) AS total_orders,
    ROUND(AVG(weight_kg), 3) AS avg_weight_kg,
    SUM(CASE WHEN net_shipping_brl < 0 THEN 1 ELSE 0 END) AS negative_freight_orders,
    ROUND(100.0 * SUM(CASE WHEN net_shipping_brl < 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS freight_loss_incidence_pct,
    ROUND(AVG(CASE WHEN net_shipping_brl < 0 THEN net_shipping_brl ELSE 0 END), 2) AS avg_freight_deduction_brl,
    ROUND(AVG(net_margin_pct), 2) AS avg_net_margin_pct
FROM fact_orders_reconciliation
WHERE audit_classification <> 'SUSPEITA'
GROUP BY weight_bracket
ORDER BY weight_bracket;

-- QUERY_ID: q3_cpf_vs_cnpj_regime
-- Comparação estrutural de carga de taxas, tributos e margem líquida: CPF vs CNPJ Simples Nacional
SELECT
    account_regime,
    COUNT(*) AS total_orders,
    ROUND(AVG(item_price_brl), 2) AS avg_ticket_brl,
    ROUND(AVG(expected_fees_brl / NULLIF(item_price_brl, 0) * 100.0), 2) AS avg_contractual_fee_pct,
    ROUND(AVG(COALESCE(reported_fees_brl, expected_fees_brl) / NULLIF(item_price_brl, 0) * 100.0), 2) AS avg_effective_fee_pct,
    ROUND(AVG(tax_amount_brl / NULLIF(item_price_brl, 0) * 100.0), 2) AS avg_tax_pct,
    ROUND(AVG(net_margin_pct), 2) AS avg_net_margin_pct,
    ROUND(100.0 * SUM(is_healthy_margin) / COUNT(*), 2) AS healthy_rate_pct
FROM fact_orders_reconciliation
WHERE audit_classification <> 'SUSPEITA'
GROUP BY account_regime
ORDER BY avg_net_margin_pct DESC;

-- QUERY_ID: q4_financial_divergence_audit
-- Taxa de divergência financeira por origem dos dados e classificação da auditoria
SELECT
    data_origin,
    audit_classification,
    COUNT(*) AS orders_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY data_origin), 2) AS share_within_origin_pct,
    ROUND(SUM(CASE WHEN fee_delta_brl > 0.01 THEN fee_delta_brl ELSE 0 END), 2) AS total_fee_overcharge_brl,
    ROUND(SUM(CASE WHEN net_shipping_brl < 0 THEN ABS(net_shipping_brl) ELSE 0 END), 2) AS total_freight_loss_brl,
    ROUND(AVG(net_margin_pct), 2) AS avg_net_margin_pct
FROM fact_orders_reconciliation
GROUP BY data_origin, audit_classification
ORDER BY data_origin, orders_count DESC;

-- QUERY_ID: q5_sku_cpa_roi_and_elasticity_ranking
-- Ranking dos Top 15 SKUs por retorno sobre CPA (cpa_roi_ratio), margem realizada e elasticidade
SELECT
    sku,
    category,
    competition_level,
    orders_count,
    ROUND(avg_realized_margin_pct, 2) AS avg_realized_margin_pct,
    ROUND(cpa_target_brl, 2) AS cpa_target_brl,
    ROUND(cpa_roi_ratio, 3) AS cpa_roi_ratio,
    ROUND(price_elasticity_coef, 3) AS price_elasticity_coef,
    RANK() OVER (ORDER BY cpa_roi_ratio DESC, avg_realized_margin_pct DESC) AS cpa_roi_rank
FROM (
    SELECT
        d.sku,
        d.category,
        d.competition_level,
        d.cpa_target_brl,
        d.cpa_roi_ratio,
        d.price_elasticity_coef,
        COUNT(f.order_id) AS orders_count,
        AVG(f.net_margin_pct) AS avg_realized_margin_pct
    FROM dim_sku_catalog d
    INNER JOIN fact_orders_reconciliation f ON d.sku = f.sku
    WHERE f.audit_classification <> 'SUSPEITA'
    GROUP BY d.sku, d.category, d.competition_level, d.cpa_target_brl, d.cpa_roi_ratio, d.price_elasticity_coef
) sub
ORDER BY cpa_roi_rank
LIMIT 15;

-- QUERY_ID: q6_mart_margin_risk_tiers
-- Consolidação do Data Mart mart_margin_risk por faixa de risco operacional
SELECT
    risk_tier,
    COUNT(*) AS sku_regime_pairs,
    SUM(orders_count) AS total_orders,
    ROUND(SUM(gmv_brl), 2) AS total_gmv_brl,
    ROUND(AVG(avg_net_margin_pct), 2) AS mean_margin_pct,
    ROUND(AVG(divergence_rate_pct), 2) AS mean_divergence_rate_pct,
    ROUND(AVG(healthy_margin_rate_pct), 2) AS mean_healthy_rate_pct
FROM mart_margin_risk
GROUP BY risk_tier
ORDER BY mean_margin_pct DESC;

-- QUERY_ID: q7_price_tier_boundary_anomaly
-- Análise de saltos na estrutura de comissão por faixa de preço unitário
SELECT
    CASE
        WHEN (item_price_brl / quantity) <= 79.99 THEN '1_ATE_79_99'
        WHEN (item_price_brl / quantity) <= 99.99 THEN '2_80_A_99_99'
        WHEN (item_price_brl / quantity) <= 199.99 THEN '3_100_A_199_99'
        ELSE '4_ACIMA_200'
    END AS unit_price_tier,
    COUNT(*) AS orders_count,
    ROUND(AVG(expected_fees_brl / NULLIF(item_price_brl, 0) * 100.0), 2) AS avg_expected_fee_pct,
    ROUND(AVG(fee_delta_brl), 2) AS avg_fee_delta_brl,
    ROUND(100.0 * SUM(CASE WHEN fee_delta_brl >= 0.02 THEN 1 ELSE 0 END) / COUNT(*), 2) AS fee_overcharge_rate_pct,
    ROUND(AVG(net_margin_pct), 2) AS avg_net_margin_pct
FROM fact_orders_reconciliation
WHERE audit_classification <> 'SUSPEITA'
GROUP BY unit_price_tier
ORDER BY unit_price_tier;
