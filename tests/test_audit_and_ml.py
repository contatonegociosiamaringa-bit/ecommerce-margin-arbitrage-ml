"""Automated test suite for deterministic audit engine, NLP matcher, privacy, SQL, and ML."""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_engine import (
    calculate_expected_marketplace_fees,
    extract_tokens,
    jaccard_token_similarity,
    match_title_to_catalog,
    parse_weight_kg,
    reconcile_order,
    to_decimal,
)
from src.run_sql_validation import run_sql_validation
from src.train_margin_models import CATEGORICAL_FEATURES, NUMERIC_FEATURES

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def test_decimal_parsing_and_weight_conversion() -> None:
    assert to_decimal("R$ 1.234,56") == Decimal("1234.56")
    assert to_decimal("22,90") == Decimal("22.90")
    assert to_decimal(None) is None
    assert to_decimal("N/A") is None

    kg_from_grams, was_g = parse_weight_kg(450)
    assert kg_from_grams == Decimal("0.450")
    assert was_g is True

    kg_direct, was_g2 = parse_weight_kg("0.350")
    assert kg_direct == Decimal("0.350")
    assert was_g2 is False


def test_marketplace_fee_tiers_and_regimes() -> None:
    # Tier 1 <= 79.99: 20% + 4.50
    bd1 = calculate_expected_marketplace_fees("69.90", account_regime="CPF_PADRAO")
    assert bd1.unit_commission == Decimal("13.98")
    assert bd1.unit_fixed_fee == Decimal("4.50")
    assert bd1.expected_total_fees == Decimal("18.48")

    # CPF Alto Volume adds R$ 3.00 per unit
    bd_hv = calculate_expected_marketplace_fees("69.90", account_regime="CPF_ALTO_VOLUME", quantity=2)
    assert bd_hv.unit_cpf_surcharge == Decimal("3.00")
    # unit_price = 34.95 -> 20% = 6.99 + 4.50 + 3.00 = 14.49 * 2 = 28.98
    assert bd_hv.expected_total_fees == Decimal("28.98")

    # Tier 2 (80..99.99): 14% + 16.00
    bd2 = calculate_expected_marketplace_fees("89.90", account_regime="CNPJ_SIMPLES_4PCT")
    assert bd2.unit_commission == Decimal("12.59")
    assert bd2.unit_fixed_fee == Decimal("16.00")
    assert bd2.expected_total_fees == Decimal("28.59")

    # Tier 3 (100..199.99): 14% + 20.00
    bd3 = calculate_expected_marketplace_fees("150.00", account_regime="CNPJ_SIMPLES_4PCT")
    assert bd3.expected_total_fees == Decimal("41.00")

    # Micro-item < 8.00: fixed fee = half of unit price
    bd_low = calculate_expected_marketplace_fees("6.00", account_regime="CPF_PADRAO")
    assert bd_low.unit_fixed_fee == Decimal("3.00")
    assert bd_low.expected_total_fees == Decimal("4.20")


def test_reconcile_order_exact_pilot_cases() -> None:
    # M02 case: cash flow adjustment (10.49) retained by platform for Ads recharge
    m02 = reconcile_order(
        order_id="OBS-M02",
        sku="DIN24116",
        item_price="69.90",
        quantity=1,
        supplier_cost="28.99",
        reported_commission_fee="10.40",
        reported_service_fee="4.58",
        coupon="3.50",
        net_shipping="-2.42",
        cash_flow_adjustment="10.49",
        account_regime="CPF_PADRAO",
    )
    assert m02.observed_payout_brl == Decimal("38.51")
    assert m02.partial_contribution_brl == Decimal("20.01")
    assert m02.audit_classification == "ERRO_COMPROVADO"
    assert "RULE_02_FREIGHT_AND_RETENTION_AUDIT" in m02.triggered_rules

    # Refunded order must be SUSPEITA and non-healthy
    m05 = reconcile_order(
        order_id="OBS-M05",
        sku="TRC0304",
        item_price="22.90",
        supplier_cost="8.99",
        order_status="Reembolsado",
        return_status="REEMBOLSADO",
    )
    assert m05.audit_classification == "SUSPEITA"
    assert m05.partial_contribution_brl is None
    assert m05.is_healthy_margin is False

    # Fee overcharge detection
    overcharge = reconcile_order(
        order_id="TEST-OVER",
        sku="TEST-SKU",
        item_price="50.00",
        supplier_cost="15.00",
        reported_commission_fee="12.00",
        reported_service_fee="5.50",  # 17.50 vs expected 14.50
    )
    assert overcharge.fee_delta_brl == Decimal("3.00")
    assert "RULE_01_PAYOUT_RECONCILIATION" in overcharge.triggered_rules
    assert overcharge.has_financial_divergence is True


def test_nlp_title_matcher() -> None:
    tokens = extract_tokens("Forma de Pão de Queijo 7 Cavidades Silicone Antiaderente [2]")
    assert "forma" in tokens
    assert "queijo" in tokens
    assert "7" in tokens
    assert "de" not in tokens

    sim = jaccard_token_similarity(
        "Dinossauro Rugido Selvagem Som e Luz",
        "Brinquedo Dinossauro T-Rex com Som e Luz Rugido",
    )
    assert sim >= 0.45

    catalog = [
        {"sku": "DIN24116", "title": "Dinossauro Rugido Selvagem Som e Luz Unik", "keywords": ["dinossauro", "t-rex"]},
        {"sku": "TRC0304", "title": "Forma de Pão de Queijo 7 Cavidades Silicone Antiaderente", "keywords": ["air fryer"]},
    ]
    matched_sku, score, layer = match_title_to_catalog("Dinossauro T-Rex Rugido Som Luz", catalog)
    assert matched_sku == "DIN24116"
    assert score >= 0.45
    assert layer == "LAYER_3_NLP_TOKENS"


def test_dataset_schema_and_privacy_compliance() -> None:
    sku_df = pd.read_csv(DATA_DIR / "sku_catalog.csv")
    orders_df = pd.read_csv(DATA_DIR / "orders_reconciliation.csv")

    assert len(sku_df) >= 60
    assert len(orders_df) >= 450
    assert {"OBSERVED_SANITIZED", "SYNTHETIC_AUGMENTED"}.issubset(set(sku_df["data_origin"].unique()))
    assert {"OBSERVED_SANITIZED", "SYNTHETIC_AUGMENTED"}.issubset(set(orders_df["data_origin"].unique()))
    assert (orders_df["data_origin"] == "OBSERVED_SANITIZED").sum() == 16

    forbidden_terms = [
        "".join(["loja", " do ", "brimo"]),
        "".join(["eduardo", "gross"]),
        "".join(["we", "drop"]),
        "".join(["sho", "pee"]),
        "".join(["big", "boss"]),
        "".join(["261002", "gk4sr0j0"]),
        "".join(["users", "\\", "corre"]),
    ]
    files_to_scan = (
        list((PROJECT_ROOT / "data").rglob("*.csv"))
        + list((PROJECT_ROOT / "data").rglob("*.json"))
        + list((PROJECT_ROOT / "src").rglob("*.py"))
        + list((PROJECT_ROOT / "sql").rglob("*.sql"))
        + list((PROJECT_ROOT / "reports").glob("*.json"))
        + list((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
        + ([PROJECT_ROOT / "README.md"] if (PROJECT_ROOT / "README.md").exists() else [])
    )
    for fpath in files_to_scan:
        content = fpath.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in content, f"Forbidden sensitive term '{term}' found in {fpath}"


def test_sql_validation_executes_cleanly() -> None:
    summary = run_sql_validation()
    assert summary["queries_passed"] == summary["queries_executed"]
    assert summary["queries_passed"] >= 6
    for qid, res in summary["results"].items():
        assert res["row_count"] > 0, f"SQL query {qid} returned empty result"


def test_ml_artifacts_and_leak_free_features() -> None:
    forbidden_leak_cols = {
        "reported_fees_brl",
        "fee_delta_brl",
        "net_shipping_brl",
        "observed_payout_brl",
        "partial_contribution_brl",
        "net_profit_brl",
        "net_margin_pct",
        "audit_classification",
        "has_financial_divergence",
        "is_healthy_margin",
    }
    used_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert used_features.isdisjoint(forbidden_leak_cols)

    metrics = json.loads((REPORTS_DIR / "ml_metrics.json").read_text(encoding="utf-8"))
    best_name = metrics["supervised_classification"]["best_model_name"]
    best_holdout = metrics["supervised_classification"]["models"][best_name]["holdout"]
    assert best_holdout["roc_auc"] >= 0.70
    assert 0.0 < best_holdout["accuracy"] < 1.0  # Confirms realistic non-trivial generalization
    assert metrics["unsupervised_clustering"]["silhouette_score"] > 0.20

    clf = joblib.load(MODELS_DIR / "margin_risk_classifier.joblib")
    orders_df = pd.read_csv(DATA_DIR / "orders_reconciliation.csv")
    probs = clf.predict_proba(orders_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(10))
    assert probs.shape == (10, 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

    for fig_name in [
        "margin_distribution_by_category.png",
        "fee_freight_impact.png",
        "sku_clusters_matrix.png",
        "classifier_roc_and_importance.png",
    ]:
        assert (REPORTS_DIR / "figures" / fig_name).exists()
