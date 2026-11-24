"""Sanitize observed audit fixtures and build reproducible analytical datasets.

Reads the sanitized catalog seed and observed order audit records, applies
privacy scrubbing (replacing any real platform/supplier/seller names with generic
identifiers), and deterministically augments the dataset using fixed seeds and
the exact Decimal audit_engine rules for SQL and ML modeling.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_engine import (  # noqa: E402
    TWO_PLACES,
    calculate_expected_marketplace_fees,
    extract_tokens,
    reconcile_order,
    to_decimal,
)

RANDOM_SEED = 42
TARGET_SKUS = 64
TARGET_ORDERS = 500

RAW_SANITIZED_DIR = PROJECT_ROOT / "data" / "raw_sanitized"
CATALOG_SEED_PATH = RAW_SANITIZED_DIR / "catalog_seed.json"
OBSERVED_ORDERS_PATH = RAW_SANITIZED_DIR / "observed_orders.json"
SKU_CATALOG_CSV = PROJECT_ROOT / "data" / "sku_catalog.csv"
ORDERS_CSV = PROJECT_ROOT / "data" / "orders_reconciliation.csv"

DEFAULT_GENERIC_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    (r"(?i)\bmerchant[_ ]demo\b", "Merchant_Demo"),
)


def scrub_text(
    value: Optional[str],
    extra_replacements: Optional[Tuple[Tuple[str, str], ...]] = None,
) -> Optional[str]:
    """Normalize and scrub partner/seller free-text identifiers into generic placeholders."""
    if value is None:
        return None
    cleaned = str(value)
    for pattern, repl in (extra_replacements or ()) + DEFAULT_GENERIC_REPLACEMENTS:
        cleaned = re.sub(pattern, repl, cleaned)
    return cleaned


def _parse_brazilian_int(raw: Any) -> int:
    """Parse strings like '14.800 buscas' or 'ALTA (24.000 buscas)' into an integer."""
    if raw is None:
        return 10000
    if isinstance(raw, (int, float)):
        return int(raw)
    match = re.search(r"([\d\.]+)", str(raw))
    if not match:
        return 10000
    digits = match.group(1).replace(".", "")
    return int(digits) if digits.isdigit() else 10000


def _infer_competition_from_demand(demand_label: str) -> str:
    upper = (demand_label or "").upper()
    if "EXPLOSIVA" in upper:
        return "HIGH"
    if "ALTA" in upper:
        return "MODERATE_HIGH"
    return "MODERATE"


def _parse_dimensions_cm3(dim_str: str) -> float:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", dim_str or "")]
    if len(nums) >= 3:
        return round(nums[0] * nums[1] * nums[2], 2)
    return 1800.0


def sync_sanitized_fixtures_from_local(
    source_catalog_path: Path,
    source_audit_dir: Path,
    extra_replacements: Optional[Tuple[Tuple[str, str], ...]] = None,
) -> None:
    """Ingest local pilot JSONs, scrub sensitive identifiers, and write sanitized seeds."""
    RAW_SANITIZED_DIR.mkdir(parents=True, exist_ok=True)

    with open(source_catalog_path, "r", encoding="utf-8") as fh:
        raw_cat = json.load(fh)

    sanitized_items: List[Dict[str, Any]] = []
    seen_skus = set()

    for item in raw_cat.get("catalog", []):
        sku = str(item["sku"]).strip().upper()
        mkt = next((v for k, v in item.items() if k.endswith("_market") and isinstance(v, dict)), {})
        search_vol = _parse_brazilian_int(mkt.get("monthly_search_volume"))
        niche_sales = _parse_brazilian_int(mkt.get("estimated_niche_sales_month"))
        weight_kg = float(item.get("weight_kg", 0.25))
        vol_cm3 = _parse_dimensions_cm3(str(item.get("dimensions_cm", "20x15x6")))
        p_avg = float(mkt.get("top_competitor_price_avg", 39.90))
        p_min = float(mkt.get("top_competitor_price_min", round(p_avg * 0.82, 2)))
        p_max = float(mkt.get("top_competitor_price_max", round(p_avg * 1.18, 2)))
        raw_title = next((v for k, v in item.items() if k.endswith("_title") or k == "title"), "")
        raw_cost = next(v for k, v in item.items() if k.endswith("_cost"))

        if sku not in seen_skus:
            seen_skus.add(sku)
            sanitized_items.append(
                {
                    "sku": sku,
                    "product_title": scrub_text(raw_title, extra_replacements),
                    "category": scrub_text(item.get("category", "Utilidades"), extra_replacements),
                    "supplier_name": "Supplier_B2B",
                    "marketplace_name": "Marketplace_BR",
                    "supplier_cost_brl": round(float(raw_cost), 2),
                    "stock_units": int(item.get("stock_units", 100)),
                    "weight_kg": round(weight_kg, 3),
                    "volumetric_cm3": vol_cm3,
                    "monthly_search_volume": search_vol,
                    "estimated_niche_sales_month": niche_sales,
                    "competitor_price_min_brl": round(p_min, 2),
                    "competitor_price_avg_brl": round(p_avg, 2),
                    "competitor_price_max_brl": round(p_max, 2),
                    "cpa_target_brl": round(float(mkt.get("cpa_target_recommended", 4.00)), 2),
                    "competition_level": str(mkt.get("competition_level", "MODERATE")).upper(),
                    "keywords": [scrub_text(k, extra_replacements) for k in item.get("keywords", [])],
                    "data_origin": "OBSERVED_SANITIZED",
                }
            )

        for alt in item.get("similar_alternatives", []):
            alt_sku = str(alt["sku"]).strip().upper()
            if alt_sku in seen_skus:
                continue
            seen_skus.add(alt_sku)
            alt_avg = float(next((v for k, v in alt.items() if k.endswith("_benchmark_price")), 34.90))
            alt_cost = float(next((v for k, v in alt.items() if k.endswith("_cost")), 15.00))
            alt_search = _parse_brazilian_int(alt.get("estimated_monthly_demand"))
            alt_comp = _infer_competition_from_demand(str(alt.get("estimated_monthly_demand", "")))
            sanitized_items.append(
                {
                    "sku": alt_sku,
                    "product_title": scrub_text(alt.get("title", ""), extra_replacements),
                    "category": scrub_text(item.get("category", "Utilidades"), extra_replacements),
                    "supplier_name": "Supplier_B2B",
                    "marketplace_name": "Marketplace_BR",
                    "supplier_cost_brl": round(alt_cost, 2),
                    "stock_units": int(alt.get("stock_units", 150)),
                    "weight_kg": round(max(0.12, weight_kg * (0.85 if alt_cost < 20 else 1.05)), 3),
                    "volumetric_cm3": round(vol_cm3 * 0.95, 2),
                    "monthly_search_volume": alt_search,
                    "estimated_niche_sales_month": int(round(alt_search * 0.28)),
                    "competitor_price_min_brl": round(alt_avg * 0.83, 2),
                    "competitor_price_avg_brl": round(alt_avg, 2),
                    "competitor_price_max_brl": round(alt_avg * 1.22, 2),
                    "cpa_target_brl": round(max(2.50, min(7.00, alt_avg * 0.11)), 2),
                    "competition_level": alt_comp,
                    "keywords": extract_tokens(alt.get("title", "")),
                    "data_origin": "OBSERVED_SANITIZED",
                }
            )

    catalog_seed_payload = {
        "schema_version": "1.0",
        "supplier": "Supplier_B2B",
        "marketplace": "Marketplace_BR",
        "commission_structure": raw_cat.get("commission_structure", {}),
        "items": sanitized_items,
    }
    with open(CATALOG_SEED_PATH, "w", encoding="utf-8") as fh:
        json.dump(catalog_seed_payload, fh, indent=2, ensure_ascii=False)

    audit_files = sorted(glob.glob(str(source_audit_dir / "M*.audit.json")))
    sanitized_orders: List[Dict[str, Any]] = []
    for fpath in audit_files:
        with open(fpath, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        fees = rec.get("fees") or {}
        is_refund = str(rec.get("classification", "")).upper() == "INDETERMINATE"
        repasse_val = next((v for k, v in rec.items() if k.endswith("_repasse_received")), None)
        applied_cost_val = next((v for k, v in rec.items() if k.endswith("_cost_applied")), None)
        sanitized_orders.append(
            {
                "order_id": f"OBS-{rec['order_alias']}",
                "seller_id": "Merchant_Demo",
                "sku": str(rec["sku"]).strip().upper(),
                "product_title": scrub_text(rec.get("product_name", ""), extra_replacements),
                "liquidation_window": rec.get("liquidation_window", "2026-10-01_to_2026-10-15"),
                "item_price_brl": rec.get("item_price"),
                "quantity": 1,
                "coupon_brl": rec.get("coupon") or 0.0,
                "net_shipping_brl": rec.get("net_shipping") if rec.get("net_shipping") is not None else 0.0,
                "reported_commission_brl": fees.get("commission"),
                "reported_service_brl": fees.get("service"),
                "reported_total_fees_brl": fees.get("total"),
                "observed_payout_brl": repasse_val,
                "supplier_cost_brl": applied_cost_val,
                "cash_flow_adjustment_brl": rec.get("cash_flow_adjustment") or 0.0,
                "order_status": "Reembolsado" if is_refund else "Concluído",
                "return_status": "REEMBOLSADO" if is_refund else "",
                "notes": scrub_text(rec.get("notes"), extra_replacements),
                "data_origin": "OBSERVED_SANITIZED",
            }
        )

    with open(OBSERVED_ORDERS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"orders": sanitized_orders}, fh, indent=2, ensure_ascii=False)


SYNTHETIC_TEMPLATES: List[Tuple[str, str, Tuple[float, float], Tuple[float, float], Tuple[int, int]]] = [
    ("Brinquedos e Hobbies", "Kit Blocos Montar Criativo Magnético", (18.0, 42.0), (0.28, 0.75), (8000, 32000)),
    ("Cozinha e Utilidades", "Processador Manual Alho Mini Triturador Inox", (7.5, 24.0), (0.14, 0.42), (14000, 52000)),
    ("Limpeza e Utilidades", "Escova Esfregão Elétrico Rotativo Sem Fio", (14.0, 36.0), (0.22, 0.68), (12000, 46000)),
    ("Esportes e Fitness", "Faixa Elástica Mini Band Kit 5 Intensidades", (9.0, 26.0), (0.12, 0.35), (15000, 48000)),
    ("Iluminação e Casa", "Projetor Estrelas Astronauta LED RGB Controle", (19.0, 48.0), (0.20, 0.55), (16000, 55000)),
    ("Beleza e Cosméticos", "Massageador Facial Elétrico Limpeza Sônica", (12.0, 44.0), (0.10, 0.30), (11000, 62000)),
    ("Brinquedos Educativos", "Lousa Mágica Tablet LCD Infantil Escrita", (11.0, 29.0), (0.15, 0.38), (18000, 44000)),
    ("Eletrônicos e Casa Inteligente", "Tomada Inteligente Wi-Fi Medidor Consumo", (16.0, 45.0), (0.11, 0.32), (10000, 38000)),
]

COMPETITION_CHOICES = ["LOW", "MODERATE", "MODERATE_HIGH", "HIGH", "VIRAL_TIKTOK"]
COMPETITION_PROBS = [0.10, 0.28, 0.24, 0.28, 0.10]


def _compute_sku_catalog_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute theoretical CPF vs CNPJ margins, CPA ROI, and elasticity for a SKU."""
    avg_price = to_decimal(row["competitor_price_avg_brl"]) or Decimal("39.90")
    cost = to_decimal(row["supplier_cost_brl"]) or Decimal("15.00")
    cpa = to_decimal(row["cpa_target_brl"]) or Decimal("4.00")

    cpf_fees = calculate_expected_marketplace_fees(avg_price, account_regime="CPF_PADRAO")
    cnpj_fees = calculate_expected_marketplace_fees(avg_price, account_regime="CNPJ_SIMPLES_4PCT")
    tax_cnpj = (avg_price * Decimal("0.04")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    cpf_contrib = avg_price - cpf_fees.expected_total_fees - cost
    cnpj_contrib = avg_price - cnpj_fees.expected_total_fees - cost - tax_cnpj

    row["estimated_net_margin_cpf_pct"] = float(((cpf_contrib / avg_price) * Decimal("100")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP))
    row["estimated_net_margin_cnpj_pct"] = float(((cnpj_contrib / avg_price) * Decimal("100")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP))
    row["cpa_roi_ratio"] = round(float(cpf_contrib / max(Decimal("0.50"), cpa)), 3)

    price_spread = (row["competitor_price_max_brl"] - row["competitor_price_min_brl"]) / max(1.0, row["competitor_price_avg_brl"])
    comp_factor = {"LOW": -0.8, "MODERATE": -1.2, "MODERATE_HIGH": -1.55, "HIGH": -1.95, "VIRAL_TIKTOK": -2.25}.get(
        row["competition_level"], -1.4
    )
    row["price_elasticity_coef"] = round(comp_factor - (price_spread * 0.6), 3)
    return row


def build_sku_catalog(seed_items: List[Dict[str, Any]], rng: np.random.Generator) -> pd.DataFrame:
    """Combine observed sanitized SKUs with reproducible synthetic SKUs to reach TARGET_SKUS."""
    rows: List[Dict[str, Any]] = []
    for item in seed_items:
        row = {
            "sku": item["sku"],
            "product_title": item["product_title"],
            "category": item["category"],
            "supplier_name": "Supplier_B2B",
            "marketplace_name": "Marketplace_BR",
            "supplier_cost_brl": float(item["supplier_cost_brl"]),
            "stock_units": int(item["stock_units"]),
            "weight_kg": float(item["weight_kg"]),
            "volumetric_cm3": float(item["volumetric_cm3"]),
            "monthly_search_volume": int(item["monthly_search_volume"]),
            "estimated_niche_sales_month": int(item["estimated_niche_sales_month"]),
            "competitor_price_min_brl": float(item["competitor_price_min_brl"]),
            "competitor_price_avg_brl": float(item["competitor_price_avg_brl"]),
            "competitor_price_max_brl": float(item["competitor_price_max_brl"]),
            "cpa_target_brl": float(item["cpa_target_brl"]),
            "competition_level": str(item["competition_level"]),
            "data_origin": "OBSERVED_SANITIZED",
        }
        rows.append(_compute_sku_catalog_metrics(row))

    needed = max(0, TARGET_SKUS - len(rows))
    for idx in range(1, needed + 1):
        cat, base_title, cost_range, weight_range, search_range = SYNTHETIC_TEMPLATES[idx % len(SYNTHETIC_TEMPLATES)]
        variant_num = (idx // len(SYNTHETIC_TEMPLATES)) + 1
        sku = f"SYN-{cat[:3].upper()}-{idx:03d}"

        cost = round(float(rng.uniform(*cost_range)), 2)
        # Create realistic spread of healthy, borderline, and margin-squeezed SKUs
        regime_tier = idx % 5
        if regime_tier == 0:  # Tight/high-cost SKU
            markup = float(rng.uniform(1.42, 1.75))
        elif regime_tier in (1, 2):  # Moderate competitive SKU
            markup = float(rng.uniform(1.85, 2.35))
        else:  # High-margin winner SKU
            markup = float(rng.uniform(2.35, 3.15))

        p_avg = round(max(16.90, min(189.90, cost * markup)) - 0.10, 2)
        if abs(p_avg - round(p_avg)) < 0.05:
            p_avg = round(p_avg - 0.10, 2)
        p_min = round(p_avg * float(rng.uniform(0.78, 0.88)), 2)
        p_max = round(p_avg * float(rng.uniform(1.14, 1.32)), 2)

        weight_kg = round(float(rng.uniform(*weight_range)), 3)
        vol_cm3 = round(float(weight_kg * rng.uniform(4200, 9500)), 1)
        search_vol = int(rng.integers(search_range[0], search_range[1]))
        niche_sales = int(round(search_vol * float(rng.uniform(0.20, 0.36))))
        comp_level = str(rng.choice(COMPETITION_CHOICES, p=COMPETITION_PROBS))
        cpa_mult = {"LOW": 0.07, "MODERATE": 0.09, "MODERATE_HIGH": 0.11, "HIGH": 0.13, "VIRAL_TIKTOK": 0.12}[comp_level]
        cpa_target = round(max(2.20, min(14.00, p_avg * cpa_mult + float(rng.uniform(-0.5, 0.8)))), 2)
        stock_units = int(rng.integers(45, 750))

        row = {
            "sku": sku,
            "product_title": f"{base_title} Modelo V{variant_num}",
            "category": cat,
            "supplier_name": "Supplier_B2B",
            "marketplace_name": "Marketplace_BR",
            "supplier_cost_brl": cost,
            "stock_units": stock_units,
            "weight_kg": weight_kg,
            "volumetric_cm3": vol_cm3,
            "monthly_search_volume": search_vol,
            "estimated_niche_sales_month": niche_sales,
            "competitor_price_min_brl": p_min,
            "competitor_price_avg_brl": p_avg,
            "competitor_price_max_brl": p_max,
            "cpa_target_brl": cpa_target,
            "competition_level": comp_level,
            "data_origin": "SYNTHETIC_AUGMENTED",
        }
        rows.append(_compute_sku_catalog_metrics(row))

    return pd.DataFrame(rows)


def _audit_to_order_row(
    raw_order: Dict[str, Any],
    sku_meta: Dict[str, Any],
    account_regime: str,
    traffic_channel: str,
    order_date: str,
    data_origin: str,
) -> Dict[str, Any]:
    """Convert raw order attributes + Decimal audit result into an analytical CSV record."""
    audit = reconcile_order(
        order_id=raw_order["order_id"],
        sku=raw_order["sku"],
        item_price=raw_order["item_price_brl"],
        quantity=int(raw_order["quantity"]),
        supplier_cost=raw_order["supplier_cost_brl"],
        reported_commission_fee=raw_order["reported_commission_brl"],
        reported_service_fee=raw_order["reported_service_brl"],
        coupon=raw_order["coupon_brl"],
        net_shipping=raw_order["net_shipping_brl"],
        cash_flow_adjustment=raw_order["cash_flow_adjustment_brl"],
        account_regime=account_regime,
        tax_pct=raw_order.get("tax_pct"),
        ads_spend_attributed=raw_order.get("ads_spend_brl", 0.0),
        order_status=raw_order.get("order_status", "Concluído"),
        return_status=raw_order.get("return_status", ""),
    )

    price_f = float(audit.item_price_brl)
    cost_f = float(audit.supplier_cost_brl) if audit.supplier_cost_brl is not None else float(sku_meta["supplier_cost_brl"])
    p_avg = float(sku_meta["competitor_price_avg_brl"])
    expected_fee_ratio = float(audit.expected_fees_brl / audit.item_price_brl) if audit.item_price_brl > 0 else 0.0
    tax_pct_val = 4.0 if "4PCT" in account_regime else (6.0 if "6PCT" in account_regime else 0.0)

    return {
        "order_id": audit.order_id,
        "order_date": order_date,
        "seller_id": raw_order["seller_id"],
        "supplier_name": "Supplier_B2B",
        "marketplace_name": "Marketplace_BR",
        "sku": audit.sku,
        "category": sku_meta["category"],
        "account_regime": account_regime,
        "traffic_channel": traffic_channel,
        "order_status": raw_order.get("order_status", "Concluído"),
        "quantity": audit.quantity,
        "item_price_brl": price_f,
        "supplier_cost_brl": cost_f,
        "weight_kg": float(sku_meta["weight_kg"]),
        "volumetric_cm3": float(sku_meta["volumetric_cm3"]),
        "monthly_search_volume": int(sku_meta["monthly_search_volume"]),
        "competition_level": sku_meta["competition_level"],
        "cpa_target_brl": float(sku_meta["cpa_target_brl"]),
        "price_vs_market_avg_ratio": round(price_f / max(1.0, p_avg), 4),
        "markup_ratio": round(price_f / max(0.01, cost_f), 4),
        "gross_spread_pct": round(((price_f - cost_f) / max(0.01, price_f)) * 100.0, 2),
        "coupon_brl": float(audit.coupon_brl),
        "coupon_pct": round((float(audit.coupon_brl) / max(0.01, price_f)) * 100.0, 2),
        "expected_tax_pct": tax_pct_val,
        "expected_fee_ratio": round(expected_fee_ratio, 4),
        "net_shipping_brl": float(audit.net_shipping_brl),
        "expected_fees_brl": float(audit.expected_fees_brl),
        "reported_fees_brl": float(audit.reported_fees_brl) if audit.reported_fees_brl is not None else np.nan,
        "fee_delta_brl": float(audit.fee_delta_brl) if audit.fee_delta_brl is not None else np.nan,
        "tax_amount_brl": float(audit.tax_amount_brl),
        "ads_spend_brl": float(audit.ads_spend_brl),
        "cash_flow_adjustment_brl": float(audit.cash_flow_adjustment_brl),
        "observed_payout_brl": float(audit.observed_payout_brl) if audit.observed_payout_brl is not None else np.nan,
        "partial_contribution_brl": float(audit.partial_contribution_brl) if audit.partial_contribution_brl is not None else np.nan,
        "net_profit_brl": float(audit.net_profit_brl) if audit.net_profit_brl is not None else np.nan,
        "net_margin_pct": float(audit.net_margin_pct) if audit.net_margin_pct is not None else np.nan,
        "audit_classification": audit.audit_classification,
        "triggered_rules": "|".join(audit.triggered_rules) if audit.triggered_rules else "NONE",
        "has_financial_divergence": int(audit.has_financial_divergence),
        "is_healthy_margin": int(audit.is_healthy_margin),
        "audit_sha256": audit.audit_sha256,
        "data_origin": data_origin,
    }


def build_orders_dataset(
    observed_orders: List[Dict[str, Any]],
    sku_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create full orders dataset combining sanitized pilot orders and deterministic expansion."""
    sku_lookup = sku_df.set_index("sku").to_dict(orient="index")
    all_skus = list(sku_lookup.keys())

    rows: List[Dict[str, Any]] = []
    base_date = pd.Timestamp("2026-10-01")

    for idx, obs in enumerate(observed_orders):
        sku_meta = sku_lookup[obs["sku"]]
        order_date = (base_date + pd.Timedelta(days=idx % 15)).strftime("%Y-%m-%d")
        rows.append(
            _audit_to_order_row(
                raw_order=obs,
                sku_meta=sku_meta,
                account_regime="CPF_PADRAO",
                traffic_channel="UNKNOWN",
                order_date=order_date,
                data_origin="OBSERVED_SANITIZED",
            )
        )

    regimes = ["CPF_PADRAO", "CPF_ALTO_VOLUME", "CNPJ_SIMPLES_4PCT", "CNPJ_SIMPLES_6PCT"]
    regime_probs = [0.25, 0.25, 0.30, 0.20]
    channels = ["ORGANIC", "PAID_ADS", "PROMO_CAMPAIGN"]
    channel_probs = [0.42, 0.38, 0.20]
    sellers = ["Seller_001", "Seller_002", "Seller_003", "Seller_004", "Seller_005", "Seller_006"]

    # Demand weights so higher-demand SKUs appear more realistically while keeping broad coverage
    demand_weights = np.array([max(1000, sku_lookup[s]["monthly_search_volume"]) ** 0.45 for s in all_skus])
    demand_probs = demand_weights / demand_weights.sum()

    needed = max(0, TARGET_ORDERS - len(rows))
    for i in range(1, needed + 1):
        sku = str(rng.choice(all_skus, p=demand_probs))
        meta = sku_lookup[sku]
        account_regime = str(rng.choice(regimes, p=regime_probs))
        traffic_channel = str(rng.choice(channels, p=channel_probs))
        seller_id = str(rng.choice(sellers))
        qty = int(rng.choice([1, 1, 1, 1, 2, 3]))
        order_date = (base_date + pd.Timedelta(days=int(rng.integers(0, 75)))).strftime("%Y-%m-%d")

        # Pricing elasticity around competitor average
        p_min = float(meta["competitor_price_min_brl"])
        p_avg = float(meta["competitor_price_avg_brl"])
        p_max = float(meta["competitor_price_max_brl"])
        unit_price = round(float(np.clip(rng.normal(p_avg, (p_max - p_min) * 0.28), p_min * 0.94, p_max * 1.05)), 2)
        item_price_dec = (Decimal(str(unit_price)) * Decimal(str(qty))).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

        coupon_val = 0.0
        if traffic_channel == "PROMO_CAMPAIGN" or rng.random() < 0.18:
            coupon_pct = float(rng.uniform(0.03, 0.09))
            coupon_val = float((item_price_dec * Decimal(str(coupon_pct))).quantize(TWO_PLACES, rounding=ROUND_HALF_UP))

        # Freight/cubage retention risk rises strongly with weight & volumetric size
        weight_kg = float(meta["weight_kg"])
        vol_cm3 = float(meta["volumetric_cm3"])
        freight_risk_prob = min(0.68, 0.06 + max(0.0, weight_kg - 0.18) * 0.72 + max(0.0, vol_cm3 - 2200.0) / 14000.0)
        net_shipping_val = 0.0
        if rng.random() < freight_risk_prob:
            freight_loss = max(0.75, (weight_kg * float(rng.uniform(2.8, 6.2))) + float(rng.uniform(0.20, 1.40)))
            net_shipping_val = -round(freight_loss, 2)

        # Deterministic contractual fee baseline + stochastic marketplace overcharge anomaly
        fee_bd = calculate_expected_marketplace_fees(item_price_dec, account_regime=account_regime, quantity=qty)
        expected_total = fee_bd.expected_total_fees

        overcharge_prob = 0.10 + (0.08 if "CPF_ALTO" in account_regime else 0.0) + (0.06 if traffic_channel == "PROMO_CAMPAIGN" else 0.0)
        if 76.0 <= unit_price <= 83.0 or 97.0 <= unit_price <= 103.0:
            overcharge_prob += 0.12

        if rng.random() < overcharge_prob:
            delta_err = Decimal(str(round(float(rng.uniform(0.35, 3.80)), 2)))
            reported_total = expected_total + delta_err
        elif rng.random() < 0.15:
            # Slight favorable fee discount
            discount = Decimal(str(round(float(rng.uniform(0.10, min(1.20, float(expected_total) * 0.10))), 2)))
            reported_total = expected_total - discount
        else:
            reported_total = expected_total

        comm_share = Decimal("0.74")
        rep_comm = (reported_total * comm_share).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        rep_serv = (reported_total - rep_comm).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

        ads_spend = 0.0
        if traffic_channel == "PAID_ADS":
            comp_mult = {"LOW": 0.70, "MODERATE": 0.90, "MODERATE_HIGH": 1.05, "HIGH": 1.25, "VIRAL_TIKTOK": 1.15}.get(
                meta["competition_level"], 1.0
            )
            ads_spend = round(max(0.50, float(meta["cpa_target_brl"]) * comp_mult * float(rng.uniform(0.70, 1.45))), 2)

        is_refund = rng.random() < 0.035
        raw_order = {
            "order_id": f"SYN-ORD-{i:04d}",
            "seller_id": seller_id,
            "sku": sku,
            "item_price_brl": float(item_price_dec),
            "quantity": qty,
            "coupon_brl": coupon_val,
            "net_shipping_brl": net_shipping_val,
            "reported_commission_brl": float(rep_comm),
            "reported_service_brl": float(rep_serv),
            "supplier_cost_brl": float(meta["supplier_cost_brl"]),
            "cash_flow_adjustment_brl": 0.0,
            "ads_spend_brl": ads_spend,
            "order_status": "Reembolsado" if is_refund else "Concluído",
            "return_status": "REEMBOLSADO" if is_refund else "",
        }
        rows.append(
            _audit_to_order_row(
                raw_order=raw_order,
                sku_meta=meta,
                account_regime=account_regime,
                traffic_channel=traffic_channel,
                order_date=order_date,
                data_origin="SYNTHETIC_AUGMENTED",
            )
        )

    return pd.DataFrame(rows)


def run_prepare_dataset(
    source_catalog: Optional[Path] = None,
    source_audit_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sku_catalog.csv and orders_reconciliation.csv deterministically."""
    if source_catalog and source_audit_dir:
        sync_sanitized_fixtures_from_local(source_catalog, source_audit_dir)

    if not CATALOG_SEED_PATH.exists() or not OBSERVED_ORDERS_PATH.exists():
        raise FileNotFoundError(
            f"Sanitized seed files not found in {RAW_SANITIZED_DIR}. "
            "Provide --source-catalog and --source-audit-dir to initialize them."
        )

    with open(CATALOG_SEED_PATH, "r", encoding="utf-8") as fh:
        seed_cat = json.load(fh)
    with open(OBSERVED_ORDERS_PATH, "r", encoding="utf-8") as fh:
        seed_orders = json.load(fh)

    rng = np.random.default_rng(RANDOM_SEED)
    sku_df = build_sku_catalog(seed_cat["items"], rng)
    orders_df = build_orders_dataset(seed_orders["orders"], sku_df, rng)

    SKU_CATALOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    sku_df.to_csv(SKU_CATALOG_CSV, index=False, encoding="utf-8")
    orders_df.to_csv(ORDERS_CSV, index=False, encoding="utf-8")
    return sku_df, orders_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sanitized and augmented e-commerce margin datasets.")
    parser.add_argument("--source-catalog", type=Path, default=None, help="Optional raw catalog JSON to sanitize.")
    parser.add_argument("--source-audit-dir", type=Path, default=None, help="Optional directory containing M*.audit.json.")
    args = parser.parse_args()

    sku_df, orders_df = run_prepare_dataset(args.source_catalog, args.source_audit_dir)
    print(f"SKU Catalog generated: {len(sku_df)} rows ({(sku_df['data_origin']=='OBSERVED_SANITIZED').sum()} observed sanitized, {(sku_df['data_origin']=='SYNTHETIC_AUGMENTED').sum()} synthetic augmented)")
    print(f"Orders Reconciliation generated: {len(orders_df)} rows ({(orders_df['data_origin']=='OBSERVED_SANITIZED').sum()} observed sanitized, {(orders_df['data_origin']=='SYNTHETIC_AUGMENTED').sum()} synthetic augmented)")
    print(f"Healthy margin share: {orders_df['is_healthy_margin'].mean():.2%}")


if __name__ == "__main__":
    main()
