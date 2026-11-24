"""Deterministic Decimal Financial Audit Engine & NLP Token Matcher.

Adapted for e-commerce marketplace reconciliation (CPF vs CNPJ Simples Nacional),
weight/cubage freight auditing, and Portuguese SEO title matching without any
floating-point rounding drift.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

TWO_PLACES = Decimal("0.01")
THREE_PLACES = Decimal("0.001")
TOLERANCE = Decimal("0.01")
ENGINE_VERSION = "Ecommerce-Margin-Audit-v2.0-Decimal"
RULE_VERSION = "2.1"

STOPWORDS_PT: Set[str] = {
    "de", "da", "do", "das", "dos", "para", "com", "sem", "em", "no", "na",
    "nos", "nas", "por", "um", "uma", "e", "ou", "o", "a", "os", "as", "que",
    "tipo", "ja", "cor", "kit", "pecas", "peca", "unidades", "unidade",
    "infantil", "portatil", "manual", "automatica", "automatico", "original",
    "novo", "nova", "alta", "qualidade", "pratica", "pratico", "facil", "usar",
    "limpar", "sortida", "sortido", "cores", "envio", "imediato", "oferta",
}

ACCOUNT_REGIME_DEFAULTS: Dict[str, Dict[str, Decimal]] = {
    "CPF_PADRAO": {
        "cpf_surcharge": Decimal("0.00"),
        "tax_pct": Decimal("0.00"),
    },
    "CPF_ALTO_VOLUME": {
        "cpf_surcharge": Decimal("3.00"),
        "tax_pct": Decimal("0.00"),
    },
    "CNPJ_SIMPLES_4PCT": {
        "cpf_surcharge": Decimal("0.00"),
        "tax_pct": Decimal("4.00"),
    },
    "CNPJ_SIMPLES_6PCT": {
        "cpf_surcharge": Decimal("0.00"),
        "tax_pct": Decimal("6.00"),
    },
}


@dataclass(frozen=True)
class FeeBreakdown:
    """Breakdown of expected contractual marketplace fees for an order."""

    unit_price: Decimal
    quantity: int
    unit_commission: Decimal
    unit_fixed_fee: Decimal
    unit_cpf_surcharge: Decimal
    expected_total_fees: Decimal
    commission_rate_applied: Decimal
    tier_label: str


@dataclass(frozen=True)
class OrderAuditResult:
    """Deterministic financial reconciliation output for a single order."""

    order_id: str
    sku: str
    account_regime: str
    item_price_brl: Decimal
    quantity: int
    supplier_cost_brl: Optional[Decimal]
    coupon_brl: Decimal
    net_shipping_brl: Decimal
    cash_flow_adjustment_brl: Decimal
    expected_fees_brl: Decimal
    reported_fees_brl: Optional[Decimal]
    fee_delta_brl: Optional[Decimal]
    tax_amount_brl: Decimal
    ads_spend_brl: Decimal
    observed_payout_brl: Optional[Decimal]
    partial_contribution_brl: Optional[Decimal]
    net_profit_brl: Optional[Decimal]
    net_margin_pct: Optional[Decimal]
    audit_classification: str  # OK, ERRO_COMPROVADO, SUSPEITA
    triggered_rules: Tuple[str, ...]
    has_financial_divergence: bool
    is_healthy_margin: bool
    audit_sha256: str


def sha256_text(text: str) -> str:
    """Return lowercase hexadecimal SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_text(val: Any) -> str:
    """Strip invisible chars/accents, lowercase, and normalize whitespace."""
    if val is None:
        return ""
    s = (
        str(val)
        .replace("\u2060", "")
        .replace("\ufeff", "")
        .strip()
        .strip('"')
        .strip("'")
    )
    nfkd = unicodedata.normalize("NFKD", s)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_accents.lower().split())


def clean_ad_title(val: Any) -> str:
    """Normalize marketplace listing title removing variant suffixes like [1] and punctuation."""
    s = norm_text(val)
    s = re.sub(r"\[\d+\]\s*$", "", s).strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def extract_tokens(val: Any) -> List[str]:
    """Extract informative Portuguese tokens (keeping numbers/dimensions) for SEO matching."""
    cleaned = clean_ad_title(val)
    tokens: List[str] = []
    for word in cleaned.split():
        if word in STOPWORDS_PT:
            continue
        if word.isdigit() or len(word) >= 3:
            tokens.append(word)
    return tokens


def jaccard_token_similarity(title_a: str, title_b: str) -> float:
    """Compute Jaccard similarity over extracted Portuguese informative tokens."""
    toks_a = set(extract_tokens(title_a))
    toks_b = set(extract_tokens(title_b))
    if not toks_a or not toks_b:
        return 0.0
    union = toks_a | toks_b
    return len(toks_a & toks_b) / len(union)


def match_title_to_catalog(
    ad_title: str,
    catalog_entries: Sequence[Dict[str, Any]],
    min_score: float = 0.45,
) -> Tuple[Optional[str], float, str]:
    """Match an ad/listing title against catalog entries using 3-layer deterministic + NLP scoring.

    Layer 1: Exact normalized title match.
    Layer 2: Prefix match (>= 30 chars) on cleaned title.
    Layer 3: Weighted token overlap + Jaccard similarity.
    """
    cleaned_ad = clean_ad_title(ad_title)
    if not cleaned_ad:
        return None, 0.0, "NO_TITLE"

    for entry in catalog_entries:
        title = entry.get("title") or entry.get("product_name") or ""
        if clean_ad_title(title) == cleaned_ad:
            return str(entry["sku"]), 1.0, "LAYER_1_EXACT_TITLE"

    for entry in catalog_entries:
        title = entry.get("title") or entry.get("product_name") or ""
        cleaned_cat = clean_ad_title(title)
        if len(cleaned_cat) >= 24 and (
            cleaned_ad.startswith(cleaned_cat[:24]) or cleaned_cat.startswith(cleaned_ad[:24])
        ):
            return str(entry["sku"]), 0.92, "LAYER_2_PREFIX"

    ad_tokens = set(extract_tokens(ad_title))
    if len(ad_tokens) < 2:
        return None, 0.0, "INSUFFICIENT_TOKENS"

    best_sku: Optional[str] = None
    best_score = 0.0
    for entry in catalog_entries:
        title = entry.get("title") or entry.get("product_name") or ""
        kw_list = entry.get("keywords") or []
        cat_tokens = set(extract_tokens(title))
        for kw in kw_list:
            cat_tokens.update(extract_tokens(kw))
        if not cat_tokens:
            continue
        inter = len(ad_tokens & cat_tokens)
        if inter < 2:
            continue
        jaccard = inter / len(ad_tokens | cat_tokens)
        recall_ad = inter / len(ad_tokens)
        score = (0.60 * jaccard) + (0.40 * recall_ad)
        if score > best_score:
            best_score = score
            best_sku = str(entry["sku"])

    if best_sku and best_score >= min_score:
        return best_sku, round(best_score, 4), "LAYER_3_NLP_TOKENS"
    return None, round(best_score, 4), "UNMATCHED"


def to_decimal(val: Any) -> Optional[Decimal]:
    """Safely parse monetary/numeric values into 2-decimal Decimal. Never coerces missing to 0."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, Decimal):
        return val.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    if isinstance(val, (int, float)):
        if val != val:  # NaN guard
            return None
        return Decimal(str(val)).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    s = str(val).strip().replace("R$", "").replace("r$", "").replace("%", "").replace(" ", "")
    if not s or s.lower() in {"-", "—", "null", "none", "n/a", "nan"}:
        return None

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")

    try:
        return Decimal(s).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def parse_weight_kg(val: Any) -> Tuple[Decimal, bool]:
    """Normalize weight to kg; values > 50 are interpreted as grams and divided by 1000."""
    dec_val = to_decimal(val)
    if dec_val is None or dec_val <= Decimal("0"):
        return Decimal("0.250"), False
    if dec_val > Decimal("50"):
        converted = (dec_val / Decimal("1000")).quantize(THREE_PLACES, rounding=ROUND_HALF_UP)
        return converted, True
    return dec_val.quantize(THREE_PLACES, rounding=ROUND_HALF_UP), False


def get_regime_params(account_regime: str) -> Dict[str, Decimal]:
    """Return CPF high-volume surcharge and Simples Nacional tax rate for a regime key."""
    key = (account_regime or "CPF_PADRAO").strip().upper()
    if key in ACCOUNT_REGIME_DEFAULTS:
        return ACCOUNT_REGIME_DEFAULTS[key]
    if "ALTO" in key or "450" in key:
        return ACCOUNT_REGIME_DEFAULTS["CPF_ALTO_VOLUME"]
    if "CNPJ" in key:
        if "6" in key:
            return ACCOUNT_REGIME_DEFAULTS["CNPJ_SIMPLES_6PCT"]
        return ACCOUNT_REGIME_DEFAULTS["CNPJ_SIMPLES_4PCT"]
    return ACCOUNT_REGIME_DEFAULTS["CPF_PADRAO"]


def calculate_expected_marketplace_fees(
    sell_price: Any,
    account_regime: str = "CPF_PADRAO",
    comm_pct: Decimal = Decimal("20.00"),
    fixed_fee_unit: Decimal = Decimal("4.50"),
    quantity: int = 1,
    use_tiers: bool = True,
) -> FeeBreakdown:
    """Calculate exact contractual marketplace commission and per-item fixed fee.

    Implements the 2026 tiered schedule:
    - Unit price <= R$ 79.99: 20% (cap R$ 100) + R$ 4.50 fixed (or price/2 if < R$ 8.00)
    - Unit price <= R$ 99.99: 14% (cap R$ 100) + R$ 16.00 fixed
    - Unit price <= R$ 199.99: 14% (cap R$ 100) + R$ 20.00 fixed
    - Unit price > R$ 199.99: 14% (cap R$ 100) + R$ 26.00 fixed
    Plus R$ 3.00/unit surcharge for CPF high-volume accounts (>450 orders/90d).
    """
    price_dec = to_decimal(sell_price)
    if price_dec is None or price_dec <= Decimal("0.00"):
        raise ValueError(f"Invalid sell_price for fee calculation: {sell_price!r}")

    qty = max(1, int(quantity))
    qty_dec = Decimal(str(qty))
    unit_price = (price_dec / qty_dec).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    regime_cfg = get_regime_params(account_regime)
    extra_cpf = regime_cfg["cpf_surcharge"]

    if not use_tiers:
        rate = (comm_pct / Decimal("100.0")).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        unit_comm = min(
            Decimal("100.00"),
            (unit_price * rate).quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
        )
        unit_fixed = (
            (unit_price / Decimal("2.0")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            if unit_price < Decimal("8.00")
            else fixed_fee_unit
        )
        tier_label = "CUSTOM_FLAT"
    elif unit_price <= Decimal("79.99"):
        rate = Decimal("0.20")
        unit_comm = min(
            Decimal("100.00"),
            (unit_price * rate).quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
        )
        unit_fixed = (
            (unit_price / Decimal("2.0")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            if unit_price < Decimal("8.00")
            else fixed_fee_unit
        )
        tier_label = "TIER_1_UP_TO_79_99"
    elif unit_price <= Decimal("99.99"):
        rate = Decimal("0.14")
        unit_comm = min(
            Decimal("100.00"),
            (unit_price * rate).quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
        )
        unit_fixed = Decimal("16.00")
        tier_label = "TIER_2_80_TO_99_99"
    elif unit_price <= Decimal("199.99"):
        rate = Decimal("0.14")
        unit_comm = min(
            Decimal("100.00"),
            (unit_price * rate).quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
        )
        unit_fixed = Decimal("20.00")
        tier_label = "TIER_3_100_TO_199_99"
    else:
        rate = Decimal("0.14")
        unit_comm = min(
            Decimal("100.00"),
            (unit_price * rate).quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
        )
        unit_fixed = Decimal("26.00")
        tier_label = "TIER_4_ABOVE_200"

    total_unit = unit_comm + unit_fixed + extra_cpf
    total_fees = (total_unit * qty_dec).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    return FeeBreakdown(
        unit_price=unit_price,
        quantity=qty,
        unit_commission=unit_comm,
        unit_fixed_fee=unit_fixed,
        unit_cpf_surcharge=extra_cpf,
        expected_total_fees=total_fees,
        commission_rate_applied=(rate * Decimal("100")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP),
        tier_label=tier_label,
    )


def reconcile_order(
    order_id: str,
    sku: str,
    item_price: Any,
    quantity: int = 1,
    supplier_cost: Any = None,
    reported_commission_fee: Any = None,
    reported_service_fee: Any = None,
    coupon: Any = Decimal("0.00"),
    net_shipping: Any = Decimal("0.00"),
    cash_flow_adjustment: Any = Decimal("0.00"),
    account_regime: str = "CPF_PADRAO",
    tax_pct: Any = None,
    ads_spend_attributed: Any = Decimal("0.00"),
    order_status: str = "Concluído",
    return_status: str = "",
    fixed_fee_unit: Decimal = Decimal("4.50"),
    use_tiers: bool = True,
) -> OrderAuditResult:
    """Reconcile a single marketplace order with exact Decimal arithmetic and 3-state classification."""
    price_dec = to_decimal(item_price)
    qty = max(1, int(quantity))
    unit_cost_dec = to_decimal(supplier_cost)
    total_supplier_cost = (
        (unit_cost_dec * Decimal(str(qty))).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        if unit_cost_dec is not None
        else None
    )
    comm_dec = to_decimal(reported_commission_fee)
    serv_dec = to_decimal(reported_service_fee)
    coupon_dec = to_decimal(coupon) or Decimal("0.00")
    shipping_dec = to_decimal(net_shipping) or Decimal("0.00")
    cash_adj_dec = to_decimal(cash_flow_adjustment) or Decimal("0.00")
    ads_dec = to_decimal(ads_spend_attributed) or Decimal("0.00")

    regime_cfg = get_regime_params(account_regime)
    effective_tax_pct = to_decimal(tax_pct)
    if effective_tax_pct is None:
        effective_tax_pct = regime_cfg["tax_pct"]

    triggered: List[str] = []
    status_upper = (order_status or "").upper()
    is_refunded = bool((return_status or "").strip()) or any(
        kw in status_upper for kw in ("REEMBOLSADO", "CANCELADO", "REFUND", "DEVOLU")
    )

    if price_dec is None or price_dec <= Decimal("0.00") or is_refunded:
        triggered.append("RULE_03_CONTRIBUTION_AND_REFUND_GUARD")
        payload = f"{order_id}|{sku}|{account_regime}|SUSPEITA|{price_dec}"
        return OrderAuditResult(
            order_id=order_id,
            sku=sku,
            account_regime=account_regime,
            item_price_brl=price_dec or Decimal("0.00"),
            quantity=qty,
            supplier_cost_brl=total_supplier_cost,
            coupon_brl=coupon_dec,
            net_shipping_brl=shipping_dec,
            cash_flow_adjustment_brl=cash_adj_dec,
            expected_fees_brl=Decimal("0.00"),
            reported_fees_brl=None,
            fee_delta_brl=None,
            tax_amount_brl=Decimal("0.00"),
            ads_spend_brl=ads_dec,
            observed_payout_brl=Decimal("0.00"),
            partial_contribution_brl=None,
            net_profit_brl=None,
            net_margin_pct=None,
            audit_classification="SUSPEITA",
            triggered_rules=tuple(triggered),
            has_financial_divergence=True,
            is_healthy_margin=False,
            audit_sha256=sha256_text(payload),
        )

    fee_breakdown = calculate_expected_marketplace_fees(
        sell_price=price_dec,
        account_regime=account_regime,
        fixed_fee_unit=fixed_fee_unit,
        quantity=qty,
        use_tiers=use_tiers,
    )
    expected_fees = fee_breakdown.expected_total_fees

    reported_fees: Optional[Decimal] = None
    fee_delta: Optional[Decimal] = None
    actual_fees = expected_fees
    if comm_dec is not None:
        reported_fees = (comm_dec + (serv_dec or Decimal("0.00"))).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )
        actual_fees = reported_fees
        fee_delta = (reported_fees - expected_fees).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        if fee_delta >= Decimal("0.02"):
            triggered.append("RULE_01_PAYOUT_RECONCILIATION")

    if shipping_dec < Decimal("0.00"):
        triggered.append("RULE_02_FREIGHT_AND_RETENTION_AUDIT")

    # Theoretical payout from item price minus fees, coupon, and freight adjustments;
    # observed settlement payout equals theoretical payout net of retained cash transfers (e.g. Ads recharge).
    effective_payout = (price_dec - actual_fees - coupon_dec + shipping_dec).quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP
    )
    observed_payout = (effective_payout - cash_adj_dec).quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP
    )

    if total_supplier_cost is None:
        triggered.append("RULE_03_MISSING_SUPPLIER_COST")
        payload = f"{order_id}|{sku}|{account_regime}|SUSPEITA|{observed_payout}"
        return OrderAuditResult(
            order_id=order_id,
            sku=sku,
            account_regime=account_regime,
            item_price_brl=price_dec,
            quantity=qty,
            supplier_cost_brl=None,
            coupon_brl=coupon_dec,
            net_shipping_brl=shipping_dec,
            cash_flow_adjustment_brl=cash_adj_dec,
            expected_fees_brl=expected_fees,
            reported_fees_brl=reported_fees,
            fee_delta_brl=fee_delta,
            tax_amount_brl=Decimal("0.00"),
            ads_spend_brl=ads_dec,
            observed_payout_brl=observed_payout,
            partial_contribution_brl=None,
            net_profit_brl=None,
            net_margin_pct=None,
            audit_classification="SUSPEITA",
            triggered_rules=tuple(triggered),
            has_financial_divergence=True,
            is_healthy_margin=False,
            audit_sha256=sha256_text(payload),
        )

    tax_amount = Decimal("0.00")
    if effective_tax_pct > Decimal("0.00"):
        tax_amount = (
            price_dec * (effective_tax_pct / Decimal("100.0"))
        ).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    partial_contrib = (effective_payout - total_supplier_cost - tax_amount).quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP
    )
    net_profit = (partial_contrib - ads_dec).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    net_margin_pct = ((net_profit / price_dec) * Decimal("100.0")).quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP
    )

    if net_profit < Decimal("0.00"):
        triggered.append("RULE_03_NEGATIVE_NET_CONTRIBUTION")

    has_divergence = bool(
        (fee_delta is not None and fee_delta >= Decimal("0.02"))
        or (shipping_dec < Decimal("0.00"))
    )

    if triggered:
        audit_class = "ERRO_COMPROVADO"
    else:
        audit_class = "OK"

    is_healthy = bool((net_margin_pct >= Decimal("15.00")) and not has_divergence)

    payload = (
        f"{order_id}|{sku}|{account_regime}|{audit_class}|"
        f"{price_dec}|{actual_fees}|{shipping_dec}|{total_supplier_cost}|{tax_amount}|{ads_dec}|{net_profit}"
    )
    return OrderAuditResult(
        order_id=order_id,
        sku=sku,
        account_regime=account_regime,
        item_price_brl=price_dec,
        quantity=qty,
        supplier_cost_brl=total_supplier_cost,
        coupon_brl=coupon_dec,
        net_shipping_brl=shipping_dec,
        cash_flow_adjustment_brl=cash_adj_dec,
        expected_fees_brl=expected_fees,
        reported_fees_brl=reported_fees,
        fee_delta_brl=fee_delta,
        tax_amount_brl=tax_amount,
        ads_spend_brl=ads_dec,
        observed_payout_brl=observed_payout,
        partial_contribution_brl=partial_contrib,
        net_profit_brl=net_profit,
        net_margin_pct=net_margin_pct,
        audit_classification=audit_class,
        triggered_rules=tuple(triggered),
        has_financial_divergence=has_divergence,
        is_healthy_margin=is_healthy,
        audit_sha256=sha256_text(payload),
    )
