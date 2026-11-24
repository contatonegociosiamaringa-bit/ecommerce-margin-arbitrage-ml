"""Validate PostgreSQL DDL and analytical queries against an in-memory SQLite mirror."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL_PATH = PROJECT_ROOT / "sql" / "schema_postgresql.sql"
QUERIES_SQL_PATH = PROJECT_ROOT / "sql" / "analytical_queries.sql"
SKU_CSV_PATH = PROJECT_ROOT / "data" / "sku_catalog.csv"
ORDERS_CSV_PATH = PROJECT_ROOT / "data" / "orders_reconciliation.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "sql_validation_summary.json"


def _adapt_postgres_ddl_to_sqlite(ddl: str) -> str:
    """Translate PostgreSQL-specific column types to SQLite-compatible DDL."""
    adapted = re.sub(r"NUMERIC\(\s*\d+\s*,\s*\d+\s*\)", "REAL", ddl)
    adapted = re.sub(r"VARCHAR\(\s*\d+\s*\)", "TEXT", adapted)
    adapted = re.sub(r"CHAR\(\s*\d+\s*\)", "TEXT", adapted)
    adapted = adapted.replace("SMALLINT", "INTEGER")
    return adapted


def _parse_named_queries(sql_text: str) -> List[Dict[str, str]]:
    """Split analytical_queries.sql into named query blocks using `-- QUERY_ID:` markers."""
    blocks = re.split(r"(?=--\s*QUERY_ID:\s*)", sql_text)
    parsed: List[Dict[str, str]] = []
    for block in blocks:
        match = re.match(r"--\s*QUERY_ID:\s*([a-zA-Z0-9_]+)\s*\n(.*)", block.strip(), re.DOTALL)
        if not match:
            continue
        qid = match.group(1)
        sql_body = match.group(2).strip()
        if sql_body:
            parsed.append({"query_id": qid, "sql": sql_body})
    return parsed


def _populate_mart_margin_risk(conn: sqlite3.Connection) -> int:
    """Populate mart_margin_risk aggregating fact_orders_reconciliation by SKU and regime."""
    insert_sql = """
    INSERT INTO mart_margin_risk (
        sku, category, account_regime, orders_count, gmv_brl,
        avg_net_margin_pct, divergence_rate_pct, freight_loss_rate_pct,
        avg_fee_delta_brl, healthy_margin_rate_pct, risk_tier
    )
    SELECT
        sku,
        category,
        account_regime,
        COUNT(*) AS orders_count,
        ROUND(SUM(item_price_brl), 2) AS gmv_brl,
        ROUND(AVG(net_margin_pct), 2) AS avg_net_margin_pct,
        ROUND(100.0 * SUM(has_financial_divergence) / COUNT(*), 2) AS divergence_rate_pct,
        ROUND(100.0 * SUM(CASE WHEN net_shipping_brl < 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS freight_loss_rate_pct,
        ROUND(AVG(fee_delta_brl), 2) AS avg_fee_delta_brl,
        ROUND(100.0 * SUM(is_healthy_margin) / COUNT(*), 2) AS healthy_margin_rate_pct,
        CASE
            WHEN AVG(COALESCE(net_margin_pct, 0)) < 12.0 OR (100.0 * SUM(has_financial_divergence) / COUNT(*)) >= 45.0 THEN 'ALTO_RISCO'
            WHEN AVG(COALESCE(net_margin_pct, 0)) < 20.0 OR (100.0 * SUM(has_financial_divergence) / COUNT(*)) >= 20.0 THEN 'ATENCAO'
            ELSE 'SAUDAVEL'
        END AS risk_tier
    FROM fact_orders_reconciliation
    GROUP BY sku, category, account_regime;
    """
    cur = conn.cursor()
    cur.execute(insert_sql)
    conn.commit()
    return cur.execute("SELECT COUNT(*) FROM mart_margin_risk").fetchone()[0]


def run_sql_validation() -> Dict[str, Any]:
    """Load CSVs into SQLite using the project schema and execute all analytical queries."""
    if not SKU_CSV_PATH.exists() or not ORDERS_CSV_PATH.exists():
        raise FileNotFoundError("Datasets not found. Run src/prepare_dataset.py first.")

    ddl_raw = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    ddl_sqlite = _adapt_postgres_ddl_to_sqlite(ddl_raw)
    queries_raw = QUERIES_SQL_PATH.read_text(encoding="utf-8")
    named_queries = _parse_named_queries(queries_raw)

    sku_df = pd.read_csv(SKU_CSV_PATH)
    orders_df = pd.read_csv(ORDERS_CSV_PATH)

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(ddl_sqlite)

    sku_df.to_sql("dim_sku_catalog", conn, if_exists="append", index=False)
    orders_df.to_sql("fact_orders_reconciliation", conn, if_exists="append", index=False)
    mart_rows = _populate_mart_margin_risk(conn)

    query_results: Dict[str, Any] = {}
    for item in named_queries:
        qid = item["query_id"]
        df_res = pd.read_sql_query(item["sql"], conn)
        query_results[qid] = {
            "row_count": int(len(df_res)),
            "columns": list(df_res.columns),
            "rows": df_res.to_dict(orient="records"),
        }

    conn.close()

    summary = {
        "schema_tables_validated": ["dim_sku_catalog", "fact_orders_reconciliation", "mart_margin_risk"],
        "dim_sku_catalog_rows": int(len(sku_df)),
        "fact_orders_reconciliation_rows": int(len(orders_df)),
        "mart_margin_risk_rows": int(mart_rows),
        "queries_executed": len(named_queries),
        "queries_passed": len(query_results),
        "results": query_results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    res = run_sql_validation()
    print(f"SQL validation passed: {res['queries_passed']}/{res['queries_executed']} queries executed with 0 errors.")
    print(f"Tables: dim_sku_catalog={res['dim_sku_catalog_rows']}, fact_orders_reconciliation={res['fact_orders_reconciliation_rows']}, mart_margin_risk={res['mart_margin_risk_rows']}")
    for qid, qdata in res["results"].items():
        print(f"  - {qid}: {qdata['row_count']} rows")
