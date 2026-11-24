"""Train supervised margin-risk classifiers and unsupervised SKU BCG clusters.

Part A (Supervised): Pipeline + ColumnTransformer comparing LogisticRegression,
RandomForestClassifier, and GradientBoostingClassifier with StratifiedKFold CV
and holdout evaluation (Accuracy, F1-Score, ROC-AUC) on ex-ante features only
(zero post-settlement leakage) to predict `is_healthy_margin`.

Part B (Unsupervised): StandardScaler + KMeans segmenting the 64-SKU catalog by
realized/expected net margin %, monthly demand, inventory turnover proxy, and
competition level, validated with Silhouette Score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SKU_CSV_PATH = DATA_DIR / "sku_catalog.csv"
ORDERS_CSV_PATH = DATA_DIR / "orders_reconciliation.csv"
METRICS_JSON_PATH = REPORTS_DIR / "ml_metrics.json"
SKU_CLUSTERS_CSV_PATH = REPORTS_DIR / "sku_clusters.csv"

RANDOM_STATE = 42

# Strictly ex-ante / pre-settlement features to guarantee zero target leakage
NUMERIC_FEATURES: List[str] = [
    "item_price_brl",
    "supplier_cost_brl",
    "quantity",
    "weight_kg",
    "volumetric_cm3",
    "monthly_search_volume",
    "cpa_target_brl",
    "price_vs_market_avg_ratio",
    "markup_ratio",
    "gross_spread_pct",
    "coupon_pct",
    "expected_tax_pct",
    "expected_fee_ratio",
]

CATEGORICAL_FEATURES: List[str] = [
    "category",
    "account_regime",
    "traffic_channel",
    "competition_level",
]

COMPETITION_SCORE_MAP: Dict[str, int] = {
    "LOW": 1,
    "MODERATE": 2,
    "MODERATE_HIGH": 3,
    "HIGH": 4,
    "VIRAL_TIKTOK": 5,
}

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def build_preprocessor() -> ColumnTransformer:
    """Create ColumnTransformer with scaling for numeric and one-hot encoding for categoricals."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_supervised_models(orders_df: pd.DataFrame) -> Dict[str, Any]:
    """Train, cross-validate, and compare 3 classifiers on holdout test split."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = orders_df[feature_cols].copy()
    y = orders_df["is_healthy_margin"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}

    results: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        cv_out = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        holdout_acc = float(accuracy_score(y_test, y_pred))
        holdout_f1 = float(f1_score(y_test, y_pred))
        holdout_auc = float(roc_auc_score(y_test, y_prob))
        holdout_prec = float(precision_score(y_test, y_pred))
        holdout_rec = float(recall_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, holdout_auc)
        fitted_pipelines[name] = pipe

        results[name] = {
            "cv_5fold": {
                "accuracy_mean": round(float(np.mean(cv_out["test_accuracy"])), 4),
                "accuracy_std": round(float(np.std(cv_out["test_accuracy"])), 4),
                "f1_mean": round(float(np.mean(cv_out["test_f1"])), 4),
                "f1_std": round(float(np.std(cv_out["test_f1"])), 4),
                "roc_auc_mean": round(float(np.mean(cv_out["test_roc_auc"])), 4),
                "roc_auc_std": round(float(np.std(cv_out["test_roc_auc"])), 4),
            },
            "holdout": {
                "accuracy": round(holdout_acc, 4),
                "precision": round(holdout_prec, 4),
                "recall": round(holdout_rec, 4),
                "f1_score": round(holdout_f1, 4),
                "roc_auc": round(holdout_auc, 4),
                "confusion_matrix": cm,
            },
        }

    best_name = max(
        results.keys(),
        key=lambda k: (results[k]["holdout"]["roc_auc"], results[k]["cv_5fold"]["roc_auc_mean"]),
    )
    best_pipe = fitted_pipelines[best_name]

    # Permutation importance on raw input features (leak-free & model-agnostic)
    perm = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "margin_risk_classifier.joblib"
    joblib.dump(best_pipe, model_path)

    return {
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "positive_class_rate": round(float(y.mean()), 4),
        "features_used": feature_cols,
        "best_model_name": best_name,
        "best_model_path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "models": results,
        "top_features_permutation": [
            {
                "feature": row["feature"],
                "importance_mean": round(float(row["importance_mean"]), 4),
                "importance_std": round(float(row["importance_std"]), 4),
            }
            for _, row in importance_df.head(10).iterrows()
        ],
        "_roc_curves": roc_curves,
        "_importance_df": importance_df,
    }


def _assign_cluster_labels(cluster_summary: pd.DataFrame) -> Dict[int, str]:
    """Assign distinct Portuguese BCG/Margin cluster names based on centroid ranks."""
    remaining = set(int(cid) for cid in cluster_summary.index)
    labels: Dict[int, str] = {}

    # 1. Lowest margin cluster -> Zona Crítica
    worst_cid = min(remaining, key=lambda cid: float(cluster_summary.loc[cid, "mean_margin_pct"]))
    labels[worst_cid] = "Zona Crítica (Margem Pressionada / Negativa)"
    remaining.remove(worst_cid)

    # 2. Highest turnover & margin among remaining -> Estrelas de Alto Giro
    star_cid = max(
        remaining,
        key=lambda cid: (
            float(cluster_summary.loc[cid, "mean_turnover_proxy"]),
            float(cluster_summary.loc[cid, "mean_margin_pct"]),
        ),
    )
    labels[star_cid] = "Estrelas de Alto Giro (Top Margem & ROI)"
    remaining.remove(star_cid)

    # 3. Highest search volume / competition among remaining -> Alto Volume Competitivo
    if remaining:
        vol_cid = max(
            remaining,
            key=lambda cid: (
                float(cluster_summary.loc[cid, "mean_monthly_search"]),
                float(cluster_summary.loc[cid, "mean_competition_score"]),
            ),
        )
        labels[vol_cid] = "Alto Volume & Competição Elevada"
        remaining.remove(vol_cid)

    # 4. Remaining balanced/niche cluster -> Nicho Equilibrado
    for cid in sorted(remaining):
        labels[cid] = "Nicho Equilibrado (Baixa Competição)"
    return labels


def train_unsupervised_clusters(sku_df: pd.DataFrame, orders_df: pd.DataFrame) -> Dict[str, Any]:
    """Segment SKUs into BCG/Margin clusters using StandardScaler + KMeans."""
    valid_orders = orders_df[orders_df["audit_classification"] != "SUSPEITA"].copy()
    order_stats = (
        valid_orders.groupby("sku")
        .agg(
            orders_count=("order_id", "count"),
            realized_margin_pct=("net_margin_pct", "mean"),
            divergence_rate_pct=("has_financial_divergence", lambda s: float(s.mean() * 100.0)),
        )
        .reset_index()
    )

    merged = sku_df.merge(order_stats, on="sku", how="left")
    merged["orders_count"] = merged["orders_count"].fillna(0).astype(int)
    merged["effective_margin_pct"] = merged["realized_margin_pct"].fillna(merged["estimated_net_margin_cpf_pct"])
    merged["divergence_rate_pct"] = merged["divergence_rate_pct"].fillna(0.0)
    merged["log_monthly_demand"] = np.log1p(merged["monthly_search_volume"])
    merged["turnover_proxy"] = merged["estimated_niche_sales_month"] / np.maximum(merged["stock_units"], 1)
    merged["competition_score"] = merged["competition_level"].map(COMPETITION_SCORE_MAP).fillna(3).astype(float)

    cluster_features = [
        "effective_margin_pct",
        "log_monthly_demand",
        "turnover_proxy",
        "competition_score",
    ]
    X_cluster = merged[cluster_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    silhouette_by_k: Dict[str, float] = {}
    for k in (3, 4, 5):
        km_tmp = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
        labels_tmp = km_tmp.fit_predict(X_scaled)
        silhouette_by_k[f"k={k}"] = round(float(silhouette_score(X_scaled, labels_tmp)), 4)

    chosen_k = 4
    kmeans_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=chosen_k, n_init=30, random_state=RANDOM_STATE)),
        ]
    )
    cluster_ids = kmeans_pipe.fit_predict(X_cluster)
    chosen_silhouette = float(silhouette_score(kmeans_pipe.named_steps["scaler"].transform(X_cluster), cluster_ids))

    merged["cluster_id"] = cluster_ids
    summary_df = (
        merged.groupby("cluster_id")
        .agg(
            sku_count=("sku", "count"),
            mean_margin_pct=("effective_margin_pct", "mean"),
            mean_monthly_search=("monthly_search_volume", "mean"),
            mean_turnover_proxy=("turnover_proxy", "mean"),
            mean_competition_score=("competition_score", "mean"),
            mean_cpa_roi=("cpa_roi_ratio", "mean"),
        )
    )
    label_map = _assign_cluster_labels(summary_df)
    merged["cluster_label"] = merged["cluster_id"].map(label_map)
    summary_df["cluster_label"] = summary_df.index.map(label_map)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cluster_model_path = MODELS_DIR / "sku_kmeans_clusterer.joblib"
    joblib.dump(kmeans_pipe, cluster_model_path)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cols_to_save = [
        "sku",
        "product_title",
        "category",
        "data_origin",
        "supplier_cost_brl",
        "competitor_price_avg_brl",
        "monthly_search_volume",
        "turnover_proxy",
        "competition_level",
        "effective_margin_pct",
        "cpa_roi_ratio",
        "orders_count",
        "cluster_id",
        "cluster_label",
    ]
    merged[cols_to_save].to_csv(SKU_CLUSTERS_CSV_PATH, index=False, encoding="utf-8")

    return {
        "n_skus_clustered": int(len(merged)),
        "chosen_k": chosen_k,
        "silhouette_score": round(chosen_silhouette, 4),
        "silhouette_grid": silhouette_by_k,
        "features_used": cluster_features,
        "model_path": str(cluster_model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "clusters": [
            {
                "cluster_id": int(cid),
                "cluster_label": str(row["cluster_label"]),
                "sku_count": int(row["sku_count"]),
                "mean_margin_pct": round(float(row["mean_margin_pct"]), 2),
                "mean_monthly_search": round(float(row["mean_monthly_search"]), 1),
                "mean_turnover_proxy": round(float(row["mean_turnover_proxy"]), 2),
                "mean_competition_score": round(float(row["mean_competition_score"]), 2),
                "mean_cpa_roi": round(float(row["mean_cpa_roi"]), 2),
            }
            for cid, row in summary_df.iterrows()
        ],
        "_clustered_df": merged,
    }


def generate_figures(
    orders_df: pd.DataFrame,
    sup_results: Dict[str, Any],
    clus_results: Dict[str, Any],
) -> List[str]:
    """Generate 4 clean, publication-grade visualizations under reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    valid_orders = orders_df[orders_df["audit_classification"] != "SUSPEITA"].copy()

    # Figure 1: Margin distribution by category
    fig, ax = plt.subplots(figsize=(10.5, 6))
    cat_order = (
        valid_orders.groupby("category")["net_margin_pct"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    sns.boxplot(
        data=valid_orders,
        y="category",
        x="net_margin_pct",
        order=cat_order,
        color="#38bdf8",
        width=0.58,
        fliersize=3,
        ax=ax,
    )
    ax.axvline(15.0, color="#16a34a", linestyle="--", linewidth=1.8, label="Meta Saudável (15%)")
    ax.axvline(0.0, color="#dc2626", linestyle=":", linewidth=1.8, label="Break-even (0%)")
    ax.set_title("Distribuição de Margem Líquida (%) por Categoria de Produto")
    ax.set_xlabel("Margem Líquida Realizada (%)")
    ax.set_ylabel("")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    p1 = FIGURES_DIR / "margin_distribution_by_category.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(str(p1.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Figure 2: Fee & Freight impact (CPF vs CNPJ + Weight brackets)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    reg_df = (
        valid_orders.groupby("account_regime")
        .agg(
            taxa_marketplace_pct=("expected_fees_brl", lambda s: float((s / valid_orders.loc[s.index, "item_price_brl"]).mean() * 100)),
            imposto_nf_pct=("tax_amount_brl", lambda s: float((s / valid_orders.loc[s.index, "item_price_brl"]).mean() * 100)),
            margem_liquida_pct=("net_margin_pct", "mean"),
        )
        .reindex(["CPF_PADRAO", "CPF_ALTO_VOLUME", "CNPJ_SIMPLES_4PCT", "CNPJ_SIMPLES_6PCT"])
    )
    x = np.arange(len(reg_df))
    width = 0.27
    ax1.bar(x - width, reg_df["taxa_marketplace_pct"], width, label="Taxa Marketplace (%)", color="#0284c7")
    ax1.bar(x, reg_df["imposto_nf_pct"], width, label="Imposto Simples NF (%)", color="#f59e0b")
    ax1.bar(x + width, reg_df["margem_liquida_pct"], width, label="Margem Líquida (%)", color="#10b981")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["CPF\nPadrão", "CPF Alto\nVolume", "CNPJ\n4% SN", "CNPJ\n6% SN"])
    ax1.set_title("Estrutura de Taxas vs Margem: CPF vs CNPJ")
    ax1.set_ylabel("% sobre Preço de Venda")
    ax1.legend(frameon=True, fontsize=8.5)

    valid_orders["weight_bracket"] = pd.cut(
        valid_orders["weight_kg"],
        bins=[0.0, 0.20, 0.35, 10.0],
        labels=["Leve (<=200g)", "Médio (201-350g)", "Pesado (>350g)"],
    )
    wt_df = (
        valid_orders.groupby("weight_bracket", observed=False)
        .agg(
            freight_loss_pct=("net_shipping_brl", lambda s: float((s < 0).mean() * 100)),
            avg_margin_pct=("net_margin_pct", "mean"),
        )
        .reset_index()
    )
    ax2_twin = ax2.twinx()
    bars = ax2.bar(
        wt_df["weight_bracket"].astype(str),
        wt_df["freight_loss_pct"],
        color="#ef4444",
        alpha=0.80,
        width=0.45,
        label="Incidência Frete Negativo (%)",
    )
    ax2_twin.plot(
        wt_df["weight_bracket"].astype(str),
        wt_df["avg_margin_pct"],
        color="#0f172a",
        marker="o",
        linewidth=2.2,
        label="Margem Líquida Média (%)",
    )
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_title("Impacto do Peso na Retenção de Frete & Margem")
    ax2.set_ylabel("Pedidos com Frete Negativo (%)", color="#b91c1c")
    ax2_twin.set_ylabel("Margem Líquida Média (%)", color="#0f172a")
    ax2.set_ylim(0, max(55, float(wt_df["freight_loss_pct"].max()) + 12))
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper center", frameon=True, fontsize=8.5)
    fig.tight_layout()
    p2 = FIGURES_DIR / "fee_freight_impact.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_paths.append(str(p2.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Figure 3: SKU BCG/Margin KMeans Matrix
    clus_df = clus_results["_clustered_df"].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    palette = sns.color_palette("Set2", n_colors=clus_results["chosen_k"])
    for idx, label in enumerate(sorted(clus_df["cluster_label"].unique())):
        sub = clus_df[clus_df["cluster_label"] == label]
        ax.scatter(
            sub["monthly_search_volume"],
            sub["effective_margin_pct"],
            s=np.clip(sub["turnover_proxy"] * 2.5, 50, 420),
            alpha=0.85,
            edgecolors="#0f172a",
            linewidth=0.8,
            color=palette[idx],
            label=f"{label} (n={len(sub)})",
        )
    ax.axhline(15.0, color="#16a34a", linestyle="--", linewidth=1.3, alpha=0.8)
    ax.axvline(clus_df["monthly_search_volume"].median(), color="#64748b", linestyle="--", linewidth=1.2, alpha=0.7)
    for pilot_sku in ["DIN24116", "TRC0304", "15-64150", "CO0961295"]:
        hit = clus_df[clus_df["sku"] == pilot_sku]
        if not hit.empty:
            r = hit.iloc[0]
            ax.annotate(
                pilot_sku,
                (r["monthly_search_volume"], r["effective_margin_pct"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color="#0f172a",
            )
    ax.set_xscale("log")
    ax.set_title(
        f"Matriz BCG / Margem de Catálogo (KMeans k={clus_results['chosen_k']}, Silhouette = {clus_results['silhouette_score']:.3f})"
    )
    ax.set_xlabel("Demanda Mensal de Busca (Escala Log)")
    ax.set_ylabel("Margem Líquida Realizada / Estimada (%)")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    fig.tight_layout()
    p3 = FIGURES_DIR / "sku_clusters_matrix.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(str(p3.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Figure 4: ROC Curves + Permutation Feature Importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"LogisticRegression": "#64748b", "RandomForestClassifier": "#0284c7", "GradientBoostingClassifier": "#10b981"}
    for mname, (fpr, tpr, auc_val) in sup_results["_roc_curves"].items():
        ax1.plot(fpr, tpr, label=f"{mname} (AUC = {auc_val:.3f})", linewidth=2.2, color=colors.get(mname))
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório (AUC = 0.500)")
    ax1.set_title("Curvas ROC no Conjunto Holdout (n=125)")
    ax1.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax1.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    imp_df = sup_results["_importance_df"].head(10).iloc[::-1]
    ax2.barh(
        imp_df["feature"],
        imp_df["importance_mean"],
        xerr=imp_df["importance_std"],
        color="#0284c7",
        alpha=0.88,
        ecolor="#334155",
    )
    ax2.set_title(f"Importância por Permutação ({sup_results['best_model_name']})")
    ax2.set_xlabel("Queda Média no ROC-AUC após Permutação")
    fig.tight_layout()
    p4 = FIGURES_DIR / "classifier_roc_and_importance.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_paths.append(str(p4.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    return saved_paths


def run_training_pipeline() -> Dict[str, Any]:
    """Execute full ML training, clustering, visualization, and JSON reporting."""
    if not SKU_CSV_PATH.exists() or not ORDERS_CSV_PATH.exists():
        raise FileNotFoundError("Datasets not found. Run src/prepare_dataset.py first.")

    sku_df = pd.read_csv(SKU_CSV_PATH)
    orders_df = pd.read_csv(ORDERS_CSV_PATH)

    sup_res = train_supervised_models(orders_df)
    clus_res = train_unsupervised_clusters(sku_df, orders_df)
    fig_paths = generate_figures(orders_df, sup_res, clus_res)

    payload = {
        "dataset_summary": {
            "sku_catalog_rows": int(len(sku_df)),
            "orders_reconciliation_rows": int(len(orders_df)),
            "observed_sanitized_orders": int((orders_df["data_origin"] == "OBSERVED_SANITIZED").sum()),
            "synthetic_augmented_orders": int((orders_df["data_origin"] == "SYNTHETIC_AUGMENTED").sum()),
        },
        "supervised_classification": {
            k: v for k, v in sup_res.items() if not k.startswith("_")
        },
        "unsupervised_clustering": {
            k: v for k, v in clus_res.items() if not k.startswith("_")
        },
        "figures": fig_paths,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


if __name__ == "__main__":
    summary = run_training_pipeline()
    sup = summary["supervised_classification"]
    clus = summary["unsupervised_clustering"]
    print(f"Best supervised classifier: {sup['best_model_name']}")
    for mname, mvals in sup["models"].items():
        ho = mvals["holdout"]
        cv = mvals["cv_5fold"]
        print(
            f"  - {mname}: CV_AUC={cv['roc_auc_mean']:.4f} | "
            f"Holdout Acc={ho['accuracy']:.4f}, F1={ho['f1_score']:.4f}, AUC={ho['roc_auc']:.4f}"
        )
    print(f"Clustering (KMeans k={clus['chosen_k']}): Silhouette={clus['silhouette_score']:.4f} (grid={clus['silhouette_grid']})")
    print(f"Figures saved: {summary['figures']}")
