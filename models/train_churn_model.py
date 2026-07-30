"""
Train an XGBoost churn model on BigQuery telco_churn.features.

Saves:
  models/churn_model.json       — XGBoost booster
  models/metrics.json           — holdout metrics
  models/global_shap_summary.json — mean |SHAP| by feature + category (for UI)

Usage:
  export PROJECT_ID=your-project
  python models/train_churn_model.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Allow `python models/train_churn_model.py` without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.explain import shap_values_for_matrix  # noqa: E402
from models.preprocess import build_feature_matrix, load_feature_config  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = "telco_churn"
DEFAULT_TABLE = "features"
RANDOM_SEED = 42


def get_project_id(explicit: str | None = None) -> str:
    project_id = explicit or os.environ.get("PROJECT_ID")
    if not project_id:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("PROJECT_ID=") and not line.startswith("#"):
                    project_id = line.split("=", 1)[1].strip()
                    break
    if not project_id:
        raise SystemExit("PROJECT_ID is required.")
    return project_id


def load_features(project_id: str, dataset: str, table: str) -> pd.DataFrame:
    client = bigquery.Client(project=project_id)
    fqn = f"{project_id}.{dataset}.{table}"
    print(f"Reading {fqn}...")
    df = client.query(f"SELECT * FROM `{fqn}`").to_dataframe()
    print(f"  {len(df):,} rows × {len(df.columns)} columns")
    return df


def train_model(
    df: pd.DataFrame,
    config: dict,
    seed: int = RANDOM_SEED,
) -> tuple[xgb.XGBClassifier, dict, pd.DataFrame, np.ndarray, np.ndarray]:
    target = config["target_column"]
    # Only rows with a fully observed 90-day window (landmark construction).
    if "label_eligible" in df.columns:
        df = df.loc[df["label_eligible"] == 1].copy()
        print(f"Training on {len(df):,} label-eligible rows (full 90-day follow-up)")

    X = build_feature_matrix(df, config)
    y = df[target].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    # scale_pos_weight helps with ~26% churn base rate without SMOTE complexity.
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - pos), 1)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=neg / pos,
        random_state=seed,
        n_jobs=4,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
        "feature_count": int(X.shape[1]),
    }
    return model, metrics, X_test, y_test, proba


def compute_global_shap_summary(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    config: dict,
    max_rows: int = 500,
) -> dict:
    """Mean |SHAP| per feature, tagged with driver_category for the UI."""
    sample = X_sample.sample(n=min(max_rows, len(X_sample)), random_state=RANDOM_SEED)
    shap_values = shap_values_for_matrix(model, sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    category_map = config.get("driver_category_map", {})
    rows = []
    for feat, importance in zip(sample.columns, mean_abs):
        rows.append(
            {
                "feature": feat,
                "mean_abs_shap": float(importance),
                "driver_category": category_map.get(feat, "other"),
            }
        )
    rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)

    # Also roll up by category for executive charts.
    by_cat: dict[str, float] = {}
    for r in rows:
        by_cat[r["driver_category"]] = by_cat.get(r["driver_category"], 0.0) + r["mean_abs_shap"]
    category_summary = [
        {"driver_category": k, "mean_abs_shap": float(v)}
        for k, v in sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {"features": rows, "categories": category_summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train telco 6G churn XGBoost model")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--dataset", default=os.environ.get("BIGQUERY_DATASET", DEFAULT_DATASET))
    parser.add_argument("--table", default=os.environ.get("BIGQUERY_TABLE", DEFAULT_TABLE))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    project_id = get_project_id(args.project_id)
    config = load_feature_config()
    df = load_features(project_id, args.dataset, args.table)

    model, metrics, X_test, _, _ = train_model(df, config, seed=args.seed)
    print("Holdout metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    model_path = MODELS_DIR / "churn_model.json"
    model.get_booster().save_model(model_path)
    print(f"Saved model → {model_path}")

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics → {metrics_path}")

    shap_summary = compute_global_shap_summary(model, X_test, config)
    shap_path = MODELS_DIR / "global_shap_summary.json"
    shap_path.write_text(json.dumps(shap_summary, indent=2))
    print(f"Saved global SHAP summary → {shap_path}")
    print("Top 5 drivers:")
    for row in shap_summary["features"][:5]:
        print(f"  {row['feature']} ({row['driver_category']}): {row['mean_abs_shap']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
