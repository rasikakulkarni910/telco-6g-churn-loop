"""
Reusable churn risk + SHAP driver service.

Public API:
  load_model()
  load_feature_config()
  predict_risk_and_drivers(user_row) -> {risk, drivers, ...}
  get_global_shap_summary() -> global importance for UI

Driver categories (network / billing / usage / support / migration) come from
feature_config.json and are what the Decision Agent maps to playbooks.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.explain import shap_values_for_matrix  # noqa: E402
from models.preprocess import (  # noqa: E402
    load_feature_config as _load_feature_config,
    matrix_from_user_row,
)

MODELS_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODELS_DIR / "churn_model.json"
SHAP_SUMMARY_PATH = MODELS_DIR / "global_shap_summary.json"


def load_feature_config(path: Path | str | None = None) -> dict[str, Any]:
    return _load_feature_config(path)


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None) -> xgb.XGBClassifier:
    """Load the trained XGBoost classifier (cached)."""
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python models/train_churn_model.py"
        )
    # Recreate sklearn wrapper so predict_proba stays consistent.
    clf = xgb.XGBClassifier()
    clf.load_model(str(path))
    return clf


def predict_risk_and_drivers(
    user_row: pd.Series | dict[str, Any],
    *,
    top_k: int = 5,
    model_path: str | None = None,
    min_abs_shap: float = 0.0,
) -> dict[str, Any]:
    """
    Score one subscriber and return risk + categorized SHAP drivers.

    Returns
    -------
    {
      "customer_id": str | None,
      "risk": float,  # P(churn in next 90 days)
      "drivers": [
        {"feature": str, "shap_value": float, "driver_category": str},
        ...
      ],
      "top_driver_category": str | None,
    }
    """
    config = load_feature_config()
    model = load_model(model_path)
    X = matrix_from_user_row(user_row, config=config)

    risk = float(model.predict_proba(X)[0, 1])
    values = shap_values_for_matrix(model, X).reshape(-1)

    category_map = config.get("driver_category_map", {})
    drivers: list[dict[str, Any]] = []
    for feat, sv in zip(config["feature_columns"], values):
        if abs(float(sv)) < min_abs_shap:
            continue
        drivers.append(
            {
                "feature": feat,
                "shap_value": float(sv),
                # Positive contribution → pushes predicted risk up.
                "driver_category": category_map.get(feat, "other"),
            }
        )

    # Rank by contribution to risk (positive first), then magnitude.
    drivers.sort(key=lambda d: (d["shap_value"], abs(d["shap_value"])), reverse=True)
    top_drivers = drivers[:top_k]

    # Strongest category among risk-increasing drivers (Decision Agent input).
    positive = [d for d in drivers if d["shap_value"] > 0]
    top_category = positive[0]["driver_category"] if positive else (
        top_drivers[0]["driver_category"] if top_drivers else None
    )

    if isinstance(user_row, dict):
        customer_id = user_row.get("customer_id") or user_row.get("customerID")
    else:
        customer_id = user_row.get("customer_id") if "customer_id" in user_row.index else None
        if customer_id is None and "customerID" in user_row.index:
            customer_id = user_row.get("customerID")

    return {
        "customer_id": None if customer_id is None else str(customer_id),
        "risk": risk,
        "drivers": top_drivers,
        "top_driver_category": top_category,
    }


def get_global_shap_summary(path: Path | str | None = None) -> dict[str, Any]:
    """Load precomputed global SHAP summary written by train_churn_model.py."""
    summary_path = Path(path) if path else SHAP_SUMMARY_PATH
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Global SHAP summary not found at {summary_path}. "
            "Run: python models/train_churn_model.py"
        )
    return json.loads(summary_path.read_text())


def clear_caches() -> None:
    """Useful in tests / after retraining in the same process."""
    load_model.cache_clear()
