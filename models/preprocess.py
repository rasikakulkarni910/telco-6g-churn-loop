"""
Shared feature matrix construction for churn model train + inference.

Keeps one-hot / binary encoding aligned with models/feature_config.json so
training columns match what predict_risk_and_drivers() expects at serve time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODELS_DIR / "feature_config.json"


def load_feature_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open() as f:
        return json.load(f)


def build_feature_matrix(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Transform a features-table row/frame into the model design matrix.

    Missing categorical levels become 0 (safe for unseen one-hot columns).
    """
    config = config or load_feature_config()
    out = pd.DataFrame(index=df.index)

    for col in config["numeric"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0

    for col in config["binary_yes_no"]:
        key = f"{col}_Yes"
        if col in df.columns:
            out[key] = (df[col].astype(str).str.strip() == "Yes").astype(float)
        else:
            out[key] = 0.0

    for col in config["categorical_one_hot"]:
        # Emit every expected dummy from config so column order is stable.
        expected = [c for c in config["feature_columns"] if c.startswith(f"{col}_")]
        series = df[col].astype(str) if col in df.columns else pd.Series("", index=df.index)
        for dummy in expected:
            level = dummy[len(col) + 1 :]
            out[dummy] = (series == level).astype(float)

    # Enforce exact training column order (drops unexpected / fills missing with 0).
    for col in config["feature_columns"]:
        if col not in out.columns:
            out[col] = 0.0
    return out[config["feature_columns"]].astype(float)


def matrix_from_user_row(
    user_row: pd.Series | dict[str, Any],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Single-user convenience wrapper → 1×p DataFrame."""
    if isinstance(user_row, dict):
        frame = pd.DataFrame([user_row])
    elif isinstance(user_row, pd.Series):
        frame = user_row.to_frame().T
    else:
        raise TypeError("user_row must be a dict or pandas Series")
    return build_feature_matrix(frame, config=config)
