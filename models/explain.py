"""SHAP-style attributions via XGBoost pred_contribs (TreeExplainer-compatible)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def shap_values_for_matrix(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Per-row tree-path contributions (same family as SHAP for trees).

    Uses booster.predict(..., pred_contribs=True) instead of shap.TreeExplainer
    because XGBoost 2+/3 encodes base_score as '[5E-1]', which breaks SHAP's loader.
    """
    dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    return np.asarray(contribs)[:, :-1]  # drop bias column
