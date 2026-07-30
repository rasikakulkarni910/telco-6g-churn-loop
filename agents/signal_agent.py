"""
Signal Agent — score subscribers and apply loop entry / exit conditions.

Entry (all required):
  1. migration_flag == 1          — focus on 6G-migrated cohort
  2. risk >= RISK_THRESHOLD       — numeric P(churn in 90d) from XGBoost
  3. at least one strong driver   — |SHAP| >= DRIVER_THRESHOLD

Exit / skip:
  - status == "churned"     → skip
  - status == "stabilized"  → skip

risk is always computed via churn_model_service.predict_risk_and_drivers()
(predict_proba), never a high/low label.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.state import LoopState, new_state  # noqa: E402
from models.churn_model_service import predict_risk_and_drivers  # noqa: E402

DEFAULT_DATASET = "telco_churn"
DEFAULT_TABLE = "features"
# CRM playbooks typically act on elevated near-term risk.
RISK_THRESHOLD = 0.60
# “Strong” driver: material contribution on the log-odds / margin scale.
DRIVER_ABS_THRESHOLD = 0.25


def _project_id(explicit: str | None = None) -> str:
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


def load_features(
    project_id: str | None = None,
    dataset: str = DEFAULT_DATASET,
    table: str = DEFAULT_TABLE,
    limit: int | None = None,
) -> pd.DataFrame:
    pid = _project_id(project_id)
    fqn = f"{pid}.{dataset}.{table}"
    sql = f"SELECT * FROM `{fqn}`"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    client = bigquery.Client(project=pid)
    return client.query(sql).to_dataframe()


def _has_strong_driver(drivers: list[dict[str, Any]], threshold: float) -> bool:
    return any(abs(float(d.get("shap_value", 0.0))) >= threshold for d in drivers)


def score_user_row(
    row: pd.Series | dict[str, Any],
    *,
    risk_threshold: float = RISK_THRESHOLD,
    driver_threshold: float = DRIVER_ABS_THRESHOLD,
    prior_status: str = "active",
) -> LoopState:
    """
    Run model scoring for one user and apply entry/exit rules.

    Returns a LoopState with numeric `risk` always populated when scored.
    """
    if isinstance(row, dict):
        customer_id = str(row.get("customer_id") or row.get("customerID") or "")
        get = row.get
    else:
        customer_id = str(row.get("customer_id") if "customer_id" in row.index else row.get("customerID", ""))
        get = row.get

    # Exit conditions — do not re-enter the loop.
    if prior_status == "churned":
        return new_state(
            customer_id,
            status="churned",
            eligible=False,
            entry_reason="exit: already churned",
        )
    if prior_status == "stabilized":
        return new_state(
            customer_id,
            status="stabilized",
            eligible=False,
            entry_reason="exit: stabilized",
        )

    prediction = predict_risk_and_drivers(row)
    risk = float(prediction["risk"])  # P(churn in next 90 days) ∈ [0, 1]
    drivers = prediction["drivers"]
    top_cat = prediction.get("top_driver_category")

    migration_flag = int(get("migration_flag") or 0)
    reasons: list[str] = []
    eligible = True

    if migration_flag != 1:
        eligible = False
        reasons.append("fail: not migrated")
    if risk < risk_threshold:
        eligible = False
        reasons.append(f"fail: risk {risk:.3f} < {risk_threshold}")
    if not _has_strong_driver(drivers, driver_threshold):
        eligible = False
        reasons.append(f"fail: no strong driver (|SHAP|>={driver_threshold})")

    if eligible:
        reasons = [f"enter: migrant risk={risk:.3f} with strong drivers"]

    return new_state(
        customer_id,
        migration_flag=migration_flag,
        value_segment=str(get("value_segment") or "Medium"),
        region=str(get("region") or ""),
        device_type=str(get("device_type") or ""),
        risk=risk,
        drivers=drivers,
        top_driver_category=top_cat,
        status="active" if eligible else "skipped",
        eligible=eligible,
        entry_reason="; ".join(reasons),
        bill_change_pct=float(get("bill_change_pct") or 0.0),
        usage_change_30d=float(get("usage_change_30d") or 0.0),
        post_migration_qos=float(get("post_migration_qos") or 0.0),
        support_sentiment=float(get("support_sentiment") or 0.0),
        MonthlyCharges=float(get("MonthlyCharges") or 0.0),
    )


def run_signal_for_all_users(
    *,
    project_id: str | None = None,
    dataset: str = DEFAULT_DATASET,
    table: str = DEFAULT_TABLE,
    limit: int | None = None,
    risk_threshold: float = RISK_THRESHOLD,
    driver_threshold: float = DRIVER_ABS_THRESHOLD,
    prior_states: dict[str, LoopState] | None = None,
    only_eligible: bool = True,
) -> list[LoopState]:
    """
    Score features from BigQuery and return loop states.

    prior_states: optional map customer_id → previous LoopState for exit checks.
    only_eligible: if True, return only users who pass entry conditions.
    """
    df = load_features(project_id=project_id, dataset=dataset, table=table, limit=limit)
    prior_states = prior_states or {}
    results: list[LoopState] = []

    for _, row in df.iterrows():
        cid = str(row.get("customer_id", ""))
        prior = prior_states.get(cid, {})
        prior_status = str(prior.get("status") or "active")
        state = score_user_row(
            row,
            risk_threshold=risk_threshold,
            driver_threshold=driver_threshold,
            prior_status=prior_status,
        )
        if only_eligible and not state.get("eligible"):
            continue
        results.append(state)

    results.sort(key=lambda s: float(s.get("risk") or 0.0), reverse=True)
    return results
