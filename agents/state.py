"""
Shared loop-state schema for the churn intervention loop.

v1 keeps state in memory (list[dict]). Later checkpoints can persist to BigQuery.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


Status = Literal["active", "churned", "stabilized", "skipped"]
Arm = Literal["treated", "control", "unassigned"]
Intensity = Literal["digital", "outbound"]


class Driver(TypedDict):
    feature: str
    shap_value: float
    driver_category: str


class LoopState(TypedDict, total=False):
    customer_id: str
    migration_flag: int
    value_segment: str
    region: str
    device_type: str
    # Model outputs — risk is P(churn in next 90 days) in [0, 1]
    risk: float
    drivers: list[Driver]
    top_driver_category: str | None
    # Loop control
    status: Status
    eligible: bool
    entry_reason: str
    # Decision
    segment: str
    playbook: str
    intensity: Intensity
    # Outreach
    intervention_message: str
    intervention_channel: str
    # Learning / experiment
    arm: Arm
    outcome_churn: int | None
    outcome_usage_change: float | None
    # Passthrough features useful for decision rules
    bill_change_pct: float
    usage_change_30d: float
    post_migration_qos: float
    support_sentiment: float
    MonthlyCharges: float


def new_state(customer_id: str, **kwargs: Any) -> LoopState:
    state: LoopState = {
        "customer_id": customer_id,
        "status": "active",
        "eligible": False,
        "arm": "unassigned",
        "outcome_churn": None,
        "outcome_usage_change": None,
    }
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state
