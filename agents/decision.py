"""
Decision Agent — assign segment, playbook, and outreach intensity.

Playbook mapping (explicit):
  network   → Network Rescue
  billing   → Bill Clarity + Price Protection
  usage     → 6G Value Unlock
  support   → VIP Rescue (especially with high value)
  migration → VIP Rescue when High value, else 6G Value Unlock

Intensity:
  risk × value → outbound for high risk + High value; else digital.
"""

from __future__ import annotations

from typing import Any

from agents.state import Intensity, LoopState

PLAYBOOK_NETWORK = "Network Rescue"
PLAYBOOK_BILLING = "Bill Clarity + Price Protection"
PLAYBOOK_VALUE = "6G Value Unlock"
PLAYBOOK_VIP = "VIP Rescue"

VALUE_RANK = {"Low": 1, "Medium": 2, "High": 3}


def assign_segment(state: LoopState) -> str:
    """
    Business segment for reporting / experiment strata.

    Combines migration + value + dominant stressor (usage drop vs bill shock).
    """
    value = str(state.get("value_segment") or "Medium")
    usage = float(state.get("usage_change_30d") or 0.0)
    bill = float(state.get("bill_change_pct") or 0.0)
    migrant = int(state.get("migration_flag") or 0)

    if migrant != 1:
        return f"NonMigrant_{value}"

    if bill >= 0.15 and usage <= -0.10:
        stress = "BillShock_UsageDrop"
    elif bill >= 0.15:
        stress = "BillShock"
    elif usage <= -0.10:
        stress = "UsageDrop"
    elif float(state.get("post_migration_qos") or 1.0) < 0.55:
        stress = "QoSPain"
    else:
        stress = "GeneralRisk"

    return f"Migrant_{value}_{stress}"


def select_playbook(state: LoopState) -> str:
    """Map top SHAP driver category → intervention playbook."""
    category = (state.get("top_driver_category") or "").lower()
    value = str(state.get("value_segment") or "Medium")
    high_value = value == "High"

    # Prefer strongest risk-increasing driver category already set by Signal.
    if category == "network":
        return PLAYBOOK_NETWORK
    if category == "billing":
        return PLAYBOOK_BILLING
    if category == "usage":
        return PLAYBOOK_VALUE
    if category == "support":
        return PLAYBOOK_VIP if high_value else PLAYBOOK_NETWORK
    if category == "migration":
        return PLAYBOOK_VIP if high_value else PLAYBOOK_VALUE

    # Fallback: inspect ranked drivers list.
    for d in state.get("drivers") or []:
        cat = str(d.get("driver_category", "")).lower()
        if float(d.get("shap_value", 0.0)) <= 0:
            continue
        if cat == "network":
            return PLAYBOOK_NETWORK
        if cat == "billing":
            return PLAYBOOK_BILLING
        if cat == "usage":
            return PLAYBOOK_VALUE
        if cat in {"support", "migration"} and high_value:
            return PLAYBOOK_VIP

    return PLAYBOOK_VALUE


def select_intensity(state: LoopState, risk_outbound: float = 0.75) -> Intensity:
    """
    High residual value × high risk → human outbound; otherwise digital.

    Why: outbound capacity is scarce; spend it on High ARPU migrants most likely
    to churn in the 90-day window.
    """
    risk = float(state.get("risk") or 0.0)
    value = str(state.get("value_segment") or "Medium")
    if value == "High" and risk >= risk_outbound:
        return "outbound"
    if value == "High" and risk >= 0.60:
        return "outbound"
    if VALUE_RANK.get(value, 2) >= 2 and risk >= 0.85:
        return "outbound"
    return "digital"


def run_decision(states: list[LoopState]) -> list[LoopState]:
    """Attach segment / playbook / intensity to each active eligible state."""
    updated: list[LoopState] = []
    for state in states:
        if not state.get("eligible", False) and state.get("status") != "active":
            updated.append(state)
            continue
        s: LoopState = dict(state)  # shallow copy
        s["segment"] = assign_segment(s)
        s["playbook"] = select_playbook(s)
        s["intensity"] = select_intensity(s)
        updated.append(s)
    return updated
