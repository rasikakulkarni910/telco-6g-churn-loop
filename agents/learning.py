"""
Learning Agent — treated/control assignment, outcome simulation, uplift metrics.

Experimental design (v1):
  - Within each (segment, playbook) stratum, randomly assign treated vs control 50/50.
  - Control receives no intervention effect in the simulator.
  - Treated gets a playbook-specific reduction in churn probability and a usage lift.
  - Outcomes are simulated Bernoulli / Gaussian draws — replace with real telemetry later.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from agents.state import LoopState

# Expected absolute reduction in P(churn) when treated, by playbook.
PLAYBOOK_CHURN_UPLIFT = {
    "Network Rescue": 0.12,
    "Bill Clarity + Price Protection": 0.10,
    "6G Value Unlock": 0.08,
    "VIP Rescue": 0.15,
}

# Expected usage_change lift (additive) when treated.
PLAYBOOK_USAGE_UPLIFT = {
    "Network Rescue": 0.04,
    "Bill Clarity + Price Protection": 0.02,
    "6G Value Unlock": 0.10,
    "VIP Rescue": 0.06,
}


def assign_treatment_arms(
    states: list[LoopState],
    *,
    seed: int = 42,
    treated_rate: float = 0.5,
) -> list[LoopState]:
    """Random treated/control assignment stratified by segment × playbook."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, s in enumerate(states):
        key = (str(s.get("segment") or "NA"), str(s.get("playbook") or "NA"))
        buckets[key].append(i)

    updated = [dict(s) for s in states]
    for idxs in buckets.values():
        for i in idxs:
            updated[i]["arm"] = "treated" if rng.random() < treated_rate else "control"
    return updated  # type: ignore[return-value]


def simulate_outcomes(
    states: list[LoopState],
    *,
    seed: int = 42,
) -> list[LoopState]:
    """
    Simulate post-intervention outcomes from baseline risk.

    outcome_churn ~ Bernoulli(p)
      control: p = risk
      treated: p = max(0, risk - playbook_uplift)
    outcome_usage_change ~ baseline usage_change_30d (+ uplift if treated)
    """
    rng = np.random.default_rng(seed + 7)
    updated: list[LoopState] = []
    for state in states:
        s: LoopState = dict(state)
        risk = float(s.get("risk") or 0.0)
        playbook = str(s.get("playbook") or "")
        arm = str(s.get("arm") or "control")
        base_usage = float(s.get("usage_change_30d") or 0.0)

        if arm == "treated":
            p_churn = max(0.0, risk - PLAYBOOK_CHURN_UPLIFT.get(playbook, 0.08))
            usage = base_usage + PLAYBOOK_USAGE_UPLIFT.get(playbook, 0.05) + float(rng.normal(0, 0.02))
        else:
            p_churn = risk
            usage = base_usage + float(rng.normal(0, 0.02))

        s["outcome_churn"] = int(rng.random() < p_churn)
        s["outcome_usage_change"] = round(float(usage), 4)
        # Soft exit signal for later loops: treated + no churn → may stabilize.
        if arm == "treated" and s["outcome_churn"] == 0 and risk >= 0.6:
            s["status"] = "stabilized"
        elif s["outcome_churn"] == 1:
            s["status"] = "churned"
        updated.append(s)
    return updated


def compute_uplift_metrics(states: list[LoopState]) -> list[dict[str, Any]]:
    """Treated vs control churn rate and usage change by playbook and segment."""
    groups: dict[tuple[str, str, str], list[LoopState]] = defaultdict(list)
    for s in states:
        arm = str(s.get("arm") or "unassigned")
        if arm not in {"treated", "control"}:
            continue
        key = (str(s.get("playbook") or "NA"), str(s.get("segment") or "NA"), arm)
        groups[key].append(s)

    # Aggregate to playbook × segment rows with both arms when possible.
    pair_keys = {(p, seg) for (p, seg, _) in groups}
    rows: list[dict[str, Any]] = []
    for playbook, segment in sorted(pair_keys):
        treated = groups.get((playbook, segment, "treated"), [])
        control = groups.get((playbook, segment, "control"), [])

        def _churn_rate(items: list[LoopState]) -> float | None:
            if not items:
                return None
            return float(np.mean([int(x.get("outcome_churn") or 0) for x in items]))

        def _usage(items: list[LoopState]) -> float | None:
            if not items:
                return None
            return float(np.mean([float(x.get("outcome_usage_change") or 0.0) for x in items]))

        c_churn = _churn_rate(control)
        t_churn = _churn_rate(treated)
        c_usage = _usage(control)
        t_usage = _usage(treated)
        rows.append(
            {
                "playbook": playbook,
                "segment": segment,
                "n_treated": len(treated),
                "n_control": len(control),
                "churn_control": c_churn,
                "churn_treated": t_churn,
                "churn_uplift": None if c_churn is None or t_churn is None else c_churn - t_churn,
                "usage_control": c_usage,
                "usage_treated": t_usage,
                "usage_uplift": None if c_usage is None or t_usage is None else t_usage - c_usage,
            }
        )
    return rows


def run_learning(states: list[LoopState], *, seed: int = 42) -> tuple[list[LoopState], list[dict[str, Any]]]:
    """Assign arms → simulate outcomes → return states + uplift table."""
    with_arms = assign_treatment_arms(states, seed=seed)
    with_outcomes = simulate_outcomes(with_arms, seed=seed)
    metrics = compute_uplift_metrics(with_outcomes)
    return with_outcomes, metrics
