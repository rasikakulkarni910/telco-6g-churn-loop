"""
Checkpoint 4 demo: chain Signal → Decision → Outreach → Learning.

Usage:
  export PROJECT_ID=your-project
  python agents/run_loop_demo.py --limit 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.decision import run_decision  # noqa: E402
from agents.learning import run_learning  # noqa: E402
from agents.outreach import run_outreach  # noqa: E402
from agents.signal_agent import RISK_THRESHOLD, run_signal_for_all_users  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run churn loop agents demo")
    parser.add_argument("--project-id", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--limit", type=int, default=500, help="Max feature rows to score")
    parser.add_argument("--risk-threshold", type=float, default=RISK_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-print", type=int, default=8)
    args = parser.parse_args(argv)

    print(f"=== Signal (risk threshold={args.risk_threshold}) ===")
    eligible = run_signal_for_all_users(
        project_id=args.project_id,
        limit=args.limit,
        risk_threshold=args.risk_threshold,
        only_eligible=True,
    )
    print(f"Loop-eligible users: {len(eligible)}")
    if not eligible:
        print("No eligible users — try lowering --risk-threshold or raising --limit")
        return 0

    # Checklist proof: risk is a float probability, not a high/low label.
    risks = [float(s["risk"]) for s in eligible]
    print(
        f"risk range: min={min(risks):.4f} max={max(risks):.4f} "
        f"mean={sum(risks)/len(risks):.4f} (all in [0,1] numeric)"
    )

    print("\n=== Decision ===")
    decided = run_decision(eligible)
    playbooks: dict[str, int] = {}
    for s in decided:
        playbooks[str(s.get("playbook"))] = playbooks.get(str(s.get("playbook")), 0) + 1
    print("playbook counts:", playbooks)

    print("\n=== Outreach ===")
    outreached = run_outreach(decided)

    print("\n=== Learning ===")
    final_states, uplift = run_learning(outreached, seed=args.seed)
    treated = sum(1 for s in final_states if s.get("arm") == "treated")
    control = sum(1 for s in final_states if s.get("arm") == "control")
    print(f"arms: treated={treated} control={control}")
    print("uplift by playbook × segment (top rows):")
    for row in uplift[:10]:
        cu = row["churn_uplift"]
        uu = row["usage_uplift"]
        print(
            f"  {row['playbook'][:28]:28} | n_t={row['n_treated']:3} n_c={row['n_control']:3} | "
            f"churn_uplift={None if cu is None else round(cu, 3)} | "
            f"usage_uplift={None if uu is None else round(uu, 3)}"
        )

    print("\n=== Sample user states ===")
    for s in final_states[: args.max_print]:
        sample = {
            "customer_id": s.get("customer_id"),
            "risk": round(float(s.get("risk") or 0.0), 4),
            "top_driver_category": s.get("top_driver_category"),
            "drivers": [
                {
                    "feature": d["feature"],
                    "shap_value": round(float(d["shap_value"]), 3),
                    "driver_category": d["driver_category"],
                }
                for d in (s.get("drivers") or [])[:3]
            ],
            "segment": s.get("segment"),
            "playbook": s.get("playbook"),
            "intensity": s.get("intensity"),
            "arm": s.get("arm"),
            "outcome_churn": s.get("outcome_churn"),
            "status": s.get("status"),
            "message": (s.get("intervention_message") or "")[:110] + "...",
        }
        print(json.dumps(sample, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
