"""
FastAPI backend for the telco 6G churn loop.

Endpoints:
  POST /signal   — Signal Agent
  POST /decide   — Decision Agent
  POST /outreach — Outreach Agent
  POST /learn    — Learning Agent
  GET  /users    — in-memory state snapshot
  GET  /health   — liveness

Run:
  uvicorn app.api:app --reload --port 8080
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google.cloud import bigquery
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.decision import run_decision  # noqa: E402
from agents.learning import run_learning  # noqa: E402
from agents.outreach import run_outreach  # noqa: E402
from agents.signal_agent import (  # noqa: E402
    DRIVER_ABS_THRESHOLD,
    RISK_THRESHOLD,
    run_signal_for_all_users,
)
from app.loop_store import LoopStore, store  # noqa: E402
from models.churn_model_service import get_global_shap_summary  # noqa: E402


class SignalRequest(BaseModel):
    limit: int | None = Field(default=500, description="Max feature rows to score")
    risk_threshold: float = Field(default=RISK_THRESHOLD, ge=0.0, le=1.0)
    driver_threshold: float = Field(default=DRIVER_ABS_THRESHOLD, ge=0.0)
    only_eligible: bool = True
    project_id: str | None = None


class LearnRequest(BaseModel):
    seed: int = 42


class LoopRunRequest(BaseModel):
    """Convenience: run Signal → Decision → Outreach → Learning in one call."""

    limit: int | None = 500
    risk_threshold: float = RISK_THRESHOLD
    driver_threshold: float = DRIVER_ABS_THRESHOLD
    seed: int = 42
    project_id: str | None = None


def get_project_id() -> str:
    project_id = os.environ.get("PROJECT_ID")
    if not project_id:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("PROJECT_ID=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip()
    if not project_id:
        raise HTTPException(status_code=500, detail="PROJECT_ID is not configured")
    return project_id


def get_bq_client() -> bigquery.Client:
    return _bq_client


def get_store() -> LoopStore:
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bq_client
    project_id = os.environ.get("PROJECT_ID")
    if not project_id:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("PROJECT_ID=") and not line.startswith("#"):
                    project_id = line.split("=", 1)[1].strip()
                    os.environ.setdefault("PROJECT_ID", project_id)
                    break
    _bq_client = bigquery.Client(project=project_id) if project_id else bigquery.Client()
    # Warm model load so first /signal is faster.
    try:
        from models.churn_model_service import load_model

        load_model()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: model warm-load failed: {exc}")
    yield


_bq_client: bigquery.Client | None = None

app = FastAPI(
    title="Telco 6G Churn Loop API",
    description="Signal → Decision → Outreach → Learning control-plane for post-migration churn.",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Browser landing page — bare `/` used to 404 because only API routes existed."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Telco 6G Churn Loop</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 42rem; margin: 3rem auto; padding: 0 1.25rem; line-height: 1.5; color: #111; }
    a { color: #0b57d0; }
    code { background: #f2f4f7; padding: 0.1rem 0.35rem; border-radius: 4px; }
    .card { border: 1px solid #e4e7ec; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
  </style>
</head>
<body>
  <h1>Telco 6G Churn Loop API</h1>
  <p>Churn risk &amp; intervention control plane for post–5G→6G migration.</p>
  <div class="card">
    <p><strong>Dashboard UI:</strong> <a href="https://telco-6g-churn-ui-454334461204.us-central1.run.app">Streamlit control center</a></p>
    <p><strong>Interactive docs:</strong> <a href="/docs">/docs</a></p>
    <p><strong>Health:</strong> <a href="/health">/health</a></p>
    <p><strong>Users snapshot:</strong> <a href="/users">/users</a></p>
  </div>
  <p>Core loop: <code>POST /signal</code> → <code>/decide</code> → <code>/outreach</code> → <code>/learn</code>
     (or <code>POST /loop/run</code> for the full chain).</p>
</body>
</html>"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/signal")
def signal(
    body: SignalRequest,
    loop_store: LoopStore = Depends(get_store),
) -> dict[str, Any]:
    """Run Signal Agent; return loop-eligible user states with numeric risk."""
    project_id = body.project_id or os.environ.get("PROJECT_ID")
    states = run_signal_for_all_users(
        project_id=project_id,
        limit=body.limit,
        risk_threshold=body.risk_threshold,
        driver_threshold=body.driver_threshold,
        only_eligible=body.only_eligible,
    )
    loop_store.set_users(states, step="signal")
    risks = [float(s.get("risk") or 0.0) for s in states]
    return {
        "step": "signal",
        "n_users": len(states),
        "risk_threshold": body.risk_threshold,
        "risk_min": min(risks) if risks else None,
        "risk_max": max(risks) if risks else None,
        "risk_mean": (sum(risks) / len(risks)) if risks else None,
        "users": states,
    }


@app.post("/decide")
def decide(loop_store: LoopStore = Depends(get_store)) -> dict[str, Any]:
    """Run Decision Agent on current in-memory eligible users."""
    snap = loop_store.snapshot()
    if not snap["users"]:
        raise HTTPException(status_code=400, detail="No users in state. Call POST /signal first.")
    states = run_decision(snap["users"])
    loop_store.set_users(states, step="decide")
    playbooks: dict[str, int] = {}
    for s in states:
        pb = str(s.get("playbook") or "NA")
        playbooks[pb] = playbooks.get(pb, 0) + 1
    return {"step": "decide", "n_users": len(states), "playbook_counts": playbooks, "users": states}


@app.post("/outreach")
def outreach(loop_store: LoopStore = Depends(get_store)) -> dict[str, Any]:
    """Run Outreach Agent; attach intervention messages."""
    snap = loop_store.snapshot()
    if not snap["users"]:
        raise HTTPException(status_code=400, detail="No users in state. Call POST /signal first.")
    states = run_outreach(snap["users"])
    loop_store.set_users(states, step="outreach")
    return {"step": "outreach", "n_users": len(states), "users": states}


@app.post("/learn")
def learn(
    body: LearnRequest | None = None,
    loop_store: LoopStore = Depends(get_store),
) -> dict[str, Any]:
    """Run Learning Agent; return uplift metrics + updated states."""
    body = body or LearnRequest()
    snap = loop_store.snapshot()
    if not snap["users"]:
        raise HTTPException(status_code=400, detail="No users in state. Call POST /signal first.")
    states, uplift = run_learning(snap["users"], seed=body.seed)
    loop_store.set_users(states, step="learn")
    loop_store.set_uplift(uplift)
    return {
        "step": "learn",
        "n_users": len(states),
        "n_treated": sum(1 for s in states if s.get("arm") == "treated"),
        "n_control": sum(1 for s in states if s.get("arm") == "control"),
        "uplift": uplift,
        "users": states,
    }


@app.post("/loop/run")
def run_full_loop(
    body: LoopRunRequest,
    loop_store: LoopStore = Depends(get_store),
) -> dict[str, Any]:
    """One-shot Signal → Decision → Outreach → Learning for the dashboard."""
    project_id = body.project_id or os.environ.get("PROJECT_ID")
    states = run_signal_for_all_users(
        project_id=project_id,
        limit=body.limit,
        risk_threshold=body.risk_threshold,
        driver_threshold=body.driver_threshold,
        only_eligible=True,
    )
    states = run_decision(states)
    states = run_outreach(states)
    states, uplift = run_learning(states, seed=body.seed)
    loop_store.set_users(states, step="learn")
    loop_store.set_uplift(uplift)
    risks = [float(s.get("risk") or 0.0) for s in states]
    return {
        "step": "loop",
        "n_users": len(states),
        "risk_min": min(risks) if risks else None,
        "risk_max": max(risks) if risks else None,
        "risk_mean": (sum(risks) / len(risks)) if risks else None,
        "n_treated": sum(1 for s in states if s.get("arm") == "treated"),
        "n_control": sum(1 for s in states if s.get("arm") == "control"),
        "uplift": uplift,
        "users": states,
    }


@app.get("/users")
def users(
    loop_store: LoopStore = Depends(get_store),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    """Return in-memory user-state snapshot (includes numeric risk)."""
    snap = loop_store.snapshot()
    return {
        "last_step": snap["last_step"],
        "n_users": snap["n_users"],
        "users": snap["users"][:limit],
        "uplift": snap["uplift"],
    }


@app.get("/metrics/shap")
def shap_summary() -> dict[str, Any]:
    """Global SHAP summary for Segments & Drivers view."""
    try:
        return get_global_shap_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/metrics/population")
def population_metrics(
    client: bigquery.Client = Depends(get_bq_client),
) -> dict[str, Any]:
    """Population KPIs from BigQuery features (executive dashboard)."""
    project_id = client.project
    dataset = os.environ.get("BIGQUERY_DATASET", "telco_churn")
    table = os.environ.get("BIGQUERY_TABLE", "features")
    fqn = f"`{project_id}.{dataset}.{table}`"
    sql = f"""
    SELECT
      COUNT(*) AS n_customers,
      COUNTIF(migration_flag = 1) AS n_migrants,
      AVG(target_churn_90d) AS overall_churn_90d,
      AVG(IF(migration_flag = 1, target_churn_90d, NULL)) AS migrant_churn_90d,
      AVG(IF(migration_flag = 0, target_churn_90d, NULL)) AS non_migrant_churn_90d,
      AVG(MonthlyCharges) AS avg_arpu,
      AVG(IF(migration_flag = 1, MonthlyCharges, NULL)) AS migrant_arpu,
      SUM(IF(migration_flag = 1, MonthlyCharges, 0)) AS migrant_revenue_base
    FROM {fqn}
    WHERE label_eligible = 1 OR label_eligible IS NULL
    """
    try:
        row = dict(list(client.query(sql).result())[0])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"BigQuery query failed: {exc}") from exc

    # Revenue at risk proxy: migrant ARPU × migrant 90d churn rate × 3 months.
    migrant_churn = float(row.get("migrant_churn_90d") or 0.0)
    migrant_rev = float(row.get("migrant_revenue_base") or 0.0)
    row["revenue_at_risk_90d"] = migrant_rev * migrant_churn * 3.0
    # Simulated migration NPS proxy (scaled from QoS-ish story).
    row["migration_nps_proxy"] = round(40.0 - 100.0 * migrant_churn, 1)
    return row
