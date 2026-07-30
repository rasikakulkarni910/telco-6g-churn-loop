"""
Build telco_churn.features from telco_churn.raw_customers (BigQuery → BigQuery).

Label definition — landmark 90-day churn (NOT a rename of Churn)
----------------------------------------------------------------
The IBM Telco table gives:
  - tenure  = months from start until churn OR censoring
  - Churn   = whether an event occurred at that tenure

That is survival data, not a 90-day ahead label. We *construct* one:

  1. Interpret (tenure, Churn) as (duration_months, event_churn).
  2. Draw a scoring landmark month tenure_at_score while the customer
     was still active, with enough follow-up to observe a 3-month window
     (3 months ≈ 90 days — the CRM intervention horizon).
  3. Set
       target_churn_90d = 1  iff  event occurs in (tenure_at_score, tenure_at_score+3]
       target_churn_90d = 0  iff  customer is observed event-free through +3 months

So a long-tenured eventual churner scored early can be label 0, and the model
must learn P(churn within ~90 days | state at score time) — not P(ever churned).

risk from the model = predict_proba = that probability.

Synthetic migration features are correlated with this horizon label (not with
ever-churn alone) so SHAP drivers stay aligned to the intervention window.

Re-run safely: WRITE_TRUNCATE on features.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

DEFAULT_DATASET = "telco_churn"
DEFAULT_SOURCE_TABLE = "raw_customers"
DEFAULT_FEATURES_TABLE = "features"
DEFAULT_LOCATION = "US"
RANDOM_SEED = 42
# 90 days ≈ 3 billing months — explicit so the probability horizon is not vague.
HORIZON_MONTHS = 3


def get_project_id(explicit: str | None = None) -> str:
    project_id = explicit or os.environ.get("PROJECT_ID")
    if not project_id:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("PROJECT_ID=") and not line.startswith("#"):
                    project_id = line.split("=", 1)[1].strip()
                    break
    if not project_id:
        raise SystemExit("PROJECT_ID is required (env, --project-id, or .env).")
    return project_id


def read_raw_customers(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> pd.DataFrame:
    table_fqn = f"{project_id}.{dataset_id}.{table_id}"
    try:
        client.get_table(table_fqn)
    except NotFound as exc:
        raise SystemExit(
            f"Source table not found: {table_fqn}\n"
            "Run Checkpoint 1 first: python data/load_raw_to_bigquery.py"
        ) from exc

    print(f"Reading {table_fqn}...")
    df = client.query(f"SELECT * FROM `{table_fqn}`").to_dataframe()
    print(f"  {len(df):,} rows × {len(df.columns)} columns")
    return df


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def assign_landmark_90d_label(
    duration_months: np.ndarray,
    event_churn: np.ndarray,
    rng: np.random.Generator,
    horizon_months: int = HORIZON_MONTHS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each customer, sample a scoring month and a 90-day-ahead label.

    Returns
    -------
    tenure_at_score : int months at which we score the customer
    target_churn_90d : 1 if churn event falls inside (t, t+horizon]
    label_eligible : 1 if follow-up is sufficient to define the label
    """
    n = len(duration_months)
    tenure_at_score = np.zeros(n, dtype=int)
    target = np.zeros(n, dtype=int)
    eligible = np.zeros(n, dtype=int)

    for i in range(n):
        t_end = int(max(duration_months[i], 0))
        event = int(event_churn[i])

        if event == 1:
            # Event at month t_end. Still active for landmarks t in [0, t_end - 1]
            # (and t=0 when t_end==0: churned in first month → label 1).
            if t_end <= 0:
                tenure_at_score[i] = 0
                target[i] = 1
                eligible[i] = 1
                continue
            t = int(rng.integers(0, t_end))  # 0 .. t_end-1 inclusive
            tenure_at_score[i] = t
            # Churn within the next `horizon_months` months after score.
            target[i] = 1 if t_end <= t + horizon_months else 0
            eligible[i] = 1
        else:
            # Censored at t_end: only landmarks with full horizon follow-up.
            # Need t_end >= t + horizon  ⇒  t <= t_end - horizon.
            max_t = t_end - horizon_months
            if max_t < 0:
                # Observed lifetime shorter than the horizon — cannot certify
                # "did not churn in 90 days". Mark ineligible for training.
                tenure_at_score[i] = t_end
                target[i] = 0
                eligible[i] = 0
                continue
            t = int(rng.integers(0, max_t + 1))
            tenure_at_score[i] = t
            target[i] = 0
            eligible[i] = 1

    return tenure_at_score, target, eligible


def build_features(raw: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Construct landmark 90-day label + synthetic migration features."""
    rng = np.random.default_rng(seed)
    df = raw.copy()

    # --- Survival framing from the raw Telco fields ---
    df["duration_months"] = _coerce_numeric(df["tenure"]).fillna(0).astype(int)
    df["event_churn"] = (
        df["Churn"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)
    df["horizon_months"] = HORIZON_MONTHS

    tenure_at_score, target_90d, label_eligible = assign_landmark_90d_label(
        df["duration_months"].to_numpy(),
        df["event_churn"].to_numpy(),
        rng,
        horizon_months=HORIZON_MONTHS,
    )
    df["tenure_at_score"] = tenure_at_score
    df["target_churn_90d"] = target_90d
    df["label_eligible"] = label_eligible

    # Modeling feature: tenure known at score time (not the eventual duration).
    df["tenure"] = df["tenure_at_score"]

    # Monthly / total charges cleaning — as-of the landmark, not end of life.
    # Using raw TotalCharges would leak post-score billing and inflate AUC.
    df["MonthlyCharges"] = _coerce_numeric(df["MonthlyCharges"]).fillna(0.0)
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure_at_score"]).round(2)

    # Value segment from ARPU terciles (Decision Agent / VIP plays).
    try:
        df["value_segment"] = pd.qcut(
            df["MonthlyCharges"],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        df["value_segment"] = "Medium"

    # --- migration_flag (early 6G adopter simulation) ---
    internet = df["InternetService"].astype(str)
    base_prob = np.where(
        internet.str.contains("Fiber", case=False, na=False),
        0.55,
        np.where(internet.str.contains("DSL", case=False, na=False), 0.35, 0.15),
    )
    base_prob = np.clip(
        base_prob + np.where(df["value_segment"] == "High", 0.10, 0.0),
        0.05,
        0.85,
    )
    df["migration_flag"] = (rng.random(len(df)) < base_prob).astype(int)
    df["migration_date_offset_days"] = np.where(
        df["migration_flag"] == 1,
        rng.integers(0, 121, size=len(df)),
        0,
    ).astype(int)

    # Stress signals correlated with the *90-day horizon label*, not ever-churn.
    migrant = df["migration_flag"].to_numpy() == 1
    risk_90 = df["target_churn_90d"].to_numpy() == 1

    qos = rng.normal(loc=0.72, scale=0.12, size=len(df))
    qos = np.where(migrant & risk_90, qos - 0.18, qos)
    qos = np.where(migrant & ~risk_90, qos + 0.05, qos)
    qos = np.where(~migrant, rng.normal(loc=0.80, scale=0.08, size=len(df)), qos)
    df["post_migration_qos"] = np.clip(qos, 0.05, 0.99).round(4)

    bill_shock = rng.normal(loc=0.04, scale=0.08, size=len(df))
    bill_shock = np.where(migrant, bill_shock + 0.06, bill_shock * 0.3)
    bill_shock = np.where(migrant & risk_90, bill_shock + 0.10, bill_shock)
    df["bill_change_pct"] = np.clip(bill_shock, -0.25, 0.60).round(4)

    usage = rng.normal(loc=0.02, scale=0.15, size=len(df))
    usage = np.where(migrant & risk_90, usage - 0.20, usage)
    usage = np.where(migrant & ~risk_90, usage + 0.05, usage)
    df["usage_change_30d"] = np.clip(usage, -0.80, 0.80).round(4)

    sentiment = rng.normal(loc=0.15, scale=0.35, size=len(df))
    sentiment = np.where(migrant & risk_90, sentiment - 0.45, sentiment)
    df["support_sentiment"] = np.clip(sentiment, -1.0, 1.0).round(4)

    regions = np.array(["Northeast", "Midwest", "South", "West"])
    devices = np.array(["Smartphone", "Hotspot", "Tablet", "CPE-Router"])
    df["region"] = regions[rng.integers(0, len(regions), size=len(df))]
    df["device_type"] = devices[rng.integers(0, len(devices), size=len(df))]

    if "customerID" in df.columns:
        df["customer_id"] = df["customerID"].astype(str)
    else:
        df["customer_id"] = [f"cust_{i:05d}" for i in range(len(df))]

    preferred_front = [
        "customer_id",
        "tenure_at_score",
        "duration_months",
        "event_churn",
        "horizon_months",
        "label_eligible",
        "migration_flag",
        "migration_date_offset_days",
        "post_migration_qos",
        "bill_change_pct",
        "usage_change_30d",
        "support_sentiment",
        "value_segment",
        "region",
        "device_type",
        "target_churn_90d",
    ]
    other_cols = [c for c in df.columns if c not in preferred_front and c != "customerID"]
    return df[preferred_front + other_cols]


def write_features(
    client: bigquery.Client,
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> str:
    table_fqn = f"{project_id}.{dataset_id}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    print(f"Writing {len(df):,} rows → {table_fqn} (WRITE_TRUNCATE)...")
    job = client.load_table_from_dataframe(df, table_fqn, job_config=job_config)
    job.result()
    table = client.get_table(table_fqn)
    print(f"Done: {table.full_table_id} — {table.num_rows:,} rows, {len(table.schema)} columns")
    return table_fqn


def summarize(df: pd.DataFrame) -> None:
    eligible = df["label_eligible"] == 1
    sub = df.loc[eligible]
    agree = float((sub["target_churn_90d"] == sub["event_churn"]).mean()) if len(sub) else 0.0
    ever_churners = sub["event_churn"] == 1
    early_score_ok = float((sub.loc[ever_churners, "target_churn_90d"] == 0).mean()) if ever_churners.any() else 0.0
    migrants = int(df["migration_flag"].sum())
    print(
        f"Eligible for 90d label: {eligible.sum():,}/{len(df):,} | "
        f"P(target_churn_90d=1)={sub['target_churn_90d'].mean():.1%} | "
        f"agreement with ever-churn={agree:.1%} "
        f"(ever-churners labeled 0 when scored early={early_score_ok:.1%})"
    )
    print(
        f"Migrants={migrants:,} ({migrants / len(df):.1%}) | "
        f"90d rate migrant={df.loc[df.migration_flag == 1, 'target_churn_90d'].mean():.1%} vs "
        f"non={df.loc[df.migration_flag == 0, 'target_churn_90d'].mean():.1%}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build BigQuery features with landmark 90d churn label")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--dataset", default=os.environ.get("BIGQUERY_DATASET", DEFAULT_DATASET))
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--features-table", default=DEFAULT_FEATURES_TABLE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    project_id = get_project_id(args.project_id)
    client = bigquery.Client(project=project_id)

    raw = read_raw_customers(client, project_id, args.dataset, args.source_table)
    features = build_features(raw, seed=args.seed)
    summarize(features)
    write_features(client, features, project_id, args.dataset, args.features_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
