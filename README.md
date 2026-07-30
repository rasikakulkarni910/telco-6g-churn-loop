# telco_6g_churn_loop

Portfolio-grade **churn risk & intervention control center** for a telecom operator after a 5G→6G migration.

This repo is built in **slices/checkpoints**. Each checkpoint ends in something runnable.

| Checkpoint | Status | What you get |
|---|---|---|
| 1 – Scaffolding + BigQuery raw load | Done | Project layout + `raw_customers` in BigQuery |
| 2 – Features + migration simulation | Done | `features` table with churn label + 6G fields |
| 3 – Model + SHAP | Done | Risk score + driver explanations |
| 4 – Agents (pure Python) | Done | Signal → Decision → Outreach → Learning |
| 5 – FastAPI | Done | `/signal`, `/decide`, `/outreach`, `/learn`, `/users` |
| 6 – Streamlit dashboard | Done | Executive / Segments / Loop / User explorer |
| 7 – Cloud Run | **Current** | Public demo URL |
| 8 – Documentation polish | Pending | Architecture diagram + full README |

---

## Problem (preview)

After migrating subscribers to 6G, high-value customers can churn due to bill shock, QoS regressions, usage drops, or weak perceived value. This system predicts **P(churn in next 90 days)**, explains drivers with SHAP, and runs an agentic intervention loop.

*(Full problem statement, label definitions, and architecture land in later checkpoints.)*

---

## Project structure

```
telco_6g_churn_loop/
├── data/                 # Download + BigQuery load scripts
├── notebooks/            # EDA / training notebooks (Checkpoint 3+)
├── models/               # Model train + risk/driver service (Checkpoint 3+)
├── agents/               # Signal / Decision / Outreach / Learning (Checkpoint 4+)
├── app/                  # FastAPI + Streamlit (Checkpoints 5–6)
├── infra/                # Cloud Run / IAM notes (Checkpoint 7)
├── requirements.txt
└── README.md
```

---

## Checkpoint 1 — BigQuery setup

### Prerequisites

1. A Google Cloud project with **BigQuery API** enabled.
2. Authentication via one of:
   - `gcloud auth application-default login` (local ADC), or
   - `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`
3. Python 3.10+ and pip.

Install Google Cloud SDK if needed: https://cloud.google.com/sdk/docs/install

### 1. Install Python deps

```bash
cd telco_6g_churn_loop
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the Telco Customer Churn CSV

```bash
python data/download_telco_dataset.py
```

This writes `data/telco_churn.csv` from the public IBM mirror.  
If the download fails, place any Telco Customer Churn CSV at that path manually.

### 3. Set your GCP project

```bash
export PROJECT_ID=your-gcp-project-id
# optional overrides:
# export BIGQUERY_DATASET=telco_churn
# export BIGQUERY_LOCATION=US
```

### 4. Create dataset + load `raw_customers`

```bash
python data/load_raw_to_bigquery.py
```

What this does:

- Creates dataset `PROJECT_ID.telco_churn` if missing (location `US` by default).
- Loads `data/telco_churn.csv` into `PROJECT_ID.telco_churn.raw_customers`.
- Uses **schema autodetection** and `WRITE_TRUNCATE` (safe to re-run).

### 5. Verify in BigQuery

Console SQL:

```sql
SELECT COUNT(*) AS n_rows
FROM `your-gcp-project-id.telco_churn.raw_customers`;

SELECT *
FROM `your-gcp-project-id.telco_churn.raw_customers`
LIMIT 5;
```

Or CLI:

```bash
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) FROM `'"$PROJECT_ID"'.telco_churn.raw_customers`'
```

### Done when

You can run `python data/load_raw_to_bigquery.py` and see table **`raw_customers`** in dataset **`telco_churn`**.

---

## Checkpoint 2 — Features + 5G→6G migration simulation

### Label definition: `target_churn_90d` (landmark construction)

The Telco CSV is **survival data**, not a 90-day label:

- `tenure` = months until churn **or** censoring  
- `Churn` = whether an event happened at that tenure  

We **construct** a forward 90-day outcome (≈ 3 billing months):

1. Treat `(tenure, Churn)` as `(duration_months, event_churn)`.
2. Sample a scoring landmark `tenure_at_score` while the customer was still active, with enough follow-up to observe the next 3 months.
3. Set:

```text
target_churn_90d = 1  if churn event falls in (tenure_at_score, tenure_at_score + 3]
target_churn_90d = 0  if observed event-free through +3 months
```

Rows without full follow-up get `label_eligible = 0` and are excluded from training.

So an eventual churner scored early can be **label 0**. The model’s `risk = predict_proba` is **P(churn within ~90 days | state at score time)** — not P(ever churned).

Features at score time avoid leakage (e.g. `TotalCharges ≈ MonthlyCharges × tenure_at_score`).

### Migration feature construction

`data/features_builder.py` reads `raw_customers` and writes `PROJECT_ID.telco_churn.features` with:

| Field | Meaning |
|---|---|
| `tenure_at_score` | Landmark month used for the 90-day label |
| `duration_months` / `event_churn` | Original survival pair from the CSV |
| `migration_flag` | Simulated 6G migrant (~35–45%); biased toward Fiber / higher ARPU |
| `migration_date_offset_days` | Days since simulated cutover (0–120) |
| `post_migration_qos` | QoS score 0–1 (worse more often when 90d label = 1) |
| `bill_change_pct` | % bill change after migration pricing |
| `usage_change_30d` | 30-day usage delta proxy |
| `support_sentiment` | Synthetic ticket sentiment (−1…1) |
| `value_segment` | Low / Medium / High from MonthlyCharges terciles |
| `region`, `device_type` | Synthetic filter dimensions for the UI |

Seeded RNG (`seed=42`) keeps regenerations reproducible.
### Run

```bash
export PROJECT_ID=project-5506c1fb-580d-4ca3-b04
source .venv/bin/activate
python data/features_builder.py
```

Verify:

```sql
SELECT
  COUNT(*) AS n,
  COUNTIF(migration_flag = 1) AS migrants,
  AVG(IF(migration_flag = 1, target_churn_90d, NULL)) AS migrant_churn_rate
FROM `project-5506c1fb-580d-4ca3-b04.telco_churn.features`;
```

### Done when

You can query `telco_churn.features` and see migration + churn features.

---

## Checkpoint 3 — Modeling + SHAP drivers

### Train

```bash
export PROJECT_ID=project-5506c1fb-580d-4ca3-b04
source .venv/bin/activate
pip install -r requirements.txt
python models/train_churn_model.py
```

Artifacts written under `models/`:

| File | Purpose |
|---|---|
| `churn_model.json` | Trained XGBoost booster |
| `metrics.json` | Holdout ROC-AUC / F1 / precision / recall |
| `global_shap_summary.json` | Mean \|SHAP\| by feature + driver category |
| `feature_config.json` | Feature list + driver category map |

Interactive EDA: `notebooks/model_eda_and_train.ipynb`

### Risk + drivers API

```python
from google.cloud import bigquery
from models.churn_model_service import predict_risk_and_drivers, get_global_shap_summary

client = bigquery.Client(project="project-5506c1fb-580d-4ca3-b04")
row = client.query("""
  SELECT * FROM `project-5506c1fb-580d-4ca3-b04.telco_churn.features`
  WHERE migration_flag = 1 LIMIT 1
""").to_dataframe().iloc[0]

print(predict_risk_and_drivers(row))
# {"risk": 0.99, "drivers": [{"feature": "post_migration_qos", "shap_value": ..., "driver_category": "network"}, ...], ...}
```

Driver categories (`network`, `billing`, `usage`, `support`, `migration`) are defined in `feature_config.json` and feed Checkpoint 4 playbook selection.

**Note:** Attributions use XGBoost `pred_contribs` (tree-path SHAP). `shap.TreeExplainer` is avoided because XGBoost 3’s `base_score` encoding breaks it; the notebook still uses the `shap` package for summary plots.

**Demo note:** synthetic migration stressors are correlated with the landmark label so SHAP/playbooks have signal; still a portfolio dataset, not production telemetry.
### Done when

You can run `predict_risk_and_drivers()` and get a risk score + categorized drivers for a sample user.

---

## Checkpoint 4 — Agentic loop (pure Python)

```bash
export PROJECT_ID=project-5506c1fb-580d-4ca3-b04
source .venv/bin/activate
python agents/run_loop_demo.py --limit 400
```

| Agent | Module | Responsibility |
|---|---|---|
| Signal | `agents/signal_agent.py` | `predict_proba` risk + SHAP drivers; entry/exit gates |
| Decision | `agents/decision.py` | segment → playbook → intensity |
| Outreach | `agents/outreach.py` | deterministic scripts (`generate_intervention_llm` stubbed) |
| Learning | `agents/learning.py` | treated/control, simulated outcomes, uplift |

**Signal entry (all required):** `migration_flag==1`, `risk >= 0.60` (numeric probability), ≥1 strong driver (`|SHAP| >= 0.25`).  
**Exit:** `status in {churned, stabilized}` → skip.

**Driver → playbook:** network→Network Rescue · billing→Bill Clarity + Price Protection · usage→6G Value Unlock · support/migration+High→VIP Rescue.

### Done when

Demo prints numeric `risk`, drivers, playbooks, arms, and uplift.

---

## Checkpoint 5 — FastAPI backend

```bash
export PROJECT_ID=project-5506c1fb-580d-4ca3-b04
source .venv/bin/activate
uvicorn app.api:app --reload --port 8080
```

Open http://127.0.0.1:8080/docs

| Method | Path | Purpose |
|---|---|---|
| POST | `/signal` | Score users; store eligible states (`risk` float) |
| POST | `/decide` | Assign segment / playbook / intensity |
| POST | `/outreach` | Attach intervention scripts |
| POST | `/learn` | Treated/control + uplift metrics |
| POST | `/loop/run` | Full chain (dashboard convenience) |
| GET | `/users` | In-memory snapshot (numeric `risk`) |
| GET | `/metrics/population` | BigQuery executive KPIs |
| GET | `/metrics/shap` | Global driver summary |
| GET | `/health` | Liveness |

### Done when

`/docs` works and each loop endpoint returns successfully.

---

## Checkpoint 6 — Streamlit control center

In a second terminal (API must be running):

```bash
source .venv/bin/activate
export API_BASE_URL=http://127.0.0.1:8080
streamlit run app/ui.py
```

Views: Executive · Segments & Drivers · Loop Performance · **User Explorer** (column `risk (P90d)` formatted as `0.0000`–`1.0000`, not high/low).

Sidebar **Run full loop** calls `POST /loop/run`.

### Done when

Dashboard loads, loop runs via API, and User Explorer shows numeric risk.

---

## Checkpoint 7 — Cloud Run (deploy from source)

**Live demo (API):** https://telco-6g-churn-loop-454334461204.us-central1.run.app  
**Live demo (UI):** https://telco-6g-churn-ui-454334461204.us-central1.run.app  
**API docs:** https://telco-6g-churn-loop-454334461204.us-central1.run.app/docs

No local Docker required. Cloud Build builds from the repo `Dockerfile` (or buildpacks) when you run:

```bash
export PROJECT_ID=project-5506c1fb-580d-4ca3-b04

gcloud run deploy telco-6g-churn-loop \
  --source . \
  --region=us-central1 \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --set-env-vars=PROJECT_ID=$PROJECT_ID,BIGQUERY_DATASET=telco_churn,BIGQUERY_TABLE=features
```

Env vars on the service: `PROJECT_ID`, `BIGQUERY_DATASET`, `BIGQUERY_TABLE`.  
Runtime SA needs BigQuery Job User + Data Viewer (granted on the default compute SA for this project).

### Done when

Public Cloud Run URL serves `/docs` and loop endpoints.

---

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `PROJECT_ID` | Yes | — | GCP project for BigQuery |
| `BIGQUERY_DATASET` | No | `telco_churn` | Dataset id |
| `BIGQUERY_TABLE` | No | `features` | Features table (train script) |
| `BIGQUERY_LOCATION` | No | `US` | Dataset location |
| `API_BASE_URL` | No | `http://127.0.0.1:8080` | Streamlit → FastAPI |
| `GOOGLE_APPLICATION_CREDENTIALS` | No* | ADC | Service account JSON path |

\* Not required if Application Default Credentials are already configured via `gcloud`.

---

## Next up

**Checkpoint 8** — Polish README: problem statement, architecture diagram, driver→playbook table, experimental design, stack + run/deploy guide, live demo link.
