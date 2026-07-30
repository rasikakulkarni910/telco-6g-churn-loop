"""
Load data/telco_churn.csv into BigQuery as PROJECT_ID.telco_churn.raw_customers.

Creates the dataset if it does not exist. Schema is autodetected from the CSV.

Environment:
  PROJECT_ID          GCP project (required)
  BIGQUERY_DATASET    default: telco_churn
  BIGQUERY_LOCATION   default: US
  GOOGLE_APPLICATION_CREDENTIALS  path to a service-account JSON (or use gcloud ADC)

Usage:
  python data/download_telco_dataset.py
  export PROJECT_ID=your-gcp-project
  python data/load_raw_to_bigquery.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = DATA_DIR / "telco_churn.csv"
DEFAULT_DATASET = "telco_churn"
DEFAULT_TABLE = "raw_customers"
DEFAULT_LOCATION = "US"


def get_project_id(explicit: str | None = None) -> str:
    project_id = explicit or os.environ.get("PROJECT_ID")
    if not project_id:
        raise SystemExit(
            "PROJECT_ID is required. Example:\n"
            "  export PROJECT_ID=your-gcp-project\n"
            "  python data/load_raw_to_bigquery.py"
        )
    return project_id


def ensure_dataset(
    client: bigquery.Client,
    dataset_id: str,
    location: str,
) -> bigquery.Dataset:
    """Create the BigQuery dataset if missing."""
    dataset_ref = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset_ref.location = location
    try:
        return client.get_dataset(dataset_ref)
    except NotFound:
        print(f"Creating dataset {client.project}.{dataset_id} in {location}...")
        return client.create_dataset(dataset_ref, exists_ok=True)


def load_csv_to_bigquery(
    csv_path: Path,
    project_id: str,
    dataset_id: str = DEFAULT_DATASET,
    table_id: str = DEFAULT_TABLE,
    location: str = DEFAULT_LOCATION,
) -> str:
    """
    Read the local CSV and load it into BigQuery with autodetected schema.

    Returns the fully-qualified table id: project.dataset.table
    """
    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            "Run: python data/download_telco_dataset.py"
        )

    # pandas read validates the file early and surfaces encoding issues clearly.
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {csv_path}")

    client = bigquery.Client(project=project_id)
    ensure_dataset(client, dataset_id, location)

    table_fqn = f"{project_id}.{dataset_id}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        # Autodetect from pandas / CSV types — fine for Checkpoint 1 raw landing.
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.CSV,
    )

    # load_table_from_dataframe is more reliable than CSV upload for mixed types
    # (e.g. TotalCharges often arrives as object due to blank strings).
    print(f"Loading into {table_fqn} (WRITE_TRUNCATE)...")
    job = client.load_table_from_dataframe(df, table_fqn, job_config=job_config)
    job.result()

    table = client.get_table(table_fqn)
    print(
        f"Done: {table.full_table_id} — {table.num_rows:,} rows, "
        f"{len(table.schema)} columns"
    )
    return table_fqn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load Telco CSV into BigQuery raw_customers")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to telco_churn.csv")
    parser.add_argument("--project-id", default=None, help="GCP project (or set PROJECT_ID)")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("BIGQUERY_DATASET", DEFAULT_DATASET),
        help="BigQuery dataset id",
    )
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Destination table id")
    parser.add_argument(
        "--location",
        default=os.environ.get("BIGQUERY_LOCATION", DEFAULT_LOCATION),
        help="Dataset location (e.g. US, EU, us-central1)",
    )
    args = parser.parse_args(argv)

    project_id = get_project_id(args.project_id)
    load_csv_to_bigquery(
        csv_path=args.csv,
        project_id=project_id,
        dataset_id=args.dataset,
        table_id=args.table,
        location=args.location,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
