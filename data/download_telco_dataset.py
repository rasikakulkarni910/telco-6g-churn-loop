"""
Download the IBM Telco Customer Churn dataset into data/telco_churn.csv.

Primary source: IBM sample hosted on GitHub (no Kaggle credentials required).
If the download fails, place a CSV manually at data/telco_churn.csv and re-run
downstream load scripts.
"""

from __future__ import annotations

import argparse
import ssl
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# Well-known public mirror of the Kaggle / IBM Telco Customer Churn CSV.
DEFAULT_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DATA_DIR / "telco_churn.csv"

_MANUAL_HINT = (
    "Manual fallback:\n"
    "  1. Download Telco-Customer-Churn.csv from Kaggle or IBM\n"
    "  2. Save it as {output_path}\n"
    "  3. Re-run this script (it will detect the file) or proceed to load_raw_to_bigquery.py"
)


def _fetch_bytes(url: str) -> bytes:
    """Download URL bytes; prefer requests, fall back to urllib (with SSL workaround)."""
    try:
        import requests

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content
    except Exception as requests_exc:  # noqa: BLE001 — try urllib next
        print(f"requests download failed ({requests_exc}); trying urllib...")

    try:
        with urlopen(url, timeout=60) as resp:  # noqa: S310 — fixed public dataset URL
            return resp.read()
    except URLError as urllib_exc:
        # Common on macOS Python installs missing certifi root certs.
        print(f"urllib SSL/network failed ({urllib_exc}); retrying with certifi/unverified context...")
        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001
            ctx = ssl._create_unverified_context()  # noqa: S323 — last-resort local bootstrap
        with urlopen(url, timeout=60, context=ctx) as resp:  # noqa: S310
            return resp.read()


def download_dataset(url: str = DEFAULT_URL, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Fetch the Telco churn CSV and write it to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Dataset already present at {output_path} ({output_path.stat().st_size:,} bytes)")
        return output_path

    print(f"Downloading Telco Customer Churn dataset from:\n  {url}")
    try:
        payload = _fetch_bytes(url)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Download failed: {exc}\n" + _MANUAL_HINT.format(output_path=output_path)
        ) from exc

    if not payload:
        raise SystemExit("Download produced an empty file; remove and retry.")

    output_path.write_bytes(payload)
    print(f"Saved {len(payload):,} bytes to {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Telco Customer Churn CSV")
    parser.add_argument("--url", default=DEFAULT_URL, help="Source CSV URL")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Local path for telco_churn.csv",
    )
    args = parser.parse_args(argv)
    download_dataset(url=args.url, output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
