"""Optional real-time market-data updater with a reproducible offline fallback.

The online path downloads JPM history from Yahoo's public chart endpoint, VIX
history from Cboe, and DGS10/DGS1 Treasury rates from FRED. The offline
path reads the latest complete Week 2 row. FRED uses a local API key when
available, with public CSV access otherwise. Every snapshot records its mode and
source status so cached/fallback data cannot be mistaken for live data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def offline_snapshot(project_root: Path):
    path = Path(project_root) / "Week2" / "processed_data" / "market_data_processed.csv"
    data = pd.read_csv(path, parse_dates=["Date"])
    row = data.dropna(subset=["Close", "VIX_Close", "Treasury_Rate", "Rolling_Volatility_20D"]).iloc[-1]
    return {
        "as_of_date": str(row["Date"].date()),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_week2_fallback",
        "jpm_close": float(row["Close"]),
        "vix_close": float(row["VIX_Close"]),
        "risk_free_rate": float(row["Treasury_Rate"]) / 100.0,
        "historical_volatility_20d": float(row["Rolling_Volatility_20D"]),
        "sources": {
            "JPM_and_VIX": str(path),
            "risk_free_rate": str(path),
        },
    }


def online_snapshot(project_root, previous, output_dir):
    from market_refresh import refresh
    return refresh(previous, Path(output_dir) / "runtime_sources")


def update_snapshot(project_root: Path, output_dir: Path, online=True):
    output_dir = Path(output_dir)
    previous = offline_snapshot(project_root)
    cache_path = output_dir / "latest_market_snapshot.json"
    if online and cache_path.exists():
        try:
            previous = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    status = "online_success"
    error = None
    if online:
        try:
            snapshot = online_snapshot(project_root, previous, output_dir)
            status = snapshot.get("update_status", "online_success")
            error = snapshot.get("online_error")
        except Exception as exc:
            snapshot = previous
            status = "online_failed_offline_fallback"
            error = type(exc).__name__
    else:
        snapshot = offline_snapshot(project_root)
        status = "offline_requested"
    snapshot["update_status"] = status
    snapshot["online_error"] = error
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from market_refresh import atomic_json
    atomic_json(output_dir / "latest_market_snapshot.json", snapshot)
    flat = {key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in snapshot.items()}
    pd.DataFrame([flat]).to_csv(
        output_dir / "latest_market_snapshot.csv", index=False
    )
    return snapshot


def locate_project_root():
    candidates = [Path.cwd(), *Path.cwd().parents, Path(r"G:\JPM-Chooser Option Pricing")]
    for candidate in candidates:
        if (candidate / "Week2" / "processed_data" / "market_data_processed.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use the reproducible Week 2 fallback without network access.")
    parser.add_argument("--output-dir", default="live_data")
    args = parser.parse_args()
    result = update_snapshot(locate_project_root(), Path(args.output_dir), online=not args.offline)
    print(json.dumps(result, indent=2))
