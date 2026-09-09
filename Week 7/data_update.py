"""Optional real-time market-data updater with a reproducible offline fallback.

The online path downloads JPM history from Yahoo's public chart endpoint, VIX
history from Cboe, and the DGS1 one-year Treasury rate from FRED.  The offline
path reads the latest complete
Week 2 row.  No API key is required.  Every snapshot records its mode and
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


def online_snapshot():
    def download_bytes(url):
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=25) as response:
            return response.read()

    yahoo_url = "https://query1.finance.yahoo.com/v8/finance/chart/JPM?range=3mo&interval=1d"
    payload = json.loads(download_bytes(yahoo_url))
    chart = payload["chart"]["result"][0]
    timestamps = pd.to_datetime(chart["timestamp"], unit="s", utc=True).tz_convert(None)
    adjusted = chart.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
    values = adjusted or chart["indicators"]["quote"][0]["close"]
    close = pd.Series(values, index=timestamps, name="JPM_Close").dropna()
    if len(close) < 21:
        raise RuntimeError("Yahoo chart endpoint returned fewer than 21 JPM observations.")
    log_returns = np.log(close / close.shift(1))
    volatility = float(log_returns.tail(20).std(ddof=1) * np.sqrt(252))

    cboe_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    vix = pd.read_csv(BytesIO(download_bytes(cboe_url)))
    vix.columns = [column.strip().upper() for column in vix.columns]
    vix["DATE"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y")
    vix["CLOSE"] = pd.to_numeric(vix["CLOSE"], errors="coerce")
    vix = vix.dropna(subset=["CLOSE"])

    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1"
    fred = pd.read_csv(BytesIO(download_bytes(fred_url)))
    fred.columns = ["Date", "DGS1"]
    fred["Date"] = pd.to_datetime(fred["Date"])
    fred["DGS1"] = pd.to_numeric(fred["DGS1"], errors="coerce")
    fred = fred.dropna(subset=["DGS1"])

    return {
        "as_of_date": str(close.index[-1].date()),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "online",
        "jpm_close": float(close.iloc[-1]),
        "vix_close": float(vix["CLOSE"].iloc[-1]),
        "risk_free_rate": float(fred["DGS1"].iloc[-1]) / 100.0,
        "historical_volatility_20d": volatility,
        "source_dates": {
            "JPM": str(close.index[-1].date()),
            "VIX": str(vix["DATE"].iloc[-1].date()),
            "DGS1": str(fred["Date"].iloc[-1].date()),
        },
        "sources": {
            "JPM": yahoo_url,
            "VIX": cboe_url,
            "risk_free_rate": fred_url,
        },
    }


def update_snapshot(project_root: Path, output_dir: Path, online=True):
    status = "online_success"
    error = None
    if online:
        try:
            snapshot = online_snapshot()
        except Exception as exc:
            snapshot = offline_snapshot(project_root)
            status = "online_failed_offline_fallback"
            error = f"{type(exc).__name__}: {exc}"
    else:
        snapshot = offline_snapshot(project_root)
        status = "offline_requested"
    snapshot["update_status"] = status
    snapshot["online_error"] = error
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_market_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    pd.DataFrame([snapshot | {"sources": json.dumps(snapshot["sources"], sort_keys=True)}]).to_csv(
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
