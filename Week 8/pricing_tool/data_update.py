"""Key-free live data updater with an explicit cached fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
CACHE = BASE / "assets" / "data" / "latest_market_snapshot.json"


def _download(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=25) as response:
        return response.read()


def online_snapshot():
    yahoo_url = "https://query1.finance.yahoo.com/v8/finance/chart/JPM?range=3mo&interval=1d"
    payload = json.loads(_download(yahoo_url))
    chart = payload["chart"]["result"][0]
    dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True).tz_convert(None)
    adjusted = chart.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
    values = adjusted or chart["indicators"]["quote"][0]["close"]
    close = pd.Series(values, index=dates).dropna()
    if len(close) < 21:
        raise RuntimeError("JPM endpoint returned fewer than 21 observations.")
    volatility = float(np.log(close / close.shift(1)).tail(20).std(ddof=1) * np.sqrt(252))

    cboe_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    vix = pd.read_csv(BytesIO(_download(cboe_url)))
    vix.columns = [column.strip().upper() for column in vix.columns]
    vix["DATE"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y")
    vix["CLOSE"] = pd.to_numeric(vix["CLOSE"], errors="coerce")
    vix = vix.dropna(subset=["CLOSE"])

    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1"
    fred = pd.read_csv(BytesIO(_download(fred_url)))
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
        "sources": {"JPM": yahoo_url, "VIX": cboe_url, "DGS1": fred_url},
        "update_status": "online_success",
        "online_error": None,
    }


def load_cached_snapshot():
    snapshot = json.loads(CACHE.read_text(encoding="utf-8"))
    snapshot["update_status"] = "cached_fallback"
    return snapshot


def update_snapshot():
    try:
        snapshot = online_snapshot()
        CACHE.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return snapshot
    except Exception as exc:
        snapshot = load_cached_snapshot()
        snapshot["online_error"] = f"{type(exc).__name__}: {exc}"
        return snapshot
