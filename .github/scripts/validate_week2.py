from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Week2" / "processed_data"

required_files = [
    "market_data_processed.csv",
    "jpm_outlier_summary.csv",
    "vix_outlier_summary.csv",
    "treasury_outlier_summary.csv",
    "feature_outlier_summary.csv",
]
for name in required_files:
    path = OUTPUT / name
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty output: {path}")

data = pd.read_csv(OUTPUT / "market_data_processed.csv", parse_dates=["Date"])
required_columns = {
    "Date",
    "Close",
    "VIX_Close",
    "Treasury_Rate",
    "Daily_Return",
    "Rolling_Volatility_20D",
    "Moving_Average_20D",
    "Moving_Average_50D",
    "Price_Momentum_20D",
    "Interest_Rate_Momentum",
    "VIX_JPM_Correlation_20D",
}
missing = sorted(required_columns.difference(data.columns))
if missing:
    raise AssertionError(f"Missing processed columns: {missing}")
if len(data) != 1760:
    raise AssertionError(f"Expected 1,760 aligned rows, found {len(data)}")
if data["Date"].min().date().isoformat() != "2018-01-02":
    raise AssertionError("Unexpected first processed date")
if data["Date"].max().date().isoformat() != "2024-12-30":
    raise AssertionError("Unexpected last processed date")
numeric = data.select_dtypes(include=[np.number])
if np.isinf(numeric.to_numpy()).any():
    raise AssertionError("Processed data contains infinite values")

print({
    "status": "PASS",
    "rows": len(data),
    "columns": len(data.columns),
    "first_date": data["Date"].min().date().isoformat(),
    "last_date": data["Date"].max().date().isoformat(),
})
