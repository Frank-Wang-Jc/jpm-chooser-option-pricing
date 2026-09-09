"""Purged time-series feature preparation for Week 5 and later weeks."""

from __future__ import annotations

import numpy as np
import pandas as pd


VOLATILITY_FEATURES = [
    "Daily_Return",
    "Abs_Return_1D",
    "Rolling_Volatility_5D",
    "Rolling_Volatility_10D",
    "Rolling_Volatility_20D",
    "Rolling_Volatility_60D",
    "VIX_Close",
    "VIX_Return",
    "VIX_JPM_Correlation_20D",
    "Treasury_Rate_Decimal",
    "Interest_Rate_Momentum",
    "Price_Momentum_20D",
    "MA20_Gap",
    "MA50_Gap",
    "Intraday_Range",
    "Overnight_Gap",
    "Volume_ZScore_20D",
]

DIRECT_PRICING_FEATURES = VOLATILITY_FEATURES + [
    "Close",
    "Log_Moneyness",
]

# Backward-compatible name used by the Week 6-8 notebooks.
PRICING_FEATURES = DIRECT_PRICING_FEATURES

BASE_FEATURES = VOLATILITY_FEATURES


def build_ml_dataset(market, chooser_price_function, strike=150.0, dividend_yield=0.0233,
                     choice_time=0.5, maturity=1.0, horizon=20):
    data = market.copy().sort_values("Date").reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Abs_Return_1D"] = data["Daily_Return"].abs()
    for window in (5, 10, 60):
        data[f"Rolling_Volatility_{window}D"] = data["Log_Return"].rolling(window).std(ddof=1) * np.sqrt(252)
    data["Treasury_Rate_Decimal"] = data["Treasury_Rate"] / 100.0
    data["Log_Moneyness"] = np.log(data["Close"] / strike)
    data["MA20_Gap"] = data["Close"] / data["Moving_Average_20D"] - 1.0
    data["MA50_Gap"] = data["Close"] / data["Moving_Average_50D"] - 1.0
    data["Intraday_Range"] = (data["High"] - data["Low"]) / data["Close"]
    data["Overnight_Gap"] = data["Open"] / data["Close"].shift(1) - 1.0
    rolling_volume = data["Volume"].rolling(20)
    data["Volume_ZScore_20D"] = (data["Volume"] - rolling_volume.mean()) / rolling_volume.std(ddof=1)

    # At row t, this contains returns observed on t+1 ... t+horizon.
    data["Target_Forward_Volatility_20D"] = (
        data["Log_Return"].rolling(horizon).std(ddof=1).shift(-horizon) * np.sqrt(252)
    )
    data["Target_End_Date"] = data["Date"].shift(-horizon)
    target_mask = data["Target_Forward_Volatility_20D"].notna()
    data.loc[target_mask, "Target_Chooser_Proxy_Price"] = chooser_price_function(
        data.loc[target_mask, "Close"],
        strike,
        data.loc[target_mask, "Treasury_Rate_Decimal"],
        dividend_yield,
        data.loc[target_mask, "Target_Forward_Volatility_20D"].clip(lower=1e-6),
        choice_time,
        maturity,
    )
    current_mask = data[["Close", "Treasury_Rate_Decimal", "Rolling_Volatility_20D"]].notna().all(axis=1)
    data.loc[current_mask, "Current_Chooser_BSM_Price"] = chooser_price_function(
        data.loc[current_mask, "Close"],
        strike,
        data.loc[current_mask, "Treasury_Rate_Decimal"],
        dividend_yield,
        data.loc[current_mask, "Rolling_Volatility_20D"].clip(lower=1e-6),
        choice_time,
        maturity,
    )
    data["Target_Chooser_Price_Adjustment"] = data["Target_Chooser_Proxy_Price"] - data["Current_Chooser_BSM_Price"]
    required = PRICING_FEATURES + ["Target_End_Date", "Current_Chooser_BSM_Price", "Target_Forward_Volatility_20D", "Target_Chooser_Proxy_Price", "Target_Chooser_Price_Adjustment"]
    return data.dropna(subset=required).reset_index(drop=True)


def chronological_split(data, train_fraction=0.70, validation_fraction=0.15, purge_horizon=20):
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave a non-empty test set.")
    if purge_horizon < 1:
        raise ValueError("purge_horizon must be at least one trading day.")
    n = len(data)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    if train_end <= purge_horizon or validation_end - train_end <= purge_horizon:
        raise ValueError("Each development block must be longer than purge_horizon.")
    block_labels = np.full(n, "test", dtype=object)
    block_labels[:train_end] = "train"
    block_labels[train_end:validation_end] = "validation"
    labels = block_labels.copy()
    labels[train_end - purge_horizon:train_end] = "purged_train_validation"
    labels[validation_end - purge_horizon:validation_end] = "purged_validation_test"
    result = data.copy()
    result["Chronological_Block"] = block_labels
    result["Split"] = labels
    train = result[result["Split"] == "train"]
    validation = result[result["Split"] == "validation"]
    test = result[result["Split"] == "test"]
    if train["Target_End_Date"].max() >= validation["Date"].min():
        raise AssertionError("Training labels overlap the validation feature period.")
    if validation["Target_End_Date"].max() >= test["Date"].min():
        raise AssertionError("Validation labels overlap the test feature period.")
    return result
