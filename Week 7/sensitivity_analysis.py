"""Scenario and counterfactual helpers for Week 7."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stress_scenarios(row, chooser_price_function, strike=150.0, dividend_yield=0.0233,
                     choice_time=0.5, maturity=1.0):
    base_sigma = float(row["Rolling_Volatility_20D"])
    base_rate = float(row["Treasury_Rate_Decimal"])
    scenarios = [
        ("Base", base_sigma, base_rate),
        ("50% Volatility Spike", base_sigma * 1.50, base_rate),
        ("2% Rate Hike", base_sigma, base_rate + 0.02),
        ("Combined Shock", base_sigma * 1.50, base_rate + 0.02),
    ]
    values = []
    base_price = None
    for name, sigma, rate in scenarios:
        price = float(chooser_price_function(
            float(row["Close"]), strike, rate, dividend_yield,
            sigma, choice_time, maturity,
        ))
        if base_price is None:
            base_price = price
        values.append({
            "Scenario": name,
            "Spot": float(row["Close"]),
            "Volatility": sigma,
            "Risk_Free_Rate": rate,
            "Chooser_BSM_Price": price,
            "Price_Change": price - base_price,
            "Price_Change_Percent": (price / base_price - 1.0) if base_price else np.nan,
        })
    return pd.DataFrame(values)


def apply_ml_scenario(row, volatility_model, pricing_model, volatility_features,
                      pricing_features, chooser_price_function, scenario,
                      strike=150.0, dividend_yield=0.0233, choice_time=0.5, maturity=1.0):
    scenario_row = row.copy()
    if scenario in {"50% Volatility Spike", "Combined Shock"}:
        for feature in [name for name in volatility_features if name.startswith("Rolling_Volatility_")]:
            scenario_row[feature] *= 1.50
        scenario_row["VIX_Close"] *= 1.50
        scenario_row["VIX_Return"] = max(float(scenario_row["VIX_Return"]), 0.50)
    if scenario in {"2% Rate Hike", "Combined Shock"}:
        scenario_row["Treasury_Rate_Decimal"] += 0.02
        scenario_row["Interest_Rate_Momentum"] += 2.0

    current_bsm = float(chooser_price_function(
        scenario_row["Close"], strike, scenario_row["Treasury_Rate_Decimal"],
        dividend_yield, scenario_row["Rolling_Volatility_20D"], choice_time, maturity,
    ))
    scenario_row["Current_Chooser_BSM_Price"] = current_bsm
    predicted_volatility = float(np.clip(
        volatility_model.predict(pd.DataFrame([scenario_row[volatility_features].to_dict()]))[0],
        0.01, 2.0,
    ))
    approach1_price = float(chooser_price_function(
        scenario_row["Close"], strike, scenario_row["Treasury_Rate_Decimal"],
        dividend_yield, predicted_volatility, choice_time, maturity,
    ))
    approach2_price = max(0.0, float(pricing_model.predict(pd.DataFrame([scenario_row[pricing_features].to_dict()]))[0]))
    return {
        "Scenario": scenario,
        "Input_VIX": float(scenario_row["VIX_Close"]),
        "Input_Rate": float(scenario_row["Treasury_Rate_Decimal"]),
        "Input_Historical_Volatility": float(scenario_row["Rolling_Volatility_20D"]),
        "Predicted_Forward_Volatility": predicted_volatility,
        "BSM_Price": current_bsm,
        "Approach1_Price": approach1_price,
        "Approach2_Price": approach2_price,
    }
