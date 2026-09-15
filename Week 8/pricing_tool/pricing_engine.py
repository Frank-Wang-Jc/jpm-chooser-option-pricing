"""Standalone dual-model chooser pricing engine for the Week 8 app."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bsm_chooser import chooser_greeks_fd, simple_chooser_price


BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
with open(ASSETS / "models" / "model_bundle_metadata.json", encoding="utf-8") as stream:
    METADATA = json.load(stream)
with open(ASSETS / "data" / "latest_feature_context.json", encoding="utf-8") as stream:
    FEATURE_CONTEXT = json.load(stream)
with open(ASSETS / "data" / "error_interval.json", encoding="utf-8") as stream:
    ERROR_INTERVAL = json.load(stream)
with open(ASSETS / "data" / "training_domain.json", encoding="utf-8") as stream:
    TRAINING_DOMAIN = json.load(stream)

VOLATILITY_MODEL = joblib.load(ASSETS / "models" / "best_volatility_model.joblib")
PRICING_MODEL = joblib.load(ASSETS / "models" / "best_direct_pricing_model.joblib")


def _updated_feature_row(spot, strike, rate, volatility, vix, context):
    values = dict(context["values"])
    model_rate = values['Treasury_Rate_Decimal'] + rate - context.get('pricing_rate', values['Treasury_Rate_Decimal'])
    current_bsm = float(simple_chooser_price(
        spot, strike, rate, METADATA["contract"]["q"], volatility,
        METADATA["contract"]["T1"], METADATA["contract"]["T2"]
    ))
    values.update({
        "Close": float(spot),
        "Log_Moneyness": float(np.log(spot / strike)),
        "Treasury_Rate_Decimal": float(model_rate),
        "Rolling_Volatility_20D": float(volatility),
        "VIX_Close": float(vix),
        "Current_Chooser_BSM_Price": current_bsm,
    })
    return values, current_bsm


def price_contract(spot, strike=150.0, rate=0.0455, dividend_yield=0.0233,
                   volatility=0.20, choice_time=0.5, maturity=1.0, vix=17.4, feature_context=None):
    context = FEATURE_CONTEXT if feature_context is None else feature_context
    if not 0 < choice_time < maturity:
        raise ValueError("Choice time must satisfy 0 < choice_time < maturity.")
    if min(spot, strike, volatility) <= 0:
        raise ValueError("Spot, strike, and volatility must be positive.")

    # The paper-matched contract is the ML model's training contract. BSM accepts
    # user contract inputs; ML output is disabled outside that trained contract.
    contract = METADATA["contract"]
    ml_contract_match = (
        abs(strike - contract["K"]) < 1e-12
        and abs(dividend_yield - contract["q"]) < 1e-12
        and abs(choice_time - contract["T1"]) < 1e-12
        and abs(maturity - contract["T2"]) < 1e-12
    )
    bsm = float(simple_chooser_price(spot, strike, rate, dividend_yield, volatility, choice_time, maturity))
    raw_greeks = {key: float(np.asarray(value)) for key, value in chooser_greeks_fd(
        spot, strike, rate, dividend_yield, volatility, choice_time, maturity
    ).items()}
    greeks = {
        "delta": raw_greeks["delta"],
        "gamma": raw_greeks["gamma"],
        "vega_per_1pct": raw_greeks["vega"] * 0.01,
        "rho_per_1pct": raw_greeks["rho"] * 0.01,
    }
    live_values = {
        "Close": float(spot), "Treasury_Rate_Decimal": float(rate),
        "Rolling_Volatility_20D": float(volatility), "VIX_Close": float(vix),
    }
    ood_flags = {
        name: value for name, value in live_values.items()
        if value < TRAINING_DOMAIN[name]["min"] or value > TRAINING_DOMAIN[name]["max"]
    }
    result = {
        "bsm_price": bsm,
        "greeks": greeks,
        "ml_available": ml_contract_match,
        "feature_context_date": context["feature_as_of_date"],
        "out_of_training_range": ood_flags,
        "warning": (
            "All ML features share the displayed feature date. Input changes are counterfactual scenarios. "
            "DGS10 remains the trained ML rate feature; DGS1 supplies the online BSM discount rate."
        ),
    }
    if not ml_contract_match:
        result["ml_warning"] = "ML models are restricted to the trained K=150, q=2.33%, T1=0.5, T2=1.0 contract."
        return result

    values, current_bsm = _updated_feature_row(spot, strike, rate, volatility, vix, context)
    vol_frame = pd.DataFrame([[values[name] for name in METADATA["volatility_features"]]], columns=METADATA["volatility_features"])
    predicted_vol = float(np.clip(VOLATILITY_MODEL.predict(vol_frame)[0], 1e-4, 3.0))
    approach1 = float(simple_chooser_price(
        spot, strike, rate, dividend_yield, predicted_vol, choice_time, maturity
    ))
    price_frame = pd.DataFrame([[values[name] for name in METADATA["pricing_features"]]], columns=METADATA["pricing_features"])
    # Keep the DGS10 feature semantics, but use the actual pricing rate for
    # structural discounting and forward moneyness (DGS1 for online pricing).
    approach2 = float(PRICING_MODEL.predict(price_frame, pricing_rate=rate)[0])
    direct_lower, direct_upper = PRICING_MODEL.price_bounds(price_frame, pricing_rate=rate)
    if not np.isfinite(approach2) or not direct_lower[0] - 1e-8 <= approach2 <= direct_upper[0] + 1e-8:
        raise ValueError('Direct pricing model returned an invalid bounded estimate.')
    best_ml = approach1
    lower = max(0.0, best_ml + ERROR_INTERVAL["lower_residual_quantile"])
    upper = max(lower, best_ml + ERROR_INTERVAL["upper_residual_quantile"])
    result.update({
        "ml_predicted_forward_volatility": predicted_vol,
        "approach1_ml_vol_bsm_price": approach1,
        "best_ml_price": best_ml,
        "best_ml_model": f"Approach 1 - {METADATA['selected_volatility_model']} volatility forecast plus BSM",
        "approach2_direct_price": approach2,
        "direct_price_bounds": [float(direct_lower[0]), float(direct_upper[0])],
        "direct_model_note": "Direct normalized time-value regression with structural bounds; no zero clipping. Out-of-training estimates still require validation.",
        "error_margin_90": [lower, upper],
        "error_margin_note": ERROR_INTERVAL["limitation"],
    })
    return result
