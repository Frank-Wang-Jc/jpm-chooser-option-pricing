"""Vectorized Black-Scholes-Merton and simple chooser pricing functions.

This module is a production-oriented refactor of the formulas validated in
Week 3.  It accepts scalars or NumPy arrays, validates financial inputs, and
keeps every parameter explicit so later ML models can reuse the same pricing
engine without changing the Week 3 economics.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _arrays(*values):
    return np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in values])


def validate_model_inputs(S, K, sigma, T):
    S, K, sigma, T = _arrays(S, K, sigma, T)
    if np.any(~np.isfinite(S)) or np.any(S <= 0):
        raise ValueError("Underlying price S must be finite and positive.")
    if np.any(~np.isfinite(K)) or np.any(K <= 0):
        raise ValueError("Strike price K must be finite and positive.")
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Volatility sigma must be finite and positive.")
    if np.any(~np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("Time to maturity T must be finite and positive.")


def calculate_d1_d2(S, K, r, q, sigma, T):
    validate_model_inputs(S, K, sigma, T)
    S, K, r, q, sigma, T = _arrays(S, K, r, q, sigma, T)
    root_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * root_t)
    return d1, d1 - sigma * root_t


def bsm_call_price(S, K, r, q, sigma, T):
    d1, d2 = calculate_d1_d2(S, K, r, q, sigma, T)
    S, K, r, q, T = _arrays(S, K, r, q, T)
    return S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


def bsm_put_price(S, K, r, q, sigma, T):
    d1, d2 = calculate_d1_d2(S, K, r, q, sigma, T)
    S, K, r, q, T = _arrays(S, K, r, q, T)
    return K * np.exp(-r * T) * ndtr(-d2) - S * np.exp(-q * T) * ndtr(-d1)


def chooser_choice_boundary(K, r, q, T1, T2):
    K, r, q, T1, T2 = _arrays(K, r, q, T1, T2)
    if np.any((T1 <= 0) | (T2 <= T1)):
        raise ValueError("Choice time must satisfy 0 < T1 < T2.")
    return K * np.exp(-(r - q) * (T2 - T1))


def simple_chooser_price(S, K, r, q, sigma, T1, T2):
    validate_model_inputs(S, K, sigma, T2)
    critical_strike = chooser_choice_boundary(K, r, q, T1, T2)
    call_component = bsm_call_price(S, K, r, q, sigma, T2)
    put_component = np.exp(-np.asarray(q, dtype=float) * (np.asarray(T2) - np.asarray(T1))) * bsm_put_price(
        S, critical_strike, r, q, sigma, T1
    )
    return call_component + put_component


def chooser_greeks_fd(S, K, r, q, sigma, T1, T2, spot_step=0.01, vol_step=1e-4, rate_step=1e-4):
    """Central finite-difference delta, gamma, vega, and rho for the chooser."""
    S = np.asarray(S, dtype=float)
    h_s = np.maximum(np.abs(S) * spot_step, 1e-4)
    base = simple_chooser_price(S, K, r, q, sigma, T1, T2)
    up_s = simple_chooser_price(S + h_s, K, r, q, sigma, T1, T2)
    down_s = simple_chooser_price(S - h_s, K, r, q, sigma, T1, T2)
    delta = (up_s - down_s) / (2 * h_s)
    gamma = (up_s - 2 * base + down_s) / (h_s**2)
    vega = (
        simple_chooser_price(S, K, r, q, np.asarray(sigma) + vol_step, T1, T2)
        - simple_chooser_price(S, K, r, q, np.asarray(sigma) - vol_step, T1, T2)
    ) / (2 * vol_step)
    rho = (
        simple_chooser_price(S, K, np.asarray(r) + rate_step, q, sigma, T1, T2)
        - simple_chooser_price(S, K, np.asarray(r) - rate_step, q, sigma, T1, T2)
    ) / (2 * rate_step)
    return {"price": base, "delta": delta, "gamma": gamma, "vega": vega, "rho": rho}
