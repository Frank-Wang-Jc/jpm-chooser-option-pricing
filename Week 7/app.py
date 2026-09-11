"""Week 7 Streamlit pricing-tool prototype."""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def locate_project_root():
    candidates = [Path.cwd(), *Path.cwd().parents, Path(r"G:\JPM-Chooser Option Pricing")]
    for candidate in candidates:
        if (candidate / "Week 4" / "bsm_chooser.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root.")


ROOT = locate_project_root()
sys.path.insert(0, str(ROOT / "Week 4"))
from bsm_chooser import chooser_greeks_fd, simple_chooser_price
from data_update import update_snapshot


def main():
    st.title("JPM Chooser Option Pricing Prototype")
    st.caption("Week 7 prototype. Prices are research estimates, not trade recommendations.")

    base = Path(__file__).resolve().parent / "live_data"
    snapshot_path = base / "runtime" / "latest_market_snapshot.json"
    auto_due = time.monotonic() - st.session_state.get("last_refresh", -1000) >= 900
    requested = st.sidebar.button("Refresh online market snapshot")
    if requested or (auto_due and os.environ.get("JPM_OFFLINE") != "1"):
        st.session_state.last_refresh = time.monotonic()
        snapshot = update_snapshot(ROOT, snapshot_path.parent, online=True)
        st.sidebar.info(f"Update status: {snapshot['update_status']}")
    read_path = base / "latest_market_snapshot.json" if os.environ.get("JPM_OFFLINE") == "1" and not requested else snapshot_path
    if read_path.exists():
        import json
        snapshot = json.loads(read_path.read_text(encoding="utf-8"))
    else:
        import json
        snapshot = json.loads((base / "latest_market_snapshot.json").read_text(encoding="utf-8"))
    st.sidebar.caption("Automatic refresh every 15 minutes while open. Completed daily data; source dates shown below.")

    st.sidebar.header("Contract and market inputs")
    spot = st.sidebar.number_input("JPM spot price ($)", min_value=0.01, value=float(snapshot["jpm_close"]), step=1.0)
    strike = st.sidebar.number_input("Strike price ($)", min_value=0.01, value=150.0, step=1.0)
    rate = st.sidebar.number_input("Risk-free rate", min_value=-0.10, max_value=0.30, value=float(snapshot["risk_free_rate"]), step=0.001, format="%.4f")
    dividend = st.sidebar.number_input("Dividend yield", min_value=0.0, max_value=0.30, value=0.0233, step=0.001, format="%.4f")
    volatility = st.sidebar.number_input("Volatility", min_value=0.001, max_value=3.0, value=float(snapshot["historical_volatility_20d"]), step=0.01, format="%.4f")
    choice_time = st.sidebar.number_input("Choice time T1 (years)", min_value=0.01, value=0.5, step=0.05)
    maturity = st.sidebar.number_input("Maturity T2 (years)", min_value=0.02, value=1.0, step=0.05)

    if choice_time >= maturity:
        st.error("Choice time T1 must be earlier than maturity T2.")
        return

    values = chooser_greeks_fd(spot, strike, rate, dividend, volatility, choice_time, maturity)
    columns = st.columns(5)
    for column, (label, value) in zip(columns, [("Chooser price", values["price"]), ("Delta", values["delta"]), ("Gamma", values["gamma"]), ("Vega", values["vega"]), ("Rho", values["rho"])]):
        column.metric(label, f"{float(value):.4f}")

    spot_grid = np.linspace(0.7 * spot, 1.3 * spot, 80)
    sensitivity = pd.DataFrame({
        "Spot": spot_grid,
        "Base": simple_chooser_price(spot_grid, strike, rate, dividend, volatility, choice_time, maturity),
        "Volatility +50%": simple_chooser_price(spot_grid, strike, rate, dividend, volatility * 1.5, choice_time, maturity),
        "Rate +2%": simple_chooser_price(spot_grid, strike, rate + 0.02, dividend, volatility, choice_time, maturity),
    }).set_index("Spot")
    st.subheader("Sensitivity")
    st.line_chart(sensitivity)
    st.subheader("Market snapshot")
    st.json(snapshot)
    st.info("The prototype uses the Week 3/4 BSM chooser engine. ML dual pricing is added in Week 8 after model packaging checks.")


@st.fragment(run_every=float(os.environ.get('JPM_REFRESH_SECONDS', '900')))
def refresh_timer():
    interval = float(os.environ.get('JPM_REFRESH_SECONDS', '900'))
    if os.environ.get('JPM_OFFLINE') != '1' and time.monotonic() - st.session_state.get('last_refresh', 0) >= interval:
        update_snapshot(ROOT, Path(__file__).resolve().parent / 'live_data/runtime', online=True)
        st.session_state.last_refresh = time.monotonic()
        st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="JPM Chooser Option Prototype", layout="wide")
    main()
    refresh_timer()
