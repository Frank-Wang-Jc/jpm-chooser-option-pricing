from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_update import load_cached_snapshot, update_snapshot
from pricing_engine import METADATA, price_contract


BASE = Path(__file__).resolve().parent
DATA = BASE / "assets" / "data"
st.set_page_config(page_title="JPM Chooser Option Lab", page_icon="◈", layout="wide")
st.markdown("""
<style>
div[data-testid="stMetric"] {background:#101827;border:1px solid #24344d;padding:14px;border-radius:12px}
.small-note {color:#8493a8;font-size:.86rem}
</style>
""", unsafe_allow_html=True)

st.title("JPM Chooser Option Lab")
st.caption("Black-Scholes-Merton baseline and supervised machine-learning comparison")

if "snapshot" not in st.session_state:
    st.session_state.snapshot = load_cached_snapshot()
snapshot = st.session_state.snapshot
with st.sidebar:
    st.header("Market snapshot")
    if st.button("Refresh public data", width="stretch"):
        with st.spinner("Updating JPM, VIX and DGS1..."):
            st.session_state.snapshot = update_snapshot()
        snapshot = st.session_state.snapshot
    st.write(f"Mode: **{snapshot.get('update_status', snapshot.get('mode'))}**")
    st.write(f"As of: **{snapshot.get('as_of_date')}**")
    if snapshot.get("online_error"):
        st.warning("Online update failed; cached values are shown.")
    st.divider()
    st.header("Contract inputs")
    spot = st.number_input("JPM spot ($)", min_value=1.0, value=float(snapshot["jpm_close"]), step=1.0)
    strike = st.number_input("Strike ($)", min_value=1.0, value=150.0, step=5.0)
    rate_pct = st.number_input("Risk-free rate (%)", min_value=-5.0, max_value=30.0, value=float(snapshot["risk_free_rate"] * 100), step=0.10)
    dividend_pct = st.number_input("Dividend yield (%)", min_value=0.0, max_value=20.0, value=2.33, step=0.05)
    vol_pct = st.number_input("Annualized volatility (%)", min_value=0.1, max_value=300.0, value=float(snapshot["historical_volatility_20d"] * 100), step=1.0)
    vix = st.number_input("VIX", min_value=1.0, max_value=150.0, value=float(snapshot["vix_close"]), step=0.5)
    choice_time = st.number_input("Choice time T1 (years)", min_value=0.01, value=0.50, step=0.05)
    maturity = st.number_input("Maturity T2 (years)", min_value=0.02, value=1.00, step=0.05)

try:
    result = price_contract(
        spot, strike, rate_pct / 100, dividend_pct / 100,
        vol_pct / 100, choice_time, maturity, vix
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

tab_price, tab_sensitivity, tab_performance, tab_notes = st.tabs([
    "Dual pricing", "Sensitivity", "Model performance", "Methods & limits"
])
with tab_price:
    col1, col2, col3 = st.columns(3)
    col1.metric("BSM chooser", f"${result['bsm_price']:,.2f}")
    if result["ml_available"]:
        col2.metric("Best ML estimate", f"${result['best_ml_price']:,.2f}", f"{result['best_ml_price']-result['bsm_price']:+.2f} vs BSM")
        lower, upper = result["error_margin_90"]
        col3.metric("Empirical 90% band", f"${lower:,.2f} – ${upper:,.2f}")
        st.info(result["warning"])
        table = pd.DataFrame({
            "Route": ["Week 4 BSM", "Approach 1: ML volatility + BSM (selected)", "Approach 2: direct proxy price"],
            "Price": [result["bsm_price"], result["approach1_ml_vol_bsm_price"], result["approach2_direct_price"]],
        })
        st.plotly_chart(px.bar(table, x="Route", y="Price", color="Route", text_auto=".2f"), width="stretch")
        st.caption(result["error_margin_note"])
        if result["out_of_training_range"]:
            st.warning("ML extrapolation warning — outside training range: " + ", ".join(result["out_of_training_range"].keys()))
    else:
        col2.metric("ML estimate", "Unavailable")
        col3.metric("Reason", "Out-of-training contract")
        st.warning(result["ml_warning"])
    st.subheader("Finite-difference chooser Greeks")
    greek_cols = st.columns(4)
    greek_labels = {
        "delta": "Delta",
        "gamma": "Gamma",
        "vega_per_1pct": "Vega per +1 vol point",
        "rho_per_1pct": "Rho per +1 rate point",
    }
    for col, key in zip(greek_cols, greek_labels):
        col.metric(greek_labels[key], f"{result['greeks'][key]:,.4f}")

with tab_sensitivity:
    vol_grid = np.linspace(max(0.01, vol_pct / 200), min(3.0, vol_pct / 100 * 1.75), 60)
    rate_grid = np.linspace(rate_pct / 100 - 0.02, rate_pct / 100 + 0.02, 60)
    vol_prices = [price_contract(spot, strike, rate_pct/100, dividend_pct/100, x, choice_time, maturity, vix)["bsm_price"] for x in vol_grid]
    rate_prices = [price_contract(spot, strike, x, dividend_pct/100, vol_pct/100, choice_time, maturity, vix)["bsm_price"] for x in rate_grid]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(x=vol_grid * 100, y=vol_prices, labels={"x":"Volatility (%)","y":"BSM chooser price ($)"}), width="stretch")
    c2.plotly_chart(px.line(x=rate_grid * 100, y=rate_prices, labels={"x":"Risk-free rate (%)","y":"BSM chooser price ($)"}), width="stretch")
    shock_rows = []
    for label, vol_mult, rate_add in [("Base",1,0),("50% volatility spike",1.5,0),("2% rate hike",1,0.02),("Combined",1.5,0.02)]:
        priced = price_contract(spot, strike, rate_pct/100+rate_add, dividend_pct/100, vol_pct/100*vol_mult, choice_time, maturity, vix*vol_mult)
        shock_rows.append({"Scenario":label,"BSM":priced["bsm_price"],"Best ML":priced.get("best_ml_price", np.nan)})
    st.dataframe(pd.DataFrame(shock_rows), hide_index=True, width="stretch")

with tab_performance:
    metrics = pd.read_csv(DATA / "model_comparison.csv")
    st.dataframe(metrics.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R2":"{:.3f}"}), hide_index=True, width="stretch")
    st.plotly_chart(px.bar(metrics, x="Approach", y="RMSE", color="Approach", text_auto=".3f"), width="stretch")
    history = pd.read_csv(DATA / "test_predictions.csv", parse_dates=["Date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["Date"], y=history["Target_Chooser_Proxy_Price"], name="Forward-volatility proxy target"))
    fig.add_trace(go.Scatter(x=history["Date"], y=history["Current_Chooser_BSM_Price"], name="Week 4 BSM"))
    fig.add_trace(go.Scatter(x=history["Date"], y=history["Approach1_MLVol_BSM_Price"], name="Selected ML"))
    fig.add_trace(go.Scatter(x=history["Date"], y=history["Approach2_Direct_Price"], name="Direct-price benchmark"))
    fig.update_layout(yaxis_title="Chooser price ($)", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

with tab_notes:
    st.markdown(f"""
    **Model chain.** Week 2 market data → Week 3 chooser formula → Week 4 vectorized BSM baseline →
    Week 5 leakage-safe features → Week 6 trained models → Week 7 stress tests/data updater → this app.

    **Selected ML model.** {result.get('best_ml_model', 'Restricted outside the trained contract')}.

    **Feature context.** Remaining engineered inputs are anchored to {result['feature_context_date']}.
    User-supplied spot, rate, volatility and VIX overwrite their matching inputs.

    **Validation boundary.** The target is an ex-post forward-volatility chooser BSM proxy, because public
    historical OTC chooser transaction prices were not available. The displayed band uses purge-aware
    out-of-fold training residuals and is evaluated on the later test period. It is not a market bid/ask
    range or a guaranteed confidence interval.
    """)
