# Week 8 — JPM Chooser Option Pricing Tool

This self-contained Streamlit prototype exposes the project’s two pricing routes:

1. the Week 3/4 Black–Scholes–Merton simple chooser model; and
2. the Week 6 selected ML model, a Random Forest volatility forecast passed through BSM.

It also displays the direct supervised price benchmark, finite-difference Greeks with explicit units,
stress scenarios, held-out performance and an empirical error band.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Public deployment

The folder is ready for Streamlit Community Cloud or an equivalent Python host. Push
`pricing_tool` to a GitHub repository, choose `app.py` as the entrypoint and deploy.
Refresh runs on opening, every 15 minutes during an active session, and on manual
request. JPM OHLCV, Cboe VIX, FRED DGS10 and DGS1 histories rebuild every original
model feature at a common completed date. DGS10 preserves the trained ML feature
meaning; DGS1 supplies the one-year pricing rate. The model and its 17/19 feature
lists remain unchanged.

FRED uses `FRED_API_KEY` from the environment or a local `.env` when available,
otherwise the public CSV endpoint. Requests use a 12-second timeout and two attempts.
Sources refresh independently and fall back to their own caches. The app retains
the last complete snapshot if the required histories cannot be assembled.
`JPM_OFFLINE=1` uses the bundled 2024-12-30 context for reproducible demonstrations.
Runtime caches are separate from the committed research sample and are gitignored.

## Important interpretation

- The ML model is restricted to the paper-matched contract K=$150, q=2.33%, T1=0.5,
  T2=1.0. BSM remains available for other valid contract inputs.
- Public historical chooser-option transaction prices were not available. The ML target
  is an ex-post forward-volatility BSM proxy. The 90% band uses purge-aware out-of-fold
  training residuals and is evaluated on the later test period. It is not a market-price
  confidence interval.
- The model is an educational/research prototype, not trading or investment advice.
