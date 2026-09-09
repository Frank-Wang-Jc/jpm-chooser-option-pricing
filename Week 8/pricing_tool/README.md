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
No API key is required. The refresh button uses public JPM, Cboe VIX and FRED DGS1
endpoints and falls back to the bundled, explicitly labelled cache if a source fails.

## Important interpretation

- The ML model is restricted to the paper-matched contract K=$150, q=2.33%, T1=0.5,
  T2=1.0. BSM remains available for other valid contract inputs.
- Public historical chooser-option transaction prices were not available. The ML target
  is an ex-post forward-volatility BSM proxy. The 90% band uses purge-aware out-of-fold
  training residuals and is evaluated on the later test period. It is not a market-price
  confidence interval.
- The model is an educational/research prototype, not trading or investment advice.
