# Week 7 Pricing Tool Prototype

## Run the prototype

From the project root, install the project requirements and run:

```powershell
streamlit run "Week 7/app.py"
```

The sidebar controls the JPM spot, strike, rate, dividend yield, volatility, choice date, and maturity. The app shows the Week 3/4 chooser price, finite-difference Greeks, and stress-sensitivity curves.

The app refreshes on opening, every 15 minutes while the session is active, and on manual request. It downloads JPM OHLCV from Yahoo Finance, VIX history from Cboe, and DGS10/DGS1 Treasury series from FRED. All 17 volatility features and 19 direct-pricing features are rebuilt at a common completed observation date using the original Week 2/5 formulas. No model retraining occurs.

Each request has a 12-second timeout and at most two attempts. Independent source caches retain usable histories if one provider fails. Incomplete updates retain the previous complete snapshot and display the fallback status. FRED uses `FRED_API_KEY` from the environment or local `.env` when available, otherwise its public CSV endpoint. Keys are not written into logs or snapshots.

`Reports/Week7 Report/Comprehensive Sensitivity Analysis Report.docx` documents the structural shocks, ML counterfactuals, VIX interpretation, and update controls. The publication copy is exported to PDF separately.

Week 8 adds dual BSM/ML prices, model error margins, and performance dashboards.
