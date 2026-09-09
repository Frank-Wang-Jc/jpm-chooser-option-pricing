# Week 7 Pricing Tool Prototype

## Run the prototype

From the project root, install the project requirements and run:

```powershell
streamlit run "Week 7/app.py"
```

The sidebar controls the JPM spot, strike, rate, dividend yield, volatility, choice date, and maturity. The app shows the Week 3/4 chooser price, finite-difference Greeks, and stress-sensitivity curves.

The **Refresh online market snapshot** button attempts to download recent JPM and VIX values from Yahoo Finance and the one-year DGS1 Treasury rate from FRED. If the update fails, the tool clearly falls back to the latest complete Week 2 row and records the failure status. The prototype does not silently label cached data as live.

Week 8 adds dual BSM/ML prices, model error margins, and performance dashboards.
