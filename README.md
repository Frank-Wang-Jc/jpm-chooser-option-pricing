# Advanced Chooser Option Pricing Model

This repository contains a continuous eight-week quantitative research project for a JPM-underlying simple chooser option. The implementation preserves one auditable pipeline from raw market data to preprocessing, the analytical Black–Scholes–Merton (BSM) model, baseline evaluation, machine-learning extensions, stress analysis, and a final interactive pricing tool.

## End-to-end workflow

1. **Week 1 — Data acquisition:** JPM equity data, Cboe VIX, and U.S. Treasury yields.
2. **Week 2 — Preprocessing and features:** aligned trading dates, cleaning, outlier review, returns, rolling volatility, momentum, moving averages, rates, and correlations.
3. **Week 3 — Original BSM replication:** vanilla BSM functions and the simple chooser closed-form model using the paper-matched contract configuration (`K = $150`, `T1 = 0.5`, `T2 = 1.0`).
4. **Week 4 — Baseline evaluation:** chronological chooser-price proxy evaluation, volatility-regime analysis, parameter sensitivity, runtime/memory benchmarks, and a separate listed-JPM vanilla-option market sanity check.
5. **Week 5 — ML design:** two purge-aware routes: forecast forward volatility before BSM, and predict the chooser proxy price directly from observable market features.
6. **Week 6 — Training and comparison:** 20-day-gap expanding-window cross-validation, separate validation-based model selection, one-time chronological test evaluation, and interpretation of the exact selected models.
7. **Week 7 — Advanced analysis:** required volatility/rate stress scenarios and public-data updating with explicit fallback status.
8. **Week 8 — Integration:** a self-contained Streamlit application showing BSM and ML prices together, unit-labelled Greeks, sensitivities, performance evidence, and a purge-aware out-of-fold proxy-error band.

## Key result

On the 252-row held-out chronological proxy test set, the selected Random Forest volatility-forecast-plus-BSM route achieved RMSE **4.3203**, compared with **4.8706** for the Week 4 BSM baseline—an **11.30%** reduction. The direct-price Linear Regression benchmark produced RMSE **49.9891** and is retained as a documented extrapolation failure. These are theoretical proxy-target results, not claims of accuracy against public historical chooser transactions.

## Important validation boundary

Public historical OTC chooser-option transactions were not available. Therefore:

- JPM stock close is never used as an option-price actual.
- Chooser MAE/RMSE use a clearly labelled forward-volatility theoretical proxy.
- A delayed Cboe JPM vanilla-option snapshot provides only a separate vanilla-BSM market sanity check.
- The empirical 90% band is calibrated from purge-aware out-of-fold training residuals and evaluated once on the later test period; it is not a market bid/ask spread or chooser market-price confidence interval.

## Run the final application

From `Week 8/pricing_tool`:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The packaged app contains the trained models, cached data, configuration, and supporting modules required for local execution. Its submitted fallback snapshot uses the final Week 2 observation so the artefacts remain reproducible without a network connection; the app can refresh public market inputs when online. Public deployment and the final demonstration recording remain user-account actions; follow the checklists in `Week 8`.

## Automated validation

GitHub Actions provides two reproducibility checks:

- `Weekly preprocessing` runs every Monday at 10:17 Asia/Shanghai time and can also be started manually. It rebuilds the Week 2 processed dataset from the archived Week 1 raw inputs, validates the dates, row count, columns, and finite numeric values, then publishes the outputs as a 30-day workflow artifact.
- `Project validation` runs on pushes and pull requests to `main`. It checks the executed notebooks and required deliverables, then loads the packaged models and performs a Streamlit smoke test in the pinned application environment.

The scheduled workflow reproduces the fixed 2018–2024 research sample; it does not silently extend the historical study period. The Week 8 application has a separate public-data refresh path for current JPM, Cboe VIX, and FRED DGS1 inputs.

## Reports and presentation

Each `WeekN Report` folder contains the weekly, learning, and understanding reports. Week 4 also includes validation and performance documentation; Week 6 includes the model-performance report; Week 8 includes the final project report and presentation.
