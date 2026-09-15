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

On the 252-row historical chronological proxy test set, the selected Random Forest volatility-forecast-plus-BSM route achieved RMSE **4.3203**, compared with **4.8706** for the Week 4 BSM baseline, an **11.30%** reduction. After an invalid live extrapolation, bounded direct time-value regression replaced the unbounded Linear Regression. Its retrospective test RMSE is **4.5404** (legacy: **49.9891**) and MAE **2.8501**. Its validation RMSE is higher than the legacy model, so this is not a uniform accuracy improvement. The historical test is reused for this repair, not a new untouched holdout. These are theoretical proxy-target results, not accuracy claims against actual chooser transactions.

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

The packaged app contains the trained models, cached data, configuration, and supporting modules required for local execution. Its submitted fallback snapshot uses the final Week 2 observation so the artefacts remain reproducible without a network connection; the app can refresh public market inputs when online. The project brief requires a deployable tool in a GitHub repository with a README, not a publicly hosted application. The remaining demo can be recorded from the locally running app; follow the checklists in `Week 8`.

## Automated validation

GitHub Actions provides research validation and a separate current-data refresh:

- `Weekly preprocessing` runs every Monday at 10:17 Asia/Shanghai time and can also be started manually. It rebuilds the Week 2 processed dataset from the archived Week 1 raw inputs, validates the dates, row count, columns, and finite numeric values, then publishes the outputs as a 30-day workflow artifact.
- `Project validation` runs on pushes and pull requests to `main`. It checks the executed notebooks and required deliverables, then loads the packaged models and performs a Streamlit smoke test in the pinned application environment.
- `Market data refresh` is configured for Tuesday–Saturday at 19:17 Asia/Shanghai and manual runs. It rebuilds all current model features from completed daily source histories and uploads a dated snapshot artifact. The repository secret supplies the FRED API key. Its first manual run passed on 2026-09-11; the schedule is configured on the default branch. It does not retrain models or commit current observations into the historical sample.

The Monday workflow reproduces the fixed 2018–2024 research sample. The Week 7/8 applications independently refresh on opening, every 15 minutes while active, and on manual request. They rebuild all 17 volatility and 19 direct-price features at a common completed date. JPM, Cboe VIX, FRED DGS10 and DGS1 requests have bounded retries and independent caches. DGS10 remains the trained ML rate feature; DGS1 is the live pricing rate. No retraining or new dividend features are introduced.

## Reports

The publication set contains 15 PDFs: weekly reports for Weeks 1–8, data specification, model validation, performance benchmark documentation, model performance report, comprehensive sensitivity analysis, final project report, and project closure report. Revised Word files are maintained locally and exported to PDF before publication. Learning and Understanding documents are excluded.
