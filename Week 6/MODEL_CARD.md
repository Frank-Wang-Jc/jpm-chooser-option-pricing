# Week 6 Model Card

## Purpose

The saved estimators compare two ML extensions of the Week 3 chooser BSM model. Approach 1 forecasts 20-day forward realized volatility and inserts the prediction into the unchanged chooser formula. Approach 2 predicts the chooser proxy price directly from observable market features and excludes the current BSM price.

## Training protocol

- Source data: Week 5 dataset built from Week 2 JPM, VIX, and Treasury features.
- Boundary control: 20 rows are purged before validation and another 20 before test because each target uses the next 20 trading days.
- Hyperparameter search: four-fold expanding-window TimeSeriesSplit with `gap=20` inside the training block.
- Selection: lowest validation RMSE within each approach after CV tuning.
- Revision evaluation: the historical test period is reused retrospectively after the September 2026 live extrapolation failure. It is not a new untouched holdout for this repair.

## Direct pricing repair

The old unbounded Linear Regression predicted -111.5589 dollars at the 2026-09-10 snapshot. Dollar spot and log moneyness had 0.9964 correlation in the training block and contributed -103.7549 and -42.6280 dollars to that prediction. Output clipping concealed the invalid prediction as a zero-height bar.

The replacement `BoundedChooserRegressor` learns a normalized time-value fraction through a logistic link. Discounted spot and strike determine structural lower and upper bounds. Regularization and non-positive moneyness-distance coefficients control the tails. It predicts price directly, without calling BSM or estimating a forward-volatility target. The original 19-column interface, upstream research sample, fixed contract and Approach 1 remain in place.

Purged training CV selects alpha=0.1 and historical-volatility-scaled moneyness. Validation RMSE is 7.3983 versus 5.4837 for the legacy regression: the repair is not uniformly more accurate. The retrospective 252-row test improves from RMSE 49.9891 to 4.5404, with MAE 2.8501 and R2 0.9530. Approach 1 remains the primary route by RMSE, while Approach 2 has lower test MAE. Dollar-unit permutation SHAP explains the complete bounded predictor.

Pointwise price bounds and sensible extreme-moneyness limits address the invalid-price failure. They do not prove an arbitrage-free surface or accuracy on real chooser transactions. Out-of-training warnings remain visible. Online structural discounting uses the actual pricing rate separately from the trained DGS10 feature.

## Target limitation

The price target is generated with ex-post 20-day realized volatility and the validated chooser BSM formula. It is a controlled research proxy, not an observed OTC chooser transaction price. Model comparisons quantify performance on this stated proxy and must not be presented as verified live-market chooser accuracy.

## Saved models

- best_volatility_model.pkl: required Week 6 pickle deliverable for the selected Approach 1 volatility forecaster.
- best_direct_pricing_model.pkl: required Week 6 pickle deliverable for the selected Approach 2 proxy-price model.
- best_volatility_model.joblib: runtime copy used by the Week 8 pricing application.
- best_direct_pricing_model.joblib: runtime copy used by the Week 8 pricing application.
- model_bundle_metadata.json: features, contract assumptions, selected model names, artifact manifest, and round-trip validation results.
