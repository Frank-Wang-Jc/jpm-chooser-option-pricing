# Week 6 Model Card

## Purpose

The saved estimators compare two ML extensions of the Week 3 chooser BSM model. Approach 1 forecasts 20-day forward realized volatility and inserts the prediction into the unchanged chooser formula. Approach 2 predicts the chooser proxy price directly from observable market features and excludes the current BSM price.

## Training protocol

- Source data: Week 5 dataset built from Week 2 JPM, VIX, and Treasury features.
- Boundary control: 20 rows are purged before validation and another 20 before test because each target uses the next 20 trading days.
- Hyperparameter search: four-fold expanding-window TimeSeriesSplit with `gap=20` inside the training block.
- Selection: lowest validation RMSE within each approach after CV tuning.
- Final evaluation: the chronological test period is evaluated once after selection.

## Target limitation

The price target is generated with ex-post 20-day realized volatility and the validated chooser BSM formula. It is a controlled research proxy, not an observed OTC chooser transaction price. Model comparisons quantify performance on this stated proxy and must not be presented as verified live-market chooser accuracy.

## Saved models

- best_volatility_model.pkl: required Week 6 pickle deliverable for the selected Approach 1 volatility forecaster.
- best_direct_pricing_model.pkl: required Week 6 pickle deliverable for the selected Approach 2 proxy-price model.
- best_volatility_model.joblib: runtime copy used by the Week 8 pricing application.
- best_direct_pricing_model.joblib: runtime copy used by the Week 8 pricing application.
- model_bundle_metadata.json: features, contract assumptions, selected model names, artifact manifest, and round-trip validation results.
