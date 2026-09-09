# Week 5 Machine Learning Architecture Design

## Continuity with Weeks 1-4

The ML system does not replace the existing project. It consumes the Week 2 processed JPM, VIX, and Treasury features, uses the vectorized Week 4 refactor of the Week 3 chooser formula, and preserves the paper contract settings of K = $150, T1 = 0.5 year, and T2 = 1 year.

## Approach 1: Forecast volatility, then price with BSM

1. Use information available on date t to predict JPM realized volatility over the next 20 trading days.
2. Insert the predicted volatility into the validated chooser BSM model.
3. Compare the resulting chooser price with the ex-post forward-volatility proxy price.

This approach is economically interpretable because ML changes only the volatility input. It directly addresses the constant-volatility limitation while retaining the Week 3 pricing logic.

## Approach 2: Direct supervised proxy pricing

1. Use the same observable market features to predict the chooser proxy price directly.
2. Exclude the current BSM price so this route tests a genuinely end-to-end supervised alternative.
3. Compare linear regression, tree ensembles, and a neural network, while treating the target as a model-derived research proxy rather than a public chooser transaction price.

This approach is more flexible but less structurally constrained. Because public historical chooser transactions are unavailable, it is a proof-of-concept supervised pricing route rather than a claim of live OTC market accuracy.

## Time-series split

Rows remain in chronological order. The first 70% define the training block, the next 15% define the validation block, and the final 15% define the test block. The last 20 rows of the training and validation blocks are purged because their forward targets extend into the following block. No shuffling is used. All preprocessing is fitted only on training observations inside each model pipeline.

## Model families

- Linear Regression: transparent benchmark.
- Random Forest: nonlinear interactions with robust bagging.
- Histogram Gradient Boosting: efficient boosted-tree alternative.
- Neural Network: flexible nonlinear benchmark.

Week 5 establishes the data contract and initial models. Week 6 performs time-series cross-validation, hyperparameter tuning, test evaluation, model persistence, and feature-importance analysis.
