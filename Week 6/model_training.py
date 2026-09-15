"""Training, tuning, selection, and evaluation helpers for Week 6."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from bounded_pricing import BoundedChooserRegressor


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "N": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
        "Mean_Error": error.mean(),
    }


def _scaled(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def _tree(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def candidate_specs(random_state):
    return {
        "Linear Regression": {
            "estimator": _scaled(LinearRegression()),
            "grid": None,
        },
        "Random Forest": {
            "estimator": _tree(RandomForestRegressor(random_state=random_state, n_jobs=-1)),
            "grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [2, 5],
            },
        },
        "XGBoost": {
            "estimator": _tree(XGBRegressor(
                objective="reg:squarederror", random_state=random_state,
                n_jobs=1, tree_method="hist", reg_lambda=1.0,
            )),
            "grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [2, 4],
                "model__learning_rate": [0.03, 0.08],
                "model__subsample": [0.8],
            },
        },
        "Neural Network": {
            "estimator": _scaled(MLPRegressor(
                hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                max_iter=1200, random_state=random_state,
            )),
            "grid": None,
        },
    }


def tune_and_validate(X_train, y_train, X_validation, y_validation, random_state, target_name,
                      cv_splits=4, purge_gap=20):
    splitter = TimeSeriesSplit(n_splits=cv_splits, gap=purge_gap)
    fitted = {}
    rows = []
    specs = candidate_specs(random_state)
    if target_name == 'Direct chooser proxy price':
        # The legacy unbounded candidates remain documented in the revision
        # audit. Deployment now requires structural pointwise price bounds.
        specs = {'Bounded time-value regression': {
            'estimator': BoundedChooserRegressor(),
            'grid': {'alpha': [.1, 1., 10., 100., 1000.],
                     'distance_scale': ['raw', 'historical']},
        }}
    for name, spec in specs.items():
        start = time.perf_counter()
        if spec["grid"]:
            search = GridSearchCV(
                spec["estimator"], spec["grid"], scoring="neg_mean_squared_error",
                cv=splitter, n_jobs=1, refit=True,
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            cv_rmse = float(np.sqrt(-search.best_score_))
            best_parameters = search.best_params_
        else:
            estimator = clone(spec["estimator"])
            scores = cross_val_score(estimator, X_train, y_train, scoring="neg_mean_squared_error", cv=splitter, n_jobs=1)
            cv_rmse = float(np.sqrt(-scores.mean()))
            estimator.fit(X_train, y_train)
            best_parameters = {}
        predictions = estimator.predict(X_validation)
        elapsed = time.perf_counter() - start
        fitted[name] = estimator
        rows.append({
            "Target": target_name,
            "Model": name,
            "CV_RMSE": cv_rmse,
            "Validation_MAE": mean_absolute_error(y_validation, predictions),
            "Validation_RMSE": mean_squared_error(y_validation, predictions) ** 0.5,
            "Validation_R2": r2_score(y_validation, predictions),
            "Training_Time_Seconds": elapsed,
            "Best_Parameters": json_dumps(best_parameters),
        })
    # CV tunes hyperparameters inside the training block. The separate validation
    # block selects the candidate. The test block is not used here.
    results = pd.DataFrame(rows).sort_values(["Validation_RMSE", "CV_RMSE"]).reset_index(drop=True)
    return fitted, results


def json_dumps(value):
    import json
    return json.dumps(value, sort_keys=True)


def refit_selected(estimator, X, y):
    fitted = clone(estimator)
    fitted.fit(X, y)
    return fitted


def purged_oof_predictions(estimator, X, y, cv_splits=4, purge_gap=20):
    """Return expanding-window out-of-fold predictions for interval calibration."""
    splitter = TimeSeriesSplit(n_splits=cv_splits, gap=purge_gap)
    predictions = np.full(len(X), np.nan, dtype=float)
    for train_index, validation_index in splitter.split(X):
        fitted = clone(estimator)
        fitted.fit(X.iloc[train_index], y.iloc[train_index])
        predictions[validation_index] = fitted.predict(X.iloc[validation_index])
    return predictions
