"""Initial machine-learning frameworks for both Week 5 approaches."""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _scaled(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _tree(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def build_initial_models(random_state=20260806):
    return {
        "Linear Regression": _scaled(LinearRegression()),
        "Random Forest": _tree(RandomForestRegressor(
            n_estimators=250, max_depth=8, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        )),
        "Histogram GBDT": _tree(HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=0.1, random_state=random_state,
        )),
        "Neural Network": _scaled(MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
            early_stopping=True, max_iter=1000, random_state=random_state,
        )),
    }
