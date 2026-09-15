"""Direct chooser proxy regression with structural bounds and controlled tails.

Let A=S exp(-qT), B=K exp(-rT). A chooser costs at least |A-B|
(the holder can choose either vanilla option) and at most A+B (a straddle
superhedges the choice payoff). Learn the fraction of the remaining span,
not dollar price. This is a direct price model: no forecast volatility or
BSM pricing function is called. These pointwise bounds do not prove that
the entire learned surface is arbitrage-free or accurate out of sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, least_squares
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class BoundedChooserRegressor(RegressorMixin, BaseEstimator):
    """Ridge-regularized logit time value with non-increasing moneyness tails.

    The two distance coefficients are constrained non-positive; a strictly
    negative quadratic coefficient ensures the learned fraction tends to
    zero at either extreme moneyness, holding other features fixed. Scaling
    and regularization are fitted inside each CV fold. Only logit encoding
    of machine-zero target fractions uses epsilon, never output prices.
    """
    def __init__(self, alpha=10.0, strike=150.0, dividend_yield=0.0233,
                 maturity=1.0, epsilon=1e-8, tail_floor=0.01, distance_scale='historical', loss='logit'):
        self.alpha = alpha
        self.strike = strike
        self.dividend_yield = dividend_yield
        self.maturity = maturity
        self.epsilon = epsilon
        self.tail_floor = tail_floor
        self.distance_scale = distance_scale
        self.loss = loss

    def _frame(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            check_is_fitted(self, 'feature_names_in_')
            frame = pd.DataFrame(X, columns=self.feature_names_in_)
        required = {'Close', 'Treasury_Rate_Decimal', 'Log_Moneyness'}
        if not required.issubset(frame.columns):
            raise ValueError('Direct pricing requires spot, decimal rate and log moneyness.')
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError('Direct pricing features must be finite.')
        if (frame['Close'] <= 0).any():
            raise ValueError('Spot must be positive.')
        return frame

    def price_bounds(self, X, pricing_rate=None):
        frame = self._frame(X)
        a = frame['Close'].to_numpy() * np.exp(-self.dividend_yield * self.maturity)
        rate = frame['Treasury_Rate_Decimal'].to_numpy() if pricing_rate is None else np.asarray(pricing_rate)
        if not np.isfinite(rate).all():
            raise ValueError('Pricing rate must be finite.')
        b = self.strike * np.exp(-rate * self.maturity)
        return np.abs(a - b), a + b

    def _features(self, frame, pricing_rate=None):
        # Do not fit competing dollar-spot and logarithmic-spot coefficients.
        z = frame.drop(columns=['Close', 'Log_Moneyness']).copy()
        m = np.log(frame['Close'].to_numpy() / self.strike)
        rate = frame['Treasury_Rate_Decimal'].to_numpy() if pricing_rate is None else np.asarray(pricing_rate)
        m += (rate - self.dividend_yield) * self.maturity
        if self.distance_scale == 'historical':
            m /= np.maximum(frame['Rolling_Volatility_20D'].to_numpy(), .01) * np.sqrt(self.maturity)
        elif self.distance_scale != 'raw':
            raise ValueError('Unknown distance scaling.')
        z['Abs_Forward_Log_Moneyness'] = np.abs(m)
        z['Forward_Log_Moneyness_Squared'] = m ** 2
        return z

    def fit(self, X, y):
        if self.alpha <= 0 or self.tail_floor <= 0 or not 0 < self.epsilon < .01:
            raise ValueError('Require positive regularization/tail floor and small epsilon.')
        frame = self._frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        lower, upper = self.price_bounds(frame)
        target = np.asarray(y, dtype=float)
        if target.shape != lower.shape or not np.isfinite(target).all():
            raise ValueError('Invalid direct pricing targets.')
        if ((target < lower - 1e-7) | (target > upper + 1e-7)).any():
            raise ValueError('Training target violates chooser price bounds.')
        fraction = (target - lower) / (upper - lower)
        latent = logit(np.clip(fraction, self.epsilon, 1 - self.epsilon))
        z = self._features(frame)
        self.transformed_feature_names_ = np.asarray(z.columns, dtype=object)
        self.scaler_ = StandardScaler().fit(z)
        matrix = np.column_stack([np.ones(len(z)), self.scaler_.transform(z)])
        penalty = np.sqrt(self.alpha) * np.eye(matrix.shape[1])
        penalty[0, 0] = 0  # Do not regularize the intercept.
        augmented = np.vstack([matrix, penalty])
        rhs = np.concatenate([latent, np.zeros(matrix.shape[1])])
        lo = np.full(matrix.shape[1], -np.inf)
        hi = np.full(matrix.shape[1], np.inf)
        hi[-2] = 0.0
        hi[-1] = -self.tail_floor * self.scaler_.scale_[-1]
        solution = lsq_linear(augmented, rhs, bounds=(lo, hi), tol=1e-10)
        if not solution.success:
            raise RuntimeError(f'Constrained direct regression failed: {solution.message}')
        weights = solution.x
        if self.loss == 'price':
            span = upper - lower
            def residual(w):
                return np.r_[lower + span * expit(matrix @ w) - target, penalty @ w]
            def jacobian(w):
                fraction = expit(matrix @ w)
                return np.vstack([(span * fraction * (1-fraction))[:, None] * matrix, penalty])
            refined = least_squares(residual, weights, jac=jacobian, bounds=(lo, hi), max_nfev=1000)
            if not refined.success:
                raise RuntimeError('Bounded price-loss optimization did not converge.')
            weights = refined.x
        elif self.loss != 'logit':
            raise ValueError('Unknown fitting loss.')
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X, pricing_rate=None):
        check_is_fitted(self, 'coef_')
        frame = self._frame(X)
        lower, upper = self.price_bounds(frame, pricing_rate=pricing_rate)
        latent = self.intercept_ + self.scaler_.transform(self._features(frame, pricing_rate=pricing_rate)) @ self.coef_
        return lower + (upper - lower) * expit(latent)
