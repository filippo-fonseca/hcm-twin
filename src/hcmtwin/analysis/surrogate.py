"""A cheap stand-in for the forward model, so posterior sampling is affordable.

Markov-chain sampling needs tens of thousands of forward evaluations per patient. At
37 ms a solve that is hours per patient, and the analysis needs fifty of them. So each
patient gets a local surrogate: a polynomial fitted over the *hidden-parameter* prior box
with that patient's measured geometry and loading held fixed, which is exactly the slice
the posterior explores.

Why a polynomial and not something cleverer. The map from five smooth material parameters
to a steady-state beat is itself smooth and mildly nonlinear over the prior box, which is
the regime a low-order polynomial handles well and a flexible learner would only overfit.
It also has two properties that matter more here than accuracy alone: it is deterministic,
so a posterior is reproducible, and its error is easy to measure honestly on held-out
points.

**The surrogate error is not swept under the rug.** It is measured on a held-out split and
added in quadrature to the measurement noise in the likelihood. So the posterior is
*widened* by surrogate imprecision rather than being silently corrupted by it, and
:meth:`Surrogate.error_report` makes the size of that widening visible. If the surrogate
error were ever comparable to the measurement error, the right response would be to
abandon the surrogate rather than to caveat it, and
``tests/test_analysis.py`` asserts it is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DEGREE: int = 3
HOLDOUT_FRACTION: float = 0.25


@dataclass
class Surrogate:
    """Polynomial ridge regression from log-parameters to a vector of observables."""

    names: tuple[str, ...]
    """Output names, in the order the prediction columns appear."""

    degree: int = DEFAULT_DEGREE
    _model: object | None = field(default=None, repr=False)
    _holdout_rmse: np.ndarray | None = field(default=None, repr=False)
    _holdout_r2: np.ndarray | None = field(default=None, repr=False)
    _output_scale: np.ndarray | None = field(default=None, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)
    _powers: np.ndarray | None = field(default=None, repr=False)
    _coef: np.ndarray | None = field(default=None, repr=False)
    _intercept: np.ndarray | None = field(default=None, repr=False)

    def fit(self, log_params: np.ndarray, outputs: np.ndarray, seed: int = 0) -> Surrogate:
        """Fit on a design, holding out a random split to measure the error.

        Args:
            log_params: ``(n, d)`` natural-log parameter values.
            outputs: ``(n, m)`` observable values.
            seed: Controls the held-out split only.
        """
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        log_params = np.asarray(log_params, dtype=float)
        outputs = np.asarray(outputs, dtype=float)
        finite = np.isfinite(log_params).all(axis=1) & np.isfinite(outputs).all(axis=1)
        log_params, outputs = log_params[finite], outputs[finite]
        if len(log_params) < 50:
            raise ValueError(
                f"only {len(log_params)} usable design points; refusing to fit a surrogate"
            )

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(log_params))
        n_holdout = max(20, int(HOLDOUT_FRACTION * len(order)))
        test_idx, train_idx = order[:n_holdout], order[n_holdout:]

        def build():  # type: ignore[no-untyped-def]
            return make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=self.degree, include_bias=False),
                Ridge(alpha=1e-6),
            )

        trial = build()
        trial.fit(log_params[train_idx], outputs[train_idx])
        predicted = np.asarray(trial.predict(log_params[test_idx]))
        residual = predicted - outputs[test_idx]
        self._holdout_rmse = np.sqrt(np.mean(residual**2, axis=0))
        spread = np.std(outputs[test_idx], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._holdout_r2 = 1.0 - (self._holdout_rmse**2) / np.maximum(spread**2, 1e-30)
        self._output_scale = spread

        # Refit on everything now that the error is measured; the reported error belongs
        # to the held-out fit and is therefore, if anything, slightly pessimistic for the
        # model actually used.
        final = build()
        final.fit(log_params, outputs)
        self._model = final
        self._compile(final)
        return self

    def _compile(self, pipeline: object) -> None:
        """Flatten the fitted pipeline into three arrays for a fast prediction path.

        A scikit-learn ``Pipeline.predict`` call costs a few milliseconds regardless of
        batch size, almost all of it dispatch and validation. Posterior sampling makes
        millions of those calls, which turned an eight-second job into a two-hour one.
        Extracting the scaler statistics, the monomial exponents and the ridge
        coefficients lets prediction be three NumPy operations.

        ``tests/test_analysis.py`` asserts the compiled path agrees with the pipeline it
        came from, so this stays an optimisation rather than a second implementation.
        """
        scaler = pipeline.named_steps["standardscaler"]  # type: ignore[attr-defined]
        poly = pipeline.named_steps["polynomialfeatures"]  # type: ignore[attr-defined]
        ridge = pipeline.named_steps["ridge"]  # type: ignore[attr-defined]
        self._mean = np.asarray(scaler.mean_, dtype=float)
        self._scale = np.asarray(scaler.scale_, dtype=float)
        self._powers = np.asarray(poly.powers_, dtype=np.intp)
        coef = np.atleast_2d(np.asarray(ridge.coef_, dtype=float))
        self._coef = coef.T
        self._intercept = np.atleast_1d(np.asarray(ridge.intercept_, dtype=float))

    def predict(self, log_params: np.ndarray) -> np.ndarray:
        """Predict observables for a batch of log-parameter vectors."""
        if self._coef is None or self._powers is None:
            raise RuntimeError("surrogate has not been fitted")
        query = np.atleast_2d(np.asarray(log_params, dtype=float))
        standardised = (query - self._mean) / self._scale

        # The exponents are small integers, so building a per-variable power table and
        # gathering from it beats a general float power: one multiply per power level,
        # then a gather and a product. Raising a (batch, terms, vars) array to a float
        # exponent array, the obvious formulation, was the single slowest line in the
        # posterior sampler.
        max_power = int(self._powers.max())
        n_samples, n_vars = standardised.shape
        table = np.empty((n_vars, max_power + 1, n_samples), dtype=float)
        table[:, 0, :] = 1.0
        transposed = standardised.T
        for level in range(1, max_power + 1):
            table[:, level, :] = table[:, level - 1, :] * transposed
        gathered = table[np.arange(n_vars)[:, None], self._powers.T]
        monomials = np.prod(gathered, axis=0).T
        return monomials @ self._coef + self._intercept

    def predict_reference(self, log_params: np.ndarray) -> np.ndarray:
        """Prediction through the original scikit-learn pipeline. Testing only."""
        if self._model is None:
            raise RuntimeError("surrogate has not been fitted")
        query = np.atleast_2d(np.asarray(log_params, dtype=float))
        return np.asarray(self._model.predict(query))  # type: ignore[attr-defined]

    @property
    def holdout_rmse(self) -> np.ndarray:
        if self._holdout_rmse is None:
            raise RuntimeError("surrogate has not been fitted")
        return self._holdout_rmse

    @property
    def holdout_r2(self) -> np.ndarray:
        if self._holdout_r2 is None:
            raise RuntimeError("surrogate has not been fitted")
        return self._holdout_r2

    def error_report(self) -> dict[str, float]:
        """Worst-case held-out accuracy across outputs, for logging and for the writeup."""
        return {
            "min_r2": float(np.nanmin(self.holdout_r2)),
            "median_r2": float(np.nanmedian(self.holdout_r2)),
            "worst_output": self.names[int(np.nanargmin(self.holdout_r2))],
        }


def latin_hypercube(
    lows: np.ndarray,
    highs: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Space-filling design over a box, for training the surrogate.

    Latin hypercube rather than the Saltelli design used for the population: this design
    exists to *cover* the hidden-parameter box evenly for interpolation, not to estimate
    variance decompositions, and those are different jobs.
    """
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=len(lows), seed=seed)
    unit = sampler.random(n_samples)
    return qmc.scale(unit, lows, highs)
