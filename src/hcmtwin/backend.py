"""A two-element numeric backend so the model is written once and runs two ways.

The single-patient path wants plain Python floats: a right-hand-side evaluation costs a
few microseconds and NumPy's per-call overhead would dominate. The population path wants
NumPy arrays so that thousands of virtual patients advance through the same integration
step together, which is the only reason a 5000-patient dose ladder finishes in minutes.

Writing the physics twice would guarantee the two drift apart. Instead the physics is
written once against this tiny interface, and ``tests/test_backend.py`` asserts the two
paths agree to solver tolerance on a real beat.
"""

from __future__ import annotations

import math
from typing import Protocol, TypeVar, Union

import numpy as np

Numeric = Union[float, np.ndarray]
T = TypeVar("T", bound=Numeric)


class Backend(Protocol):
    """The operations the model's right-hand side is allowed to use."""

    def exp(self, x: T) -> T: ...

    def log1p(self, x: T) -> T: ...

    def sqrt(self, x: T) -> T: ...

    def clip(self, x: T, lo: float, hi: float) -> T: ...

    def maximum(self, x: T, lo: float) -> T: ...

    def softplus(self, x: T, width: float) -> T:
        """Smooth, strictly positive approximation to ``max(0, x)``.

        ``width * log(1 + exp(x / width))``. As ``width -> 0`` it recovers the hard
        rectifier; at the width used for the valves it lets a fraction of a millilitre
        per second leak backwards through a shut valve, which is the price of a
        differentiable right-hand side and is small enough to leave volume conservation
        inside tolerance.
        """
        ...


class ScalarBackend:
    """Plain-float backend for single-patient simulation."""

    __slots__ = ()

    def exp(self, x: float) -> float:  # type: ignore[override]
        return math.exp(x)

    def log1p(self, x: float) -> float:  # type: ignore[override]
        return math.log1p(x)

    def sqrt(self, x: float) -> float:  # type: ignore[override]
        return math.sqrt(x)

    def clip(self, x: float, lo: float, hi: float) -> float:  # type: ignore[override]
        return lo if x < lo else (hi if x > hi else x)

    def maximum(self, x: float, lo: float) -> float:  # type: ignore[override]
        return x if x > lo else lo

    def softplus(self, x: float, width: float) -> float:  # type: ignore[override]
        z = x / width
        if z > 30.0:
            return x
        if z < -30.0:
            return 0.0
        return width * math.log1p(math.exp(z))


class ArrayBackend:
    """NumPy backend for advancing a whole virtual cohort in lockstep."""

    __slots__ = ()

    def exp(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.exp(x)

    def log1p(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.log1p(x)

    def sqrt(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.sqrt(x)

    def clip(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:  # type: ignore[override]
        return np.clip(x, lo, hi)

    def maximum(self, x: np.ndarray, lo: float) -> np.ndarray:  # type: ignore[override]
        return np.maximum(x, lo)

    def softplus(self, x: np.ndarray, width: float) -> np.ndarray:  # type: ignore[override]
        z = x / width
        return np.where(z > 30.0, x, width * np.log1p(np.exp(np.minimum(z, 30.0))))


SCALAR: Backend = ScalarBackend()
ARRAY: Backend = ArrayBackend()
