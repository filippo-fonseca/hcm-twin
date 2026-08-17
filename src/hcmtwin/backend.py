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
from typing import Protocol

import numpy as np

Numeric = float | np.ndarray


class Backend(Protocol):
    """The operations the model's right-hand side is allowed to use.

    Typed against ``Numeric`` rather than a type variable: a scalar backend cannot honour
    a promise to return an array for an array, and pretending otherwise would only move
    the inaccuracy into the type checker's blind spot.
    """

    def exp(self, x: Numeric) -> Numeric: ...

    def log1p(self, x: Numeric) -> Numeric: ...

    def sqrt(self, x: Numeric) -> Numeric: ...

    def clip(self, x: Numeric, lo: float, hi: float) -> Numeric: ...

    def maximum(self, x: Numeric, lo: float) -> Numeric: ...

    def softplus(self, x: Numeric, width: float) -> Numeric:
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

    def exp(self, x: Numeric) -> Numeric:
        return math.exp(x)

    def log1p(self, x: Numeric) -> Numeric:
        return math.log1p(x)

    def sqrt(self, x: Numeric) -> Numeric:
        return math.sqrt(x)

    def clip(self, x: Numeric, lo: float, hi: float) -> Numeric:
        return lo if x < lo else (hi if x > hi else x)

    def maximum(self, x: Numeric, lo: float) -> Numeric:
        return x if x > lo else lo

    def softplus(self, x: Numeric, width: float) -> Numeric:
        z = x / width
        if z > 30.0:
            return x
        if z < -30.0:
            return 0.0
        return width * math.log1p(math.exp(z))


class ArrayBackend:
    """NumPy backend for advancing a whole virtual cohort in lockstep."""

    __slots__ = ()

    def exp(self, x: Numeric) -> Numeric:
        return np.exp(x)

    def log1p(self, x: Numeric) -> Numeric:
        return np.log1p(x)

    def sqrt(self, x: Numeric) -> Numeric:
        return np.sqrt(x)

    def clip(self, x: Numeric, lo: float, hi: float) -> Numeric:
        return np.clip(x, lo, hi)

    def maximum(self, x: Numeric, lo: float) -> Numeric:
        return np.maximum(x, lo)

    def softplus(self, x: Numeric, width: float) -> Numeric:
        z = x / width
        return np.where(z > 30.0, x, width * np.log1p(np.exp(np.minimum(z, 30.0))))


SCALAR: Backend = ScalarBackend()
ARRAY: Backend = ArrayBackend()
