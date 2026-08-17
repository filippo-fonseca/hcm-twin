"""Quantities the model knows and a clinic cannot. Ground truth for scoring, never a predictor.

A module of its own so that the boundary is visible in the import graph and not only in a
docstring. Anything that imports from here is scoring an inference; anything that imports
from :mod:`hcmtwin.observables` is doing the inference. A file that imports both is doing
something that deserves a second look, and ``tests/test_observables.py`` and
``tests/test_hidden_only.py`` between them assert that no hidden name can reach a feature
matrix.

The definitions live in :mod:`hcmtwin.observables` next to the observable ones, because the
test that matters most is that the two name sets are disjoint and that test is easier to
trust when it can see both. This module is the access point, not a second definition.
"""

from __future__ import annotations

import numpy as np

from .observables import (
    HIDDEN_NAMES,
    OBSERVABLE_NAMES,
    HiddenTruth,
    hidden_truth,
    hidden_truth_arrays,
)

__all__ = [
    "HIDDEN_NAMES",
    "HiddenTruth",
    "assert_not_a_predictor",
    "hidden_truth",
    "hidden_truth_arrays",
]


def assert_not_a_predictor(names: tuple[str, ...] | list[str]) -> None:
    """Raise if any name in a proposed feature set is a hidden ground-truth quantity.

    Call this at the top of anything that builds a design matrix. It is cheap, and the
    failure it prevents is the kind that produces a beautiful result which means nothing:
    an inference that recovers a parameter because the parameter was handed to it.
    """
    offenders = sorted(set(names) & set(HIDDEN_NAMES))
    if offenders:
        raise ValueError(
            "these are hidden ground-truth quantities and must never enter a predictor: "
            + ", ".join(offenders)
            + f". Measurable quantities are: {', '.join(OBSERVABLE_NAMES)}"
        )


def score_recovery(
    estimated: dict[str, float] | np.ndarray,
    truth: HiddenTruth,
    names: tuple[str, ...],
) -> dict[str, float]:
    """Relative error of an estimate against ground truth, per parameter.

    The only sanctioned use of the hidden block: comparing what an inference recovered
    with what was actually there.
    """
    assert_not_a_predictor(())  # no-op, kept so the guard is exercised on every import path
    values = (
        estimated
        if isinstance(estimated, dict)
        else dict(zip(names, (float(v) for v in np.asarray(estimated)), strict=True))
    )
    scores: dict[str, float] = {}
    for name in names:
        actual = float(getattr(truth, name))
        scores[name] = (values[name] - actual) / actual if actual else float("nan")
    return scores
