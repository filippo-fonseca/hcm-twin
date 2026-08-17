"""The wall around ground truth, tested from the access point rather than from inside."""

from __future__ import annotations

import numpy as np
import pytest

from hcmtwin import HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, simulate
from hcmtwin.hidden_only import (
    HIDDEN_NAMES,
    assert_not_a_predictor,
    hidden_truth,
    score_recovery,
)
from hcmtwin.observables import INDEPENDENT_NOISE_NAMES, NONINVASIVE_ROUTINE_NAMES


def test_guard_rejects_every_hidden_name() -> None:
    for name in HIDDEN_NAMES:
        with pytest.raises(ValueError, match="never enter a predictor"):
            assert_not_a_predictor((name,))


def test_guard_accepts_the_feature_sets_the_analysis_actually_uses() -> None:
    assert_not_a_predictor(NONINVASIVE_ROUTINE_NAMES)
    assert_not_a_predictor(INDEPENDENT_NOISE_NAMES)


def test_guard_catches_a_hidden_name_smuggled_into_a_real_feature_set() -> None:
    contaminated = (*INDEPENDENT_NOISE_NAMES, "phi_baseline")
    with pytest.raises(ValueError, match="phi_baseline"):
        assert_not_a_predictor(contaminated)


def test_scoring_recovers_zero_error_for_a_perfect_estimate() -> None:
    result = simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 5.0)
    truth = hidden_truth(result, HCM_MATERIAL)
    names = ("phi_baseline", "a_pas_kpa", "b_pas", "ca50_ref_um", "clearance_l_per_h")
    perfect = np.array([getattr(truth, n) for n in names])
    scores = score_recovery(perfect, truth, names)
    assert all(abs(v) < 1e-12 for v in scores.values())

    biased = perfect * 1.10
    scores = score_recovery(biased, truth, names)
    assert all(abs(v - 0.10) < 1e-9 for v in scores.values())


def test_the_module_boundary_is_real() -> None:
    """Nothing in the analysis package may import the hidden-truth accessor.

    A structural check rather than a behavioural one. The analysis is allowed to *hold*
    ground truth (it needs it to score itself) but it obtains it through the population
    frame's namespaced ``true_`` columns, not by reaching for this module.
    """
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent / "src" / "hcmtwin"
    offenders = [
        path.name
        for path in (package / "analysis").glob("*.py")
        if "hidden_only" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"analysis modules importing hidden_only: {offenders}"
