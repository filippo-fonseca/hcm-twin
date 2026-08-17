"""The wall between what a clinic can measure and what only the model knows.

If a hidden ground-truth quantity ever reaches a feature matrix, every identifiability
result in the project becomes meaningless: the analysis would be recovering a parameter
from itself. These tests exist to make that failure impossible rather than unlikely.
"""

from __future__ import annotations

import numpy as np
import pytest

from hcmtwin import HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, hidden_truth, observe, simulate
from hcmtwin.observables import (
    HIDDEN_NAMES,
    INDEPENDENT_NOISE_NAMES,
    NONINVASIVE_ROUTINE_NAMES,
    OBSERVABLE_NAMES,
    SPECS,
    noise_sigma,
    to_vector,
)


def test_observable_and_hidden_names_are_disjoint() -> None:
    overlap = set(OBSERVABLE_NAMES) & set(HIDDEN_NAMES)
    assert not overlap, f"a quantity is both measurable and hidden: {overlap}"


def test_every_observable_has_a_spec() -> None:
    missing = set(OBSERVABLE_NAMES) - set(SPECS)
    assert not missing, f"observables with no modality or noise entry: {missing}"
    extra = set(SPECS) - set(OBSERVABLE_NAMES)
    assert not extra, f"noise specs for things that are not observables: {extra}"


def test_every_observable_documents_its_modality_and_noise() -> None:
    for name, spec in SPECS.items():
        assert spec.units, f"{name} has no units"
        assert len(spec.modality) > 10, f"{name} does not name a real-world modality"
        assert spec.noise_realistic > 0.0, f"{name} has no realistic-level noise"
        assert spec.noise_optimistic > 0.0, f"{name} has no optimistic-level noise"
        assert spec.noise_optimistic <= spec.noise_realistic, (
            f"{name}: the optimistic noise level must not exceed the realistic one"
        )


def test_feature_vector_refuses_hidden_names() -> None:
    result = simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 0.0)
    observed = observe(result, HCM_GEOMETRY, RESTING_LOADING)
    for hidden_name in HIDDEN_NAMES:
        with pytest.raises(KeyError):
            to_vector(observed, (hidden_name,))


def test_default_feature_sets_contain_no_hidden_and_no_invasive_names() -> None:
    for name in NONINVASIVE_ROUTINE_NAMES:
        assert name not in HIDDEN_NAMES
        assert not SPECS[name].invasive, f"{name} is invasive but in the routine set"
        assert SPECS[name].routine
    for name in INDEPENDENT_NOISE_NAMES:
        assert not SPECS[name].derived, (
            f"{name} is derived and must not carry independent noise in the likelihood"
        )


def test_hidden_truth_is_reachable_only_through_its_own_accessor() -> None:
    result = simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 10.0)
    observed = observe(result, HCM_GEOMETRY, RESTING_LOADING)
    truth = hidden_truth(result, HCM_MATERIAL)
    for hidden_name in HIDDEN_NAMES:
        assert not hasattr(observed, hidden_name), (
            f"{hidden_name} leaked onto the observable dataclass"
        )
        assert hasattr(truth, hidden_name)
    assert truth.phi_effective < truth.phi_baseline


def test_noise_sigma_shapes_and_levels() -> None:
    result = simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 0.0)
    observed = observe(result, HCM_GEOMETRY, RESTING_LOADING)
    values = to_vector(observed, INDEPENDENT_NOISE_NAMES)
    realistic = noise_sigma(INDEPENDENT_NOISE_NAMES, values, "realistic")
    optimistic = noise_sigma(INDEPENDENT_NOISE_NAMES, values, "optimistic")
    assert realistic.shape == values.shape
    assert np.all(realistic > 0.0)
    assert np.all(optimistic <= realistic + 1e-12)
    with pytest.raises(ValueError):
        noise_sigma(INDEPENDENT_NOISE_NAMES, values, "wishful")


def test_cohort_and_single_patient_observables_agree() -> None:
    """The vectorised reduction and the scalar one must be the same arithmetic."""
    from hcmtwin.observables import observe_arrays

    result = simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 5.0)
    scalar = observe(result, HCM_GEOMETRY, RESTING_LOADING)
    arrays = observe_arrays(
        result.summary,
        wall_volume_ml=np.array([HCM_GEOMETRY.wall_volume_ml]),
        body_surface_area_m2=np.array([HCM_GEOMETRY.body_surface_area_m2]),
        heart_rate_bpm=np.array([RESTING_LOADING.heart_rate_bpm]),
    )
    for name in OBSERVABLE_NAMES:
        value = float(np.asarray(arrays[name]).reshape(-1)[0])
        assert value == pytest.approx(getattr(scalar, name), rel=1e-12)
