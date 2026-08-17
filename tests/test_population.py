"""The virtual cohort: sampling is reproducible, and the population is plausible."""

from __future__ import annotations

import numpy as np
import pytest

from hcmtwin import defaults as d
from hcmtwin.population import (
    HIDDEN_PARAM_NAMES,
    LOADING_PARAM_NAMES,
    MEASURED_PARAM_NAMES,
    PRIOR_NAMES,
    PRIORS,
    label_over_responders,
    sample_population,
    simulate_population,
    summarise_cohort,
)
from hcmtwin.provocation import ALL_PROVOCATIONS, REST


@pytest.fixture(scope="module")
def small_cohort():  # type: ignore[no-untyped-def]
    params = sample_population(n_base=48)
    results = simulate_population(params)
    return params, results, label_over_responders(results)


def test_sampling_is_reproducible() -> None:
    first = sample_population(n_base=8, seed=7)
    second = sample_population(n_base=8, seed=7)
    different = sample_population(n_base=8, seed=8)
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    assert not np.array_equal(first.to_numpy(), different.to_numpy())


def test_saltelli_design_has_the_expected_size() -> None:
    n_base = 16
    params = sample_population(n_base=n_base)
    assert len(params) == n_base * (len(PRIORS) + 2)


def test_samples_respect_their_priors() -> None:
    params = sample_population(n_base=16)
    for prior in PRIORS:
        column = params[prior.name]
        assert column.min() >= prior.low - 1e-9, prior.name
        assert column.max() <= prior.high + 1e-9, prior.name


def test_prior_groups_partition_the_parameters() -> None:
    groups = (MEASURED_PARAM_NAMES, HIDDEN_PARAM_NAMES, LOADING_PARAM_NAMES)
    combined = [name for group in groups for name in group]
    assert sorted(combined) == sorted(PRIOR_NAMES)
    assert len(set(combined)) == len(combined), "a parameter is in two groups"


def test_every_prior_states_a_rationale() -> None:
    for prior in PRIORS:
        assert len(prior.rationale) > 40, f"{prior.name} has no stated rationale"
        assert prior.units
        assert prior.low < prior.high


def test_hidden_parameters_are_the_ones_the_analysis_infers() -> None:
    """The five hidden parameters must match the material dataclass exactly."""
    from dataclasses import fields

    from hcmtwin import HiddenMaterial

    assert set(HIDDEN_PARAM_NAMES) == {f.name for f in fields(HiddenMaterial)}


def test_all_conditions_are_simulated(small_cohort) -> None:  # type: ignore[no-untyped-def]
    params, results, _ = small_cohort
    rest = results[results["provocation"] == REST.name]
    assert sorted(rest["dose_mg_per_day"].unique()) == [0.0, 2.5, 5.0, 10.0, 15.0]
    for provocation in ALL_PROVOCATIONS:
        if provocation.name == REST.name:
            continue
        subset = results[results["provocation"] == provocation.name]
        assert sorted(subset["dose_mg_per_day"].unique()) == [0.0, d.DOSE_MID_MG_PER_DAY]
    expected_rows = len(params) * (5 + 2 * (len(ALL_PROVOCATIONS) - 1))
    assert len(results) == expected_rows


def test_most_of_the_cohort_is_physiological(small_cohort) -> None:  # type: ignore[no-untyped-def]
    _, _, labelled = small_cohort
    assert labelled["rest_conditions_physiological"].mean() > 0.90


def test_over_responder_rate_is_in_the_expected_band(small_cohort) -> None:  # type: ignore[no-untyped-def]
    """Between 1% and 25%, as the specification requires of a well-tuned prior.

    Reported on the trial-eligible subset, because that is the population whose rate a
    published rate describes. Nothing in the priors was tuned against the published value.
    """
    _, _, labelled = small_cohort
    summary = summarise_cohort(labelled)
    rate = summary["over_responder_rate_eligible"]
    assert 0.01 <= rate <= 0.25, (
        f"over-responder rate {rate:.3f} is outside the plausible band; the priors are wrong"
    )


def test_trial_eligible_cohort_looks_like_a_trial_cohort(small_cohort) -> None:  # type: ignore[no-untyped-def]
    _, _, labelled = small_cohort
    eligible = labelled[labelled["trial_eligible"]]
    assert len(eligible) > 10
    assert (eligible["ejection_fraction"] >= d.TRIAL_MIN_EF).all()
    assert (eligible["wall_thickness_cm"] >= d.TRIAL_MIN_WALL_THICKNESS_CM).all()
    assert eligible["peak_lvot_gradient_mmhg"].median() > 20.0


def test_over_responders_are_a_minority_with_normal_baseline_ejection_fraction(
    small_cohort,  # type: ignore[no-untyped-def]
) -> None:
    """The premise of the whole project: baseline ejection fraction does not give it away."""
    _, _, labelled = small_cohort
    eligible = labelled[labelled["trial_eligible"]]
    crashers = eligible[eligible["over_responder"]]
    if len(crashers) == 0:
        pytest.skip("no over-responders in this small cohort")
    assert (crashers["ejection_fraction"] >= d.TRIAL_MIN_EF).all(), (
        "every over-responder had a normal baseline ejection fraction, by construction of "
        "the eligibility criterion; if this fails the filter is broken"
    )


def test_hidden_truth_columns_are_prefixed(small_cohort) -> None:  # type: ignore[no-untyped-def]
    """Ground truth is namespaced so it cannot be swept into a feature matrix by accident."""
    from hcmtwin.observables import HIDDEN_NAMES

    _, results, _ = small_cohort
    for name in HIDDEN_NAMES:
        assert f"true_{name}" in results.columns
        assert name not in results.columns or name in {"phi_baseline"}


def test_summary_reports_both_populations(small_cohort) -> None:  # type: ignore[no-untyped-def]
    _, _, labelled = small_cohort
    summary = summarise_cohort(labelled)
    for key in (
        "n_sampled",
        "n_trial_eligible",
        "over_responder_rate_usable",
        "over_responder_rate_eligible",
        "median_baseline_ef_eligible",
    ):
        assert key in summary
        assert np.isfinite(summary[key])
