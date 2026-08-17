"""The analysis layer: the emulator is faithful, and the inference is not fooling itself."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcmtwin import RESTING_LOADING, HiddenMaterial, MeasuredGeometry
from hcmtwin import defaults as d
from hcmtwin.analysis import identifiability as idn
from hcmtwin.analysis import sensitivity as sens
from hcmtwin.analysis import tiebreaker as tb
from hcmtwin.analysis.surrogate import Surrogate, latin_hypercube


@pytest.fixture(scope="module")
def case() -> idn.PatientCase:
    return idn.PatientCase(
        patient_id=0,
        geometry=MeasuredGeometry(wall_volume_ml=245.0, ref_cavity_volume_ml=60.0),
        loading=RESTING_LOADING,
        truth=np.array([d.PHI_HCM, d.A_PAS_HCM_KPA, d.B_PAS_HCM, d.CA50_REF_UM, d.DRUG_CL_L_PER_H]),
    )


# ======================================================================================
# Surrogate
# ======================================================================================


@pytest.fixture(scope="module")
def fitted_surrogate() -> Surrogate:
    lows, highs = idn.hidden_bounds()
    design = latin_hypercube(lows, highs, 300, seed=3)
    outputs = np.column_stack(
        [
            np.log(design[:, 0]) * 2.0 + design[:, 1] ** 0.5,
            np.exp(-design[:, 2] / 10.0) + design[:, 3] * design[:, 4],
        ]
    )
    return Surrogate(names=("a", "b")).fit(np.log(design), outputs, seed=1)


def test_compiled_prediction_matches_the_pipeline_it_came_from(
    fitted_surrogate: Surrogate,
) -> None:
    """The fast path is an optimisation, not a second implementation."""
    lows, highs = idn.hidden_bounds()
    query = np.log(latin_hypercube(lows, highs, 40, seed=99))
    fast = fitted_surrogate.predict(query)
    reference = fitted_surrogate.predict_reference(query)
    assert np.allclose(fast, reference, rtol=1e-10, atol=1e-12)


def test_surrogate_reports_its_own_error(fitted_surrogate: Surrogate) -> None:
    report = fitted_surrogate.error_report()
    assert 0.0 <= report["min_r2"] <= 1.0
    assert report["worst_output"] in fitted_surrogate.names
    assert np.all(fitted_surrogate.holdout_rmse >= 0.0)


def test_surrogate_refuses_a_design_too_small_to_fit() -> None:
    with pytest.raises(ValueError, match="refusing to fit"):
        Surrogate(names=("a",)).fit(np.zeros((10, 5)), np.zeros((10, 1)))


def test_latin_hypercube_covers_the_box() -> None:
    lows, highs = idn.hidden_bounds()
    design = latin_hypercube(lows, highs, 200, seed=5)
    assert design.shape == (200, len(lows))
    assert np.all(design >= lows - 1e-9)
    assert np.all(design <= highs + 1e-9)
    # A space-filling design should reach both ends of every axis.
    spread = (design.max(axis=0) - design.min(axis=0)) / (highs - lows)
    assert np.all(spread > 0.9)


# ======================================================================================
# Fisher information
# ======================================================================================


def test_fisher_information_is_stable_under_step_halving(case: idn.PatientCase) -> None:
    """A finite difference that moves when the step changes is not a derivative."""
    coarse = idn.fisher_information(case, idn.BASELINE, "realistic", relative_step=0.02)
    fine = idn.fisher_information(case, idn.BASELINE, "realistic", relative_step=0.01)
    # Compare the informative directions; the null direction is exactly zero in both.
    ratio = coarse.eigenvalues[:4] / fine.eigenvalues[:4]
    assert np.all(np.abs(ratio - 1.0) < 0.25), f"eigenvalue ratios {ratio}"


def test_clearance_is_structurally_invisible_before_the_first_dose(
    case: idn.PatientCase,
) -> None:
    """The sharpest result in the project, asserted so it cannot quietly stop being true.

    Drug clearance enters the model only through drug exposure. At zero dose every partial
    derivative with respect to it is exactly zero, the Fisher information matrix is
    singular, and its null direction is pure clearance. No maneuver can amplify a signal
    that does not exist; the remedy is a genotype or a probe dose.
    """
    result = idn.fisher_information(case, idn.BASELINE, "realistic")
    clearance_index = idn.HIDDEN_ORDER.index("clearance_l_per_h")
    assert np.allclose(result.jacobian[:, clearance_index], 0.0, atol=1e-12)
    assert result.eigenvalues[-1] == pytest.approx(0.0, abs=1e-9)
    composition = result.stiffest_invisible_direction()
    assert composition["clearance_l_per_h"] > 0.99


def test_fisher_information_sees_the_other_four_parameters(case: idn.PatientCase) -> None:
    result = idn.fisher_information(case, idn.BASELINE, "realistic")
    for index, name in enumerate(idn.HIDDEN_ORDER):
        if name == "clearance_l_per_h":
            continue
        assert np.abs(result.jacobian[:, index]).max() > 1e-6, f"{name} moves nothing"


def test_better_measurements_carry_more_information(case: idn.PatientCase) -> None:
    optimistic = idn.fisher_information(case, idn.BASELINE, "optimistic")
    realistic = idn.fisher_information(case, idn.BASELINE, "realistic")
    assert optimistic.eigenvalues[0] > realistic.eigenvalues[0]


# ======================================================================================
# Conditions and case selection
# ======================================================================================


def test_stress_conditions_are_all_untreated() -> None:
    """A maneuver performed on treatment answers an easier and different question."""
    for condition in idn.STRESS_CONDITIONS:
        assert condition.dose_mg_per_day == 0.0
    assert idn.BASELINE.dose_mg_per_day == 0.0
    assert len(idn.ALL_CONDITIONS) == len(idn.STRESS_CONDITIONS) + 1


def test_hidden_order_matches_the_material_dataclass() -> None:
    from dataclasses import fields

    assert set(idn.HIDDEN_ORDER) == {f.name for f in fields(HiddenMaterial)}


def test_hidden_bounds_come_from_the_population_priors() -> None:
    from hcmtwin.population import PRIORS

    lows, highs = idn.hidden_bounds()
    by_name = {p.name: p for p in PRIORS}
    for index, name in enumerate(idn.HIDDEN_ORDER):
        assert lows[index] == by_name[name].low
        assert highs[index] == by_name[name].high


def test_forward_evaluation_round_trips_through_the_material_type(
    case: idn.PatientCase,
) -> None:
    from hcmtwin.observables import INDEPENDENT_NOISE_NAMES

    vector = idn.forward(case, case.truth, idn.BASELINE)
    assert vector.shape == (len(INDEPENDENT_NOISE_NAMES),)
    assert np.all(np.isfinite(vector))


def test_a_maneuver_changes_the_observables(case: idn.PatientCase) -> None:
    rest = idn.forward(case, case.truth, idn.BASELINE)
    for condition in idn.STRESS_CONDITIONS:
        stressed = idn.forward(case, case.truth, condition)
        assert not np.allclose(rest, stressed), f"{condition.key} changed nothing"


# ======================================================================================
# Tie-breaker mechanics
# ======================================================================================


def test_candidate_pairs_are_returned_even_when_nothing_crosses_the_threshold() -> None:
    """A project that reported "nothing was confounded" because the worst pair was 0.78
    rather than 0.80 would have learned nothing."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "noise_level": ["realistic"] * 3,
            "parameter_a": ["phi_baseline", "a_pas_kpa", "b_pas"],
            "parameter_b": ["a_pas_kpa", "b_pas", "ca50_ref_um"],
            "median_abs_correlation": [0.78, 0.55, 0.10],
            "confounded": [False, False, False],
        }
    )
    pairs = tb.candidate_pairs(frame, top_k=2)
    assert len(pairs) == 2
    assert pairs[0].a == "phi_baseline" and pairs[0].b == "a_pas_kpa"


def test_pair_indices_point_into_the_hidden_order() -> None:
    pair = tb.Pair("a_pas_kpa", "b_pas")
    assert idn.HIDDEN_ORDER[pair.index_a] == "a_pas_kpa"
    assert idn.HIDDEN_ORDER[pair.index_b] == "b_pas"
    assert str(pair) == "a_pas_kpa / b_pas"


def test_discriminating_signal_is_reported_against_measurement_error(
    case: idn.PatientCase,
) -> None:
    direction = np.zeros(len(idn.HIDDEN_ORDER))
    direction[idn.HIDDEN_ORDER.index("a_pas_kpa")] = 0.15
    direction[idn.HIDDEN_ORDER.index("b_pas")] = -0.15
    signal = tb.discriminating_signal(case, direction, idn.STRESS_CONDITIONS[0], "realistic")
    assert {"observable", "units", "signal", "measurement_sigma", "signal_to_noise"} <= set(
        signal.columns
    )
    assert np.all(signal["measurement_sigma"] > 0)
    assert signal["signal_to_noise"].is_monotonic_decreasing
    # A zero displacement must produce no signal at all.
    none = tb.discriminating_signal(
        case, np.zeros(len(idn.HIDDEN_ORDER)), idn.STRESS_CONDITIONS[0], "realistic"
    )
    assert float(none["abs_signal"].max()) == pytest.approx(0.0, abs=1e-9)


def test_structural_note_separates_impossible_from_merely_hard() -> None:
    import pandas as pd

    fisher_table = pd.DataFrame(
        {
            "invisible_weight_phi_baseline": [0.0],
            "invisible_weight_a_pas_kpa": [0.0],
            "invisible_weight_b_pas": [0.0],
            "invisible_weight_ca50_ref_um": [0.0],
            "invisible_weight_clearance_l_per_h": [1.0],
        }
    )
    sobol = pd.DataFrame(
        {
            "group": ["hidden"] * 4,
            "parameter": ["clearance_l_per_h", "clearance_l_per_h", "phi_baseline", "phi_baseline"],
            "quantity": ["edv_ml", "ef_drop_at_mid_dose", "edv_ml", "ef_drop_at_mid_dose"],
            "ST": [0.0, 0.63, 0.30, 0.19],
        }
    )
    note = tb.structural_unidentifiability_note(fisher_table, sobol).set_index("parameter")
    assert bool(note.loc["clearance_l_per_h", "structurally_invisible_at_baseline"])
    assert bool(note.loc["clearance_l_per_h", "matters_for_outcome"])
    assert "genotype" in str(note.loc["clearance_l_per_h", "remedy"])
    assert not bool(note.loc["phi_baseline", "structurally_invisible_at_baseline"])


# ======================================================================================
# Sensitivity
# ======================================================================================


def test_sobol_imputation_is_reported_not_hidden() -> None:
    from hcmtwin.analysis.sensitivity import _impute_for_sobol

    values = np.array([1.0, 2.0, np.nan, 4.0])
    usable = np.array([True, True, True, False])
    filled, count = _impute_for_sobol(values, usable)
    assert count == 2
    assert np.all(np.isfinite(filled))
    assert filled[2] == pytest.approx(np.median([1.0, 2.0]))


def test_outcome_names_are_not_observables() -> None:
    from hcmtwin.analysis.sensitivity import OUTCOME_NAMES
    from hcmtwin.observables import OBSERVABLE_NAMES

    assert not set(OUTCOME_NAMES) & set(OBSERVABLE_NAMES)


# =====================================================================================
# Visibility versus importance: the inversion table
# =====================================================================================


def _inversion_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A minimal Sobol/summary pair with a known, deliberately reversed ranking."""
    sobol = pd.DataFrame(
        {
            "quantity": [sens.OUTCOME_FOR_RANKING] * 3,
            "parameter": ["a", "b", "c"],
            "group": ["hidden"] * 3,
            "S1": [0.5, 0.2, 0.1],
            "ST": [0.6, 0.3, 0.1],
        }
    )
    summary = pd.DataFrame(
        {
            "parameter": ["a", "b", "c"],
            "best_observable": ["x", "y", "z"],
            "best_total_order": [0.01, 0.20, 0.90],
            "best_routine_total_order": [0.01, 0.20, 0.90],
        }
    )
    return sobol, summary


def test_visibility_versus_importance_ranks_both_directions() -> None:
    table = sens.visibility_versus_importance(*_inversion_inputs())
    assert list(table["parameter"]) == ["a", "b", "c"]  # sorted by outcome importance
    assert list(table["rank_drives"]) == [1, 2, 3]
    assert list(table["rank_visible"]) == [3, 2, 1]


def test_inversion_statistics_detects_a_perfect_reversal() -> None:
    stats = sens.inversion_statistics(sens.visibility_versus_importance(*_inversion_inputs()))
    assert stats["spearman_any"] == pytest.approx(-1.0)
    assert stats["spearman_routine"] == pytest.approx(-1.0)
    assert stats["n_parameters"] == 3


def test_inversion_statistics_detects_agreement() -> None:
    """Sign is the message, so a model where visibility tracked importance must flip it."""
    sobol, summary = _inversion_inputs()
    summary["best_total_order"] = [0.90, 0.20, 0.01]
    summary["best_routine_total_order"] = [0.90, 0.20, 0.01]
    stats = sens.inversion_statistics(sens.visibility_versus_importance(sobol, summary))
    assert stats["spearman_any"] == pytest.approx(1.0)


def test_visibility_versus_importance_requires_the_outcome() -> None:
    sobol, summary = _inversion_inputs()
    sobol["quantity"] = "not_the_outcome"
    with pytest.raises(ValueError, match="no hidden-parameter Sobol rows"):
        sens.visibility_versus_importance(sobol, summary)
