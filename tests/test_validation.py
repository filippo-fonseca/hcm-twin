"""The Section 7 validation gates. This file is the claim that the heart works.

``pytest tests/test_validation.py`` is the single command that proves it. Every range
asserted here traces to a row in ``docs/research/05_validation_targets.md``; the ranges
are duplicated in :data:`GATES` so the generated validation table (D2) and these tests
cannot disagree.

Two gates deserve a note before you read them.

The HCM gate asserts that the disease phenotype *emerges*. Nothing in the model is told
that HCM has a high ejection fraction, a small stroke volume, a high filling pressure, or
a high energy cost. The inputs are a higher unloaded myosin availability, stiffer passive
tissue, and a thicker wall. Everything else is a consequence, and the test checks it
against the healthy reference computed from the same code path.

The exposure-response gate is a *prediction*, not a fit. No parameter in this model was
calibrated against a published dose-to-ejection-fraction curve. The test therefore records
the comparison and asserts only the direction and an order-of-magnitude bound; the
quantitative agreement is reported in the validation table for the reader to judge.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hcmtwin import (
    HCM_GEOMETRY,
    HCM_MATERIAL,
    HEALTHY_GEOMETRY,
    HEALTHY_MATERIAL,
    RESTING_LOADING,
    Observables,
    observe,
    simulate,
)
from hcmtwin import defaults as d
from hcmtwin.drug import APPROVED_DOSE_LADDER_MG_PER_DAY
from hcmtwin.validation import HEALTHY_GATES, exposure_response_comparison as _exposure

GATES: dict[str, tuple[float, float]] = {
    name: (low, high) for name, (low, high, _units) in HEALTHY_GATES.items()
}
"""Healthy resting haemodynamics, imported from :mod:`hcmtwin.validation` rather than
restated, so a target cannot be tightened in the report and left alone in the test."""


def _run(geometry, material, loading, dose=0.0):  # type: ignore[no-untyped-def]
    result = simulate(geometry, material, loading, dose)
    return observe(result, geometry, loading), result


# ======================================================================================
# Gate 1: healthy baseline
# ======================================================================================


def test_healthy_baseline_converges(healthy_beat) -> None:  # type: ignore[no-untyped-def]
    assert healthy_beat.converged, "healthy baseline hit the beat cap without converging"
    assert healthy_beat.beats_used < d.MAX_BEATS


@pytest.mark.parametrize("quantity", sorted(GATES))
def test_healthy_baseline_in_range(healthy, healthy_beat, quantity: str) -> None:  # type: ignore[no-untyped-def]
    low, high = GATES[quantity]
    if quantity == "peak_lv_pressure_mmhg":
        value = float(healthy_beat.summary.peak_lv_pressure_mmhg)
    else:
        value = getattr(healthy, quantity)
    assert low <= value <= high, f"{quantity} = {value:.3f}, outside [{low}, {high}]"


def test_healthy_wall_thickness_is_normal(healthy: Observables) -> None:
    assert 0.6 <= healthy.wall_thickness_cm <= 1.1, (
        f"healthy wall thickness {healthy.wall_thickness_cm:.2f} cm is not normal"
    )


def test_healthy_is_not_obstructive(healthy: Observables) -> None:
    assert healthy.peak_lvot_gradient_mmhg < d.LVOT_OBSTRUCTIVE_THRESHOLD_MMHG, (
        "a structurally normal ventricle must not report an obstructive gradient"
    )


# ======================================================================================
# Gate 2: Frank-Starling
# ======================================================================================


def test_frank_starling_is_monotone() -> None:
    """Raising stressed blood volume must raise both end-diastolic and stroke volume."""
    edvs, svs = [], []
    for factor in (0.85, 0.95, 1.05, 1.15):
        loading = RESTING_LOADING.scaled(blood_volume_factor=factor)
        observed, _ = _run(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, loading)
        edvs.append(observed.edv_ml)
        svs.append(observed.stroke_volume_ml)
    assert all(b > a for a, b in zip(edvs, edvs[1:], strict=False)), f"EDV not monotone: {edvs}"
    assert all(b > a for a, b in zip(svs, svs[1:], strict=False)), f"SV not monotone: {svs}"


# ======================================================================================
# Gate 3: afterload
# ======================================================================================


def test_afterload_lowers_stroke_volume_and_raises_esv() -> None:
    svs, esvs = [], []
    for factor in (0.85, 1.0, 1.15, 1.30):
        loading = RESTING_LOADING.scaled(resistance_factor=factor)
        observed, _ = _run(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, loading)
        svs.append(observed.stroke_volume_ml)
        esvs.append(observed.esv_ml)
    assert all(b < a for a, b in zip(svs, svs[1:], strict=False)), f"SV not falling: {svs}"
    assert all(b > a for a, b in zip(esvs, esvs[1:], strict=False)), f"ESV not rising: {esvs}"


# ======================================================================================
# Gate 4: pressure-volume loop shape
# ======================================================================================


def test_loop_closes(healthy_beat) -> None:  # type: ignore[no-untyped-def]
    trace = healthy_beat.trace
    assert trace is not None
    assert abs(trace.cavity_volume_ml[0] - trace.cavity_volume_ml[-1]) < 0.5
    assert abs(trace.lv_pressure_mmhg[0] - trace.lv_pressure_mmhg[-1]) < 1.0


def test_loop_is_counter_clockwise_and_does_positive_work(healthy_beat) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    trace = healthy_beat.trace
    assert trace is not None
    # For a counter-clockwise loop in the (V, P) plane the contour integral of P dV is
    # the negative of the enclosed area, so stroke work is positive.
    signed = -np.trapezoid(trace.lv_pressure_mmhg, trace.cavity_volume_ml)
    assert signed > 0.0, "pressure-volume loop is traversed the wrong way"
    assert float(healthy_beat.summary.stroke_work_mmhg_ml) == pytest.approx(signed, rel=0.02)


def test_loop_has_recognisable_isovolumic_phases(healthy_beat) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    trace = healthy_beat.trace
    assert trace is not None
    dv = np.gradient(trace.cavity_volume_ml)
    dp = np.gradient(trace.lv_pressure_mmhg)
    scale = trace.cavity_volume_ml.max() - trace.cavity_volume_ml.min()
    quiet = np.abs(dv) < 0.002 * scale
    contracting = int(np.sum(quiet & (dp > 0)))
    relaxing = int(np.sum(quiet & (dp < 0)))
    assert contracting > 5, "no isovolumic contraction phase"
    assert relaxing > 5, "no isovolumic relaxation phase"


# ======================================================================================
# Gate 5: diastolic relation
# ======================================================================================


def test_passive_relation_is_monotone_and_convex() -> None:
    """Checked on the relation itself, at fixed volume, not through the coupled loop."""
    import numpy as np

    from hcmtwin.chamber import cavity_pressure_mmhg, stretch_from_volume
    from hcmtwin.sarcomere import passive_stress_kpa

    volumes = np.linspace(65.0, 175.0, 40)
    pressures = np.array(
        [
            cavity_pressure_mmhg(
                passive_stress_kpa(
                    stretch_from_volume(v, d.V_W_HEALTHY_ML, d.V_LV_REF_HEALTHY_ML),
                    d.A_PAS_KPA,
                    d.B_PAS,
                ),
                v,
                d.V_W_HEALTHY_ML,
            )
            for v in volumes
        ]
    )
    assert np.all(np.diff(pressures) > 0.0), "passive pressure-volume relation is not monotone"
    assert np.all(np.diff(pressures, 2) > -1e-9), "passive relation is not convex"


def test_stiffer_tissue_raises_pressure_at_fixed_volume() -> None:
    from hcmtwin.chamber import cavity_pressure_mmhg, stretch_from_volume
    from hcmtwin.sarcomere import passive_stress_kpa

    volume = 130.0
    stretch = stretch_from_volume(volume, d.V_W_HEALTHY_ML, d.V_LV_REF_HEALTHY_ML)
    pressures = [
        cavity_pressure_mmhg(passive_stress_kpa(stretch, a, d.B_PAS), volume, d.V_W_HEALTHY_ML)
        for a in (0.6, 0.9, 1.5, 2.5, 4.0)
    ]
    assert all(b > a for a, b in zip(pressures, pressures[1:], strict=False)), pressures


def test_stiffer_tissue_raises_end_diastolic_pressure_in_the_coupled_model() -> None:
    edps = []
    for a_pas in (0.6, 0.9, 1.5, 2.5, 4.0):
        material = replace(HEALTHY_MATERIAL, a_pas_kpa=a_pas)
        observed, _ = _run(HEALTHY_GEOMETRY, material, RESTING_LOADING)
        edps.append(observed.end_diastolic_pressure_mmhg)
    assert all(b > a for a, b in zip(edps, edps[1:], strict=False)), edps


# ======================================================================================
# Gate 6: the HCM phenotype emerges rather than being imposed
# ======================================================================================


def test_hcm_phenotype_emerges(healthy: Observables, hcm: Observables) -> None:
    """The inputs are availability, stiffness and wall volume. Everything else follows."""
    assert hcm.ejection_fraction >= 0.70, (
        f"HCM ejection fraction {hcm.ejection_fraction:.3f} is not supranormal"
    )
    assert hcm.ejection_fraction > healthy.ejection_fraction
    assert hcm.stroke_volume_ml < healthy.stroke_volume_ml, (
        "HCM must be hypercontractile yet deliver less blood; that is the paradox"
    )
    assert hcm.end_diastolic_pressure_mmhg > healthy.end_diastolic_pressure_mmhg
    assert hcm.atp_cost_per_stroke_work > healthy.atp_cost_per_stroke_work
    assert hcm.e_over_e_prime > healthy.e_over_e_prime
    assert hcm.peak_strain_amplitude < healthy.peak_strain_amplitude, (
        "reduced longitudinal strain despite preserved ejection fraction is the "
        "characteristic HCM finding and it has to emerge"
    )
    assert hcm.wall_thickness_cm >= d.TRIAL_MIN_WALL_THICKNESS_CM


def test_disease_requires_material_not_only_geometry() -> None:
    """Give the HCM *geometry* healthy *material*: the wall is thick but the heart is well.

    This separates the two halves of the phenotype, and the answer is instructive. Wall
    thickening alone *does* raise ejection fraction, to 0.76 here, slightly above the
    diseased reference. That is not a defect: concentric remodelling raises ejection
    fraction by geometry alone, which is exactly why a reassuring ejection fraction is
    worth so little in a thick-walled ventricle, and it is the reason this project exists.

    What the thick wall alone does *not* produce is the disease. Filling pressure, the E/e'
    surrogate, and the energetic penalty all require the material change, and all three
    are what make an HCM patient symptomatic. So the assertion is on those, not on
    ejection fraction.
    """
    benign, _ = _run(HCM_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING)
    diseased, _ = _run(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING)
    assert benign.end_diastolic_pressure_mmhg < diseased.end_diastolic_pressure_mmhg
    assert benign.e_over_e_prime < diseased.e_over_e_prime
    assert benign.atp_cost_per_stroke_work < diseased.atp_cost_per_stroke_work
    assert benign.stroke_volume_ml > diseased.stroke_volume_ml


def test_no_observable_is_assigned_a_literal_anywhere_in_the_model() -> None:
    """Structural proof that the phenotype is computed, not written down.

    Walks the package source and asserts that no clinical observable is ever the target of
    an assignment from a numeric literal. A test that merely compares two simulations
    cannot rule out a hard-coded value; this can.
    """
    import ast
    import pathlib

    from hcmtwin.observables import OBSERVABLE_NAMES

    package = pathlib.Path(__file__).resolve().parent.parent / "src" / "hcmtwin"
    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Constant) or not isinstance(
                node.value.value, (int, float)
            ):
                continue
            for target in node.targets:
                name = (
                    target.id
                    if isinstance(target, ast.Name)
                    else target.attr
                    if isinstance(target, ast.Attribute)
                    else None
                )
                if name in OBSERVABLE_NAMES:
                    offenders.append(f"{path.name}:{node.lineno} assigns {name}")
    assert not offenders, "observable assigned a literal: " + "; ".join(offenders)


def test_hcm_energy_cost_is_elevated(healthy: Observables, hcm: Observables) -> None:
    ratio = hcm.atp_cost_per_stroke_work / healthy.atp_cost_per_stroke_work
    assert ratio > 1.3, f"ATP cost per unit stroke work only {ratio:.2f}x the healthy value"


# ======================================================================================
# Gate 7: drug direction
# ======================================================================================


def test_dose_lowers_ejection_fraction_monotonically() -> None:
    efs = [
        _run(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, dose)[0].ejection_fraction
        for dose in APPROVED_DOSE_LADDER_MG_PER_DAY
    ]
    assert all(b < a for a, b in zip(efs, efs[1:], strict=False)), f"EF not monotone: {efs}"


def test_dose_lowers_peak_gradient_monotonically() -> None:
    gradients = [
        _run(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, dose)[0].peak_lvot_gradient_mmhg
        for dose in APPROVED_DOSE_LADDER_MG_PER_DAY
    ]
    assert all(b < a for a, b in zip(gradients, gradients[1:], strict=False)), gradients


def test_drug_lowers_availability_and_energy_cost() -> None:
    """The mechanistic signature: parking heads must reduce the ATP bill."""
    untreated, untreated_result = _run(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 0.0)
    treated, treated_result = _run(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 15.0)
    assert treated_result.phi_effective < untreated_result.phi_effective
    assert (
        treated_result.summary.atp_per_head < untreated_result.summary.atp_per_head
    ), "a myosin inhibitor must lower ATP consumption per head"
    assert treated.peak_lvot_gradient_mmhg < untreated.peak_lvot_gradient_mmhg


# ======================================================================================
# Gate 8: independent exposure-response shape check
# ======================================================================================


def test_exposure_response_direction_and_magnitude() -> None:
    """Direction must be right; magnitude must be the right order.

    EXPLORER-HCM reported a modest fall in ejection fraction, of order a handful of
    percentage points at 30 weeks, alongside a large fall in the outflow gradient. A model
    that reproduced a 40-point ejection-fraction collapse, or a rise, would be wrong even
    though nothing was fitted.
    """
    comparison = _exposure()
    assert comparison["ef_change_points_mid"] < 0.0, "dose must lower ejection fraction"
    assert -20.0 < comparison["ef_change_points_mid"] < -0.5, (
        f"ejection-fraction change of {comparison['ef_change_points_mid']:.1f} points at the "
        "mid dose is outside the plausible order of magnitude"
    )
    assert comparison["gradient_change_mmhg_mid"] < -10.0, (
        "the outflow gradient must fall substantially, which is the efficacy signal"
    )
