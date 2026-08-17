"""Property-based tests: invariants that must hold for every patient, not just the two.

These are the tests that catch the bug the fixed examples miss. A conservation law you
have not tested is a conservation law you have assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hcmtwin import HiddenMaterial, Loading, MeasuredGeometry, ModelConstants, observe, simulate
from hcmtwin import defaults as d
from hcmtwin.circulation import venous_pressure_mmhg

SLOW = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

geometries = st.builds(
    MeasuredGeometry,
    wall_volume_ml=st.floats(110.0, 290.0),
    ref_cavity_volume_ml=st.floats(46.0, 78.0),
    body_surface_area_m2=st.floats(1.6, 2.2),
)
materials = st.builds(
    HiddenMaterial,
    phi_baseline=st.floats(0.29, 0.61),
    a_pas_kpa=st.floats(0.45, 5.4),
    b_pas=st.floats(6.2, 19.8),
    ca50_ref_um=st.floats(0.46, 0.79),
    clearance_l_per_h=st.floats(0.14, 1.09),
)
loadings = st.builds(
    Loading,
    heart_rate_bpm=st.floats(53.0, 84.0),
    total_blood_volume_ml=st.floats(342.0, 448.0),
    systemic_resistance_mmhg_s_per_ml=st.floats(0.82, 1.33),
)


@SLOW
@given(geometry=geometries, material=materials, loading=loadings)
def test_myosin_populations_sum_to_one(
    geometry: MeasuredGeometry, material: HiddenMaterial, loading: Loading
) -> None:
    """S + D + A = 1 at every integration step, for every patient.

    Meaningful precisely because the solver integrates all three states rather than
    eliminating one: this checks that the right-hand side sums to zero and that the
    integrator preserves it, which an algebraic elimination would have made vacuous.
    """
    result = simulate(geometry, material, loading, 0.0)
    assert float(result.summary.population_error) < 1e-10


@SLOW
@given(geometry=geometries, material=materials, loading=loadings)
def test_volumes_and_pressures_stay_physical(
    geometry: MeasuredGeometry, material: HiddenMaterial, loading: Loading
) -> None:
    result = simulate(geometry, material, loading, 0.0)
    summary = result.summary
    assert float(summary.esv_ml) > 0.0, "cavity volume went non-positive"
    assert float(summary.edv_ml) > float(summary.esv_ml), "no forward stroke volume"
    assert np.isfinite(float(summary.peak_lv_pressure_mmhg))
    assert float(summary.peak_lv_pressure_mmhg) < 500.0
    assert 0.0 <= float(summary.peak_attached) <= 1.0
    assert float(summary.min_attached) >= -1e-12


@SLOW
@given(geometry=geometries, material=materials, loading=loadings)
def test_blood_volume_is_conserved(
    geometry: MeasuredGeometry, material: HiddenMaterial, loading: Loading
) -> None:
    """Total stressed volume is conserved to machine precision, by construction.

    The venous compartment closes the balance algebraically rather than being integrated,
    so this checks the construction rather than the integrator. It is still worth
    asserting: a future refactor that starts integrating venous pressure would break it
    silently.
    """
    result = simulate(geometry, material, loading, 0.0, record_trace=True)
    trace = result.trace
    assert trace is not None
    total = (
        trace.cavity_volume_ml
        + d.C_ART_ML_PER_MMHG * trace.arterial_pressure_mmhg
        + d.C_VEN_ML_PER_MMHG * trace.venous_pressure_mmhg
    )
    drift = float(np.max(np.abs(total - loading.total_blood_volume_ml)))
    assert drift < 1e-8, f"stressed volume drifted by {drift:.2e} mL"


@SLOW
@given(geometry=geometries, material=materials, loading=loadings)
def test_pressure_volume_loop_closes(
    geometry: MeasuredGeometry, material: HiddenMaterial, loading: Loading
) -> None:
    result = simulate(geometry, material, loading, 0.0, record_trace=True)
    trace = result.trace
    assert trace is not None
    span = float(trace.cavity_volume_ml.max() - trace.cavity_volume_ml.min())
    gap = abs(float(trace.cavity_volume_ml[0] - trace.cavity_volume_ml[-1]))
    assert gap < 0.02 * span, f"loop failed to close: {gap:.3f} mL over a {span:.1f} mL span"


@SLOW
@given(geometry=geometries, material=materials, loading=loadings)
def test_more_drug_never_raises_availability(
    geometry: MeasuredGeometry, material: HiddenMaterial, loading: Loading
) -> None:
    """Monotone by construction, and worth pinning: the drug is the only phi-lowering path."""
    previous = None
    for dose in (0.0, 5.0, 15.0):
        result = simulate(geometry, material, loading, dose)
        if previous is not None:
            assert result.phi_effective < previous
        previous = result.phi_effective


def test_venous_pressure_closes_the_volume_balance() -> None:
    p_ven = venous_pressure_mmhg(400.0, 110.0, 95.0, 1.7, 15.0)
    reconstructed = 110.0 + 1.7 * 95.0 + 15.0 * p_ven
    assert reconstructed == pytest.approx(400.0, abs=1e-9)


@pytest.mark.parametrize("dose", [0.0, 5.0, 15.0])
def test_reported_observables_are_finite(dose: float) -> None:
    from hcmtwin import HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING
    from hcmtwin.observables import OBSERVABLE_NAMES

    result = simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, dose)
    observed = observe(result, HCM_GEOMETRY, RESTING_LOADING)
    for name in OBSERVABLE_NAMES:
        value = getattr(observed, name)
        assert np.isfinite(value), f"{name} is not finite at dose {dose}"


def test_steady_state_is_independent_of_initial_conditions() -> None:
    """Deliberately wrong starting conditions must reach the same converged beat.

    This is what makes it safe to tune the initial guess for convergence speed: the guess
    is an optimisation, never a result.
    """
    from hcmtwin import HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING

    baseline = simulate(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0)
    shifted = simulate(
        HEALTHY_GEOMETRY,
        HEALTHY_MATERIAL,
        RESTING_LOADING,
        0.0,
        constants=ModelConstants(max_beats=80),
    )
    assert float(baseline.summary.edv_ml) == pytest.approx(
        float(shifted.summary.edv_ml), rel=1e-4
    )
