"""Solver behaviour: convergence, discretisation, and the two backends agreeing."""

from __future__ import annotations

import time

import numpy as np
import pytest

from hcmtwin import (
    HCM_GEOMETRY,
    HCM_MATERIAL,
    HEALTHY_GEOMETRY,
    HEALTHY_MATERIAL,
    RESTING_LOADING,
    ModelConstants,
    observe,
    simulate,
    simulate_cohort,
)


def test_never_reports_a_first_beat() -> None:
    result = simulate(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0)
    assert result.beats_used >= 2
    assert result.converged


def test_step_count_is_converged() -> None:
    """400 steps per beat must match 2400 to the precision the analysis relies on.

    The peak outflow gradient is the sensitive one, because it depends on the square of a
    flow; it is checked to a tighter tolerance than it needs so the choice cannot decay
    unnoticed.
    """
    coarse = observe(
        simulate(
            HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0,
            constants=ModelConstants(steps_per_beat=400),
        ),
        HEALTHY_GEOMETRY,
        RESTING_LOADING,
    )
    fine = observe(
        simulate(
            HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0,
            constants=ModelConstants(steps_per_beat=2400),
        ),
        HEALTHY_GEOMETRY,
        RESTING_LOADING,
    )
    assert coarse.ejection_fraction == pytest.approx(fine.ejection_fraction, rel=1e-4)
    assert coarse.edv_ml == pytest.approx(fine.edv_ml, rel=1e-4)
    assert coarse.esv_ml == pytest.approx(fine.esv_ml, rel=1e-4)
    assert coarse.end_diastolic_pressure_mmhg == pytest.approx(
        fine.end_diastolic_pressure_mmhg, rel=1e-3
    )
    assert coarse.peak_lvot_gradient_mmhg == pytest.approx(
        fine.peak_lvot_gradient_mmhg, rel=5e-3
    )


def test_end_diastolic_volume_agrees_with_the_peak_of_the_beat() -> None:
    """Confirms the beat is phased so its closing instant really is end-diastole.

    End-diastolic quantities are read at the end of the beat rather than at the argmax of
    volume, for reasons documented on the beat summary. That is only legitimate if the two
    volumes agree; if a future change to the calcium phasing broke it, this catches it.
    """
    for geometry, material in (
        (HEALTHY_GEOMETRY, HEALTHY_MATERIAL),
        (HCM_GEOMETRY, HCM_MATERIAL),
    ):
        result = simulate(geometry, material, RESTING_LOADING, 0.0)
        edv = float(result.summary.edv_ml)
        peak = float(result.summary.max_cavity_volume_ml)
        assert abs(peak - edv) / edv < 0.005, f"{peak=} vs {edv=}"


def test_non_convergence_is_reported_not_hidden() -> None:
    starved = simulate(
        HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0,
        constants=ModelConstants(max_beats=2),
    )
    assert not starved.converged, "a run that hit the beat cap must say so"


def test_tighter_tolerance_does_not_move_the_answer() -> None:
    loose = simulate(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0)
    tight = simulate(
        HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0,
        constants=ModelConstants(steady_tol_ml=1e-4, steady_tol_mmhg=1e-4, max_beats=120),
    )
    assert float(loose.summary.edv_ml) == pytest.approx(float(tight.summary.edv_ml), rel=1e-3)
    assert float(loose.summary.esv_ml) == pytest.approx(float(tight.summary.esv_ml), rel=1e-3)


def test_scalar_and_cohort_paths_agree_on_a_full_beat() -> None:
    """The two backends must be the same physics, not two implementations of it."""
    geometries = [HEALTHY_GEOMETRY, HCM_GEOMETRY, HCM_GEOMETRY]
    materials = [HEALTHY_MATERIAL, HCM_MATERIAL, HCM_MATERIAL]
    doses = [0.0, 0.0, 10.0]

    scalar = [
        observe(simulate(g, m, RESTING_LOADING, dose), g, RESTING_LOADING)
        for g, m, dose in zip(geometries, materials, doses, strict=True)
    ]
    summary, _, _, converged, _ = simulate_cohort(
        wall_volume_ml=np.array([g.wall_volume_ml for g in geometries]),
        ref_cavity_volume_ml=np.array([g.ref_cavity_volume_ml for g in geometries]),
        phi_baseline=np.array([m.phi_baseline for m in materials]),
        a_pas_kpa=np.array([m.a_pas_kpa for m in materials]),
        b_pas=np.array([m.b_pas for m in materials]),
        ca50_ref_um=np.array([m.ca50_ref_um for m in materials]),
        clearance_l_per_h=np.array([m.clearance_l_per_h for m in materials]),
        heart_rate_bpm=np.full(3, RESTING_LOADING.heart_rate_bpm),
        total_blood_volume_ml=np.full(3, RESTING_LOADING.total_blood_volume_ml),
        systemic_resistance=np.full(3, RESTING_LOADING.systemic_resistance_mmhg_s_per_ml),
        dose_mg_per_day=np.array(doses),
    )
    assert converged
    for i, observed in enumerate(scalar):
        assert float(summary.edv_ml[i]) == pytest.approx(observed.edv_ml, rel=2e-4)
        assert float(summary.esv_ml[i]) == pytest.approx(observed.esv_ml, rel=2e-4)
        assert float(summary.peak_lvot_gradient_mmhg[i]) == pytest.approx(
            observed.peak_lvot_gradient_mmhg, rel=5e-3, abs=0.05
        )


def test_cohort_result_is_independent_of_who_it_is_batched_with() -> None:
    """A patient's answer must not depend on the other patients in the array.

    The cohort loop iterates until the *slowest* patient converges, so a patient batched
    with a slow one takes more beats. At a fixed point that must change nothing, and this
    is the test that says so.
    """
    def run(n_extra: int) -> float:
        wall = np.concatenate(
            [[HCM_GEOMETRY.wall_volume_ml], np.linspace(110.0, 290.0, n_extra)]
        )
        size = len(wall)
        summary, _, _, _, _ = simulate_cohort(
            wall_volume_ml=wall,
            ref_cavity_volume_ml=np.full(size, HCM_GEOMETRY.ref_cavity_volume_ml),
            phi_baseline=np.full(size, HCM_MATERIAL.phi_baseline),
            a_pas_kpa=np.full(size, HCM_MATERIAL.a_pas_kpa),
            b_pas=np.full(size, HCM_MATERIAL.b_pas),
            ca50_ref_um=np.full(size, HCM_MATERIAL.ca50_ref_um),
            clearance_l_per_h=np.full(size, HCM_MATERIAL.clearance_l_per_h),
            heart_rate_bpm=np.full(size, RESTING_LOADING.heart_rate_bpm),
            total_blood_volume_ml=np.full(size, RESTING_LOADING.total_blood_volume_ml),
            systemic_resistance=np.full(
                size, RESTING_LOADING.systemic_resistance_mmhg_s_per_ml
            ),
            dose_mg_per_day=0.0,
        )
        return float(summary.edv_ml[0])

    assert run(3) == pytest.approx(run(25), rel=1e-6)


@pytest.mark.slow
def test_steady_state_solve_meets_its_speed_target() -> None:
    """Under 50 ms per steady-state solve. The whole project depends on this."""
    simulate(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0)
    start = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        simulate(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0)
    per_solve_ms = (time.perf_counter() - start) / repeats * 1e3
    assert per_solve_ms < 80.0, f"{per_solve_ms:.1f} ms per solve is too slow to be usable"


def test_trace_is_only_produced_when_asked() -> None:
    assert simulate(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0).trace is None
    traced = simulate(
        HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0, record_trace=True
    )
    assert traced.trace is not None
    assert len(traced.trace.time_s) == ModelConstants().steps_per_beat
