"""Assembly: the three layers coupled, and the steady-state beat solver.

Integration happens in *normalised beat phase* ``tau = t / T`` rather than in wall-clock
time. Two reasons. First, every heart rate then uses the same number of steps per beat,
so a whole virtual cohort with a range of heart rates can be advanced through one
integration step together in a single vectorised operation, which is what makes a
5000-patient dose ladder finish in minutes rather than hours. Second, the beat boundary
lands exactly on a step boundary, so beat-to-beat comparisons for the steady-state test
are exact rather than interpolated.

Never report a first beat. Every public result here comes from a beat that follows a
converged sequence, and a run that hit the beat cap without converging is flagged
:attr:`BeatResult.converged` ``= False`` rather than quietly returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from . import chamber, circulation, defaults as d, drug, obstruction, sarcomere
from .backend import ARRAY, SCALAR, Backend, Numeric
from .calcium import beat_period_s, calcium_um
from .parameters import HiddenMaterial, Loading, MeasuredGeometry, ModelConstants

MMHG_ML_TO_JOULE: float = 1.33322387415e-4
"""Exact conversion from mmHg*mL of stroke work to joules. Definitional, not
physiological, so it lives here rather than in :mod:`hcmtwin.defaults`."""


class _Params(NamedTuple):
    """Flat bundle handed to the right-hand side.

    Fields are floats in the single-patient path and equal-length arrays in the cohort
    path; the arithmetic is identical either way. Flat and immutable because the
    right-hand side is evaluated a few thousand times per beat and attribute lookup is
    not free.
    """

    period_s: Numeric
    wall_volume_ml: Numeric
    ref_cavity_volume_ml: Numeric
    phi_eff: Numeric
    a_pas_kpa: Numeric
    b_pas: Numeric
    ca50_ref_um: Numeric
    total_volume_ml: Numeric
    r_sys: Numeric
    k_park_off: Numeric
    k_park_on: Numeric
    # Shared across the population
    ca_diast_um: float
    ca_peak_um: float
    ca_tau_r_s: float
    k_att: float
    k_det: float
    k_force_per_kpa: float
    hill_n: float
    beta_len: float
    t_ref_kpa: float
    beta_overlap: float
    overlap_max: float
    k_xb_per_s: float
    xb_half: float
    xb_max_gain: float
    c_art: float
    c_ven: float
    r_av: float
    r_mv: float
    valve_smooth: float
    obstruction_enabled: bool
    k_obs: float
    a0_lvot: float
    crowding_ref: float
    lvot_exponent: float
    a_min_frac: float


class _Diagnostics(NamedTuple):
    """Quantities the right-hand side computes anyway and the observables need."""

    p_lv_mmhg: Numeric
    p_ven_mmhg: Numeric
    q_av: Numeric
    q_mv: Numeric
    q_sys: Numeric
    lvot_gradient_mmhg: Numeric
    stretch: Numeric
    fiber_strain: Numeric
    calcium_um: Numeric
    fiber_strain_rate_per_s: Numeric


def _evaluate(
    tau: float,
    parked: Numeric,
    available: Numeric,
    attached: Numeric,
    distortion: Numeric,
    cavity_ml: Numeric,
    p_art: Numeric,
    p: _Params,
    xp: Backend,
) -> tuple[tuple[Numeric, ...], _Diagnostics]:
    """One evaluation of the coupled right-hand side, in normalised beat phase.

    Returns the eight phase derivatives ``dy/dtau = T * dy/dt`` and the diagnostics the
    observable reductions consume, so the final beat does not have to recompute them.

    Evaluation order matters and is deliberate: stress depends only on *states*, flows
    depend on stress, the strain rate depends on the flows, and the distortion derivative
    depends on the strain rate. That chain is acyclic, which is what keeps a model with a
    genuine force-velocity relation explicit.
    """
    time_s = tau * p.period_s
    ca = calcium_um(time_s, p.ca_diast_um, p.ca_peak_um, p.ca_tau_r_s, xp)

    strain = chamber.fiber_strain(cavity_ml, p.wall_volume_ml, p.ref_cavity_volume_ml, xp)
    lam = xp.exp(strain)

    sigma_f = sarcomere.total_stress_kpa(
        attached,
        distortion,
        lam,
        p.t_ref_kpa,
        p.beta_overlap,
        p.overlap_max,
        p.xb_half,
        p.xb_max_gain,
        p.a_pas_kpa,
        p.b_pas,
        xp,
    )
    p_lv = chamber.cavity_pressure_mmhg(sigma_f, cavity_ml, p.wall_volume_ml)
    p_ven = circulation.venous_pressure_mmhg(
        p.total_volume_ml, cavity_ml, p_art, p.c_art, p.c_ven
    )

    if p.obstruction_enabled:
        area = obstruction.lvot_area_cm2(
            cavity_ml,
            p.wall_volume_ml,
            p.a0_lvot,
            p.crowding_ref,
            p.lvot_exponent,
            p.a_min_frac,
            xp,
        )
        q_av = obstruction.aortic_flow_ml_per_s(
            p_lv - p_art, area, p.r_av, p.k_obs, p.valve_smooth, xp
        )
        gradient = obstruction.lvot_gradient_mmhg(q_av, area, p.k_obs)
    else:
        q_av = circulation.valve_flow_ml_per_s(p_lv - p_art, p.r_av, p.valve_smooth, xp)
        gradient = 0.0 * q_av

    q_mv = circulation.valve_flow_ml_per_s(p_ven - p_lv, p.r_mv, p.valve_smooth, xp)
    q_sys = circulation.systemic_flow_ml_per_s(p_art, p_ven, p.r_sys)

    sarc = sarcomere.derivatives(
        parked,
        available,
        attached,
        ca,
        lam,
        sigma_f,
        p.k_park_off,
        p.k_park_on,
        p.k_att,
        p.k_det,
        p.k_force_per_kpa,
        p.ca50_ref_um,
        p.hill_n,
        p.beta_len,
        xp,
    )

    d_cavity = q_mv - q_av
    d_p_art = (q_av - q_sys) / p.c_art
    # d(eps_f)/dt follows from differentiating the one-fiber strain relation.
    d_strain = d_cavity / (3.0 * (cavity_ml + p.wall_volume_ml / 3.0))
    d_distortion = sarcomere.distortion_derivative(distortion, d_strain, p.k_xb_per_s)
    # Stroke work accumulates as the integral of p dV over the loop.
    d_work = p_lv * d_cavity

    t = p.period_s
    derivs = (
        t * sarc.ds_dt,
        t * sarc.dd_dt,
        t * sarc.da_dt,
        t * d_distortion,
        t * d_cavity,
        t * d_p_art,
        t * sarc.atp_flux_per_s,
        t * d_work,
    )
    diag = _Diagnostics(p_lv, p_ven, q_av, q_mv, q_sys, gradient, lam, strain, ca, d_strain)
    return derivs, diag


class _BeatSummary(NamedTuple):
    """Reductions taken over a single beat.

    End-diastolic quantities are read at the *final* instant of the beat rather than at
    the argmax of cavity volume. The two are the same point physiologically -- the beat is
    phased so that it begins with the calcium upstroke, so its last instant is the end of
    filling -- but they are not the same point numerically. During isovolumic contraction
    cavity volume sits on a flat plateau while pressure climbs by a hundred millimetres of
    mercury, and the softplus valve leaks a few hundredths of a millilitre inward across
    that plateau, so an argmax lands late in the upstroke and reports an end-diastolic
    pressure two to three times too high. Reading the end of the beat is unambiguous.
    ``tests/test_model.py`` asserts the two volumes agree to well under a percent, which
    is the check that the phasing assumption still holds.
    """

    edv_ml: Numeric
    max_cavity_volume_ml: Numeric
    esv_ml: Numeric
    end_diastolic_pressure_mmhg: Numeric
    peak_lv_pressure_mmhg: Numeric
    systolic_arterial_mmhg: Numeric
    diastolic_arterial_mmhg: Numeric
    mean_arterial_mmhg: Numeric
    mean_systemic_flow_ml_per_s: Numeric
    peak_lvot_gradient_mmhg: Numeric
    peak_strain: Numeric
    min_strain: Numeric
    atp_per_head: Numeric
    stroke_work_mmhg_ml: Numeric
    peak_mitral_flow_ml_per_s: Numeric
    peak_transmitral_gradient_mmhg: Numeric
    """Largest venous-minus-ventricular pressure difference during the beat, mmHg.

    This, not the mitral flow, is what drives the clinical E wave: continuous-wave Doppler
    reads a velocity and the velocity comes from the atrioventricular pressure gradient via
    ``4 v^2``. Using the gradient is what lets the E/e' surrogate respond to filling
    pressure. An earlier version divided peak mitral *flow* by peak lengthening *rate*,
    which is very nearly the same quantity over itself -- both are ``dV/dt`` -- so the
    ratio came out an almost pure function of chamber geometry and barely moved when the
    tissue was stiffened. It looked like a filling-pressure surrogate and was not one."""

    peak_lengthening_rate_per_s: Numeric
    peak_attached: Numeric
    min_attached: Numeric
    mean_parked: Numeric
    population_error: Numeric
    """Worst absolute deviation of ``S + D + A`` from 1 seen during the beat."""


@dataclass(frozen=True)
class BeatTrace:
    """Full within-beat waveforms. Only produced when explicitly requested."""

    time_s: np.ndarray
    cavity_volume_ml: np.ndarray
    lv_pressure_mmhg: np.ndarray
    arterial_pressure_mmhg: np.ndarray
    venous_pressure_mmhg: np.ndarray
    aortic_flow_ml_per_s: np.ndarray
    mitral_flow_ml_per_s: np.ndarray
    lvot_gradient_mmhg: np.ndarray
    stretch: np.ndarray
    parked: np.ndarray
    available: np.ndarray
    attached: np.ndarray
    calcium_um: np.ndarray


@dataclass(frozen=True)
class BeatResult:
    """A converged steady-state beat.

    Raw mechanics only. Turning this into the things a clinic could measure is
    :mod:`hcmtwin.observables`' job, and keeping the two apart is what lets the tests
    assert that no hidden quantity leaks into a predictor.
    """

    summary: _BeatSummary
    converged: bool
    beats_used: int
    phi_effective: float
    concentration_ng_per_ml: float
    period_s: float
    wall_volume_ml: float
    ref_cavity_volume_ml: float
    trace: BeatTrace | None = None


def _make_params(
    measured: MeasuredGeometry,
    hidden: HiddenMaterial,
    loading: Loading,
    dose_mg_per_day: float,
    constants: ModelConstants,
) -> tuple[_Params, float, float]:
    concentration = drug.steady_state_concentration_ng_per_ml(
        dose_mg_per_day, hidden.clearance_l_per_h
    )
    phi_eff = drug.effective_phi(
        hidden.phi_baseline, concentration, constants.drug_e_max, constants.drug_ec50_ng_per_ml
    )
    rates = sarcomere.rates_from_phi(phi_eff, constants.k_park_tot_per_s)
    params = _Params(
        period_s=beat_period_s(loading.heart_rate_bpm),
        wall_volume_ml=measured.wall_volume_ml,
        ref_cavity_volume_ml=measured.ref_cavity_volume_ml,
        phi_eff=phi_eff,
        a_pas_kpa=hidden.a_pas_kpa,
        b_pas=hidden.b_pas,
        ca50_ref_um=hidden.ca50_ref_um,
        total_volume_ml=loading.total_blood_volume_ml,
        r_sys=loading.systemic_resistance_mmhg_s_per_ml,
        k_park_off=rates.k_park_off_per_s,
        k_park_on=rates.k_park_on_per_s,
        ca_diast_um=constants.ca_diast_um,
        ca_peak_um=constants.ca_peak_um,
        ca_tau_r_s=constants.ca_tau_r_s,
        k_att=constants.k_att_per_s,
        k_det=constants.k_det_per_s,
        k_force_per_kpa=constants.k_force_per_kpa,
        hill_n=constants.hill_n,
        beta_len=constants.beta_len,
        t_ref_kpa=constants.t_ref_kpa,
        beta_overlap=constants.beta_overlap,
        overlap_max=constants.overlap_max,
        k_xb_per_s=constants.k_xb_per_s,
        xb_half=constants.xb_half,
        xb_max_gain=constants.xb_max_gain,
        c_art=constants.c_art_ml_per_mmhg,
        c_ven=constants.c_ven_ml_per_mmhg,
        r_av=constants.r_av_mmhg_s_per_ml,
        r_mv=constants.r_mv_mmhg_s_per_ml,
        valve_smooth=constants.valve_smooth_mmhg,
        obstruction_enabled=constants.obstruction_enabled,
        k_obs=constants.k_obs_mmhg_s2_cm4_per_ml2,
        a0_lvot=constants.a0_lvot_cm2,
        crowding_ref=constants.crowding_ref,
        lvot_exponent=constants.lvot_exponent,
        a_min_frac=constants.a_min_frac_lvot,
    )
    return params, float(phi_eff), float(concentration)


def _initial_state(
    p: _Params, xp: Backend
) -> tuple[Numeric, Numeric, Numeric, Numeric, Numeric, Numeric]:
    """A cheap analytic guess, so the beat iteration starts near its fixed point.

    Only convergence speed depends on this. ``tests/test_model.py`` starts the same
    patient from deliberately wrong initial conditions and asserts the converged beat is
    identical, which is the test that makes it safe to tune the guess for speed.
    """
    n_rest = 0.0
    if isinstance(p.phi_eff, np.ndarray):
        parked = np.empty_like(p.phi_eff)
        available = np.empty_like(p.phi_eff)
        attached = np.empty_like(p.phi_eff)
        for i, phi in enumerate(np.asarray(p.phi_eff, dtype=float)):
            parked[i], available[i], attached[i] = sarcomere.resting_populations(
                float(phi), p.k_att, p.k_det, n_rest
            )
    else:
        parked, available, attached = sarcomere.resting_populations(
            float(p.phi_eff), p.k_att, p.k_det, n_rest
        )
    del xp
    cavity = p.ref_cavity_volume_ml + d.INIT_CAVITY_OFFSET_ML
    p_art = d.INIT_ARTERIAL_PRESSURE_MMHG + 0.0 * p.ref_cavity_volume_ml
    distortion = 0.0 * cavity
    return parked, available, attached, distortion, cavity, p_art


def _run_beat(
    state: tuple[Numeric, ...],
    p: _Params,
    xp: Backend,
    n_steps: int,
    record: bool,
) -> tuple[tuple[Numeric, ...], _BeatSummary, BeatTrace | None]:
    """Advance one full cardiac cycle with fixed-step RK4 in normalised phase.

    Fixed step rather than adaptive: the system is only mildly stiff (the fastest time
    constant is the ~10 ms arterial-valve product), the step is far inside the stability
    limit, and a fixed step is what lets an entire cohort share one loop.
    ``tests/test_model.py`` halves and doubles the step count and asserts the converged
    observables move by less than the tolerance the analysis uses.
    """
    parked, available, attached, distortion, cavity, p_art = state
    atp = 0.0 * cavity
    work = 0.0 * cavity
    dtau = 1.0 / n_steps
    n_state = 8

    zero = 0.0 * cavity
    v_max = cavity + zero
    v_min = cavity + zero
    edp = zero
    p_lv_max = zero - 1.0e9
    pa_max = p_art + zero
    pa_min = p_art + zero
    pa_sum = zero
    qsys_sum = zero
    grad_max = zero
    strain_max = zero - 1.0e9
    strain_min = zero + 1.0e9
    qmv_max = zero
    av_grad_max = zero
    lengthen_max = zero
    a_max = zero
    a_min = zero + 1.0e9
    parked_sum = zero
    pop_err = zero
    first = True

    traces: dict[str, list[Numeric]] | None = None
    if record:
        traces = {
            key: []
            for key in (
                "time_s",
                "cavity_volume_ml",
                "lv_pressure_mmhg",
                "arterial_pressure_mmhg",
                "venous_pressure_mmhg",
                "aortic_flow_ml_per_s",
                "mitral_flow_ml_per_s",
                "lvot_gradient_mmhg",
                "stretch",
                "parked",
                "available",
                "attached",
                "calcium_um",
            )
        }

    for step in range(n_steps):
        tau = step * dtau
        y = (parked, available, attached, distortion, cavity, p_art, atp, work)

        k1, diag = _evaluate(tau, y[0], y[1], y[2], y[3], y[4], y[5], p, xp)
        y2 = tuple(y[i] + 0.5 * dtau * k1[i] for i in range(n_state))
        k2, _ = _evaluate(tau + 0.5 * dtau, y2[0], y2[1], y2[2], y2[3], y2[4], y2[5], p, xp)
        y3 = tuple(y[i] + 0.5 * dtau * k2[i] for i in range(n_state))
        k3, _ = _evaluate(tau + 0.5 * dtau, y3[0], y3[1], y3[2], y3[3], y3[4], y3[5], p, xp)
        y4 = tuple(y[i] + dtau * k3[i] for i in range(n_state))
        k4, _ = _evaluate(tau + dtau, y4[0], y4[1], y4[2], y4[3], y4[4], y4[5], p, xp)

        # --- reductions, taken at the start of the step where the diagnostics are exact
        total_pop = parked + available + attached
        pop_err = _maximum(pop_err, _absolute(total_pop - 1.0, xp), xp)
        if first:
            v_max = cavity
            v_min = cavity
            edp = diag.p_lv_mmhg
            p_lv_max = diag.p_lv_mmhg
            pa_max = p_art
            pa_min = p_art
            strain_max = diag.fiber_strain
            strain_min = diag.fiber_strain
            a_max = attached
            a_min = attached
            first = False
        else:
            v_max = _maximum(v_max, cavity, xp)
            v_min = _minimum(v_min, cavity, xp)
            p_lv_max = _maximum(p_lv_max, diag.p_lv_mmhg, xp)
            pa_max = _maximum(pa_max, p_art, xp)
            pa_min = _minimum(pa_min, p_art, xp)
            strain_max = _maximum(strain_max, diag.fiber_strain, xp)
            strain_min = _minimum(strain_min, diag.fiber_strain, xp)
            a_max = _maximum(a_max, attached, xp)
            a_min = _minimum(a_min, attached, xp)
        pa_sum = pa_sum + p_art
        qsys_sum = qsys_sum + diag.q_sys
        parked_sum = parked_sum + parked
        grad_max = _maximum(grad_max, diag.lvot_gradient_mmhg, xp)
        qmv_max = _maximum(qmv_max, diag.q_mv, xp)
        av_grad_max = _maximum(av_grad_max, diag.p_ven_mmhg - diag.p_lv_mmhg, xp)
        lengthen_max = _maximum(lengthen_max, diag.fiber_strain_rate_per_s, xp)

        if traces is not None:
            traces["time_s"].append(tau * p.period_s)
            traces["cavity_volume_ml"].append(cavity)
            traces["lv_pressure_mmhg"].append(diag.p_lv_mmhg)
            traces["arterial_pressure_mmhg"].append(p_art)
            traces["venous_pressure_mmhg"].append(diag.p_ven_mmhg)
            traces["aortic_flow_ml_per_s"].append(diag.q_av)
            traces["mitral_flow_ml_per_s"].append(diag.q_mv)
            traces["lvot_gradient_mmhg"].append(diag.lvot_gradient_mmhg)
            traces["stretch"].append(diag.stretch)
            traces["parked"].append(parked)
            traces["available"].append(available)
            traces["attached"].append(attached)
            traces["calcium_um"].append(diag.calcium_um)

        advance = tuple(
            y[i] + (dtau / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(n_state)
        )
        parked, available, attached, distortion, cavity, p_art, atp, work = advance

    # One extra evaluation at the closing instant of the beat: this is end-diastole.
    _, end_diag = _evaluate(1.0, parked, available, attached, distortion, cavity, p_art, p, xp)
    edp = end_diag.p_lv_mmhg

    summary = _BeatSummary(
        edv_ml=cavity,
        max_cavity_volume_ml=v_max,
        esv_ml=v_min,
        end_diastolic_pressure_mmhg=edp,
        peak_lv_pressure_mmhg=p_lv_max,
        systolic_arterial_mmhg=pa_max,
        diastolic_arterial_mmhg=pa_min,
        mean_arterial_mmhg=pa_sum / n_steps,
        mean_systemic_flow_ml_per_s=qsys_sum / n_steps,
        peak_lvot_gradient_mmhg=grad_max,
        peak_strain=strain_max,
        min_strain=strain_min,
        atp_per_head=atp,
        stroke_work_mmhg_ml=-work,
        peak_mitral_flow_ml_per_s=qmv_max,
        peak_transmitral_gradient_mmhg=av_grad_max,
        peak_lengthening_rate_per_s=lengthen_max,
        peak_attached=a_max,
        min_attached=a_min,
        mean_parked=parked_sum / n_steps,
        population_error=pop_err,
    )
    trace = None
    if traces is not None:
        trace = BeatTrace(**{k: np.asarray(v, dtype=float) for k, v in traces.items()})
    return (parked, available, attached, distortion, cavity, p_art), summary, trace


# --------------------------------------------------------------------------------------
# Backend-agnostic reduction helpers
# --------------------------------------------------------------------------------------


def _maximum(a: Numeric, b: Numeric, xp: Backend) -> Numeric:
    del xp
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.maximum(a, b)
    return a if a > b else b


def _minimum(a: Numeric, b: Numeric, xp: Backend) -> Numeric:
    del xp
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.minimum(a, b)
    return a if a < b else b


def _absolute(a: Numeric, xp: Backend) -> Numeric:
    del xp
    if isinstance(a, np.ndarray):
        return np.abs(a)
    return abs(a)


def _greater(a: Numeric, b: Numeric, xp: Backend) -> Numeric:
    del xp
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.greater(a, b)
    return a > b


def _select(condition: Numeric, if_true: Numeric, if_false: Numeric) -> Numeric:
    if isinstance(condition, np.ndarray):
        return np.where(condition, if_true, if_false)
    return if_true if condition else if_false


# --------------------------------------------------------------------------------------
# Steady-state driver
# --------------------------------------------------------------------------------------


def _iterate_to_steady_state(
    p: _Params,
    xp: Backend,
    constants: ModelConstants,
    record: bool,
) -> tuple[_BeatSummary, BeatTrace | None, bool, int]:
    state = _initial_state(p, xp)
    prev_edv: Numeric | None = None
    prev_map: Numeric | None = None
    converged = False
    beats_used = 0

    for beat in range(constants.max_beats):
        state, summary, _ = _run_beat(state, p, xp, constants.steps_per_beat, record=False)
        beats_used = beat + 1
        if prev_edv is not None and prev_map is not None:
            edv_delta = _absolute(summary.edv_ml - prev_edv, xp)
            map_delta = _absolute(summary.mean_arterial_mmhg - prev_map, xp)
            if _all_below(edv_delta, constants.steady_tol_ml) and _all_below(
                map_delta, constants.steady_tol_mmhg
            ):
                converged = True
                break
        prev_edv = summary.edv_ml
        prev_map = summary.mean_arterial_mmhg

    final_state, summary, trace = _run_beat(
        state, p, xp, constants.steps_per_beat, record=record
    )
    del final_state
    return summary, trace, converged, beats_used


def _all_below(value: Numeric, tolerance: float) -> bool:
    if isinstance(value, np.ndarray):
        return bool(np.all(value < tolerance))
    return bool(value < tolerance)


def simulate(
    measured: MeasuredGeometry,
    hidden: HiddenMaterial,
    loading: Loading,
    dose_mg_per_day: float = 0.0,
    constants: ModelConstants | None = None,
    record_trace: bool = False,
) -> BeatResult:
    """Solve one virtual patient to a steady-state beat. The public entry point.

    Args:
        measured: Chamber shape, treated as known exactly.
        hidden: Tissue material properties and drug clearance, treated as unknown.
        loading: Heart rate, stressed blood volume, systemic resistance.
        dose_mg_per_day: Maintained daily dose. Zero is the untreated baseline.
        constants: Population-fixed physiology and numerics. Defaults to
            :class:`~hcmtwin.parameters.ModelConstants`.
        record_trace: Whether to return the full within-beat waveforms. Off by default
            because storing them for a 5000-patient cohort would cost gigabytes.

    Returns:
        A :class:`BeatResult` for the beat *after* the steady-state test passed.
    """
    constants = constants or ModelConstants()
    p, phi_eff, concentration = _make_params(
        measured, hidden, loading, dose_mg_per_day, constants
    )
    summary, trace, converged, beats = _iterate_to_steady_state(p, SCALAR, constants, record_trace)
    return BeatResult(
        summary=summary,
        converged=converged,
        beats_used=beats,
        phi_effective=phi_eff,
        concentration_ng_per_ml=concentration,
        period_s=float(p.period_s),
        wall_volume_ml=float(p.wall_volume_ml),
        ref_cavity_volume_ml=float(p.ref_cavity_volume_ml),
        trace=trace,
    )


def simulate_cohort(
    wall_volume_ml: np.ndarray,
    ref_cavity_volume_ml: np.ndarray,
    phi_baseline: np.ndarray,
    a_pas_kpa: np.ndarray,
    b_pas: np.ndarray,
    ca50_ref_um: np.ndarray,
    clearance_l_per_h: np.ndarray,
    heart_rate_bpm: np.ndarray,
    total_blood_volume_ml: np.ndarray,
    systemic_resistance: np.ndarray,
    dose_mg_per_day: np.ndarray | float = 0.0,
    constants: ModelConstants | None = None,
) -> tuple[_BeatSummary, np.ndarray, np.ndarray, bool, int]:
    """Solve an entire cohort in lockstep. Same physics, NumPy instead of floats.

    Every argument is an array of the same length ``N``, one entry per virtual patient.
    All ``N`` patients advance through the same integration step together, so the cost per
    patient falls by roughly two orders of magnitude relative to looping
    :func:`simulate`. The steady-state test is applied to the whole cohort at once and
    iteration continues until the *slowest* patient converges, which wastes a little work
    on the fast ones and is far cheaper than bookkeeping a shrinking active set.

    Returns:
        The beat summary (arrays), the effective ``phi`` per patient, the steady-state
        concentration per patient, whether every patient converged, and the beat count.
    """
    constants = constants or ModelConstants()
    concentration = drug.steady_state_concentration_ng_per_ml(
        np.asarray(dose_mg_per_day, dtype=float), clearance_l_per_h
    )
    phi_eff = drug.effective_phi(
        phi_baseline, concentration, constants.drug_e_max, constants.drug_ec50_ng_per_ml
    )
    if np.any(phi_eff <= 0.0) or np.any(phi_eff >= 1.0):
        raise ValueError("effective phi left the open interval (0, 1)")

    p = _Params(
        period_s=60.0 / heart_rate_bpm,
        wall_volume_ml=wall_volume_ml,
        ref_cavity_volume_ml=ref_cavity_volume_ml,
        phi_eff=phi_eff,
        a_pas_kpa=a_pas_kpa,
        b_pas=b_pas,
        ca50_ref_um=ca50_ref_um,
        total_volume_ml=total_blood_volume_ml,
        r_sys=systemic_resistance,
        k_park_off=phi_eff * constants.k_park_tot_per_s,
        k_park_on=(1.0 - phi_eff) * constants.k_park_tot_per_s,
        ca_diast_um=constants.ca_diast_um,
        ca_peak_um=constants.ca_peak_um,
        ca_tau_r_s=constants.ca_tau_r_s,
        k_att=constants.k_att_per_s,
        k_det=constants.k_det_per_s,
        k_force_per_kpa=constants.k_force_per_kpa,
        hill_n=constants.hill_n,
        beta_len=constants.beta_len,
        t_ref_kpa=constants.t_ref_kpa,
        beta_overlap=constants.beta_overlap,
        overlap_max=constants.overlap_max,
        k_xb_per_s=constants.k_xb_per_s,
        xb_half=constants.xb_half,
        xb_max_gain=constants.xb_max_gain,
        c_art=constants.c_art_ml_per_mmhg,
        c_ven=constants.c_ven_ml_per_mmhg,
        r_av=constants.r_av_mmhg_s_per_ml,
        r_mv=constants.r_mv_mmhg_s_per_ml,
        valve_smooth=constants.valve_smooth_mmhg,
        obstruction_enabled=constants.obstruction_enabled,
        k_obs=constants.k_obs_mmhg_s2_cm4_per_ml2,
        a0_lvot=constants.a0_lvot_cm2,
        crowding_ref=constants.crowding_ref,
        lvot_exponent=constants.lvot_exponent,
        a_min_frac=constants.a_min_frac_lvot,
    )
    summary, _, converged, beats = _iterate_to_steady_state(p, ARRAY, constants, record=False)
    return summary, np.asarray(phi_eff), np.asarray(concentration), converged, beats
