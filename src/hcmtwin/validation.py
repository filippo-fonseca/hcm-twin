"""The validation gates, as data, so the tests and the report cannot disagree.

``tests/test_validation.py`` asserts these; :func:`validation_table` renders them. Both
import the same ranges from here, so a target can never be tightened in one place and left
alone in the other.

Every range traces to a row in ``docs/research/05_validation_targets.md``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import defaults as d
from .drug import APPROVED_DOSE_LADDER_MG_PER_DAY
from .model import simulate
from .observables import Observables, observe
from .parameters import (
    HCM_GEOMETRY,
    HCM_MATERIAL,
    HEALTHY_GEOMETRY,
    HEALTHY_MATERIAL,
    RESTING_LOADING,
)

HEALTHY_GATES: dict[str, tuple[float, float, str]] = {
    "ejection_fraction": (0.55, 0.70, "fraction"),
    "edv_ml": (110.0, 130.0, "mL"),
    "stroke_volume_ml": (65.0, 75.0, "mL"),
    "peak_lv_pressure_mmhg": (110.0, 130.0, "mmHg"),
    "end_diastolic_pressure_mmhg": (5.0, 12.0, "mmHg"),
    "mean_arterial_pressure_mmhg": (85.0, 95.0, "mmHg"),
    "cardiac_output_l_per_min": (4.5, 5.5, "L/min"),
    "wall_thickness_cm": (0.60, 1.10, "cm"),
}
"""Healthy resting haemodynamics and geometry."""

EJECTION_REALISM_GATES: dict[str, tuple[float, float, str]] = {
    "peak_lvot_velocity_m_per_s": (0.7, 1.4, "m/s"),
    "ejection_duration_ms": (250.0, 340.0, "ms"),
    "peak_aortic_flow_ml_per_s": (350.0, 600.0, "mL/s"),
}
"""Added after the first calibration pass, which passed every haemodynamic gate while
ejecting the whole stroke volume in 154 ms at twice physiological peak flow. A model can
satisfy its checks and still be wrong, and these are the checks that would have caught
it."""


def _run(geometry, material, loading, dose=0.0, trace=False):  # type: ignore[no-untyped-def]
    result = simulate(geometry, material, loading, dose, record_trace=trace)
    return observe(result, geometry, loading), result


@dataclass(frozen=True)
class GateRow:
    gate: str
    quantity: str
    value: float
    units: str
    target: str
    passed: bool
    note: str = ""


def _range_row(
    gate: str, quantity: str, value: float, bounds: tuple[float, float, str], note: str = ""
) -> GateRow:
    low, high, units = bounds
    return GateRow(
        gate=gate,
        quantity=quantity,
        value=round(float(value), 4),
        units=units,
        target=f"{low:g} to {high:g}",
        passed=bool(low <= value <= high),
        note=note,
    )


def _bool_row(gate: str, quantity: str, passed: bool, note: str) -> GateRow:
    return GateRow(
        gate=gate,
        quantity=quantity,
        value=float(passed),
        units="boolean",
        target="true",
        passed=bool(passed),
        note=note,
    )


def _monotone(values: list[float], increasing: bool) -> bool:
    pairs = itertools.pairwise(values)
    return all((b > a) if increasing else (b < a) for a, b in pairs)


def _sweep(  # type: ignore[no-untyped-def]
    modifier: Callable[[float], object], factors: tuple[float, ...]
):
    return [_run(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, modifier(f))[0] for f in factors]


def validation_rows() -> pd.DataFrame:
    """Evaluate every gate and return the table. This is D2."""
    rows: list[GateRow] = []

    # --- Gate 1: healthy baseline -----------------------------------------------------
    healthy, healthy_result = _run(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, trace=True)
    for quantity, bounds in HEALTHY_GATES.items():
        value = (
            float(healthy_result.summary.peak_lv_pressure_mmhg)
            if quantity == "peak_lv_pressure_mmhg"
            else getattr(healthy, quantity)
        )
        rows.append(_range_row("healthy baseline", quantity, value, bounds))
    rows.append(
        _bool_row(
            "healthy baseline",
            "not obstructive",
            healthy.peak_lvot_gradient_mmhg < d.LVOT_OBSTRUCTIVE_THRESHOLD_MMHG,
            f"peak gradient {healthy.peak_lvot_gradient_mmhg:.1f} mmHg",
        )
    )
    rows.append(
        _bool_row(
            "healthy baseline",
            "steady state reached",
            healthy_result.converged,
            f"{healthy_result.beats_used} beats",
        )
    )

    # --- Ejection realism -------------------------------------------------------------
    trace = healthy_result.trace
    assert trace is not None
    flow = trace.aortic_flow_ml_per_s
    realism = {
        "peak_lvot_velocity_m_per_s": float(np.sqrt(healthy.peak_lvot_gradient_mmhg / 4.0)),
        "ejection_duration_ms": float(
            (flow > 5.0).sum() / len(flow) * healthy_result.period_s * 1000.0
        ),
        "peak_aortic_flow_ml_per_s": float(flow.max()),
    }
    for quantity, bounds in EJECTION_REALISM_GATES.items():
        rows.append(_range_row("ejection realism", quantity, realism[quantity], bounds))

    # --- Gate 2: Frank-Starling -------------------------------------------------------
    factors = (0.85, 0.95, 1.05, 1.15)
    preload = _sweep(lambda f: RESTING_LOADING.scaled(blood_volume_factor=f), factors)
    rows.append(
        _bool_row(
            "Frank-Starling",
            "EDV rises with preload",
            _monotone([o.edv_ml for o in preload], True),
            "EDV " + ", ".join(f"{o.edv_ml:.1f}" for o in preload) + " mL",
        )
    )
    rows.append(
        _bool_row(
            "Frank-Starling",
            "SV rises with preload",
            _monotone([o.stroke_volume_ml for o in preload], True),
            "SV " + ", ".join(f"{o.stroke_volume_ml:.1f}" for o in preload) + " mL",
        )
    )

    # --- Gate 3: afterload ------------------------------------------------------------
    afterload = _sweep(
        lambda f: RESTING_LOADING.scaled(resistance_factor=f), (0.85, 1.0, 1.15, 1.30)
    )
    rows.append(
        _bool_row(
            "afterload",
            "SV falls with afterload",
            _monotone([o.stroke_volume_ml for o in afterload], False),
            "SV " + ", ".join(f"{o.stroke_volume_ml:.1f}" for o in afterload) + " mL",
        )
    )
    rows.append(
        _bool_row(
            "afterload",
            "ESV rises with afterload",
            _monotone([o.esv_ml for o in afterload], True),
            "ESV " + ", ".join(f"{o.esv_ml:.1f}" for o in afterload) + " mL",
        )
    )

    # --- Gate 4: loop shape -----------------------------------------------------------
    volume, pressure = trace.cavity_volume_ml, trace.lv_pressure_mmhg
    span = float(volume.max() - volume.min())
    closes = abs(float(volume[0] - volume[-1])) < 0.02 * span
    signed_area = float(-np.trapezoid(pressure, volume))
    dv, dp = np.gradient(volume), np.gradient(pressure)
    quiet = np.abs(dv) < 0.002 * span
    rows.append(
        _bool_row(
            "loop shape",
            "loop closes",
            closes,
            f"gap {abs(float(volume[0] - volume[-1])):.3f} mL over {span:.1f} mL",
        )
    )
    rows.append(
        _bool_row(
            "loop shape",
            "counter-clockwise",
            signed_area > 0,
            f"stroke work {signed_area:.0f} mmHg*mL",
        )
    )
    rows.append(
        _bool_row(
            "loop shape",
            "isovolumic phases present",
            bool((quiet & (dp > 0)).sum() > 5 and (quiet & (dp < 0)).sum() > 5),
            f"{int((quiet & (dp > 0)).sum())} contraction, "
            f"{int((quiet & (dp < 0)).sum())} relaxation steps",
        )
    )

    # --- Gate 5: diastolic ------------------------------------------------------------
    from .chamber import cavity_pressure_mmhg, stretch_from_volume
    from .sarcomere import passive_stress_kpa

    volumes = np.linspace(65.0, 175.0, 40)
    passive = np.array(
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
    rows.append(
        _bool_row(
            "diastolic",
            "passive relation monotone",
            bool(np.all(np.diff(passive) > 0)),
            "at fixed volume",
        )
    )
    rows.append(
        _bool_row(
            "diastolic",
            "passive relation convex",
            bool(np.all(np.diff(passive, 2) > -1e-9)),
            "at fixed volume",
        )
    )
    edps = [
        _run(HEALTHY_GEOMETRY, replace(HEALTHY_MATERIAL, a_pas_kpa=a), RESTING_LOADING)[
            0
        ].end_diastolic_pressure_mmhg
        for a in (0.6, 0.9, 1.5, 2.5, 4.0)
    ]
    rows.append(
        _bool_row(
            "diastolic",
            "stiffer tissue raises filling pressure",
            _monotone(edps, True),
            "EDP " + ", ".join(f"{e:.1f}" for e in edps) + " mmHg",
        )
    )

    # --- Gate 6: HCM phenotype emerges ------------------------------------------------
    hcm, _ = _run(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING)
    benign, _ = _run(HCM_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING)
    checks: list[tuple[str, bool, str]] = [
        (
            "supranormal ejection fraction",
            hcm.ejection_fraction >= 0.70,
            f"{hcm.ejection_fraction:.3f} vs healthy {healthy.ejection_fraction:.3f}",
        ),
        (
            "reduced stroke volume",
            hcm.stroke_volume_ml < healthy.stroke_volume_ml,
            f"{hcm.stroke_volume_ml:.1f} vs {healthy.stroke_volume_ml:.1f} mL",
        ),
        (
            "elevated filling pressure",
            hcm.end_diastolic_pressure_mmhg > healthy.end_diastolic_pressure_mmhg,
            f"{hcm.end_diastolic_pressure_mmhg:.1f} vs "
            f"{healthy.end_diastolic_pressure_mmhg:.1f} mmHg",
        ),
        (
            "elevated E/e' surrogate",
            hcm.e_over_e_prime > healthy.e_over_e_prime,
            f"{hcm.e_over_e_prime:.1f} vs {healthy.e_over_e_prime:.1f}",
        ),
        (
            "reduced strain despite preserved EF",
            hcm.peak_strain_amplitude < healthy.peak_strain_amplitude,
            f"{hcm.peak_strain_amplitude:.3f} vs {healthy.peak_strain_amplitude:.3f}",
        ),
        (
            "elevated energy cost per unit work",
            hcm.atp_cost_per_stroke_work > 1.3 * healthy.atp_cost_per_stroke_work,
            f"{hcm.atp_cost_per_stroke_work / healthy.atp_cost_per_stroke_work:.2f}x healthy",
        ),
        (
            "hypertrophic on imaging",
            hcm.wall_thickness_cm >= d.TRIAL_MIN_WALL_THICKNESS_CM,
            f"{hcm.wall_thickness_cm:.2f} cm",
        ),
        (
            "disease needs material, not only geometry",
            bool(
                benign.end_diastolic_pressure_mmhg < hcm.end_diastolic_pressure_mmhg
                and benign.atp_cost_per_stroke_work < hcm.atp_cost_per_stroke_work
            ),
            f"thick wall with healthy tissue: EDP {benign.end_diastolic_pressure_mmhg:.1f} mmHg, "
            f"EF {benign.ejection_fraction:.3f}",
        ),
    ]
    for name, passed, note in checks:
        rows.append(_bool_row("HCM phenotype emerges", name, passed, note))

    # --- Gate 7: drug direction -------------------------------------------------------
    ladder = [
        _run(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, dose)[0]
        for dose in APPROVED_DOSE_LADDER_MG_PER_DAY
    ]
    rows.append(
        _bool_row(
            "drug direction",
            "dose lowers ejection fraction",
            _monotone([o.ejection_fraction for o in ladder], False),
            "EF " + ", ".join(f"{o.ejection_fraction:.3f}" for o in ladder),
        )
    )
    rows.append(
        _bool_row(
            "drug direction",
            "dose lowers outflow gradient",
            _monotone([o.peak_lvot_gradient_mmhg for o in ladder], False),
            "gradient " + ", ".join(f"{o.peak_lvot_gradient_mmhg:.0f}" for o in ladder) + " mmHg",
        )
    )

    # --- Gate 8: independent exposure-response comparison -----------------------------
    comparison = exposure_response_comparison(ladder)
    rows.append(
        GateRow(
            gate="exposure-response (PREDICTION, not fitted)",
            quantity="ejection-fraction change at the mid dose",
            value=round(comparison["ef_change_points_mid"], 2),
            units="percentage points",
            target="-20 to -0.5; trial reference -4.8 (SEQUOIA-HCM, aficamten)",
            passed=bool(-20.0 < comparison["ef_change_points_mid"] < -0.5),
            note="no model parameter was calibrated against a published dose-response curve",
        )
    )
    rows.append(
        GateRow(
            gate="exposure-response (PREDICTION, not fitted)",
            quantity="outflow-gradient change at the mid dose",
            value=round(comparison["gradient_change_mmhg_mid"], 1),
            units="mmHg",
            target="< -10; trial reference about -35 (SEQUOIA-HCM)",
            passed=bool(comparison["gradient_change_mmhg_mid"] < -10.0),
            note="reported for comparison, not asserted quantitatively",
        )
    )

    frame = pd.DataFrame([r.__dict__ for r in rows])
    return frame.rename(columns={"passed": "pass"})


def exposure_response_comparison(ladder: list[Observables] | None = None) -> dict[str, float]:
    """Simulated drug effect at the mid dose, for comparison with published trial data."""
    if ladder is None:
        ladder = [
            _run(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, dose)[0]
            for dose in APPROVED_DOSE_LADDER_MG_PER_DAY
        ]
    doses = list(APPROVED_DOSE_LADDER_MG_PER_DAY)
    baseline = ladder[doses.index(0.0)]
    mid = ladder[doses.index(d.DOSE_MID_MG_PER_DAY)]
    top = ladder[-1]
    return {
        "ef_baseline": baseline.ejection_fraction,
        "ef_mid_dose": mid.ejection_fraction,
        "ef_change_points_mid": 100.0 * (mid.ejection_fraction - baseline.ejection_fraction),
        "ef_change_points_top": 100.0 * (top.ejection_fraction - baseline.ejection_fraction),
        "gradient_change_mmhg_mid": (
            mid.peak_lvot_gradient_mmhg - baseline.peak_lvot_gradient_mmhg
        ),
    }


def validation_table() -> pd.DataFrame:
    """Alias with the name the pipeline uses."""
    return validation_rows()
