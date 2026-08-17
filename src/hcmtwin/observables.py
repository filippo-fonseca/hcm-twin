"""What a clinic could actually measure, and a hard wall around what it could not.

Two dataclasses live here and they are not allowed to mix.

:class:`Observables`
    Quantities obtainable from a real patient with a real machine. Every field's
    docstring names the modality and points at its entry in
    ``docs/research/06_measurement_noise.md``, because the noise on these numbers is what
    ultimately decides whether the identifiability result is optimistic or pessimistic.

:class:`HiddenTruth`
    Quantities the *model* knows and a clinic cannot. These are ground truth for scoring
    an inference and must never appear in a predictor. ``tests/test_observables.py``
    asserts the two name sets are disjoint and that no hidden name reaches any feature
    matrix.

A note on "non-invasive"
------------------------

Not every field here is obtainable from a routine outpatient echo, and pretending
otherwise would quietly inflate the result. Each observable carries a
:class:`ObservableSpec` recording its modality and whether it is invasive, and the
identifiability analysis runs on the non-invasive routine subset by default. Left
ventricular end-diastolic pressure, for instance, is in the specification's observable
list and is included here, but it needs a catheter; ``e_over_e_prime`` is its
non-invasive stand-in and is the one the default feature set uses.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

from . import defaults as d
from .chamber import wall_thickness_cm
from .model import MMHG_ML_TO_JOULE, BeatResult, BeatSummary
from .parameters import HiddenMaterial, Loading, MeasuredGeometry

NoiseKind = Literal["absolute", "relative"]


@dataclass(frozen=True, slots=True)
class ObservableSpec:
    """Metadata for one measurable quantity."""

    units: str
    modality: str
    """The real-world instrument that would produce it."""

    noise_kind: NoiseKind
    noise_realistic: float
    """Standard deviation of measurement error at the pessimistic (routine two-dimensional
    echocardiography) level. Absolute in ``units``, or a fraction of the value."""

    noise_optimistic: float
    """Standard deviation at the optimistic (three-dimensional echocardiography, core-lab
    read, or averaged repeats) level."""

    invasive: bool = False
    routine: bool = True
    """Whether an ordinary outpatient study would produce it. Non-routine fields exist so
    the tie-breaker search can consider research measurements and cost them honestly."""

    derived: bool = False
    """True if the field is an algebraic function of other fields in this dataclass.

    Derived fields are informative for the sensitivity analysis but are excluded from the
    likelihood in the identifiability analysis: giving each of them independent noise
    would count the same measurement twice and make the posterior look tighter than the
    data supports."""


SPECS: dict[str, ObservableSpec] = {
    # --- Echo geometry ------------------------------------------------------------
    "edv_ml": ObservableSpec(
        "mL", "2D/3D echocardiography, biplane Simpson", "relative", 0.08, 0.05
    ),
    "esv_ml": ObservableSpec(
        "mL", "2D/3D echocardiography, biplane Simpson", "relative", 0.10, 0.06
    ),
    "stroke_volume_ml": ObservableSpec(
        "mL", "echocardiography (EDV - ESV)", "relative", 0.12, 0.07, derived=True
    ),
    "ejection_fraction": ObservableSpec("fraction", "echocardiography", "absolute", 0.050, 0.030),
    "wall_thickness_cm": ObservableSpec(
        "cm", "2D echocardiography, parasternal long axis", "absolute", 0.10, 0.06
    ),
    "lv_mass_g": ObservableSpec(
        "g", "echocardiography, linear or 3D method", "relative", 0.10, 0.06
    ),
    # --- Doppler ------------------------------------------------------------------
    "peak_lvot_gradient_mmhg": ObservableSpec(
        "mmHg",
        "continuous-wave Doppler, 4v^2",
        "relative",
        0.35,
        0.20,
    ),
    # --- Filling ------------------------------------------------------------------
    "end_diastolic_pressure_mmhg": ObservableSpec(
        "mmHg", "left heart catheterisation", "absolute", 2.0, 1.0, invasive=True, routine=False
    ),
    "e_over_e_prime": ObservableSpec(
        "dimensionless",
        "pulsed-wave Doppler mitral inflow over tissue-Doppler annular velocity (SURROGATE)",
        "relative",
        0.15,
        0.09,
    ),
    # --- Systolic function beyond ejection fraction -------------------------------
    "peak_strain_amplitude": ObservableSpec(
        "dimensionless",
        "speckle-tracking global longitudinal strain (SURROGATE: model fiber strain)",
        "relative",
        0.08,
        0.05,
    ),
    # --- Systemic -----------------------------------------------------------------
    "mean_arterial_pressure_mmhg": ObservableSpec("mmHg", "brachial cuff", "absolute", 5.0, 3.0),
    "cardiac_output_l_per_min": ObservableSpec(
        "L/min", "Doppler LVOT velocity-time integral x area x heart rate", "relative", 0.12, 0.08
    ),
    "heart_rate_bpm": ObservableSpec("bpm", "electrocardiogram", "absolute", 3.0, 2.0),
    # --- Engineered ---------------------------------------------------------------
    "thickness_to_cavity_ratio": ObservableSpec(
        "1/cm", "echocardiography (derived)", "relative", 0.12, 0.07, derived=True
    ),
    "stroke_volume_index_ml_per_m2": ObservableSpec(
        "mL/m^2",
        "echocardiography indexed to body surface area",
        "relative",
        0.12,
        0.07,
        derived=True,
    ),
    "stroke_work_j": ObservableSpec(
        "J",
        "pressure-volume catheterisation",
        "relative",
        0.15,
        0.10,
        invasive=True,
        routine=False,
    ),
    "atp_cost_per_stroke_work": ObservableSpec(
        "AU/J",
        "31P magnetic resonance spectroscopy with pressure-volume work (RESEARCH ONLY)",
        "relative",
        0.20,
        0.12,
        routine=False,
    ),
}
"""Registry of every observable, its modality, and its measurement error.

The noise figures are the load-bearing numbers of the whole project: the identifiability
conclusion is a statement about what these error bars permit. They are sourced in
``docs/research/06_measurement_noise.md``, and the analysis is run at both levels so a
reader can see how the conclusions degrade."""


@dataclass(frozen=True, slots=True)
class Observables:
    """Everything a clinic could obtain from one virtual patient at one operating point."""

    edv_ml: float
    """End-diastolic volume, mL. Echocardiography."""

    esv_ml: float
    """End-systolic volume, mL. Echocardiography."""

    stroke_volume_ml: float
    """Stroke volume, mL: EDV minus ESV."""

    ejection_fraction: float
    """Ejection fraction, dimensionless fraction (not percent). The measurement the whole
    project orbits: the trial interruption threshold is 0.50, and two patients can share
    this number at baseline and diverge completely under dosing."""

    wall_thickness_cm: float
    """End-diastolic wall thickness, cm, from the thick-walled spherical relation between
    wall volume and cavity volume, so the model reports what a sonographer measures."""

    lv_mass_g: float
    """Left-ventricular mass, g."""

    peak_lvot_gradient_mmhg: float
    """Peak instantaneous outflow tract gradient during ejection, mmHg. Continuous-wave
    Doppler. Clinically the primary efficacy measurement, and the observable most reactive
    to provocation, which makes it the obvious tie-breaker candidate. The analysis is
    what decides whether it actually is one."""

    end_diastolic_pressure_mmhg: float
    """Left-ventricular end-diastolic pressure, mmHg, taken at the instant of maximum
    cavity volume. INVASIVE: this needs a catheter. Included because the specification
    lists it; excluded from the default non-invasive feature set."""

    e_over_e_prime: float
    """Surrogate for the clinical early-filling-to-annular-velocity ratio, dimensionless.

    SURROGATE, and the word is not decoration. This model has no atrium, so there is no
    A wave and no atrial kick, and it has no explicit long-axis motion, so there is no true
    annular velocity.

    The numerator is the velocity implied by the peak atrioventricular pressure gradient
    through the simplified Bernoulli relation, which is exactly the physics behind a
    Doppler E wave, so it rises when filling pressure rises. The denominator is the peak
    myocardial lengthening rate scaled to a velocity, so it falls when relaxation is
    impaired. The single nominal constant is the long-axis length, and it sets the absolute
    scale only. Read this as a monotone stand-in for filling pressure whose *direction* is
    trustworthy and whose absolute value is not comparable to a clinical E/e'."""

    peak_strain_amplitude: float
    """Peak-to-trough representative fiber strain over the beat, dimensionless.

    SURROGATE for global longitudinal strain. Fiber strain in a one-fiber model is not
    longitudinal strain in a real ventricle; it is a whole-wall average with no regional
    or directional content."""

    mean_arterial_pressure_mmhg: float
    """Mean arterial pressure, mmHg."""

    cardiac_output_l_per_min: float
    """Cardiac output, L/min, as mean systemic flow."""

    heart_rate_bpm: float
    """Heart rate, beats per minute. An input, reported back because a real study records
    it and because every rate-dependent observable has to be read against it."""

    thickness_to_cavity_ratio: float
    """Wall thickness divided by end-diastolic volume, 1/cm. Engineered: a scalar for
    "thick wall around a small cavity", the geometry that lets a poorly filling ventricle
    report a reassuring ejection fraction."""

    stroke_volume_index_ml_per_m2: float
    """Stroke volume per square metre of body surface area, mL/m^2."""

    stroke_work_j: float
    """External stroke work, J: the area enclosed by the pressure-volume loop. INVASIVE."""

    atp_cost_per_stroke_work: float
    """ATP consumed per joule of external work, arbitrary units per joule.

    The numerator is the beat integral of the attachment flux scaled by wall volume, so it
    is proportional to true ATP consumption but not calibrated to it; only ratios between
    virtual patients are meaningful. Elevated cost per unit work is one of the three
    primary consequences of the HCM shift and has to *emerge* from raising ``phi``, which
    ``tests/test_validation.py`` checks."""


@dataclass(frozen=True, slots=True)
class HiddenTruth:
    """Ground truth the model knows and no clinic can measure. Never a predictor."""

    phi_baseline: float
    """Untreated myosin availability, dimensionless."""

    phi_effective: float
    """Availability after the drug, dimensionless."""

    a_pas_kpa: float
    """Passive stiffness scale, kPa."""

    b_pas: float
    """Passive stiffness exponent, dimensionless."""

    ca50_ref_um: float
    """Thin-filament calcium sensitivity at reference length, uM."""

    clearance_l_per_h: float
    """Apparent drug clearance, L/h."""

    concentration_ng_per_ml: float
    """Steady-state plasma concentration, ng/mL."""

    peak_attached_fraction: float
    """Peak fraction of myosin heads bound to actin during the beat, dimensionless."""

    contractile_reserve: float
    """Beat-average parked fraction, dimensionless: the heads still available to be
    recruited, and the mechanistic reading of "how much reserve is left". One of the three
    competing explanations for over-response is that this number was already low."""

    atp_per_head_per_beat: float
    """Attachment cycles per myosin head per beat, dimensionless."""


OBSERVABLE_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Observables))
HIDDEN_NAMES: tuple[str, ...] = tuple(f.name for f in fields(HiddenTruth))

NONINVASIVE_ROUTINE_NAMES: tuple[str, ...] = tuple(
    name for name in OBSERVABLE_NAMES if not SPECS[name].invasive and SPECS[name].routine
)
"""The default feature set: what an ordinary outpatient echo visit produces."""

UNINFORMATIVE_NAMES: frozenset[str] = frozenset({"heart_rate_bpm", "lv_mass_g"})
"""Observables that carry no information about the hidden parameters.

Heart rate is *reported* by any real study and is a genuine observable, but in this model
it is an input the analyst already knows exactly: it is a field of ``Loading``, it is
identical for every candidate parameter vector, and no posterior can learn anything from
it. Leaving it in the likelihood adds a term that is exactly zero for every sample.

It also breaks the surrogate's own diagnostics, which is how it was found: a coefficient of
determination is one minus the error variance over the *output* variance, and for a
constant output that denominator is zero. A handful of emulators reported an R-squared of
minus two hundred thousand, which looks like catastrophe and was floating-point noise
divided by nothing.

Left-ventricular mass is excluded for the same reason and it is the more interesting case.
It is a real measurement, and it is *the* measurement that defines the disease on imaging.
But in this project's framing wall volume is pinned as known, and mass is wall volume times
a density constant, so conditional on the geometry it is already determined and can
contribute nothing further about the tissue. That is not a defect of the measurement; it is
the whole point of separating shape from material."""

INDEPENDENT_NOISE_NAMES: tuple[str, ...] = tuple(
    name
    for name in NONINVASIVE_ROUTINE_NAMES
    if not SPECS[name].derived and name not in UNINFORMATIVE_NAMES
)
"""The subset used in the identifiability likelihood.

Derived fields are dropped so that one measurement is not counted several times under
different names, which would shrink the posterior for free. Uninformative fields are
dropped because they cannot constrain anything."""


def _safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, fallback: float = 0.0
) -> np.ndarray:
    """Elementwise division that returns ``fallback`` instead of a warning or an infinity.

    Extreme corners of the sampled parameter space can produce a virtual patient whose
    stroke work is essentially zero. That is a legitimate simulation result and it should
    not take the whole cohort run down with a divide-by-zero, but it also must not quietly
    become an infinity that poisons a Sobol index.
    """
    denom = np.asarray(denominator, dtype=float)
    return np.where(np.abs(denom) > 1e-12, np.asarray(numerator, dtype=float) / denom, fallback)


def observe_arrays(
    summary: BeatSummary,
    wall_volume_ml: np.ndarray,
    body_surface_area_m2: np.ndarray,
    heart_rate_bpm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorised reduction of a cohort beat summary to the clinical measurement set.

    The single-patient :func:`observe` is a thin wrapper around this, so the two paths
    cannot drift apart in the way two hand-written copies of the same arithmetic would.
    """
    s = summary
    edv = np.asarray(s.edv_ml, dtype=float)
    esv = np.asarray(s.esv_ml, dtype=float)
    stroke_volume = edv - esv
    stroke_work = np.asarray(s.stroke_work_mmhg_ml, dtype=float) * MMHG_ML_TO_JOULE
    thickness = wall_thickness_cm(edv, wall_volume_ml)

    # E wave: the Doppler velocity implied by the peak atrioventricular pressure gradient,
    # via the same simplified Bernoulli relation (dp = 4 v^2, v in m/s) a sonographer uses.
    # The factor of 100 converts m/s to cm/s.
    transmitral = np.maximum(
        np.asarray(s.peak_transmitral_gradient_mmhg, dtype=float),
        0.0,
    )
    e_wave_cm_per_s = 100.0 * np.sqrt(transmitral / 4.0)
    # e' wave: peak myocardial lengthening rate, scaled to an annular velocity.
    e_prime_cm_per_s = np.asarray(s.peak_lengthening_rate_per_s, dtype=float) * d.ANNULUS_LENGTH_CM
    atp_au = np.asarray(s.atp_per_head, dtype=float) * wall_volume_ml

    return {
        "edv_ml": edv,
        "esv_ml": esv,
        "stroke_volume_ml": stroke_volume,
        "ejection_fraction": _safe_divide(stroke_volume, edv),
        "wall_thickness_cm": thickness,
        "lv_mass_g": wall_volume_ml * d.MYOCARDIUM_DENSITY_G_PER_ML,
        "peak_lvot_gradient_mmhg": np.asarray(s.peak_lvot_gradient_mmhg, dtype=float),
        "end_diastolic_pressure_mmhg": np.asarray(
            s.end_diastolic_pressure_mmhg,
            dtype=float,
        ),
        "e_over_e_prime": _safe_divide(e_wave_cm_per_s, e_prime_cm_per_s),
        "peak_strain_amplitude": np.asarray(s.peak_strain - s.min_strain, dtype=float),
        "mean_arterial_pressure_mmhg": np.asarray(s.mean_arterial_mmhg, dtype=float),
        "cardiac_output_l_per_min": np.asarray(
            s.mean_systemic_flow_ml_per_s,
            dtype=float,
        )
        * 60.0
        / 1000.0,
        "heart_rate_bpm": np.asarray(heart_rate_bpm, dtype=float),
        "thickness_to_cavity_ratio": _safe_divide(thickness, edv),
        "stroke_volume_index_ml_per_m2": _safe_divide(stroke_volume, body_surface_area_m2),
        "stroke_work_j": stroke_work,
        "atp_cost_per_stroke_work": _safe_divide(atp_au, stroke_work),
    }


def physiological_mask(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Which rows of a cohort result describe a heart rather than a solver artefact.

    Extreme corners of the sampled parameter space, especially under an aggressive
    maneuver, produce solves that are arithmetically fine and physiologically meaningless:
    a ventricle that cannot fill at all, a negative cavity volume, a collapsed arterial
    pressure. These are excluded everywhere rather than trimmed, and the *count* is always
    reported, because a maneuver that is uninterpretable in a tenth of the patients it was
    proposed for is a finding about the maneuver.
    """
    values = np.column_stack([np.asarray(fields[name], dtype=float) for name in OBSERVABLE_NAMES])
    return (
        np.isfinite(values).all(axis=1)
        & (np.asarray(fields["edv_ml"]) > 5.0)
        & (np.asarray(fields["esv_ml"]) > 0.0)
        & (np.asarray(fields["stroke_volume_ml"]) > 1.0)
        & (np.asarray(fields["ejection_fraction"]) > 0.05)
        & (np.asarray(fields["ejection_fraction"]) < 0.99)
        & (np.asarray(fields["mean_arterial_pressure_mmhg"]) > 20.0)
    )


def observe(
    result: BeatResult,
    measured: MeasuredGeometry,
    loading: Loading,
) -> Observables:
    """Reduce a converged beat to the clinical measurement set."""
    one = np.ones(1)
    fields = observe_arrays(
        result.summary,
        wall_volume_ml=one * measured.wall_volume_ml,
        body_surface_area_m2=one * measured.body_surface_area_m2,
        heart_rate_bpm=one * loading.heart_rate_bpm,
    )
    return Observables(**{k: float(np.asarray(v).reshape(-1)[0]) for k, v in fields.items()})


def hidden_truth_arrays(
    summary: BeatSummary,
    phi_baseline: np.ndarray,
    phi_effective: np.ndarray,
    a_pas_kpa: np.ndarray,
    b_pas: np.ndarray,
    ca50_ref_um: np.ndarray,
    clearance_l_per_h: np.ndarray,
    concentration_ng_per_ml: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorised ground-truth block. Scoring only, never a predictor."""
    s = summary
    return {
        "phi_baseline": np.asarray(phi_baseline, dtype=float),
        "phi_effective": np.asarray(phi_effective, dtype=float),
        "a_pas_kpa": np.asarray(a_pas_kpa, dtype=float),
        "b_pas": np.asarray(b_pas, dtype=float),
        "ca50_ref_um": np.asarray(ca50_ref_um, dtype=float),
        "clearance_l_per_h": np.asarray(clearance_l_per_h, dtype=float),
        "concentration_ng_per_ml": np.asarray(concentration_ng_per_ml, dtype=float),
        "peak_attached_fraction": np.asarray(s.peak_attached, dtype=float),
        "contractile_reserve": np.asarray(s.mean_parked, dtype=float),
        "atp_per_head_per_beat": np.asarray(s.atp_per_head, dtype=float),
    }


def hidden_truth(result: BeatResult, hidden: HiddenMaterial) -> HiddenTruth:
    """Extract the ground-truth block. Scoring only."""
    s = result.summary
    return HiddenTruth(
        phi_baseline=hidden.phi_baseline,
        phi_effective=result.phi_effective,
        a_pas_kpa=hidden.a_pas_kpa,
        b_pas=hidden.b_pas,
        ca50_ref_um=hidden.ca50_ref_um,
        clearance_l_per_h=hidden.clearance_l_per_h,
        concentration_ng_per_ml=result.concentration_ng_per_ml,
        peak_attached_fraction=float(s.peak_attached),
        contractile_reserve=float(s.mean_parked),
        atp_per_head_per_beat=float(s.atp_per_head),
    )


def to_vector(observables: Observables, names: tuple[str, ...]) -> np.ndarray:
    """Pack selected observables into a feature vector.

    Raises if any requested name is not an observable. This is the choke point every
    predictor and every likelihood goes through, which is what makes the "no hidden field
    in a feature matrix" rule enforceable rather than aspirational.
    """
    unknown = [n for n in names if n not in OBSERVABLE_NAMES]
    if unknown:
        raise KeyError(
            f"not measurable, refusing to build a feature vector from: {unknown}. "
            f"Hidden ground-truth quantities are {HIDDEN_NAMES}."
        )
    return np.array([getattr(observables, n) for n in names], dtype=float)


def noise_sigma(names: tuple[str, ...], values: np.ndarray, level: str) -> np.ndarray:
    """Measurement-error standard deviations for a feature vector.

    Args:
        names: Observable names, in the same order as ``values``.
        values: The noiseless values, needed because several errors are proportional.
        level: ``"realistic"`` or ``"optimistic"``.

    Returns:
        Standard deviations in the same units as ``values``.
    """
    if level not in ("realistic", "optimistic"):
        raise ValueError(f"level must be 'realistic' or 'optimistic', got {level!r}")
    out = np.empty(len(names), dtype=float)
    for i, name in enumerate(names):
        spec = SPECS[name]
        magnitude = spec.noise_realistic if level == "realistic" else spec.noise_optimistic
        if spec.noise_kind == "absolute":
            out[i] = magnitude
        else:
            out[i] = magnitude * max(abs(float(values[i])), 1e-6)
    return out
