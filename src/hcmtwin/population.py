"""The virtual cohort: sampling, simulation, and the over-responder label.

Sampling uses SALib's Saltelli scheme so that Sobol indices come out of the same design
that generates the population, rather than requiring a second, separate set of runs. The
seed is fixed and logged; :func:`sample_population` is a pure function of it.

Three groups of inputs, and the distinction between them is the project's whole method:

**Measured geometry** is sampled and then treated as *known exactly*, because clinical
imaging measures shape well. **Hidden material** is sampled and then treated as *unknown*,
because clinical imaging measures material not at all. **Loading** is the operating
condition, and it is what a provocation maneuver changes.

Every patient is run at rest across the approved dose ladder, and at each stress maneuver
both untreated and at the mid dose. That is thirteen steady-state solves per patient, and
they are done as thirteen *cohort* solves rather than as a loop over patients, which is
the only reason a five-thousand-patient study finishes in minutes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import defaults as d
from .drug import APPROVED_DOSE_LADDER_MG_PER_DAY
from .model import simulate_cohort
from .observables import hidden_truth_arrays, observe_arrays, physiological_mask
from .parameters import Loading, ModelConstants
from .provocation import ALL_PROVOCATIONS, REST, Provocation

logger = logging.getLogger(__name__)

DEFAULT_SEED: int = 20260816
"""Fixed seed for the population draw. Logged on every run; changing it changes the
cohort, so it is reported in the writeup alongside the results."""


@dataclass(frozen=True, slots=True)
class Prior:
    """A uniform prior over one sampled input."""

    name: str
    low: float
    high: float
    group: str
    """``measured``, ``hidden``, or ``loading``."""

    units: str
    rationale: str


PRIORS: tuple[Prior, ...] = (
    # --- Measured geometry: sampled, then treated as known exactly ------------------
    Prior(
        "wall_volume_ml", 100.0, 300.0, "measured", "mL",
        "105-315 g of myocardium: spans a small normal ventricle through severe "
        "hypertrophy, straddling the 115 g/m^2 upper limit of normal for men.",
    ),
    Prior(
        "ref_cavity_volume_ml", 45.0, 80.0, "measured", "mL",
        "Unloaded cavity size. Deliberately NOT correlated with wall volume: the small "
        "cavity of HCM has to emerge from stiff tissue refusing to fill, not be assumed.",
    ),
    Prior(
        "body_surface_area_m2", 1.60, 2.20, "measured", "m^2",
        "Adult range; needed only to index volumes the way a clinical report does.",
    ),
    # --- Hidden material: sampled, then treated as unknown --------------------------
    Prior(
        "phi_baseline", 0.28, 0.62, "hidden", "dimensionless",
        "Unloaded myosin availability. The healthy reference is 0.35 and the HCM "
        "reference 0.55, so the range covers below-normal through severe "
        "hypercontractility.",
    ),
    Prior(
        "a_pas_kpa", 0.40, 5.50, "hidden", "kPa",
        "Passive stiffness scale, healthy reference 0.90. The upper end is the "
        "several-fold increase reported for fibrotic hypertrophic myocardium.",
    ),
    Prior(
        "b_pas", 6.0, 20.0, "hidden", "dimensionless",
        "Passive stiffness exponent, healthy reference 10.",
    ),
    Prior(
        "ca50_ref_um", 0.45, 0.80, "hidden", "uM",
        "Thin-filament calcium sensitivity. This is where thin-filament HCM mutations "
        "act, as distinct from thick-filament mutations which act on phi.",
    ),
    Prior(
        "clearance_l_per_h", 0.13, 1.10, "hidden", "L/h",
        "Apparent drug clearance. Spans CYP2C19 poor metaboliser (about 28% of the "
        "normal-metaboliser value) through rapid metaboliser, an eightfold range in "
        "exposure from the same pill.",
    ),
    # --- Loading --------------------------------------------------------------------
    Prior(
        "heart_rate_bpm", 52.0, 85.0, "loading", "bpm",
        "Resting adult range, mostly on background beta-blockade in this population.",
    ),
    Prior(
        "total_blood_volume_ml", 340.0, 450.0, "loading", "mL",
        "Stressed blood volume; the reference is 394 mL.",
    ),
    Prior(
        "systemic_resistance_mmhg_s_per_ml", 0.80, 1.35, "loading", "mmHg*s/mL",
        "Systemic vascular resistance; the reference is 1.04.",
    ),
)

PRIOR_NAMES: tuple[str, ...] = tuple(p.name for p in PRIORS)
HIDDEN_PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PRIORS if p.group == "hidden")
MEASURED_PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PRIORS if p.group == "measured")
LOADING_PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PRIORS if p.group == "loading")


def salib_problem() -> dict[str, object]:
    """The SALib problem definition matching :data:`PRIORS`."""
    return {
        "num_vars": len(PRIORS),
        "names": list(PRIOR_NAMES),
        "bounds": [[p.low, p.high] for p in PRIORS],
    }


def sample_population(n_base: int = 385, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Draw the virtual cohort with a Saltelli design.

    Args:
        n_base: SALib base sample size. The cohort size is ``n_base * (D + 2)`` with
            ``D = 11``, so the default 385 yields 5005 patients.
        seed: Random seed, logged.

    Returns:
        One row per virtual patient, columns named by :data:`PRIOR_NAMES`.
    """
    from SALib.sample import sobol as sobol_sample

    problem = salib_problem()
    logger.info(
        "sampling virtual population: n_base=%d seed=%d vars=%d",
        n_base,
        seed,
        len(PRIORS),
    )
    matrix = sobol_sample.sample(problem, n_base, calc_second_order=False, seed=seed)
    frame = pd.DataFrame(matrix, columns=list(PRIOR_NAMES))
    frame.insert(0, "patient_id", np.arange(len(frame), dtype=int))
    return frame


def _simulate_condition(
    params: pd.DataFrame,
    dose_mg_per_day: float,
    provocation: Provocation,
    constants: ModelConstants,
) -> pd.DataFrame:
    """Run the whole cohort at one dose under one maneuver."""
    base = Loading(
        heart_rate_bpm=1.0, total_blood_volume_ml=1.0, systemic_resistance_mmhg_s_per_ml=1.0
    )
    factors = provocation.apply(base)

    heart_rate = params["heart_rate_bpm"].to_numpy() * factors.heart_rate_bpm
    blood_volume = params["total_blood_volume_ml"].to_numpy() * factors.total_blood_volume_ml
    resistance = (
        params["systemic_resistance_mmhg_s_per_ml"].to_numpy()
        * factors.systemic_resistance_mmhg_s_per_ml
    )

    summary, phi_eff, concentration, converged, beats = simulate_cohort(
        wall_volume_ml=params["wall_volume_ml"].to_numpy(),
        ref_cavity_volume_ml=params["ref_cavity_volume_ml"].to_numpy(),
        phi_baseline=params["phi_baseline"].to_numpy(),
        a_pas_kpa=params["a_pas_kpa"].to_numpy(),
        b_pas=params["b_pas"].to_numpy(),
        ca50_ref_um=params["ca50_ref_um"].to_numpy(),
        clearance_l_per_h=params["clearance_l_per_h"].to_numpy(),
        heart_rate_bpm=heart_rate,
        total_blood_volume_ml=blood_volume,
        systemic_resistance=resistance,
        dose_mg_per_day=dose_mg_per_day,
        constants=constants,
    )
    if not converged:
        logger.warning(
            "cohort did not fully converge at dose=%.1f provocation=%s after %d beats",
            dose_mg_per_day,
            provocation.name,
            beats,
        )

    observables = observe_arrays(
        summary,
        wall_volume_ml=params["wall_volume_ml"].to_numpy(),
        body_surface_area_m2=params["body_surface_area_m2"].to_numpy(),
        heart_rate_bpm=heart_rate,
    )
    hidden = hidden_truth_arrays(
        summary,
        phi_baseline=params["phi_baseline"].to_numpy(),
        phi_effective=phi_eff,
        a_pas_kpa=params["a_pas_kpa"].to_numpy(),
        b_pas=params["b_pas"].to_numpy(),
        ca50_ref_um=params["ca50_ref_um"].to_numpy(),
        clearance_l_per_h=params["clearance_l_per_h"].to_numpy(),
        concentration_ng_per_ml=concentration,
    )

    frame = pd.DataFrame(
        {
            "patient_id": params["patient_id"].to_numpy(),
            "dose_mg_per_day": dose_mg_per_day,
            "provocation": provocation.name,
            **observables,
            **{f"true_{k}": v for k, v in hidden.items()},
        }
    )
    # Guard rails: a virtual patient whose solve produced a non-finite or physically
    # impossible result is flagged rather than silently averaged into a Sobol index. The
    # same check is used by the surrogate fit, so the two cannot diverge.
    frame["physiological"] = physiological_mask(observables)
    return frame


def simulate_population(
    params: pd.DataFrame,
    dose_ladder: tuple[float, ...] = APPROVED_DOSE_LADDER_MG_PER_DAY,
    mid_dose_mg_per_day: float = d.DOSE_MID_MG_PER_DAY,
    constants: ModelConstants | None = None,
) -> pd.DataFrame:
    """Run every patient at every condition.

    Conditions are the full dose ladder at rest, plus each stress maneuver untreated and
    at the mid dose. Returns a long frame: one row per (patient, dose, maneuver).
    """
    constants = constants or ModelConstants()
    frames: list[pd.DataFrame] = []

    for dose in dose_ladder:
        logger.info("simulating cohort: rest, dose=%.1f mg/day", dose)
        frames.append(_simulate_condition(params, dose, REST, constants))

    for provocation in ALL_PROVOCATIONS:
        if provocation.name == REST.name:
            continue
        for dose in (0.0, mid_dose_mg_per_day):
            logger.info(
                "simulating cohort: %s, dose=%.1f mg/day", provocation.name, dose
            )
            frames.append(_simulate_condition(params, dose, provocation, constants))

    return pd.concat(frames, ignore_index=True)


def label_over_responders(
    results: pd.DataFrame,
    mid_dose_mg_per_day: float = d.DOSE_MID_MG_PER_DAY,
    ef_threshold: float = d.EF_INTERRUPTION_THRESHOLD,
) -> pd.DataFrame:
    """Attach the outcome label: does this patient cross the ejection-fraction floor?

    ``over_responder`` is true when resting ejection fraction falls below the threshold at
    or below the mid dose. This is the model's analogue of the prespecified interruption
    criterion the trials use, and it is the thing the whole identifiability analysis is
    trying to predict from a *baseline* measurement.

    Returns one row per patient with the label, the baseline observables, and the
    dose-response slope.
    """
    rest = results[results["provocation"] == REST.name]
    at_or_below = rest[rest["dose_mg_per_day"] <= mid_dose_mg_per_day + 1e-9]
    treated = at_or_below[at_or_below["dose_mg_per_day"] > 0.0]

    crossed = (
        treated.assign(_below=treated["ejection_fraction"] < ef_threshold)
        .groupby("patient_id")["_below"]
        .any()
        .rename("over_responder")
    )

    baseline = rest[rest["dose_mg_per_day"] == 0.0].set_index("patient_id")
    mid = (
        rest[np.isclose(rest["dose_mg_per_day"], mid_dose_mg_per_day)]
        .set_index("patient_id")
    )

    out = baseline.copy()
    out["over_responder"] = crossed.reindex(out.index).fillna(False)
    out["ef_at_mid_dose"] = mid["ejection_fraction"].reindex(out.index)
    out["ef_drop_at_mid_dose"] = out["ejection_fraction"] - out["ef_at_mid_dose"]
    # The slope the project exists to predict: fractional ejection-fraction loss per
    # mg/day of maintained dose, over the interval a clinician would actually titrate.
    out["ef_slope_per_mg"] = out["ef_drop_at_mid_dose"] / mid_dose_mg_per_day
    out["all_conditions_physiological"] = (
        results.groupby("patient_id")["physiological"].all().reindex(out.index)
    )
    # Kept separate, because they fail for different reasons and discarding a patient for
    # the second reason would throw away a real finding. A resting solve fails only in a
    # genuinely unphysiological corner of parameter space. A *provoked* solve can fail
    # because a very stiff ventricle at 150 beats per minute cannot fill at all, which is
    # not a numerical problem: it is the maneuver being uninterpretable in exactly the
    # patients it was proposed for, and the tie-breaker analysis has to report that.
    out["rest_conditions_physiological"] = (
        rest.groupby("patient_id")["physiological"].all().reindex(out.index)
    )

    # Trial eligibility, applied as the pivotal trial applied it rather than as a
    # convenient filter. EXPLORER-HCM enrolled patients with an ejection fraction of at
    # least 55%, hypertrophy on imaging, and an outflow gradient of at least 30 mmHg at
    # rest or at least 50 mmHg provoked. Reproducing those criteria is what makes the
    # cohort's over-response rate comparable to the trial's; note that nothing in the
    # priors was tuned to that rate, so the comparison is a prediction.
    provoked = (
        results[
            (results["provocation"] == "preload_reduction")
            & (results["dose_mg_per_day"] == 0.0)
        ]
        .set_index("patient_id")["peak_lvot_gradient_mmhg"]
        .reindex(out.index)
    )
    out["provoked_gradient_mmhg"] = provoked
    out["trial_eligible"] = (
        out["rest_conditions_physiological"].fillna(False)
        & (out["ejection_fraction"] >= d.TRIAL_MIN_EF)
        & (out["wall_thickness_cm"] >= d.TRIAL_MIN_WALL_THICKNESS_CM)
        & (
            (out["peak_lvot_gradient_mmhg"] >= d.TRIAL_MIN_RESTING_GRADIENT_MMHG)
            | (provoked >= d.TRIAL_MIN_PROVOKED_GRADIENT_MMHG)
        )
    )
    return out.reset_index()


def summarise_cohort(labelled: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for the population, for the validation report and the log.

    Two populations are reported and they answer different questions. The *usable* set is
    everyone whose resting solve is physiological, and it is the set the sensitivity
    analysis runs on. The *trial-eligible* set applies the pivotal trial's enrolment
    criteria, and it is the only set whose over-response rate is comparable to a published
    rate.
    """
    usable = labelled[labelled["rest_conditions_physiological"].fillna(False)]
    eligible = labelled[labelled["trial_eligible"]]
    return {
        "n_sampled": float(len(labelled)),
        "n_usable": float(len(usable)),
        "usable_fraction": float(len(usable) / max(len(labelled), 1)),
        "n_trial_eligible": float(len(eligible)),
        "trial_eligible_fraction": float(len(eligible) / max(len(labelled), 1)),
        "all_conditions_physiological_fraction": float(
            labelled["all_conditions_physiological"].mean()
        ),
        "over_responder_rate_usable": float(usable["over_responder"].mean()),
        "over_responder_rate_eligible": float(eligible["over_responder"].mean()),
        "median_baseline_ef_eligible": float(eligible["ejection_fraction"].median()),
        "median_ef_drop_at_mid_dose_eligible": float(
            eligible["ef_drop_at_mid_dose"].median()
        ),
        "median_baseline_gradient_eligible_mmhg": float(
            eligible["peak_lvot_gradient_mmhg"].median()
        ),
        "median_wall_thickness_eligible_cm": float(eligible["wall_thickness_cm"].median()),
        "obstructive_fraction_eligible": float(
            (
                eligible["peak_lvot_gradient_mmhg"] >= d.LVOT_OBSTRUCTIVE_THRESHOLD_MMHG
            ).mean()
        ),
    }
