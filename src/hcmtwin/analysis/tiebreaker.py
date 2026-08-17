"""D5: for each confounded pair, does a provocation break the tie, and by how much?

This is the payoff. The confounding map from D4 is descriptive: it says which parameter
combinations a resting study cannot see. This module tries to convert that into something
a cardiologist could act on: *resting measurements cannot separate these two mechanisms,
but adding this specific maneuver can, and here is the expected signal in clinical units
against the documented measurement variability.*

Three things are computed per confounded pair and per candidate maneuver.

**Does the posterior correlation fall?** Re-run the inference with the baseline
observables *plus* the maneuver's observables and see whether the pair's correlation
drops.

**Do the credible intervals shrink?** A drop in correlation with no gain in precision
means the ridge rotated rather than resolved, which is not progress.

**Is the discriminating signal real?** This is the part that decides whether the proposal
survives contact with a clinic. Take two parameter settings separated along the *confounded
direction* by an amount the resting study cannot distinguish. Compute what each would
produce under the maneuver. The difference is the discriminating signal, and it is
reported in the observable's own clinical units next to that observable's measurement
error. A signal smaller than the error bar is not a test, however good the posterior
correlation looks.

A maneuver can also fail for a reason that has nothing to do with information: a very stiff
ventricle at 150 beats per minute may not fill at all, so the solve is unphysiological.
That is not a numerical inconvenience, it is the maneuver being uninterpretable in exactly
the patients it was proposed for, and it is reported as a coverage column rather than
dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..observables import INDEPENDENT_NOISE_NAMES, SPECS, noise_sigma
from ..parameters import ModelConstants
from .identifiability import (
    BASELINE,
    HIDDEN_ORDER,
    STRESS_CONDITIONS,
    Condition,
    PatientCase,
    PosteriorResult,
    forward,
    hidden_bounds,
    sample_posterior,
)
from .surrogate import Surrogate

logger = logging.getLogger(__name__)

MIN_SIGNAL_TO_NOISE: float = 1.0
"""Ratio of discriminating signal to measurement error at which a maneuver is called
useful. One standard deviation is a deliberately generous bar and is stated as such."""


@dataclass(frozen=True)
class Pair:
    """A parameter pair under investigation."""

    a: str
    b: str

    @property
    def index_a(self) -> int:
        return HIDDEN_ORDER.index(self.a)

    @property
    def index_b(self) -> int:
        return HIDDEN_ORDER.index(self.b)

    def __str__(self) -> str:
        return f"{self.a} / {self.b}"


def candidate_pairs(confounding: pd.DataFrame, top_k: int = 3) -> list[Pair]:
    """The pairs worth trying to separate.

    Every pair above the confounding threshold, plus the top few by correlation even if
    none crosses it. The second clause matters: a threshold is a convention, and a project
    that reported "nothing was confounded, so there was nothing to do" because the largest
    correlation was 0.78 rather than 0.80 would have learned nothing.
    """
    realistic = confounding[confounding["noise_level"] == "realistic"]
    ranked = realistic.sort_values("median_abs_correlation", ascending=False)
    chosen = ranked[ranked["confounded"]]
    if len(chosen) < top_k:
        chosen = ranked.head(top_k)
    return [Pair(row["parameter_a"], row["parameter_b"]) for _, row in chosen.iterrows()]


def confounded_direction(posterior: PosteriorResult, pair: Pair) -> np.ndarray:
    """The direction in the pair's plane along which the posterior is least constrained.

    The leading eigenvector of the two-by-two posterior covariance, in *relative* units so
    the two parameters are comparable whatever their scales. Returned as a full-length
    vector of relative perturbations with zeros outside the pair.
    """
    columns = np.array([pair.index_a, pair.index_b])
    samples = posterior.chain[:, columns]
    relative = samples / np.median(samples, axis=0)
    covariance = np.cov(relative, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    leading = eigenvectors[:, int(np.argmax(eigenvalues))]
    direction = np.zeros(len(HIDDEN_ORDER))
    direction[columns] = leading
    spread = float(np.sqrt(np.max(eigenvalues)))
    return direction * spread


def discriminating_signal(
    case: PatientCase,
    direction: np.ndarray,
    condition: Condition,
    noise_level: str,
    feature_names: tuple[str, ...] = INDEPENDENT_NOISE_NAMES,
    constants: ModelConstants | None = None,
) -> pd.DataFrame:
    """How far apart do two indistinguishable-at-rest patients look under this maneuver?

    The two settings are the truth displaced by plus and minus one posterior standard
    deviation along the confounded direction, which is by construction the displacement a
    resting study cannot resolve. Everything is computed on the **exact** forward model,
    not the surrogate, because this number is the clinical claim.
    """
    lows, highs = hidden_bounds()
    plus = np.clip(case.truth * (1.0 + direction), lows, highs)
    minus = np.clip(case.truth * (1.0 - direction), lows, highs)

    y_plus = forward(case, plus, condition, feature_names, constants)
    y_minus = forward(case, minus, condition, feature_names, constants)
    midpoint = 0.5 * (y_plus + y_minus)
    sigma = noise_sigma(feature_names, midpoint, noise_level)
    difference = y_plus - y_minus

    return pd.DataFrame(
        {
            "observable": list(feature_names),
            "units": [SPECS[name].units for name in feature_names],
            "value_low": y_minus,
            "value_high": y_plus,
            "signal": difference,
            "abs_signal": np.abs(difference),
            "measurement_sigma": sigma,
            "signal_to_noise": np.abs(difference) / sigma,
        }
    ).sort_values("signal_to_noise", ascending=False)


def run(
    cases: list[PatientCase],
    surrogates: dict[tuple[int, str], Surrogate],
    baseline_posteriors: dict[int, PosteriorResult],
    pairs: list[Pair],
    noise_level: str = "realistic",
    conditions: tuple[Condition, ...] = STRESS_CONDITIONS,
    seed: int = 999,
    constants: ModelConstants | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The full tie-breaker search.

    Returns:
        ``(detail, table)``. ``detail`` has one row per (patient, pair, maneuver) with the
        correlation before and after and the best discriminating observable. ``table`` is
        the D5 deliverable: one row per confounded pair naming the best maneuver.
    """
    detail_rows: list[dict[str, object]] = []

    for case_index, case in enumerate(cases):
        baseline = baseline_posteriors.get(case.patient_id)
        if baseline is None:
            continue
        for pair in pairs:
            direction = confounded_direction(baseline, pair)
            before = abs(float(baseline.correlation()[pair.index_a, pair.index_b]))
            before_widths = baseline.credible_widths()
            for condition in conditions:
                try:
                    posterior = sample_posterior(
                        case,
                        surrogates,
                        (BASELINE, condition),
                        noise_level,
                        seed=seed + 977 * case_index,
                        constants=constants,
                    )
                    after = abs(
                        float(posterior.correlation()[pair.index_a, pair.index_b])
                    )
                    after_widths = posterior.credible_widths()
                    signal = discriminating_signal(
                        case, direction, condition, noise_level, constants=constants
                    )
                    usable = True
                except Exception as error:  # noqa: BLE001
                    logger.warning(
                        "maneuver %s failed for patient %d: %s",
                        condition.key,
                        case.patient_id,
                        error,
                    )
                    after, usable = float("nan"), False
                    after_widths = np.full(len(HIDDEN_ORDER), np.nan)
                    signal = None

                best = None if signal is None else signal.iloc[0]
                detail_rows.append(
                    {
                        "patient_id": case.patient_id,
                        "pair": str(pair),
                        "parameter_a": pair.a,
                        "parameter_b": pair.b,
                        "maneuver": condition.provocation.name,
                        "noise_level": noise_level,
                        "usable": usable,
                        "correlation_before": before,
                        "correlation_after": after,
                        "correlation_drop": before - after,
                        "ci_shrinkage_a": (
                            1.0 - after_widths[pair.index_a] / before_widths[pair.index_a]
                        ),
                        "ci_shrinkage_b": (
                            1.0 - after_widths[pair.index_b] / before_widths[pair.index_b]
                        ),
                        "best_observable": None if best is None else best["observable"],
                        "best_units": None if best is None else best["units"],
                        "best_signal": float("nan") if best is None else float(best["signal"]),
                        "best_signal_to_noise": (
                            float("nan") if best is None else float(best["signal_to_noise"])
                        ),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    return detail, summarise(detail)


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    """The D5 table: one row per confounded pair, naming the best maneuver."""
    if detail.empty:
        return pd.DataFrame()

    grouped = (
        detail[detail["usable"]]
        .groupby(["pair", "maneuver"])
        .agg(
            correlation_before=("correlation_before", "median"),
            correlation_after=("correlation_after", "median"),
            correlation_drop=("correlation_drop", "median"),
            ci_shrinkage_a=("ci_shrinkage_a", "median"),
            ci_shrinkage_b=("ci_shrinkage_b", "median"),
            median_signal_to_noise=("best_signal_to_noise", "median"),
            n_patients=("patient_id", "count"),
        )
        .reset_index()
    )
    coverage = (
        detail.groupby(["pair", "maneuver"])["usable"].mean().rename("usable_fraction")
    )
    grouped = grouped.merge(coverage, on=["pair", "maneuver"])

    best_observable = (
        detail[detail["usable"]]
        .groupby(["pair", "maneuver"])["best_observable"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
        .rename("modal_best_observable")
    )
    grouped = grouped.merge(best_observable, on=["pair", "maneuver"])

    signal_units = (
        detail[detail["usable"]]
        .groupby(["pair", "maneuver"])
        .agg(
            median_abs_signal=("best_signal", lambda s: float(np.median(np.abs(s)))),
            units=("best_units", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        )
        .reset_index()
    )
    grouped = grouped.merge(signal_units, on=["pair", "maneuver"])

    rows: list[dict[str, object]] = []
    for pair_name, group in grouped.groupby("pair"):
        winner = group.loc[group["correlation_drop"].idxmax()]
        rows.append(
            {
                "confounded_pair": pair_name,
                "best_maneuver": winner["maneuver"],
                "correlation_before": round(float(winner["correlation_before"]), 3),
                "correlation_after": round(float(winner["correlation_after"]), 3),
                "correlation_drop": round(float(winner["correlation_drop"]), 3),
                "ci_shrinkage_a": round(float(winner["ci_shrinkage_a"]), 3),
                "ci_shrinkage_b": round(float(winner["ci_shrinkage_b"]), 3),
                "discriminating_observable": winner["modal_best_observable"],
                "expected_signal": round(float(winner["median_abs_signal"]), 4),
                "signal_units": winner["units"],
                "signal_to_noise": round(float(winner["median_signal_to_noise"]), 2),
                "exceeds_measurement_noise": bool(
                    winner["median_signal_to_noise"] >= MIN_SIGNAL_TO_NOISE
                ),
                "maneuver_usable_fraction": round(float(winner["usable_fraction"]), 3),
                "n_patients": int(winner["n_patients"]),
            }
        )
    return pd.DataFrame(rows).sort_values("correlation_before", ascending=False)


def structural_unidentifiability_note(
    fisher_table: pd.DataFrame,
    sobol: pd.DataFrame,
) -> pd.DataFrame:
    """Parameters that no pre-treatment maneuver can help with, and what would.

    A parameter can be unidentifiable for two very different reasons and the distinction
    is the difference between a hard problem and an impossible one.

    *Weakly identified*: the measurements respond to it, but not enough to beat the noise.
    A better maneuver, a better machine or a repeated study can help.

    *Structurally unidentified*: the measurements do not respond to it **at all** under the
    conditions available. No maneuver helps, because there is no signal to amplify. Drug
    clearance in an untreated patient is the clean example: it enters the model only
    through drug exposure, so before the first dose every derivative with respect to it is
    exactly zero and the Fisher information matrix is exactly singular in that direction.
    That is not a limitation of the measurement set; it is arithmetic.

    Naming which of the two applies is the most useful thing this analysis produces,
    because they call for completely different responses.
    """
    weights = [c for c in fisher_table.columns if c.startswith("invisible_weight_")]
    invisible = fisher_table[weights].mean().rename("mean_invisible_weight")
    invisible.index = [c.replace("invisible_weight_", "") for c in invisible.index]

    hidden_sobol = sobol[(sobol["group"] == "hidden")]
    observable_st = (
        hidden_sobol[~hidden_sobol["quantity"].str.startswith("ef_")]
        .groupby("parameter")["ST"]
        .max()
        .rename("max_total_order_on_observables")
    )
    outcome_st = (
        hidden_sobol[hidden_sobol["quantity"] == "ef_drop_at_mid_dose"]
        .set_index("parameter")["ST"]
        .rename("total_order_on_outcome")
    )

    table = pd.concat([invisible, observable_st, outcome_st], axis=1).reset_index(
        names="parameter"
    )
    table["structurally_invisible_at_baseline"] = (
        table["max_total_order_on_observables"] < 1e-3
    )
    table["matters_for_outcome"] = table["total_order_on_outcome"] > 0.10
    table["remedy"] = np.where(
        table["structurally_invisible_at_baseline"] & table["matters_for_outcome"],
        "no pre-treatment maneuver can help; needs genotype or a probe dose with a "
        "concentration measurement",
        np.where(
            table["matters_for_outcome"],
            "weakly identified; a maneuver may help, see the tie-breaker table",
            "does not drive the outcome",
        ),
    )
    return table.sort_values("total_order_on_outcome", ascending=False)
