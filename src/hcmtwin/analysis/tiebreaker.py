"""D5: for each confounded pair, does a provocation break the tie, and by how much?

This is the payoff. The confounding map from D4 is descriptive: it says which parameter
combinations a resting study cannot see. This module tries to convert that into something
a cardiologist could act on: *resting measurements cannot separate these two mechanisms,
but adding this specific maneuver can, and here is the expected signal in clinical units
against the documented measurement variability.*

Four things are computed per confounded pair and per candidate maneuver.

**Does the invisible direction narrow?** This is the criterion the table ranks on. Take the
direction in the pair's plane that the resting study could not resolve, and measure how wide
the posterior still is along it once the maneuver's observables are added.

**Does the posterior correlation fall?** Reported, but *not* used to rank, and the reason is
worth stating because it is a trap. Correlation describes the shape of the uncertainty, not
its size. Adding information that constrains the same combination the data already knew
collapses the cloud further onto its ridge and pushes the correlation *up* while the patient
becomes better characterised. That is what happens here for two of the three pairs, and an
earlier version of this table scored maneuvers by correlation drop and would have called
those maneuvers harmful.

**Do the credible intervals shrink?** The per-parameter version of the first question.

**Is the discriminating signal real?** Take two parameter settings separated along the
*confounded direction* by an amount the resting study cannot distinguish, compute what each
would produce under the maneuver, and report the difference in the observable's own clinical
units next to that observable's measurement error.

**These last two answers disagree here, and the disagreement is the result.** The maneuver
produces a signal several measurement errors wide, yet the posterior barely narrows along
the direction that signal was supposed to resolve. That is not a contradiction and it is not
a numerical artefact: the emulator's error is at most a quarter of the measurement error, so
it is not the limit either. The signal is computed with the other three hidden parameters
held at their true values, and the posterior is not allowed that luxury. A change in the
maneuver's observables that is large when everything else is known can be *mimicked* by
adjusting the nuisance parameters, and an inference that must estimate all five at once
cannot tell the two explanations apart.

So the two columns answer different questions, and both belong in the table. The signal
column answers "if the rest of the tissue were characterised, would this maneuver separate
these two mechanisms?" The narrowing column answers "given that the rest of the tissue is
also unknown, does it?" The second is the operative one for a clinic, and it is the one the
table ranks on.

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
    vector of relative perturbations with zeros outside the pair, scaled to one posterior
    standard deviation along that direction.
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


def ridge_width(posterior: PosteriorResult, direction: np.ndarray) -> float:
    """Posterior standard deviation along a fixed direction, in relative units.

    **This, not the correlation, is the honest measure of whether a maneuver helped.**

    Posterior correlation measures the *shape* of the uncertainty, not its size, and the
    two move independently. Adding a maneuver's observables tightens the posterior, and if
    the extra information constrains the two parameters in the same combination it already
    knew, the cloud collapses further onto the ridge and the correlation goes *up* while
    the patient becomes better characterised. That is not a paradox and it is not a bug: it
    is what happened here for two of the three pairs examined, and a table that scored
    maneuvers by correlation drop would have called those maneuvers harmful.

    Projecting onto the direction that was invisible at baseline, and asking how wide the
    posterior still is along it, answers the question actually being asked: is the
    combination the resting study could not see any better resolved now?
    """
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    relative = posterior.chain / np.median(posterior.chain, axis=0)
    return float(np.std(relative @ unit))


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
    resting study cannot resolve. Everything is computed on the **exact** forward model, not
    the surrogate.

    **Important caveat, and the reason this number is optimistic.** The other three hidden
    parameters are held at their true values. A real inference does not know them, and the
    observable change this function reports can be partly mimicked by adjusting them. So
    read this as an upper bound: the signal available *if the rest of the tissue were
    characterised*. The posterior narrowing computed by :func:`ridge_width` is the version
    that accounts for nuisance compensation, and where the two disagree the narrowing is the
    one to believe.
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
        before_widths = baseline.credible_widths()
        baseline_correlation = baseline.correlation()

        # The posterior depends on (patient, maneuver) and not on which pair is being
        # examined, so it is sampled once per maneuver and every pair is read off it.
        # Sampling per pair would have multiplied the cost by the number of pairs for
        # identical chains.
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
                after_correlation = posterior.correlation()
                after_widths = posterior.credible_widths()
                usable = True
            except Exception as error:
                logger.warning(
                    "maneuver %s failed for patient %d: %s",
                    condition.key,
                    case.patient_id,
                    error,
                )
                after_correlation = np.full((len(HIDDEN_ORDER), len(HIDDEN_ORDER)), np.nan)
                after_widths = np.full(len(HIDDEN_ORDER), np.nan)
                usable = False

            for pair in pairs:
                before = abs(float(baseline_correlation[pair.index_a, pair.index_b]))
                after = abs(float(after_correlation[pair.index_a, pair.index_b]))
                best = None
                ridge_before = float("nan")
                ridge_after = float("nan")
                if usable:
                    try:
                        direction = confounded_direction(baseline, pair)
                        # Both widths are measured along the *same* direction, the one that
                        # was invisible at baseline, so the comparison is like for like.
                        ridge_before = ridge_width(baseline, direction)
                        ridge_after = ridge_width(posterior, direction)
                        signal = discriminating_signal(
                            case, direction, condition, noise_level, constants=constants
                        )
                        best = signal.iloc[0]
                    except Exception as error:
                        logger.warning(
                            "signal calculation failed for patient %d, %s, %s: %s",
                            case.patient_id,
                            pair,
                            condition.key,
                            error,
                        )

                detail_rows.append(
                    {
                        "patient_id": case.patient_id,
                        "pair": str(pair),
                        "parameter_a": pair.a,
                        "parameter_b": pair.b,
                        "maneuver": condition.provocation.name,
                        "noise_level": noise_level,
                        "usable": usable and best is not None,
                        "correlation_before": before,
                        "correlation_after": after,
                        "correlation_drop": before - after,
                        "ridge_width_before": ridge_before,
                        "ridge_width_after": ridge_after,
                        "ridge_shrinkage": (
                            1.0 - ridge_after / ridge_before if ridge_before > 0 else float("nan")
                        ),
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
    """The D5 table: one row per confounded pair, naming the best maneuver.

    The maneuver is chosen by how much it narrows the posterior *along the direction that
    was invisible at baseline*, not by how much it lowers the pair's correlation. Those two
    criteria disagree here, and the correlation one is wrong: see :func:`ridge_width`.
    """
    if detail.empty:
        return pd.DataFrame()

    usable = detail[detail["usable"]]
    records: list[dict[str, object]] = []
    for (pair_name, maneuver), group in usable.groupby(["pair", "maneuver"]):
        # The modal discriminating observable, and the signal reported *for that
        # observable only*. Taking the modal observable and the median signal
        # independently pairs a value with the wrong units whenever patients disagree on
        # which observable is best, which is exactly what an earlier version did.
        modes = group["best_observable"].mode()
        modal = modes.iat[0] if len(modes) else None
        matched = group[group["best_observable"] == modal]
        records.append(
            {
                "pair": pair_name,
                "maneuver": maneuver,
                "correlation_before": float(group["correlation_before"].median()),
                "correlation_after": float(group["correlation_after"].median()),
                "median_paired_drop": float(group["correlation_drop"].median()),
                "ridge_shrinkage": float(group["ridge_shrinkage"].median()),
                "ridge_width_before": float(group["ridge_width_before"].median()),
                "ridge_width_after": float(group["ridge_width_after"].median()),
                "ci_shrinkage_a": float(group["ci_shrinkage_a"].median()),
                "ci_shrinkage_b": float(group["ci_shrinkage_b"].median()),
                "modal_best_observable": modal,
                "median_abs_signal": float(np.median(np.abs(matched["best_signal"]))),
                "units": (matched["best_units"].iat[0] if len(matched) else None),
                "median_signal_to_noise": float(matched["best_signal_to_noise"].median()),
                "modal_share": float(len(matched) / max(len(group), 1)),
                "n_patients": len(group),
            }
        )
    grouped = pd.DataFrame(records)
    coverage = detail.groupby(["pair", "maneuver"])["usable"].mean().rename("usable_fraction")
    grouped = grouped.merge(coverage, on=["pair", "maneuver"])

    rows: list[dict[str, object]] = []
    for pair_name, group in grouped.groupby("pair"):
        # Ranked by how much the *invisible direction* narrows, not by how much the
        # correlation falls. See ridge_width for why the second is the wrong criterion.
        winner = group.loc[group["ridge_shrinkage"].idxmax()]
        rows.append(
            {
                "confounded_pair": pair_name,
                "best_maneuver": winner["maneuver"],
                "correlation_before": round(float(winner["correlation_before"]), 3),
                "correlation_after": round(float(winner["correlation_after"]), 3),
                "correlation_drop": round(float(winner["median_paired_drop"]), 3),
                "ridge_width_before": round(float(winner["ridge_width_before"]), 4),
                "ridge_width_after": round(float(winner["ridge_width_after"]), 4),
                "ridge_shrinkage": round(float(winner["ridge_shrinkage"]), 3),
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
                "patients_agreeing_on_observable": round(float(winner["modal_share"]), 2),
                "n_patients": int(winner["n_patients"]),
            }
        )
    return pd.DataFrame(rows).sort_values("ridge_shrinkage", ascending=False)


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

    table = pd.concat([invisible, observable_st, outcome_st], axis=1).reset_index(names="parameter")
    table["structurally_invisible_at_baseline"] = table["max_total_order_on_observables"] < 1e-3
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
