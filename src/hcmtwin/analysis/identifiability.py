"""D4: which hidden parameters can a clinical measurement set recover, and which not.

This is the step that makes the project honest, and it is the one most likely to be
skipped, because it is the step that can return "you cannot get there from here".

Two analyses, cheap first.

**Local (Fisher information).** Finite-difference the observable vector with respect to
the hidden parameters at several representative operating points, weight each row by that
observable's measurement error, and eigen-decompose. Small eigenvalues mark directions in
parameter space the measurements cannot see. The eigenvector composition is the useful
part, because it names *which combination* is invisible rather than merely reporting that
something is.

Derivatives are taken with respect to the *logarithm* of each parameter, so the matrix is
scale-free and its eigenvectors are combinations of relative changes. Without that a
parameter measured in kPa and a dimensionless one cannot be compared, and the eigenvector
composition would be an artefact of units.

**Global (posterior sampling).** For each of about fifty virtual patients, treat their
observables plus realistic noise as data and recover the posterior over the five hidden
parameters by Markov chain Monte Carlo. The posterior correlation matrix is what the local
analysis approximates, and it is trustworthy where the local one is not: it accounts for
prior bounds, for nonlinearity across the whole prior box, and for the fact that a ridge
can be curved.

The output is a ranked list of parameter pairs whose absolute posterior correlation
exceeds a threshold. Those are the confounded pairs, and they are what
:mod:`~hcmtwin.analysis.tiebreaker` then tries to separate.

Everything is run at both noise levels from ``06_measurement_noise.md``, so a reader can
see how far the conclusions depend on measurement quality rather than on physics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..model import simulate, simulate_cohort
from ..observables import INDEPENDENT_NOISE_NAMES, noise_sigma, observe, to_vector
from ..parameters import HiddenMaterial, Loading, MeasuredGeometry, ModelConstants
from ..population import PRIORS
from ..provocation import ALL_PROVOCATIONS, REST, Provocation
from .surrogate import Surrogate, latin_hypercube

logger = logging.getLogger(__name__)

HIDDEN_ORDER: tuple[str, ...] = (
    "phi_baseline",
    "a_pas_kpa",
    "b_pas",
    "ca50_ref_um",
    "clearance_l_per_h",
)
"""The five hidden parameters, in a fixed order used by every matrix in this module."""

CONFOUNDING_THRESHOLD: float = 0.80
"""Absolute posterior correlation above which a pair is called confounded.

A convention, not a discovery. Reported alongside the full correlation distribution so a
reader can apply a different one."""

NOISE_LEVELS: tuple[str, ...] = ("optimistic", "realistic")


def hidden_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Prior box for the five hidden parameters, taken from the population priors."""
    by_name = {prior.name: prior for prior in PRIORS}
    lows = np.array([by_name[name].low for name in HIDDEN_ORDER], dtype=float)
    highs = np.array([by_name[name].high for name in HIDDEN_ORDER], dtype=float)
    return lows, highs


@dataclass(frozen=True)
class PatientCase:
    """One virtual patient held fixed while its hidden parameters are inferred."""

    patient_id: int
    geometry: MeasuredGeometry
    loading: Loading
    truth: np.ndarray
    """True hidden parameter values in :data:`HIDDEN_ORDER`. Ground truth for scoring."""


@dataclass(frozen=True)
class Condition:
    """A measurement occasion: a maneuver at a dose."""

    provocation: Provocation
    dose_mg_per_day: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.provocation.name}@{self.dose_mg_per_day:g}"


BASELINE = Condition(REST, 0.0)
STRESS_CONDITIONS: tuple[Condition, ...] = tuple(
    Condition(p, 0.0) for p in ALL_PROVOCATIONS if p.name != REST.name
)
"""Maneuvers performed *before* any drug is given.

Deliberately untreated. The clinical question is whether a pre-treatment measurement can
predict the dose-response slope; a maneuver performed on treatment would be answering a
different and much easier question."""

ALL_CONDITIONS: tuple[Condition, ...] = (BASELINE, *STRESS_CONDITIONS)


def select_cases(
    labelled: pd.DataFrame,
    params: pd.DataFrame,
    n_cases: int = 50,
    seed: int = 12345,
) -> list[PatientCase]:
    """Choose representative patients, stratified across wall thickness and stiffness.

    Stratified rather than random because a uniform draw over the sampled box
    over-represents the middle, and the interesting identifiability behaviour is not
    guaranteed to live there. Trial-eligible patients only: the question is about people
    who would actually be offered the drug.

    Args:
        labelled: Output of :func:`~hcmtwin.population.label_over_responders`.
        params: The Saltelli design, joined on ``patient_id`` so that the *exact* sampled
            geometry and loading are used rather than values reconstructed from
            observables.
        n_cases: How many patients to select.
        seed: Reproducibility.
    """
    merged = labelled.merge(params, on="patient_id", suffixes=("", "_design"))
    eligible = merged[merged["trial_eligible"]].copy()
    if len(eligible) == 0:
        raise ValueError("no trial-eligible patients in the cohort")
    if len(eligible) < n_cases:
        logger.warning(
            "only %d trial-eligible patients available, requested %d", len(eligible), n_cases
        )
        n_cases = len(eligible)

    n_thickness = min(5, max(2, n_cases // 8))
    n_stiffness = min(3, max(2, n_cases // 12))
    eligible["_thickness_bin"] = pd.qcut(
        eligible["wall_thickness_cm"], q=n_thickness, labels=False, duplicates="drop"
    )
    eligible["_stiffness_bin"] = pd.qcut(
        eligible["a_pas_kpa"], q=n_stiffness, labels=False, duplicates="drop"
    )
    per_cell = max(
        1, n_cases // max(1, eligible.groupby(["_thickness_bin", "_stiffness_bin"]).ngroups)
    )
    picked = pd.concat(
        [
            group.sample(min(len(group), per_cell), random_state=seed)
            for _, group in eligible.groupby(["_thickness_bin", "_stiffness_bin"])
        ],
        ignore_index=True,
    )
    if len(picked) > n_cases:
        picked = picked.sample(n_cases, random_state=seed).reset_index(drop=True)
    elif len(picked) < n_cases:
        remaining = eligible[~eligible["patient_id"].isin(picked["patient_id"])]
        extra = remaining.sample(min(len(remaining), n_cases - len(picked)), random_state=seed)
        picked = pd.concat([picked, extra], ignore_index=True)

    cases: list[PatientCase] = []
    for _, row in picked.iterrows():
        cases.append(
            PatientCase(
                patient_id=int(row["patient_id"]),
                geometry=MeasuredGeometry(
                    wall_volume_ml=float(row["wall_volume_ml"]),
                    ref_cavity_volume_ml=float(row["ref_cavity_volume_ml"]),
                    body_surface_area_m2=float(row["body_surface_area_m2"]),
                ),
                loading=Loading(
                    heart_rate_bpm=float(row["heart_rate_bpm_design"]),
                    total_blood_volume_ml=float(row["total_blood_volume_ml"]),
                    systemic_resistance_mmhg_s_per_ml=float(
                        row["systemic_resistance_mmhg_s_per_ml"]
                    ),
                ),
                truth=np.array([float(row[f"true_{name}"]) for name in HIDDEN_ORDER]),
            )
        )
    return sorted(cases, key=lambda c: c.patient_id)


def _material(values: np.ndarray) -> HiddenMaterial:
    return HiddenMaterial(**dict(zip(HIDDEN_ORDER, (float(v) for v in values), strict=True)))


def forward(
    case: PatientCase,
    hidden: np.ndarray,
    condition: Condition,
    feature_names: tuple[str, ...] = INDEPENDENT_NOISE_NAMES,
    constants: ModelConstants | None = None,
) -> np.ndarray:
    """One exact forward evaluation: hidden parameters to an observable vector."""
    loading = condition.provocation.apply(case.loading)
    result = simulate(
        case.geometry, _material(hidden), loading, condition.dose_mg_per_day, constants=constants
    )
    return to_vector(observe(result, case.geometry, loading), feature_names)


# =====================================================================================
# Local analysis: the Fisher information matrix
# =====================================================================================


@dataclass(frozen=True)
class FisherResult:
    patient_id: int
    condition: str
    noise_level: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    """Columns are eigenvectors, ordered with the eigenvalues (largest first)."""

    jacobian: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def condition_number(self) -> float:
        return float(self.eigenvalues[0] / max(self.eigenvalues[-1], 1e-300))

    def stiffest_invisible_direction(self) -> dict[str, float]:
        """Composition of the least-visible direction, as squared eigenvector weights."""
        vector = self.eigenvectors[:, -1]
        weights = vector**2 / np.sum(vector**2)
        return dict(zip(HIDDEN_ORDER, (float(w) for w in weights), strict=True))


def fisher_information(
    case: PatientCase,
    condition: Condition = BASELINE,
    noise_level: str = "realistic",
    relative_step: float = 0.02,
    feature_names: tuple[str, ...] = INDEPENDENT_NOISE_NAMES,
) -> FisherResult:
    """Noise-weighted Fisher information at one operating point, in log-parameters.

    Central differences with a 2% relative step. The step is a compromise: too small and
    the difference is swamped by the solver's own convergence tolerance, too large and it
    stops being a derivative. ``tests/test_analysis.py`` checks the result is stable
    when the step is halved.
    """
    base = forward(case, case.truth, condition, feature_names)
    sigma = noise_sigma(feature_names, base, noise_level)

    jacobian = np.zeros((len(feature_names), len(HIDDEN_ORDER)), dtype=float)
    for index in range(len(HIDDEN_ORDER)):
        up = case.truth.copy()
        down = case.truth.copy()
        up[index] *= 1.0 + relative_step
        down[index] *= 1.0 - relative_step
        # d(y) / d(log theta) = theta * dy/dtheta, approximated by the symmetric difference
        # over a multiplicative step, which is exactly the log-derivative.
        y_up = forward(case, up, condition, feature_names)
        y_down = forward(case, down, condition, feature_names)
        jacobian[:, index] = (y_up - y_down) / (2.0 * relative_step)

    weighted = jacobian / sigma[:, None]
    information = weighted.T @ weighted
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    order = np.argsort(eigenvalues)[::-1]
    return FisherResult(
        patient_id=case.patient_id,
        condition=condition.key,
        noise_level=noise_level,
        eigenvalues=eigenvalues[order],
        eigenvectors=eigenvectors[:, order],
        jacobian=jacobian,
        feature_names=feature_names,
    )


def fisher_table(results: list[FisherResult]) -> pd.DataFrame:
    """Eigenvalue spectra and the composition of the least-visible direction."""
    rows: list[dict[str, object]] = []
    for result in results:
        composition = result.stiffest_invisible_direction()
        row: dict[str, object] = {
            "patient_id": result.patient_id,
            "condition": result.condition,
            "noise_level": result.noise_level,
            "condition_number": result.condition_number,
        }
        for rank, value in enumerate(result.eigenvalues):
            row[f"eigenvalue_{rank}"] = float(value)
        for name, weight in composition.items():
            row[f"invisible_weight_{name}"] = weight
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================================
# Global analysis: surrogate-backed posterior sampling
# =====================================================================================


def build_surrogates(
    cases: list[PatientCase],
    conditions: tuple[Condition, ...] = ALL_CONDITIONS,
    n_design: int = 400,
    seed: int = 4242,
    feature_names: tuple[str, ...] = INDEPENDENT_NOISE_NAMES,
    constants: ModelConstants | None = None,
) -> tuple[dict[tuple[int, str], Surrogate], pd.DataFrame]:
    """Fit one local surrogate per (patient, condition).

    Every patient's design is evaluated in a single cohort solve per condition, stacking
    all patients and all design points into one array. That is the whole reason this is
    affordable: the cost is set by the number of *conditions*, not by the number of
    forward evaluations.
    """
    constants = constants or ModelConstants()
    lows, highs = hidden_bounds()
    design = latin_hypercube(lows, highs, n_design, seed=seed)
    log_design = np.log(design)

    n_cases = len(cases)
    total = n_cases * n_design
    logger.info(
        "building surrogates: %d patients x %d design points x %d conditions = %d solves",
        n_cases,
        n_design,
        len(conditions),
        total * len(conditions),
    )

    tiled = np.tile(design, (n_cases, 1))
    repeat = np.repeat(np.arange(n_cases), n_design)
    wall = np.array([c.geometry.wall_volume_ml for c in cases])[repeat]
    ref_cavity = np.array([c.geometry.ref_cavity_volume_ml for c in cases])[repeat]
    bsa = np.array([c.geometry.body_surface_area_m2 for c in cases])[repeat]

    surrogates: dict[tuple[int, str], Surrogate] = {}
    report_rows: list[dict[str, object]] = []

    for condition in conditions:
        factors = condition.provocation.apply(
            Loading(
                heart_rate_bpm=1.0, total_blood_volume_ml=1.0, systemic_resistance_mmhg_s_per_ml=1.0
            )
        )
        heart_rate = np.array([c.loading.heart_rate_bpm for c in cases])[repeat] * (
            factors.heart_rate_bpm
        )
        volume = np.array([c.loading.total_blood_volume_ml for c in cases])[repeat] * (
            factors.total_blood_volume_ml
        )
        resistance = (
            np.array([c.loading.systemic_resistance_mmhg_s_per_ml for c in cases])[repeat]
            * factors.systemic_resistance_mmhg_s_per_ml
        )

        summary, _, _, converged, beats = simulate_cohort(
            wall_volume_ml=wall,
            ref_cavity_volume_ml=ref_cavity,
            phi_baseline=tiled[:, 0],
            a_pas_kpa=tiled[:, 1],
            b_pas=tiled[:, 2],
            ca50_ref_um=tiled[:, 3],
            clearance_l_per_h=tiled[:, 4],
            heart_rate_bpm=heart_rate,
            total_blood_volume_ml=volume,
            systemic_resistance=resistance,
            dose_mg_per_day=condition.dose_mg_per_day,
            constants=constants,
        )
        if not converged:
            logger.warning(
                "surrogate design for %s did not fully converge in %d beats", condition.key, beats
            )

        from ..observables import observe_arrays, physiological_mask

        fields = observe_arrays(summary, wall, bsa, heart_rate)
        outputs = np.column_stack([fields[name] for name in feature_names])
        physiological = physiological_mask(fields)

        for index, case in enumerate(cases):
            slice_ = slice(index * n_design, (index + 1) * n_design)
            keep = physiological[slice_]
            # Fitting through unphysiological rows is what produced a held-out R^2 of -12
            # on the exercise condition in an earlier run: a handful of collapsed solves
            # dragged a smooth polynomial into nonsense across the whole box.
            if keep.sum() < 60:
                logger.warning(
                    "patient %d, %s: only %d/%d design points physiological; skipping",
                    case.patient_id,
                    condition.key,
                    int(keep.sum()),
                    n_design,
                )
                report_rows.append(
                    {
                        "patient_id": case.patient_id,
                        "condition": condition.key,
                        "min_r2": float("nan"),
                        "median_r2": float("nan"),
                        "worst_output": None,
                        "design_points_used": int(keep.sum()),
                        "design_points_dropped": int(n_design - keep.sum()),
                        "fitted": False,
                    }
                )
                continue
            surrogate = Surrogate(names=feature_names).fit(
                log_design[keep], outputs[slice_][keep], seed=seed + index
            )
            surrogates[(case.patient_id, condition.key)] = surrogate
            report = surrogate.error_report()
            report_rows.append(
                {
                    "patient_id": case.patient_id,
                    "condition": condition.key,
                    "min_r2": report["min_r2"],
                    "median_r2": report["median_r2"],
                    "worst_output": report["worst_output"],
                    "design_points_used": int(keep.sum()),
                    "design_points_dropped": int(n_design - keep.sum()),
                    "fitted": True,
                }
            )
        dropped = int((~physiological).sum())
        logger.info(
            "condition %s: surrogates fitted, %d/%d design solves unphysiological (%.1f%%)",
            condition.key,
            dropped,
            len(physiological),
            100.0 * dropped / len(physiological),
        )

    return surrogates, pd.DataFrame(report_rows)


def _log_posterior_factory(
    surrogates: list[Surrogate],
    data: list[np.ndarray],
    sigmas: list[np.ndarray],
    lows: np.ndarray,
    highs: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Vectorised log-posterior over a batch of walkers.

    Uniform priors on the natural scale, matching the population priors exactly, so the
    posterior can be read against the prior box without a change-of-variable correction.
    """
    log_lows, log_highs = np.log(lows), np.log(highs)

    def log_posterior(theta_log: np.ndarray) -> np.ndarray:
        theta_log = np.atleast_2d(theta_log)
        inside = np.all((theta_log >= log_lows) & (theta_log <= log_highs), axis=1)
        out = np.full(len(theta_log), -np.inf)
        if not inside.any():
            return out
        candidates = theta_log[inside]
        total = np.zeros(len(candidates))
        for surrogate, observed, sigma in zip(surrogates, data, sigmas, strict=True):
            predicted = surrogate.predict(candidates)
            residual = (predicted - observed) / sigma
            total += -0.5 * np.sum(residual**2, axis=1)
        # Uniform prior on the natural scale, sampled in log space: the Jacobian of the
        # transform contributes sum(log theta), which is added so the posterior is over
        # the natural-scale uniform prior rather than a log-uniform one.
        total += np.sum(candidates, axis=1)
        out[inside] = total
        return out

    return log_posterior


@dataclass(frozen=True)
class PosteriorResult:
    patient_id: int
    conditions: tuple[str, ...]
    noise_level: str
    chain: np.ndarray
    """``(n_samples, 5)`` natural-scale posterior samples."""

    truth: np.ndarray
    acceptance: float

    def correlation(self) -> np.ndarray:
        return np.corrcoef(self.chain, rowvar=False)

    def credible_widths(self, mass: float = 0.90) -> np.ndarray:
        """Width of the central credible interval per parameter, relative to the truth."""
        low = np.percentile(self.chain, 100.0 * (1.0 - mass) / 2.0, axis=0)
        high = np.percentile(self.chain, 100.0 * (1.0 + mass) / 2.0, axis=0)
        return (high - low) / np.maximum(self.truth, 1e-12)


def sample_posterior(
    case: PatientCase,
    surrogates: dict[tuple[int, str], Surrogate],
    conditions: tuple[Condition, ...],
    noise_level: str,
    seed: int,
    feature_names: tuple[str, ...] = INDEPENDENT_NOISE_NAMES,
    n_walkers: int = 40,
    n_steps: int = 3000,
    n_burn: int = 1200,
    thin: int = 6,
    constants: ModelConstants | None = None,
) -> PosteriorResult:
    """Recover the posterior over hidden parameters from noisy observables.

    The synthetic data come from the **exact** forward model, never from the surrogate, so
    that surrogate error acts as model misspecification exactly as it would in a real
    application rather than cancelling out.
    """
    import emcee

    rng = np.random.default_rng(seed)
    lows, highs = hidden_bounds()

    used: list[Surrogate] = []
    data: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    for condition in conditions:
        truth_vector = forward(case, case.truth, condition, feature_names, constants)
        sigma = noise_sigma(feature_names, truth_vector, noise_level)
        observed = truth_vector + rng.normal(0.0, sigma)
        surrogate = surrogates[(case.patient_id, condition.key)]
        # Surrogate error enters the likelihood in quadrature, so an imprecise emulator
        # widens the posterior rather than corrupting it.
        inflated = np.sqrt(sigma**2 + surrogate.holdout_rmse**2)
        used.append(surrogate)
        data.append(observed)
        sigmas.append(inflated)

    log_posterior = _log_posterior_factory(used, data, sigmas, lows, highs)

    log_lows, log_highs = np.log(lows), np.log(highs)
    start = np.log(case.truth)[None, :] + 0.05 * rng.normal(size=(n_walkers, len(HIDDEN_ORDER)))
    start = np.clip(start, log_lows + 1e-6, log_highs - 1e-6)

    sampler = emcee.EnsembleSampler(n_walkers, len(HIDDEN_ORDER), log_posterior, vectorize=True)
    sampler.run_mcmc(start, n_steps, progress=False)
    chain_log = sampler.get_chain(discard=n_burn, thin=thin, flat=True)
    return PosteriorResult(
        patient_id=case.patient_id,
        conditions=tuple(c.key for c in conditions),
        noise_level=noise_level,
        chain=np.exp(chain_log),
        truth=case.truth,
        acceptance=float(np.mean(sampler.acceptance_fraction)),
    )


def confounding_table(posteriors: list[PosteriorResult]) -> pd.DataFrame:
    """Ranked parameter pairs by median absolute posterior correlation across patients."""
    rows: list[dict[str, object]] = []
    by_level: dict[str, list[np.ndarray]] = {}
    for posterior in posteriors:
        by_level.setdefault(posterior.noise_level, []).append(posterior.correlation())

    for level, matrices in by_level.items():
        stacked = np.stack(matrices)
        for i in range(len(HIDDEN_ORDER)):
            for j in range(i + 1, len(HIDDEN_ORDER)):
                values = stacked[:, i, j]
                rows.append(
                    {
                        "noise_level": level,
                        "parameter_a": HIDDEN_ORDER[i],
                        "parameter_b": HIDDEN_ORDER[j],
                        "median_abs_correlation": float(np.median(np.abs(values))),
                        "mean_correlation": float(np.mean(values)),
                        "q25_abs": float(np.percentile(np.abs(values), 25)),
                        "q75_abs": float(np.percentile(np.abs(values), 75)),
                        "n_patients": len(values),
                        "fraction_above_threshold": float(
                            np.mean(np.abs(values) > CONFOUNDING_THRESHOLD)
                        ),
                        "confounded": bool(np.median(np.abs(values)) > CONFOUNDING_THRESHOLD),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["noise_level", "median_abs_correlation"], ascending=[True, False]
    )


def recovery_table(posteriors: list[PosteriorResult]) -> pd.DataFrame:
    """How well each parameter is recovered: credible-interval width and bias."""
    rows: list[dict[str, object]] = []
    for posterior in posteriors:
        widths = posterior.credible_widths()
        medians = np.median(posterior.chain, axis=0)
        for index, name in enumerate(HIDDEN_ORDER):
            rows.append(
                {
                    "noise_level": posterior.noise_level,
                    "patient_id": posterior.patient_id,
                    "parameter": name,
                    "truth": float(posterior.truth[index]),
                    "posterior_median": float(medians[index]),
                    "relative_bias": float(
                        (medians[index] - posterior.truth[index]) / posterior.truth[index]
                    ),
                    "relative_ci90_width": float(widths[index]),
                }
            )
    return pd.DataFrame(rows)


def recovery_summary(recovery: pd.DataFrame) -> pd.DataFrame:
    """Per parameter and noise level: is it recoverable at all?

    ``relative_ci90_width`` near 2.0 or above means the 90% credible interval spans
    essentially the whole prior box: the data said nothing.
    """
    grouped = (
        recovery.groupby(["noise_level", "parameter"])
        .agg(
            median_ci90_width=("relative_ci90_width", "median"),
            median_abs_bias=("relative_bias", lambda s: float(np.median(np.abs(s)))),
            n=("patient_id", "count"),
        )
        .reset_index()
    )
    grouped["recoverable"] = grouped["median_ci90_width"] < 0.60
    return grouped.sort_values(["noise_level", "median_ci90_width"])
