"""D3: Sobol sensitivity of every observable to every sampled parameter.

The first of the three questions, and the cheapest: **which parameters move which
measurements at all?** A hidden parameter that no observable responds to cannot be
recovered from any of them, by any inference method, ever. Establishing that costs one
pass over a design we already have to generate.

Total-order indices are the headline, not first-order. A parameter can have a negligible
first-order index and still dominate an observable through interactions, and in a
nonlinear coupled system that is common rather than exotic. First-order indices are
reported alongside, and the gap between them is itself informative: a large
``ST - S1`` means the parameter only matters in combination with others, which is an early
warning of the confounding the next stage looks for.

Plain correlation appears only as a supplementary panel. It hides nonlinearity, it is
signed where a variance decomposition is not, and using it as the headline metric would
make a non-monotone dependence look like independence.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..observables import OBSERVABLE_NAMES, SPECS
from ..population import PRIOR_NAMES, PRIORS, salib_problem

logger = logging.getLogger(__name__)

OUTCOME_NAMES: tuple[str, ...] = (
    "ef_at_mid_dose",
    "ef_drop_at_mid_dose",
    "ef_slope_per_mg",
)
"""Not observables: the things the project is trying to predict.

Included in the sensitivity sweep because knowing which hidden parameters drive the
*outcome* is what tells us which ones an estimator would need to recover. They are kept
namespaced away from :data:`~hcmtwin.observables.OBSERVABLE_NAMES` so they cannot be
mistaken for something a clinic measures."""


def _impute_for_sobol(values: np.ndarray, usable: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace unusable entries with the median of the usable ones.

    Saltelli analysis needs the full design in its original order and cannot tolerate gaps,
    so rows whose solve was non-physiological have to be filled rather than dropped. Median
    imputation biases the indices towards zero for whatever drove the failures, which is
    conservative in the right direction: it makes a parameter look *less* influential than
    it is rather than more. The count is returned and reported, and if it were ever a large
    fraction the result would have to be discarded rather than caveated.
    """
    filled = np.array(values, dtype=float, copy=True)
    bad = ~np.isfinite(filled) | ~usable
    if bad.any():
        filled[bad] = float(np.median(filled[~bad])) if (~bad).any() else 0.0
    return filled, int(bad.sum())


def sobol_indices(
    params: pd.DataFrame,
    quantities: pd.DataFrame,
    usable: np.ndarray | None = None,
) -> pd.DataFrame:
    """First- and total-order Sobol indices for every quantity against every parameter.

    Args:
        params: The Saltelli design, one row per virtual patient, columns
            :data:`~hcmtwin.population.PRIOR_NAMES`.
        quantities: One row per patient in the *same order*, columns to analyse.
        usable: Optional boolean mask of rows whose solve was physiological.

    Returns:
        Tidy frame with columns ``quantity``, ``parameter``, ``group``, ``S1``,
        ``S1_conf``, ``ST``, ``ST_conf``, ``n_imputed``.
    """
    from SALib.analyze import sobol as sobol_analyze

    problem = salib_problem()
    if len(params) != len(quantities):
        raise ValueError("params and quantities must have the same number of rows")
    mask = np.ones(len(params), dtype=bool) if usable is None else np.asarray(usable, bool)
    group_of = {prior.name: prior.group for prior in PRIORS}

    rows: list[dict[str, object]] = []
    for column in quantities.columns:
        values, n_imputed = _impute_for_sobol(quantities[column].to_numpy(), mask)
        if np.allclose(values, values[0]):
            logger.warning("quantity %s is constant across the design; skipping", column)
            continue
        result = sobol_analyze.analyze(
            problem, values, calc_second_order=False, print_to_console=False
        )
        for index, name in enumerate(PRIOR_NAMES):
            rows.append(
                {
                    "quantity": column,
                    "parameter": name,
                    "group": group_of[name],
                    "S1": float(result["S1"][index]),
                    "S1_conf": float(result["S1_conf"][index]),
                    "ST": float(result["ST"][index]),
                    "ST_conf": float(result["ST_conf"][index]),
                    "n_imputed": n_imputed,
                }
            )
    return pd.DataFrame(rows)


def spearman_panel(
    params: pd.DataFrame,
    quantities: pd.DataFrame,
    usable: np.ndarray | None = None,
) -> pd.DataFrame:
    """Supplementary rank-correlation panel. Never the headline metric.

    Signed, so it shows direction, which the variance decomposition cannot. Monotone-only,
    so it will report zero for a genuine but non-monotone dependence, which the variance
    decomposition will catch. The two are complementary and the second is the one to
    believe when they disagree.
    """
    from scipy.stats import spearmanr

    mask = np.ones(len(params), dtype=bool) if usable is None else np.asarray(usable, bool)
    rows: list[dict[str, object]] = []
    for column in quantities.columns:
        y = quantities[column].to_numpy()[mask]
        for name in PRIOR_NAMES:
            x = params[name].to_numpy()[mask]
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 10 or np.allclose(y[finite], y[finite][0]):
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(x[finite], y[finite])
            rows.append(
                {"quantity": column, "parameter": name, "spearman": float(rho),
                 "p_value": float(p_value)}
            )
    return pd.DataFrame(rows)


def run(
    params: pd.DataFrame,
    labelled: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run the whole D3 analysis on a labelled cohort.

    Args:
        params: The Saltelli design.
        labelled: Output of :func:`~hcmtwin.population.label_over_responders`, one row per
            patient in the same order as ``params``.

    Returns:
        ``{"sobol": ..., "spearman": ..., "matrix": ...}`` where ``matrix`` is the
        total-order index pivoted for the heatmap.
    """
    ordered = labelled.sort_values("patient_id").reset_index(drop=True)
    if not np.array_equal(ordered["patient_id"].to_numpy(), params["patient_id"].to_numpy()):
        raise ValueError("labelled cohort is not aligned with the Saltelli design")

    columns = [*OBSERVABLE_NAMES, *OUTCOME_NAMES]
    quantities = ordered[columns]
    usable = ordered["rest_conditions_physiological"].fillna(False).to_numpy(dtype=bool)
    logger.info(
        "Sobol analysis over %d parameters and %d quantities, %d/%d usable rows",
        len(PRIOR_NAMES),
        len(columns),
        int(usable.sum()),
        len(usable),
    )

    sobol = sobol_indices(params, quantities, usable)
    spearman = spearman_panel(params, quantities, usable)
    matrix = sobol.pivot(index="quantity", columns="parameter", values="ST")
    matrix = matrix.reindex(index=columns, columns=list(PRIOR_NAMES))
    return {"sobol": sobol, "spearman": spearman, "matrix": matrix}


def summarise(sobol: pd.DataFrame) -> pd.DataFrame:
    """Per hidden parameter: which observable sees it best, and how well.

    This is the table that decides whether the identifiability analysis has anything to
    work with. A hidden parameter whose best total-order index across every observable is
    near zero is invisible, and no posterior will recover it.
    """
    hidden = sobol[sobol["group"] == "hidden"]
    measurable = hidden[hidden["quantity"].isin(OBSERVABLE_NAMES)]
    rows: list[dict[str, object]] = []
    for name, group in measurable.groupby("parameter"):
        routine = group[group["quantity"].map(lambda q: SPECS[q].routine and not SPECS[q].invasive)]
        best = group.loc[group["ST"].idxmax()]
        best_routine = (
            routine.loc[routine["ST"].idxmax()] if len(routine) else None
        )
        rows.append(
            {
                "parameter": name,
                "best_observable": best["quantity"],
                "best_total_order": float(best["ST"]),
                "best_routine_observable": (
                    None if best_routine is None else best_routine["quantity"]
                ),
                "best_routine_total_order": (
                    float("nan") if best_routine is None else float(best_routine["ST"])
                ),
                "n_observables_above_0.05": int((group["ST"] > 0.05).sum()),
                "visible": bool(best["ST"] > 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("best_total_order", ascending=False)
