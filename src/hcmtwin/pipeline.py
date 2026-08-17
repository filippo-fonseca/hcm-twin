"""One command reproduces every figure, and every figure ships with its CSV.

Each stage writes its artifacts into ``results/`` and later stages read them from disk, so
a run can be resumed, a single stage can be re-run, and the expensive parts are not
repeated by accident. Every stage records its seed and its wall-clock time in
``results/manifest.json``.

The stages, in dependency order:

``validate``       the Section 7 gates, as a table (D2)
``population``     the virtual cohort and its outcome labels
``sensitivity``    Sobol indices and the sensitivity matrix (D3)
``identifiability``  Fisher information, posteriors, the confounding map (D4)
``tiebreaker``     the provocation search and its table (D5)
``explorer``       the self-contained interactive page (D6)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import defaults as d
from .analysis import identifiability as idn
from .analysis import sensitivity as sens
from .analysis import tiebreaker as tb
from .drug import APPROVED_DOSE_LADDER_MG_PER_DAY
from .model import simulate
from .observables import observe
from .parameters import (
    HCM_GEOMETRY,
    HCM_MATERIAL,
    HEALTHY_GEOMETRY,
    HEALTHY_MATERIAL,
    RESTING_LOADING,
)
from .population import (
    DEFAULT_SEED,
    label_over_responders,
    sample_population,
    simulate_population,
    summarise_cohort,
)
from .viz import matrices, pvloop

logger = logging.getLogger(__name__)

RESULTS = Path("results")


@dataclass
class Config:
    """Everything that changes between a smoke run and the real one."""

    results_dir: Path = RESULTS
    population_n_base: int = 385
    """Saltelli base size; the cohort is ``n_base * 13``. 385 gives 5005 patients."""

    seed: int = DEFAULT_SEED
    n_identifiability_cases: int = 50
    n_surrogate_design: int = 400
    mcmc_steps: int = 3000
    mcmc_burn: int = 1200
    force: bool = False
    timings: dict[str, float] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir / name


def _stage(config: Config, name: str, outputs: list[str]):  # type: ignore[no-untyped-def]
    """Decide whether a stage needs to run, and time it if it does."""
    existing = all((config.results_dir / o).exists() for o in outputs)
    if existing and not config.force:
        logger.info("stage %s: artifacts present, skipping (use --force to rebuild)", name)
        return False
    logger.info("stage %s: running", name)
    return True


def _record(config: Config, name: str, started: float) -> None:
    config.timings[name] = round(time.perf_counter() - started, 2)
    logger.info("stage %s: done in %.1f s", name, config.timings[name])


# =====================================================================================
# D2: validation
# =====================================================================================


def run_validation(config: Config) -> pd.DataFrame:
    """Generate the validation table: every Section 7 gate, pass or fail, explicitly."""
    started = time.perf_counter()
    from .validation import validation_rows

    table = validation_rows()
    table.to_csv(config.path("validation_table.csv"), index=False)
    config.path("validation_table.md").write_text(_markdown_table(table), encoding="utf-8")
    _record(config, "validate", started)
    return table


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body]) + "\n"


# =====================================================================================
# Population
# =====================================================================================


def run_population(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample and simulate the virtual cohort, then label the over-responders."""
    params_path = config.path("population_params.csv.gz")
    labelled_path = config.path("population_labelled.csv.gz")
    if not _stage(config, "population", [params_path.name, labelled_path.name]):
        return pd.read_csv(params_path), pd.read_csv(labelled_path)

    started = time.perf_counter()
    params = sample_population(n_base=config.population_n_base, seed=config.seed)
    results = simulate_population(params)
    labelled = label_over_responders(results)

    params.to_csv(params_path, index=False)
    labelled.to_csv(labelled_path, index=False)
    results.to_csv(config.path("population_conditions.csv.gz"), index=False)

    summary = summarise_cohort(labelled)
    summary["seed"] = float(config.seed)
    summary["n_base"] = float(config.population_n_base)
    config.path("population_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("cohort summary: %s", json.dumps(summary, indent=2))
    _record(config, "population", started)
    return params, labelled


# =====================================================================================
# D3: sensitivity
# =====================================================================================


def run_sensitivity(config: Config, params: pd.DataFrame, labelled: pd.DataFrame) -> dict[str, Any]:
    """Sobol indices, the sensitivity matrix figure, and the visibility summary."""
    started = time.perf_counter()
    output = sens.run(params, labelled)
    output["sobol"].to_csv(config.path("sensitivity_sobol.csv"), index=False)
    output["spearman"].to_csv(config.path("sensitivity_spearman.csv"), index=False)
    summary = sens.summarise(output["sobol"])
    summary.to_csv(config.path("sensitivity_summary.csv"), index=False)

    inversion = sens.visibility_versus_importance(output["sobol"], summary)
    inversion.to_csv(config.path("visibility_vs_importance.csv"), index=False)
    stats = sens.inversion_statistics(inversion)
    config.path("visibility_vs_importance_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        "visibility vs importance: spearman rho = %+.2f over %d hidden parameters",
        stats["spearman_any"],
        int(stats["n_parameters"]),
    )

    matrices.plot_sensitivity_matrix(output["matrix"], config.path("fig_sensitivity_matrix.png"))
    hidden_matrix = output["matrix"][list(idn.HIDDEN_ORDER)]
    matrices.plot_sensitivity_matrix(
        hidden_matrix,
        config.path("fig_sensitivity_hidden.png"),
        title="Total-order Sobol indices: hidden parameters only",
        subtitle="The five quantities the analysis is trying to recover",
    )
    _record(config, "sensitivity", started)
    output["summary"] = summary
    output["inversion"] = inversion
    return output


# =====================================================================================
# D4: identifiability
# =====================================================================================


def run_identifiability(
    config: Config, params: pd.DataFrame, labelled: pd.DataFrame
) -> dict[str, Any]:
    """Fisher information, surrogates, posteriors, the confounding map."""
    started = time.perf_counter()
    cases = idn.select_cases(
        labelled, params, n_cases=config.n_identifiability_cases, seed=config.seed + 1
    )
    logger.info("identifiability: %d representative patients", len(cases))

    fisher = [
        idn.fisher_information(case, idn.BASELINE, level)
        for level in idn.NOISE_LEVELS
        for case in cases
    ]
    fisher_table = idn.fisher_table(fisher)
    fisher_table.to_csv(config.path("fisher_table.csv"), index=False)

    surrogates, report = idn.build_surrogates(
        cases,
        conditions=idn.ALL_CONDITIONS,
        n_design=config.n_surrogate_design,
        seed=config.seed + 2,
    )
    report.to_csv(config.path("surrogate_report.csv"), index=False)
    logger.info(
        "surrogate quality: worst held-out R2 %.4f, median %.5f",
        report["min_r2"].min(),
        report["median_r2"].median(),
    )

    posteriors: list[idn.PosteriorResult] = []
    baseline_by_patient: dict[int, idn.PosteriorResult] = {}
    for level in idn.NOISE_LEVELS:
        for index, case in enumerate(cases):
            posterior = idn.sample_posterior(
                case,
                surrogates,
                (idn.BASELINE,),
                level,
                seed=config.seed + 100 + index,
                n_steps=config.mcmc_steps,
                n_burn=config.mcmc_burn,
            )
            posteriors.append(posterior)
            if level == "realistic":
                baseline_by_patient[case.patient_id] = posterior
        logger.info("posteriors sampled at %s noise for %d patients", level, len(cases))

    confounding = idn.confounding_table(posteriors)
    confounding.to_csv(config.path("confounding_table.csv"), index=False)
    recovery = idn.recovery_table(posteriors)
    recovery.to_csv(config.path("recovery_table.csv"), index=False)
    recovery_summary = idn.recovery_summary(recovery)
    recovery_summary.to_csv(config.path("recovery_summary.csv"), index=False)

    matrices.plot_confounding_map(
        confounding, config.path("fig_confounding_map.png"), idn.HIDDEN_ORDER, "realistic"
    )
    matrices.plot_confounding_map(
        confounding,
        config.path("fig_confounding_map_optimistic.png"),
        idn.HIDDEN_ORDER,
        "optimistic",
    )
    matrices.plot_recovery(recovery_summary, config.path("fig_recovery.png"))

    _record(config, "identifiability", started)
    return {
        "cases": cases,
        "surrogates": surrogates,
        "posteriors": posteriors,
        "baseline_by_patient": baseline_by_patient,
        "confounding": confounding,
        "recovery_summary": recovery_summary,
        "fisher_table": fisher_table,
        "surrogate_report": report,
    }


# =====================================================================================
# D5: tie-breaker
# =====================================================================================


def run_tiebreaker(
    config: Config, identifiability: dict[str, Any], sensitivity: dict[str, Any]
) -> dict[str, Any]:
    """The provocation search, its table, and the structural-unidentifiability note."""
    started = time.perf_counter()
    pairs = tb.candidate_pairs(identifiability["confounding"], top_k=3)
    logger.info("tie-breaker: examining %s", [str(p) for p in pairs])

    detail, table = tb.run(
        identifiability["cases"],
        identifiability["surrogates"],
        identifiability["baseline_by_patient"],
        pairs,
        noise_level="realistic",
        seed=config.seed + 3,
    )
    detail.to_csv(config.path("tiebreaker_detail.csv"), index=False)
    table.to_csv(config.path("tiebreaker_table.csv"), index=False)

    structural = tb.structural_unidentifiability_note(
        identifiability["fisher_table"], sensitivity["sobol"]
    )
    structural.to_csv(config.path("structural_identifiability.csv"), index=False)

    if not detail[detail["usable"]].empty:
        matrices.plot_tiebreaker(detail, config.path("fig_tiebreaker.png"))
    else:
        logger.warning("no usable tie-breaker rows; skipping the figure")

    _record(config, "tiebreaker", started)
    return {"detail": detail, "table": table, "structural": structural, "pairs": pairs}


# =====================================================================================
# Reference figures
# =====================================================================================


def run_reference_figures(config: Config, labelled: pd.DataFrame) -> None:
    """The figures that describe the model rather than the analysis."""
    started = time.perf_counter()

    healthy = simulate(HEALTHY_GEOMETRY, HEALTHY_MATERIAL, RESTING_LOADING, 0.0, record_trace=True)
    hcm = simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, 0.0, record_trace=True)
    treated = simulate(
        HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, d.DOSE_MID_MG_PER_DAY, record_trace=True
    )
    pvloop.plot_loops(
        {"healthy": healthy, "HCM, untreated": hcm, "HCM, 10 mg/day": treated},
        config.path("fig_pv_loops.png"),
        title="The phenotype is a consequence, not an input",
        subtitle=(
            "Same equations; only myosin availability, passive stiffness and wall volume differ"
        ),
    )

    doses = np.array(APPROVED_DOSE_LADDER_MG_PER_DAY)
    ejection, gradient = [], []
    for dose in doses:
        result = simulate(HCM_GEOMETRY, HCM_MATERIAL, RESTING_LOADING, float(dose))
        observed = observe(result, HCM_GEOMETRY, RESTING_LOADING)
        ejection.append(observed.ejection_fraction)
        gradient.append(observed.peak_lvot_gradient_mmhg)
    pvloop.plot_dose_response(
        doses, np.array(ejection), np.array(gradient), config.path("fig_dose_response.png")
    )

    matrices.plot_over_responder_separation(labelled, config.path("fig_over_responders.png"))
    matrices.plot_observable_noise_context(config.path("fig_measurement_noise.png"))
    _record(config, "figures", started)


# =====================================================================================
# D7: the writeup
# =====================================================================================


def run_paper(config: Config, paper_dir: Path = Path("paper")) -> Path | None:
    """Generate the paper's numbers from the results, then compile it.

    Nothing quantitative in ``main.tex`` is a literal: the prose reads macros written here
    from the result CSVs. A paper whose numbers are typed is a paper that goes stale the
    first time a parameter changes, and nobody notices.
    """
    started = time.perf_counter()
    from .report import write_macros, write_result_tables

    write_macros(config.results_dir, paper_dir / "generated.tex")
    write_result_tables(config.results_dir, paper_dir)

    pdf = _compile_latex(paper_dir / "main.tex")
    _record(config, "paper", started)
    return pdf


def _compile_latex(source: Path) -> Path | None:
    """Compile with whichever engine is present, and say clearly if none is.

    Two engines are supported because two environments are: ``tectonic`` on a developer
    machine, where it needs no TeX installation, and ``pdflatex`` in the container, where a
    pinned TeX Live is already there and no network fetch should happen during a build.
    """
    import shutil
    import subprocess

    attempts: list[list[str]] = []
    if shutil.which("tectonic"):
        attempts.append(["tectonic", "--keep-logs", source.name])
    if shutil.which("pdflatex"):
        # Twice, so \ref and \label resolve.
        attempts.append(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name])
        attempts.append(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name])
    if not attempts:
        logger.warning(
            "no LaTeX engine found (tried tectonic, pdflatex): the generated .tex files are "
            "written and the container image supplies an engine"
        )
        return None

    for command in attempts:
        logger.info("compiling: %s", " ".join(command))
        completed = subprocess.run(
            command, cwd=source.parent, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            tail = (completed.stdout or "")[-3000:] + (completed.stderr or "")[-2000:]
            logger.error("LaTeX compilation failed:\n%s", tail)
            return None

    pdf = source.with_suffix(".pdf")
    if not pdf.exists():
        logger.error("LaTeX reported success but produced no PDF")
        return None
    logger.info("wrote %s (%.0f kB)", pdf, pdf.stat().st_size / 1024)
    return pdf


# =====================================================================================
# Everything
# =====================================================================================


def run_all(config: Config | None = None) -> dict[str, Any]:
    """Every stage, in order, writing every artifact."""
    config = config or Config()
    overall = time.perf_counter()

    validation = run_validation(config)
    params, labelled = run_population(config)
    run_reference_figures(config, labelled)
    sensitivity = run_sensitivity(config, params, labelled)
    identifiability = run_identifiability(config, params, labelled)
    tiebreaker = run_tiebreaker(config, identifiability, sensitivity)

    from .viz import dashboard

    dashboard.build(config.path("explorer.html"), labelled=labelled)

    # The manifest is written *before* the paper, because the paper's numbers are read
    # from it. Writing it afterwards worked on a machine that had run the pipeline before
    # and failed on the first clean container build, which is the whole reason the
    # container build exists.
    config.timings["total"] = round(time.perf_counter() - overall, 2)
    manifest = {
        "seed": config.seed,
        "population_n_base": config.population_n_base,
        "n_patients": len(params),
        "n_identifiability_cases": config.n_identifiability_cases,
        "n_surrogate_design": config.n_surrogate_design,
        "mcmc_steps": config.mcmc_steps,
        "timings_seconds": config.timings,
        "validation_all_pass": bool(validation["pass"].all()),
    }
    config.path("manifest.json").write_text(json.dumps(manifest, indent=2))

    run_paper(config)
    manifest["timings_seconds"] = config.timings
    config.path("manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("pipeline complete in %.0f s", config.timings["total"])
    return {
        "validation": validation,
        "params": params,
        "labelled": labelled,
        "sensitivity": sensitivity,
        "identifiability": identifiability,
        "tiebreaker": tiebreaker,
        "manifest": manifest,
    }
