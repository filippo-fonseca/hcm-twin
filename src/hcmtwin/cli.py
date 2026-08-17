"""Command line entry point: ``hcmtwin <stage>``.

``hcmtwin all`` is what ``make all`` runs and what the Dockerfile runs. Individual stages
exist so a single expensive step can be re-run without repeating the others.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import (
    Config,
    run_all,
    run_identifiability,
    run_paper,
    run_population,
    run_reference_figures,
    run_sensitivity,
    run_tiebreaker,
    run_validation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcmtwin",
        description=(
            "Mechanistic HCM digital twin: run the model, the virtual population, and the "
            "identifiability analysis."
        ),
    )
    parser.add_argument(
        "stage",
        choices=[
            "all",
            "validate",
            "population",
            "figures",
            "sensitivity",
            "identifiability",
            "tiebreaker",
            "explorer",
            "paper",
        ],
        help="which stage to run",
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--n-base",
        type=int,
        default=385,
        help="Saltelli base size; the cohort is n_base * 13 patients (default 385 = 5005)",
    )
    parser.add_argument(
        "--cases", type=int, default=50, help="patients used for the identifiability analysis"
    )
    parser.add_argument(
        "--design", type=int, default=400, help="surrogate design points per patient"
    )
    parser.add_argument("--mcmc-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--force", action="store_true", help="rebuild stages whose artifacts already exist"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    config = Config(
        results_dir=args.results,
        population_n_base=args.n_base,
        seed=args.seed,
        n_identifiability_cases=args.cases,
        n_surrogate_design=args.design,
        mcmc_steps=args.mcmc_steps,
        mcmc_burn=max(400, args.mcmc_steps // 3),
        force=args.force,
    )

    if args.stage == "all":
        run_all(config)
        return 0
    if args.stage == "validate":
        table = run_validation(config)
        failures = table[~table["pass"]]
        if len(failures):
            print(f"{len(failures)} validation gates FAILED:", file=sys.stderr)
            print(failures.to_string(index=False), file=sys.stderr)
            return 1
        print(f"all {len(table)} validation gates pass")
        return 0
    if args.stage == "population":
        run_population(config)
        return 0

    params, labelled = run_population(config)
    if args.stage == "figures":
        run_reference_figures(config, labelled)
        return 0
    if args.stage == "sensitivity":
        run_sensitivity(config, params, labelled)
        return 0
    if args.stage == "explorer":
        from .viz import dashboard

        dashboard.build(config.path("explorer.html"), labelled=labelled)
        return 0
    if args.stage == "paper":
        return 0 if run_paper(config) is not None else 1

    sensitivity = run_sensitivity(config, params, labelled)
    identifiability = run_identifiability(config, params, labelled)
    if args.stage == "tiebreaker":
        run_tiebreaker(config, identifiability, sensitivity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
