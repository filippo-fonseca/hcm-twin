# hcm-twin

A mechanistic digital twin of a hypertrophic cardiomyopathy (HCM) ventricle under myosin-inhibitor
dosing, built to answer one question:

> Two patients have the same ejection fraction the day before treatment starts. One tolerates dose
> escalation; the other falls through the 50% ejection-fraction floor and has to stop. **What
> measurement, taken beforehand, could have told them apart?**

The project is a zero-dimensional, three-layer model (sarcomere → chamber → closed-loop
circulation) plus an identifiability analysis that asks which hidden tissue-level and molecular
parameters non-invasive clinical measurements can and cannot recover.

## Why mechanistic

The published industry models are statistical: drug concentration in, ejection fraction out, fitted
to trial data. They are the right tool for designing a titration schedule. But there is no heart
between input and output, so they cannot separate "too much drug" from "a ventricle with no
reserve," and they cannot leave the covariate distribution of the trial population. This repository
builds the mechanistic alternative and then, crucially, measures how much of it is actually
identifiable from a clinical echo.

## The structural trick that makes it tractable

Clinical imaging measures **shape** accurately (wall thickness, cavity volume, mass) and **material**
not at all (tissue stiffness, fraction of myosin heads available). So the model pins the shape per
virtual patient and infers only the material. Six unknowns drop to five, of which only three are
strongly coupled. The split is enforced in code: `MeasuredGeometry` and `HiddenMaterial` are
separate frozen dataclasses and no function accepts a merged dictionary.

## Layout

```
docs/research/        D0  research dossier: every number in the code traces to a row here
src/hcmtwin/          D1  the package
  calcium.py              prescribed Ca2+ transient
  sarcomere.py            Layer 1: parked / available / attached myosin
  chamber.py              Layer 2: one-fiber cavity mechanics (Arts 1991)
  circulation.py          Layer 3: closed loop, blood volume conserved
  obstruction.py          LVOT obstruction with flow-dependent narrowing
  drug.py                 dose -> steady-state exposure -> shift in head availability
  model.py                assembly, steady-state beat solver
  observables.py          only what a clinic could measure (+ a quarantined hidden-truth block)
  population.py           virtual cohort sampling (Saltelli)
  provocation.py          Valsalva / tachycardia / handgrip / exercise analogues
  analysis/               sensitivity, identifiability, tie-breaker search
  viz/                    figures and the interactive explorer
tests/                D2  validation gates
notebooks/            D2  validation, population, analysis
paper/                D7  LaTeX writeup
results/                  every figure, and the CSV behind it
```

## Quickstart

```bash
make setup      # venv + pinned dependencies
make test       # the full suite, including every Section 7 validation gate
make all        # everything: gates, figures, CSVs, explorer, PDF
```

Or reproducibly, with no local Python at all:

```bash
docker build -t hcm-twin . && docker run --rm -v "$PWD/results:/app/results" hcm-twin
```

`make all` is the definition of done: from a clean checkout it produces the validation table (D2),
the sensitivity matrix (D3), the confounding map (D4), the tie-breaker table (D5), the interactive
explorer (D6), and the compiled writeup (D7), with the CSV behind every figure committed alongside
it.

## Honesty constraints

These are load-bearing, not boilerplate. See `docs/research/` and the limitations section of the
writeup.

- No mutation-specific predictions. Claims are about regions of parameter space, never named
  variants in named patients.
- No calibration against published trial curves that is then presented as independent validation.
  The exposure-response comparison in `tests/test_validation.py` is a *prediction*, and it is
  recorded honestly whether or not it succeeds.
- Negative results carry equal weight. "Nothing is confounded" and "nothing is separable" are both
  findings.
- Every illustrative figure is labelled illustrative.

## Status

See `docs/research/00_overview.md` for the argument in one page and
`docs/research/07_gap_statement.md` for what is and is not known.

## License

MIT.
