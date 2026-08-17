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

## What it found

**The phenotype is a consequence, not an input.** Raising myosin availability and passive
stiffness and thickening the wall yields, as outputs: a supranormal ejection fraction (0.75
against a 0.66 healthy reference), a reduced stroke volume, an elevated filling pressure,
reduced longitudinal strain despite preserved ejection fraction, and roughly twice the ATP
cost per unit of external work. A test walks the package's syntax tree to prove no
observable is ever assigned a literal.

**Wall thickening alone raises ejection fraction.** Give the HCM geometry *healthy* material
and the ejection fraction is 0.758, slightly above the diseased reference. That is correct
physiology and it is the sharpest available statement of why this project exists: in a
thick-walled ventricle, a reassuring ejection fraction is close to uninformative.

**Two quantitative predictions land, and neither was fitted.** The reference patient loses
5.3 ejection-fraction points at the mid dose, against a placebo-corrected 4.8 points
reported for aficamten in SEQUOIA-HCM. No parameter in the model was calibrated against any
published dose-response curve, and `docs/research/04_model_provenance.md` records the
confidence label on all 65 constants so that claim can be checked rather than trusted.

**The headline identifiability result is negative, and it is the useful kind.** Drug
clearance explains about half the variance in who over-responds and is *structurally*
invisible before the first dose: its total-order Sobol index on every baseline observable
is exactly zero, and the Fisher information matrix has a zero eigenvalue whose null
direction is 100% clearance. It enters the model only through drug exposure, so every
derivative with respect to it is identically zero. **No provocation maneuver can help**,
because there is no signal to amplify. What that argues for is pharmacokinetic information
obtained before titration rather than inferred from its consequences.

Of the four tissue parameters, calcium sensitivity and myosin availability are recoverable
from a routine study; the two passive stiffness parameters trade off against each other.
The tie-breaker table reports, per confounded pair, which maneuver helps and whether the
discriminating signal clears the documented measurement error.

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
