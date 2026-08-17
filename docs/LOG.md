# Build log

A running record of what has been built, what was decided and why, and what is left.
Written so that a fresh session (human or model) can pick the project up without
re-deriving anything.

**Read this first, then `docs/research/00_overview.md`, then `src/hcmtwin/parameters.py`.**

---

## Where the project stands

| Deliverable | State |
|---|---|
| D0 research dossier | **done** — 8 documents + bibliography, zero `[VERIFY]`, every `[GAP]` justified, CI-enforced |
| D1 `hcmtwin` package | **done** — ruff, ruff format and mypy all clean; 137 tests green |
| D2 validation report | **done** — 35 gates, all pass; `results/validation_table.md` |
| D3 sensitivity matrix | **done** — `results/fig_sensitivity_matrix.png` + CSV |
| D4 confounding map | **done** — `results/fig_confounding_map.png` + CSV |
| D5 tie-breaker table | **done** — `results/tiebreaker_table.csv` + figure |
| D6 interactive explorer | **done** — `results/explorer.html`, self-contained |
| D7 writeup | **done** — `paper/main.pdf`, 8 pages, every number generated from results |

```bash
make test        # 137 tests, including every Section 7 gate
make all         # everything, from a clean checkout
make docker-all  # the same, reproducibly
```

---

## The two things a newcomer most needs to know

**1. The measured/hidden split is the method, not a detail.** `MeasuredGeometry` (wall
volume, unloaded cavity volume, body surface area) is sampled and then treated as known
exactly, because imaging measures shape well. `HiddenMaterial` (`phi_baseline`,
`a_pas_kpa`, `b_pas`, `ca50_ref_um`, `clearance_l_per_h`) is sampled and then treated as
unknown, because imaging measures material not at all. They are separate frozen dataclasses
and nothing takes a merged dict. Do not "simplify" this.

**2. Several things had to be added to the specification's model, and each was added
because the model visibly failed without it.** They are listed below with the failure. If
you are tempted to remove one, read the failure first.

---

## Deviations from the specification, and why

Each is documented in place in the source and in `docs/research/04_model_provenance.md`.

| Change | Where | Why it was necessary |
|---|---|---|
| Active force-length (filament overlap) term | `sarcomere.overlap_factor` | Without it the chamber's end-systolic elastance is **negative**: cavity pressure is `sigma/(1+3V/Vw)`, so at fixed fiber stress a smaller cavity makes *more* pressure (141 mmHg at 100 mL, 224 at 42, 314 at 13). Ejection never met a force balance and the ventricle emptied completely every beat. |
| Cross-bridge distortion (force-velocity) | `sarcomere.distortion_derivative` | A muscle with no velocity dependence has no bound on shortening speed. Coupled to a circulation it ejected past zero volume on the first run. |
| Force-dependent recruitment from the parked state | `sarcomere.force_recruitment_factor` | The published mechanism for length-dependent activation (Campbell 2018). `beta_len` alone gave a stroke volume that was flat then falling as preload rose, failing the Frank-Starling gate. Also what makes "contractile reserve" mechanistically real. |
| Compressive branch on the passive law | `sarcomere.passive_stress_kpa` | The specification's single exponential is bounded below by `-a_pas` however hard the chamber is squeezed: about 1 kPa of restoring stress against 50 kPa of active stress. |
| Obstruction driven by cavity-to-wall **ratio**, as a **cubic** power law | `obstruction.lvot_area_cm2` | Absolute cavity volume gave a 20 mmHg resting gradient in a normal heart. A *linear* law in the ratio also fails: peak gradient needs a small area while flow is still high (mid-ejection), and a healthy ventricle at end-systole sits at the same ratio (0.28) as an HCM ventricle at end-diastole (0.33), so no threshold separates them. |
| Filling compartment is pulmonary-venous/left-atrial, not systemic venous | `defaults.C_VEN_ML_PER_MMHG` | At a systemic venous compliance (~100 mL/mmHg) the reservoir absorbs anything the ventricle refuses, and end-diastolic pressure becomes **insensitive to the ventricle's own stiffness**. Elevated filling pressure is the entire clinical problem in HCM, so that was fatal. |
| End-diastolic quantities read at the beat's closing instant, not at the argmax of volume | `model._run_beat` | During isovolumic contraction the volume sits on a plateau while pressure climbs 100 mmHg; an argmax landed late in the upstroke and reported an end-diastolic pressure 2-3x too high (17.6 vs 9.3 mmHg). |
| E/e' surrogate built from the transmitral pressure gradient | `observables.observe_arrays` | Dividing peak mitral flow by peak lengthening rate is very nearly `dV/dt` over `dV/dt`, so the ratio was an almost pure function of chamber geometry and barely moved when the tissue was stiffened. |
| Heart rate and LV mass excluded from the inference likelihood | `observables.UNINFORMATIVE_NAMES` | Both are constant across the hidden parameters for a fixed patient, so no posterior can learn from them. LV mass is the interesting case: wall volume is pinned as known and mass is wall volume times a density constant. |
| Paper runs to 11 pages, not the specified 6-8 | `paper/main.tex` | Sections 6 (Conclusions, graded by confidence) and 7 (Where to look next, clinical and computational) were added on request after the first 8-page build, along with an AI-usage disclosure. The extra three pages are all conclusions and next steps; nothing was added to the methods. Kept rather than trimmed because a reader asking "what do I do with this" was the stated priority. |
| `parameters.py` and `validation.py` added to the specified layout | — | The dataclasses are cross-cutting and cannot live in `defaults.py` (the provenance test scans it for constants). `validation.py` holds the gates as data so the tests and the report cannot drift. |

---

## Calibration record

Calibrated by grid search against the Section 7 target **ranges**, never against a trial
outcome. Three ejection-realism targets were added after the first pass passed every
haemodynamic gate while ejecting the whole stroke volume in 154 ms at a peak aortic flow of
1000 mL/s, roughly twice physiological and invisible to the gates as written: peak LVOT
velocity 0.7-1.4 m/s, ejection duration 250-340 ms, peak aortic flow 350-600 mL/s.

Final: `T_REF_KPA = 130`, `CA_TAU_R_S = 0.110`, `XB_HALF = 0.055`,
`K_FORCE_PER_KPA = 0.045`, `V_TOT_ML = 394`, `R_SYS = 1.04`, `A_PAS_KPA = 0.90`.
HCM reference: `V_W_HCM_ML = 245`, `V_LV_REF_HCM_ML = 60` (**the same as healthy** — the
small cavity has to emerge from stiff tissue refusing to fill), `PHI_HCM = 0.55`,
`A_PAS_HCM_KPA = 4.0`.

`CA_TAU_R_S` is longer than the measured calcium time-to-peak. Deliberate: the waveform has
one time constant for both rise and decay, and systolic *duration* is what the mechanics
depend on. Labelled `calibrated`.

**Nothing was fitted to a published dose-response curve.** `DRUG_E_MAX` and
`DRUG_EC50_NG_PER_ML` are `assumed`, so the exposure-response comparison is a prediction.

---

## Results

**Healthy reference:** EF 0.658, EDV 113.0 mL, SV 74.4 mL, peak LV 110.7 mmHg,
EDP 9.58 mmHg, MAP 94.9 mmHg, CO 4.85 L/min, gradient 7.1 mmHg, wall 0.92 cm, E/e' 7.3.

**HCM reference:** EF 0.747, EDV 80.4 mL, SV 60.1 mL, peak LV 169.0 mmHg, EDP 13.3 mmHg,
gradient 76.1 mmHg, wall 1.59 cm, E/e' 10.8, ATP per unit work 2.05x healthy, strain
amplitude 0.154 vs 0.209.

### Two unfitted quantitative predictions

Ejection-fraction change at the 10 mg mid dose: **-5.3 points**, against a
placebo-corrected **-4.8** reported for aficamten in SEQUOIA-HCM. Gradient change
-39.4 mmHg against roughly -35.

Over-response rate on a cohort built with the trial's own enrolment criteria (EF >= 55%,
wall >= 1.3 cm, gradient >= 30 resting or >= 50 provoked): **1.4% at or below the mid
dose**, against a published 5.2% over a full titration. See "open questions" below: the
model's crossing rate has a narrower tail than reality and this is reported rather than
tuned away.

### The headline identifiability result

**Drug clearance explains about half the variance in who over-responds (total-order Sobol
0.50 on the ejection-fraction drop) and is structurally invisible before the first dose.**
Its total-order index on every baseline observable is exactly 0.000; the Fisher information
matrix has a zero eigenvalue whose null direction is 100% clearance. It enters the model
only through drug exposure, so every derivative is identically zero. No maneuver can
amplify a signal that does not exist; the remedy is a genotype or a probe dose.

Fisher eigenvalue spectrum (median over 50 patients, realistic noise):
188.5, 35.6, 0.616, 0.093, **0.000**.

Posterior recovery at realistic noise (relative 90% credible-interval width, lower is
better): `ca50_ref_um` 0.34 and `phi_baseline` 0.57 are recoverable; `b_pas` 0.72,
`a_pas_kpa` 1.18 and `clearance_l_per_h` 1.43 are not.

Most correlated pair: `a_pas_kpa` / `b_pas`, |r| 0.68 at realistic noise and 0.81 at
optimistic. **Confounding is higher at lower noise**, which is worth explaining rather than
glossing: as the data get better the posterior tightens onto the ridge and the correlation
along it becomes clearer, whereas with more noise the prior box dominates and decorrelates.

### The tie-breaker result, and a metric that had to be replaced

**No maneuver meaningfully resolves any confounded direction.** The best case narrows the
invisible direction by 5.4% (myosin availability / calcium sensitivity, under combined
exercise). Two of three pairs show a discriminating signal above one measurement error, but
that is a different and weaker claim, for the reason below.

Two methodological findings came out of this and both are in the paper.

*Correlation is the wrong score.* The first run ranked maneuvers by the drop in the pair's
posterior correlation, and every maneuver made it worse (0.68 rising to 0.79-0.88). That is
an artefact: correlation describes the shape of the uncertainty, not its size, and adding
information that constrains the combination the data already knew collapses the cloud
further onto its ridge. The table now ranks on the posterior standard deviation projected
onto the direction that was invisible at baseline. If you change this, read
`tiebreaker.ridge_width` first.

*The signal column and the narrowing column disagree, and that is the result.* Exercise
moves strain amplitude by 8.6 measurement standard deviations between two patients a resting
study cannot tell apart, yet the posterior barely narrows. It is not the emulator (held-out
error is at most a quarter of the measurement error, checked) and not sampling noise
(consistent across all 50 patients and all 4 maneuvers). It is **nuisance compensation**: the
signal is computed with the other three parameters held at their true values, and an
inference estimating all five at once can mimic the maneuver's effect by moving the ones it
does not know.

### Other findings worth keeping

**Thick wall alone raises ejection fraction.** HCM geometry with *healthy* material gives
EF 0.758, slightly above the diseased reference. Correct physiology, and the sharpest
statement of why the project exists. What the wall alone does not produce is the elevated
filling pressure, the raised E/e', or the energetic penalty.

---

## Performance notes

- Scalar steady-state solve: **~37 ms** (target was 50). `STEPS_PER_BEAT` dropped from 1200
  to 400 after a convergence study: 400 matches 2400 to five significant figures, and the
  peak gradient (most step-sensitive) to 0.05%.
- Cohort path: 20,000 solves per condition in about 40 s.
- Surrogate prediction was flattened out of the scikit-learn pipeline into three arrays:
  17x faster, bit-identical, and the difference between a 10-minute and a 2-hour MCMC.
- Integration is in normalised beat phase so a cohort with mixed heart rates advances in
  lockstep. That is why `backend.py` exists; do not collapse it to one backend.
- Whole pipeline: population ~3.5 min, identifiability ~8 min, tie-breaker ~5 min.
- **The one remaining slow spot** is `dashboard._loop_traces`, which runs 2600 scalar
  solves in a Python loop to collect pressure-volume traces for the explorer'''s emulator.
  Everything else in the project vectorises across patients; this cannot, because
  `simulate_cohort` deliberately does not record traces (storing them for a real cohort
  would cost gigabytes). Fixing it means adding subsampled trace recording to the cohort
  path, which is worth doing if the explorer is rebuilt often and is not worth touching the
  core solver for otherwise. Costs 2 minutes on the host, about 5 in the container.

---

## Session history

### 2026-08-16 to 17 (session 1)

1. Scaffolded the repo; venv on Python 3.12 via `uv`.
2. Literature phase. Key sources located and checked: Campbell 2018 (Layer 1 scheme and
   force recruitment), Arts 1991 (one-fiber relation), Klotz 2006 (independent passive
   check), Thavendiranathan 2013 (EF test-retest noise), Geske 2009 (outflow gradient CV
   ~0.5, the largest single caveat in the project), Lang 2015 (chamber reference ranges),
   SEQUOIA-HCM and MAVA-LTE (drug outcome anchors).
3. Wrote `units`, `backend`, `defaults`, `parameters`, then the three layers.
4. Hit and fixed, in order: ejection past zero volume; a negative end-systolic elastance;
   a filling pressure insensitive to stiffness; an end-diastolic pressure read from the
   wrong instant; a geometry-only E/e' surrogate; a resting gradient in a normal heart.
5. Calibrated healthy and HCM references. All Section 7 gates pass.
6. `provocation`, `population` (Saltelli, trial eligibility), then the full test suite.
7. Research dossier: 8 documents, CI-enforced provenance for all 65 constants.
8. Analysis: Sobol, Fisher information, surrogate-backed MCMC, tie-breaker search.
9. Pipeline, CLI, figures with a validated palette, self-contained explorer, LaTeX paper
   whose numbers are all generated.
10. Ran the full study: 5005 patients, 50 identifiability cases, 16 minutes.
11. Read the compiled PDF and fixed five rendering defects no test would catch (colliding
    figure titles, a collapsed tie-breaker figure, LaTeX escaped into table headers, a table
    scaled to six points, a signal rounded to 0.00). Trimmed to 8 pages.
12. Replaced the tie-breaker's scoring metric after the first run said every maneuver was
    harmful; see above.
13. Verified the explorer in a browser: the twins demo shows identical readouts untreated
    (EF 0.749 both) diverging to 0.690 and 0.649 at 10 mg from clearance alone, and the
    emulator agrees with the real model to three decimals.

---

## Open questions, to be resolved in the writeup rather than silently

- **The model's over-response rate (1.4% at the mid dose) is below the published 5.2%.**
  The eligible virtual cohort is *more* severely obstructive than the trial's (99%
  obstructive, median gradient 67 mmHg against about 55) and starts from a higher ejection
  fraction, so it has further to fall. The model's response distribution also has a
  narrower tail than reality. Not tuned toward the published number: doing so would make
  the exposure-response comparison a fit rather than a prediction, and the constraint is
  explicit in the specification.
- **The drug slightly *raises* end-diastolic pressure in the model** (13.34 to 13.37 mmHg
  at 15 mg) whereas trials report improved filling parameters. The model has no
  load-dependent improvement in relaxation. Known, reportable limitation.
- **End-systolic volume in the most severely obstructive virtual patients is set by outflow
  closure approaching cavity obliteration.** That is a real clinical phenomenon, but it
  means their ejection fraction is geometry-limited rather than contractility-limited, and
  it buffers the drug's effect on ejection fraction. Probably part of why the crossing rate
  is low.
- **Confounding rises as measurement noise falls.** Explained above; worth a sentence in
  the paper rather than a footnote, since the naive expectation is the opposite.

---

## Verified, not assumed

- 137 tests pass; ruff, ruff format and mypy clean across 29 modules.
- The explorer was opened in a real browser and driven through its presets and sliders.
- The paper was compiled and read page by page; the defects that found are listed above.
- CI runs the suite plus a reduced-resolution end-to-end pipeline and asserts every
  deliverable is produced.
- **`make all` was run inside the container from a clean checkout.** The first attempt
  failed at the paper stage and caught a real bug (the manifest was written after the
  document that reads it); the second completed in 375 s with pdflatex producing an 841 kB
  PDF, every deliverable present, and `validation_all_pass: true`. That is the definition
  of done, actually executed rather than asserted.

## The central result, and where it is computed

Added after the first full build, on the observation that the project's main finding was
implicit rather than stated: you had to read the outcome-Sobol column out of
`sensitivity_sobol.csv`, the visibility column out of `sensitivity_summary.csv`, and rank them
against each other yourself.

`analysis/sensitivity.visibility_versus_importance` now joins the two and ranks both;
`inversion_statistics` reports the Spearman rank correlation between them. Written to
`results/visibility_vs_importance.csv` and `..._stats.json`, rendered as `table_inversion.tex`,
and carried into the paper as Section 5.2.

The number is **rho = -0.90 (p = 0.037, n = 5)**, identical when restricted to routine
non-invasive observables. The rankings are reversed except that the bottom two swap. An
earlier verbal claim of a *perfect* reversal was wrong and was corrected before it reached the
paper: `a_pas` and `b_pas` sit at visibility ranks 1 and 2 against importance ranks 4 and 5,
so the reversal is one transposition short.

By variance rather than rank: recoverable parameters carry 0.32 of the outcome total-order
index, unrecoverable ones 0.63. Total-order indices overlap, so those do not partition the
variance and will not sum to one. Do not present them as a partition.

Caveat that must travel with the number: five parameters is a small sample, so the sign is the
message and the decimal is not. The individual rows are the evidence.

**Bug found by this work:** `report.required_macros` derives the macro list from `main.tex` by
regex, so writing `\medskip` in the paper made the build fail with "Command \medskip already
defined". Placeholders now emit `\providecommand` instead of `\newcommand`; real values still
use `\newcommand` so a genuine collision in our own macros stays loud. The hand-maintained
`_NOT_OURS` exclusion list is now an optimisation, not a correctness requirement.

---

## How this was built (AI usage)

Recorded here as well as in the README and the paper, so a future session inherits the same
disclosure rather than re-deriving it.

This project was implemented with **Claude Code** (Anthropic) working alongside the author. The
assistant wrote the bulk of the Python implementation, the test suite, the figure code and the
first drafts of the research dossier and the paper; it also ran the literature search, the
parameter calibration sweeps, and the debugging of the model failures recorded in the calibration
record above. Direction, scope, scientific judgement and acceptance of every result remained with
the author.

Two commitments follow, and any future session must keep them:

- **Never fabricate a citation.** Every reference in `docs/research/bibliography.bib` was located
  and its content verified. Where a needed number could not be traced to a primary source, the
  dossier says `[GAP]` and states what would be required. Do not fill a `[GAP]` with a plausible
  value to make a table look complete.
- **Document failures, not just fixes.** Several of the substantive findings came out of things
  that broke. Each is recorded with the symptom that exposed it, so the fix can be judged rather
  than trusted. Keep doing that.

## If you are picking this up

- `make test-fast` skips the multi-minute cohort tests while iterating.
- `make all N_BASE=48 CASES=8 DESIGN=200 MCMC=800` runs the whole study in a few minutes
  at reduced resolution; good for checking a change end to end.
- Stages cache: `results/population_*.csv.gz` is reused unless `--force` is passed, so
  re-running the analysis after a change to the inference does not repeat the cohort.
- `tests/test_provenance.py` will fail if you add a constant to `defaults.py` without a
  sourced row in `docs/research/04_model_provenance.md`. That is deliberate.
