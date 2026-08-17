# Build log

A running record of what has been built, what was decided and why, and what is left.
Written so that a fresh session (human or model) can pick the project up without
re-deriving anything. Newest entries at the bottom of each section.

**Read this first, then `docs/research/00_overview.md`, then `src/hcmtwin/parameters.py`.**

---

## Where the project stands

| Deliverable | State |
|---|---|
| D0 research dossier | in progress: bibliography done, prose files being written |
| D1 `hcmtwin` package | **done and green** |
| D2 validation report | gates all pass; generated table and notebook outstanding |
| D3 sensitivity matrix | not started |
| D4 confounding map | not started |
| D5 tie-breaker table | not started |
| D6 interactive explorer | not started |
| D7 writeup | not started |

Run `pytest -q` to confirm the state of the model. `pytest tests/test_validation.py` is
the single command that proves the heart works.

---

## The two things a newcomer most needs to know

**1. The measured/hidden split is the method, not a detail.** `MeasuredGeometry` (wall
volume, reference cavity volume, body surface area) is sampled and then treated as known
exactly, because imaging measures shape well. `HiddenMaterial` (`phi_baseline`,
`a_pas_kpa`, `b_pas`, `ca50_ref_um`, `clearance_l_per_h`) is sampled and then treated as
unknown, because imaging measures material not at all. They are separate frozen
dataclasses and nothing takes a merged dict. Do not "simplify" this.

**2. Several things had to be added to the specification's model, and each was added
because the model visibly failed without it.** They are listed below with the failure.
If you are tempted to remove one, re-read the failure first.

---

## Deviations from the specification, and why

Each is documented in place in the source and will appear in
`docs/research/04_model_provenance.md`.

| Change | Where | Why it was necessary |
|---|---|---|
| Active force-length (filament overlap) term | `sarcomere.overlap_factor` | Without it the chamber's end-systolic elastance is **negative**: cavity pressure is `sigma/(1+3V/Vw)`, so at fixed fiber stress a smaller cavity makes *more* pressure. The ventricle never met a force balance and emptied completely every beat. |
| Cross-bridge distortion (force-velocity) | `sarcomere.distortion_derivative` | A muscle with no velocity dependence has no bound on shortening speed. Coupled to a circulation it ejected past zero volume on the first run. |
| Force-dependent recruitment from the parked state | `sarcomere.force_recruitment_factor` | The published mechanism for length-dependent activation (Campbell 2018). `beta_len` alone gave a stroke volume that was flat then falling as preload rose, failing the Frank-Starling gate. Also what makes "contractile reserve" mechanistically real. |
| Compressive branch on the passive law | `sarcomere.passive_stress_kpa` | The specification's single exponential is bounded below by `-a_pas` however hard the chamber is squeezed: about 1 kPa of restoring stress against 50 kPa of active stress. |
| Obstruction driven by cavity-to-wall **ratio**, cubic power law | `obstruction.lvot_area_cm2` | Absolute cavity volume cannot distinguish a small normal ventricle from an obstructed hypertrophic one, and gave a 20 mmHg resting gradient in a normal heart. A *linear* law in the ratio also fails, because peak gradient needs a small area while flow is still high (mid-ejection), and because a healthy ventricle at end-systole and an HCM ventricle at end-diastole sit at the same ratio. |
| Filling compartment is pulmonary-venous/left-atrial, not systemic venous | `defaults.C_VEN_ML_PER_MMHG` | At a systemic venous compliance (~100 mL/mmHg) the reservoir absorbs anything the ventricle refuses, and end-diastolic pressure becomes **insensitive to the ventricle's own stiffness**. Elevated filling pressure is the entire clinical problem in HCM, so that was fatal. |
| End-diastolic quantities read at the beat's closing instant, not at the argmax of volume | `model._run_beat` | During isovolumic contraction the volume sits on a plateau while pressure climbs 100 mmHg; an argmax lands late in the upstroke and reported an end-diastolic pressure 2-3x too high (17.6 vs 9.3 mmHg). |
| E/e' surrogate built from the transmitral pressure gradient | `observables.observe_arrays` | Dividing peak mitral flow by peak lengthening rate is very nearly `dV/dt` over `dV/dt`, so the ratio was an almost pure function of chamber geometry and barely moved when the tissue was stiffened. It looked like a filling-pressure surrogate and was not one. |
| `MeasuredGeometry.body_surface_area_m2` added | `parameters.py` | Needed for stroke volume index and mass index, which are what a clinical report actually contains. |
| `parameters.py` module added to the specified layout | `parameters.py` | The dataclasses are cross-cutting; putting them in `defaults.py` would break the provenance test, which scans that file for constants. |

---

## Calibration record

The healthy baseline and the HCM reference were calibrated by grid search against the
Section 7 target ranges, **not** against any trial outcome. Calibration targets were:
resting haemodynamics from `05_validation_targets.md`, plus three ejection-realism
targets added after the first pass produced a 154 ms ejection at a peak aortic flow of
1000 mL/s (roughly twice physiological): peak LVOT velocity 0.7-1.4 m/s, ejection
duration 250-340 ms, peak aortic flow 350-600 mL/s.

Final calibrated values and the searches that produced them:

- `T_REF_KPA = 130`, `CA_TAU_R_S = 0.110`, `XB_HALF = 0.055`, `K_FORCE_PER_KPA = 0.045`,
  `V_TOT_ML = 394`, `R_SYS = 1.04`, `A_PAS_KPA = 0.90`. Zero-miss on all ten targets with
  a monotone Frank-Starling relation.
- `CA_TAU_R_S` is longer than the measured calcium time-to-peak. Deliberate: the
  prescribed waveform has one time constant for both rise and decay and cannot match
  both, and systolic *duration* is what the mechanics depend on. Labelled `calibrated`.
- HCM reference: `V_W_HCM_ML = 245`, `V_LV_REF_HCM_ML = 60` (**same as healthy** -- the
  small cavity has to emerge from stiff tissue refusing to fill, not be assumed),
  `PHI_HCM = 0.55`, `A_PAS_HCM_KPA = 4.0`, `B_PAS_HCM = 10`.

**Nothing was fitted to a published dose-response curve.** The drug parameters
(`DRUG_E_MAX`, `DRUG_EC50_NG_PER_ML`) were left where mechanism put them, so the
exposure-response comparison is a prediction.

---

## Results worth keeping

Current reference-patient values (regenerate with the gate script; see below).

**Healthy:** EF 0.658, EDV 113.0 mL, SV 74.4 mL, peak LV 110.7 mmHg, EDP 9.58 mmHg,
MAP 94.9 mmHg, CO 4.85 L/min, gradient 7.1 mmHg, wall 0.92 cm, E/e' 7.3.

**HCM:** EF 0.747, EDV 80.4 mL, SV 60.1 mL, peak LV 169.0 mmHg, EDP 13.34 mmHg,
gradient 76.1 mmHg, wall 1.59 cm, E/e' 10.8, ATP cost per unit stroke work 2.05x healthy,
strain amplitude 0.154 vs 0.209.

Two findings already worth writing down.

**Thick wall alone raises ejection fraction.** Giving the HCM geometry *healthy* material
gives EF 0.758, slightly above the diseased reference. That is correct physiology, and it
is the sharpest possible statement of why this project exists: a reassuring ejection
fraction in a thick-walled ventricle is close to uninformative. What the thick wall alone
does *not* produce is elevated filling pressure, a raised E/e', or the energetic penalty.

**The over-response rate falls out right.** Applying the pivotal trial's actual enrolment
criteria (EF >= 55%, wall >= 1.3 cm, gradient >= 30 resting or >= 50 provoked) to the
sampled cohort gives an over-responder rate near 4%, against a published 5.2% in
MAVA-LTE. Nothing in the priors was tuned to that number.

**Exposure-response magnitude matches without being fitted.** The reference HCM patient
loses 5.3 ejection-fraction points at the 10 mg mid dose. SEQUOIA-HCM reported a
placebo-corrected change of -4.8 points for aficamten. Independent, and it should be
presented as such.

---

## Performance notes

- Scalar steady-state solve: **~37 ms** (target was under 50 ms). Achieved by dropping
  `STEPS_PER_BEAT` from 1200 to 400 after a convergence study: 400 steps matches 2400 to
  five significant figures, and the peak gradient (the most step-sensitive quantity) to
  0.05%.
- Cohort path: ~10 ms per patient at N=1024, improving with N. A 5005-patient,
  13-condition study is roughly 10 minutes.
- Integration is in normalised beat phase so a cohort with mixed heart rates advances in
  lockstep. This is why `backend.py` exists; do not collapse it to one backend.

---

## Session history

### 2026-08-16 (session 1)

1. Scaffolded the repo, pinned dependencies, made the venv (Python 3.12, `uv`).
2. Literature phase: located and checked the primary sources. Key finds were
   Campbell 2018 (Layer 1 state scheme and the force-recruitment mechanism), Arts 1991
   (Layer 2 one-fiber relation), Klotz 2006 (independent passive-relation check),
   Thavendiranathan 2013 (ejection-fraction test-retest noise), Geske 2009 (outflow
   gradient variability, CV ~0.5, which is very large and matters for the tie-breaker),
   Lang 2015 (chamber reference ranges), SEQUOIA-HCM and MAVA-LTE (drug outcome anchors).
3. Wrote `units`, `backend`, `defaults`, `parameters`.
4. Wrote the three layers plus obstruction and drug. Hit and fixed, in order: ejection
   past zero volume; a negative end-systolic elastance; a filling pressure insensitive to
   stiffness; an end-diastolic pressure read from the wrong instant; a geometry-only E/e'
   surrogate; a resting gradient in a normal heart. Each fix is in the deviations table.
5. Calibrated healthy and HCM references. All Section 7 gates pass.
6. Wrote `provocation` and `population` (Saltelli, trial eligibility).
7. Wrote the test suite: 99 tests, all green.
8. Started the research dossier; bibliography complete.

---

## Reproducing the calibration and gate checks

The exploratory sweep scripts used during calibration were scratch and are not committed.
The permanent equivalents are:

- `pytest tests/test_validation.py` -- every Section 7 gate.
- `python -m hcmtwin.cli validate` -- regenerates the validation table (D2).
- `make all` -- everything, from a clean checkout.

---

## What to do next

1. Finish `docs/research/` prose. `tests/test_provenance.py` will fail until
   `04_model_provenance.md` has a row for all 65 constants in `defaults.py`.
2. `analysis/sensitivity.py` -> D3, `analysis/identifiability.py` -> D4,
   `analysis/tiebreaker.py` -> D5.
3. `viz/` and the self-contained explorer -> D6.
4. `paper/main.tex` -> D7.
5. `Makefile` + `Dockerfile` so `make all` reproduces everything.

Open questions flagged during the build, to be resolved in the writeup rather than
silently:

- The drug slightly *raises* end-diastolic pressure in the model (13.34 -> 13.37 mmHg at
  15 mg), whereas the trials report improved filling parameters. The model has no
  load-dependent improvement in relaxation, so this is a known and reportable limitation.
- End-systolic volume in the most severely obstructive virtual patients is set by outflow
  closure approaching cavity obliteration. That is a real clinical phenomenon, but it
  means their ejection fraction is geometry-limited rather than contractility-limited,
  and the identifiability analysis should be read with that in mind.
