# 04. Model provenance

One subsection per equation: what is implemented, where it comes from, what was simplified
relative to that source, and what the simplification is expected to cost. The table at the
end lists every constant in `src/hcmtwin/defaults.py` with its source and confidence, and
`tests/test_provenance.py` fails CI if any constant lacks a row, if any row is orphaned,
or if any recorded value has drifted from the code.

Confidence labels: `measured` is a direct experimental or clinical value, or an exact
consequence of one; `calibrated` is fitted inside a cited model or back-calculated here
from cited observations, and is **not** independent evidence; `assumed` is our choice, with
the rationale recorded.

---

## 4.0 The largest simplification, stated first: prescribed calcium

The model does not simulate electrophysiology. There are no ion channels, no membrane
potential, and no calcium handling. An analytic transient is imposed identically on every
beat:

```
Ca(t) = Ca_diast + (Ca_peak - Ca_diast) * (t / tau_r) * exp(1 - t / tau_r)
```

**Why.** It collapses the cell layer to three myosin states, which is what makes a
steady-state beat cost about 37 ms and a five-thousand-patient study a laptop computation.
It also means every difference between two virtual patients is attributable to
myofilament material properties rather than to electrophysiology, which is exactly the
attribution the project needs to make.

**What it costs, concretely.**

- No rate-dependent calcium handling. The tachycardia maneuver captures the shortening of
  filling time and nothing else: no positive staircase, no catecholamine effect on
  calcium load or on the rate of sequestration. Any tie-breaker conclusion that leans on
  tachycardia or exercise must be read with this attached.
- No arrhythmia, and therefore nothing to say about the arrhythmic risk that dominates
  HCM prognosis.
- No drug effect on calcium handling. Mavacamten acts on myosin, so this is defensible,
  but it does mean the model has no route to the improved relaxation the trials report.
- The transient's time constant had to be **calibrated to systolic duration rather than to
  the measured calcium upstroke**. A single time constant governs both rise and decay in
  this waveform and cannot match both; at a physiological 40-60 ms time-to-peak the model
  ejected its whole stroke volume in 154 ms at a peak aortic flow near 1000 mL/s, roughly
  twice physiological. Duration was prioritised because it is what the mechanics depend
  on. `CA_TAU_R_S` is labelled `calibrated` for this reason.

---

## 4.1 Layer 1: the three-state myosin population

**Implemented.**

```
dS/dt = -k_park_off(sigma) * S + k_park_on * D
dD/dt =  k_park_off(sigma) * S - k_park_on * D - k_att * n(Ca, lam) * D + k_det * A
dA/dt =  k_att * n(Ca, lam) * D - k_det * A
```

with `S + D + A = 1`, and the availability knob
`phi = k_park_off / (k_park_off + k_park_on)` from which both rates are derived given a
fixed total turnover.

**Source.** Campbell, Janssen and Campbell [@campbell2018recruitment]. Their model has six
states: thin-filament sites off/on/bound, and myosin OFF/ON/force-generating.

**Simplifications relative to it.**

- *Thin-filament states collapsed into an algebraic Hill function.* Their explicit
  `N_off / N_on / N_bound` kinetics with nearest-neighbour cooperativity become
  `n(Ca, lam) = Ca^h / (Ca^h + Ca50(lam)^h)`, an instantaneous equilibrium. Cost: no
  cooperative activation transient, so the rise of force is slightly too fast and the
  model cannot represent cooperativity changes, which is one route by which thin-filament
  mutations act. The thin-filament axis survives only as `Ca50_ref`.
- *No cross-bridge strain distribution.* Their force-generating state is resolved over
  spring extension `x`; here it is a single population plus a mean-distortion variable
  (section 4.2). Cost: no accurate representation of quick-release transients or of
  work-loop shape at the sub-beat scale.
- *Their fitted rate constants are not transplanted.* `k_att` and `k_det` here are
  effective whole-cell rates, calibrated so the coupled model reproduces resting
  haemodynamics. Their published values are in different units and belong to a different
  state scheme; importing them numerically would be a category error, and doing so quietly
  would be worse.

**Imported essentially unchanged: force-dependent recruitment.** Their central result is
that a model whose off-to-on rate rises linearly with force reproduces length-dependent
activation significantly better than one with a constant rate (F-test, p < 0.001). Here:

```
k_park_off(sigma) = k_park_off * (1 + k_force * max(sigma_f, 0))
```

At zero load this reduces exactly to the specification's scheme, so `phi` keeps its clean
reading as the *unloaded* resting availability.

This term is not optional. Without it the coupled model produced a stroke volume that was
flat and then falling as preload rose, failing the Frank-Starling gate, and pushing
`beta_len` high enough to compensate would have taken it far outside the range the
myofilament literature supports. It is also what makes contractile reserve a real
quantity: a ventricle whose parked pool is already depleted cannot recruit when load rises.
Campbell and colleagues later carried the same mechanism to the whole-ventricle scale and
showed it steepens the end-systolic pressure-volume relationship [@campbell2020espvr].

**Length-dependent calcium sensitivity.**

```
Ca50(lam) = Ca50_ref * (1 - beta_len * (lam - 1))
```

Clamped below at a small floor, which `tests/test_layers.py` asserts is never active in the
physiological range. Positive `beta_len` means a stretched sarcomere is more
calcium-sensitive; this is the second contributor to length-dependent activation.

**ATP accounting.** One ATP per attachment event, so the flux is `k_att * n * D` and the
per-beat cost is its time integral. Parked heads are charged nothing, which is the whole
point: the energetic penalty of HCM comes from having moved heads out of the parked state
[@anderson2018srx; @toepfer2020hcm]. Cost of the simplification: it ignores the ATP spent
on calcium handling, which is a substantial share of real cardiac energetics, so the model
speaks only to *cross-bridge* cost and its absolute scale is arbitrary. Only ratios
between virtual patients are meaningful, and the observable is named and documented as
arbitrary units.

---

## 4.2 Layer 1 additions not in the specification

Three, each added because the model visibly failed without it. They are additions rather
than simplifications, so they are recorded separately and prominently.

### 4.2.1 Active force-length (filament overlap)

```
sigma_a = T_ref * A * clip(1 + beta_overlap * (lam - 1), 0, overlap_max) * g(x)
```

**Why it is necessary.** Cavity pressure in the one-fiber relation is
`sigma_f / (1 + 3 V/V_w)`, so at a *fixed* fiber stress a smaller cavity generates a
*higher* pressure. With no length dependence in the active stress the chamber's
pressure-generating capacity therefore rises monotonically as it empties: for the healthy
reference, 141 mmHg at 100 mL, 224 mmHg at 42 mL, 314 mmHg at 13 mL. That is a **negative
end-systolic elastance**, and its consequence is that ejection never meets a force balance
and the ventricle empties until something else stops it.

**Source.** The classic sarcomere force-length relation, linearised on its ascending limb:
cardiac force rises from roughly a fifth to full over sarcomere lengths of about 1.7 to
2.1 micrometres, which at a 2.0 micrometre reference is a slope of about 4 per unit
stretch, with a plateau above.

**Cost.** A piecewise-linear caricature of a smooth relation, and no descending limb.

### 4.2.2 Cross-bridge distortion, giving force-velocity

```
dx/dt = d(eps_f)/dt - k_xb * x
g(x)  = clip(1 + x / xb_half, 0, 1 + xb_max_gain)
```

At constant shortening velocity `v` the distortion settles at `x = -v / k_xb`, so `g` falls
linearly with velocity and reaches zero at `v_max = k_xb * xb_half`: a linearised
Hill-type force-velocity relation.

**Why it is necessary.** A muscle with no velocity dependence has no upper bound on
shortening speed. Coupled to a circulation, the first version of this model ejected past
zero cavity volume.

**Why it is explicit, not implicit.** Stress depends on the *state* `x`; the strain rate is
computed from the flows, which are computed from the stress. That chain is acyclic, so a
model with a genuine force-velocity relation still needs no inner iteration.

**Cost.** A linear force-velocity relation, whereas the real one is hyperbolic. Force
enhancement during stretch is capped by a chosen constant rather than emerging from
cross-bridge detachment kinetics.

### 4.2.3 A compressive branch on the passive law

For `lam >= 1` the specification's relation is used unchanged. For `lam < 1` it is
mirrored, `sigma_p = -a_pas * (exp(b_pas * (1 - lam)) - 1)`.

**Why it is necessary.** The single exponential is bounded below by `-a_pas` however hard
the chamber is squeezed: about 1 kPa of restoring stress against 50 kPa of active stress,
which cannot resist cavity obliteration.

**Justification.** Myocardium is a nearly incompressible solid folding in on itself at
small cavity volumes and resists that at least as stiffly as it resists stretch.

**Cost.** The mirrored branch is a choice, not a measurement. It is unconstrained by data,
and every diastolic validation gate is deliberately checked on the *unmodified* `lam >= 1`
branch so that none of them rests on it.

---

## 4.3 Layer 2: the one-fiber chamber

**Implemented.**

```
eps_f = (1/3) * log((V_lv + V_w/3) / (V_lv_ref + V_w/3))
lam   = exp(eps_f)
p_lv  = sigma_f / (1 + 3 * V_lv / V_w)
```

**Source.** Arts and colleagues [@arts1991onefiber], who showed that under rotational
symmetry and homogeneous wall loading the dimensionless ratio of fiber stress to cavity
pressure depends mainly on the cavity-to-wall volume ratio and is largely independent of
other geometric detail. That independence is what makes a zero-dimensional model
defensible rather than merely convenient.

**Simplifications.**

- *Homogeneous wall load.* No regional variation, so no fiber disarray, no regional
  fibrosis, no segmental wall-motion abnormality. In HCM, where hypertrophy is
  characteristically asymmetric, this is a real loss: the model has a uniformly thickened
  wall of the same total volume as an asymmetric one.
- *Thick-walled sphere for reported wall thickness.* Cavity and epicardium are concentric
  spheres and thickness is the difference of their radii. It returns 0.92 cm for the
  healthy reference and 1.59 cm for the HCM reference, both in the range echocardiography
  reports for those groups, with neither tuned to land there. But a sphere is not a
  ventricle, and the number should be read as "the wall thickness this much muscle around
  this much cavity implies" rather than as a septal measurement.
- *The reference cavity volume is treated as measured.* It is not directly measured
  clinically. It is placed in `MeasuredGeometry` because letting it float would smuggle a
  geometric unknown into the material inference the project is about. Recorded as a
  modelling choice, not a measurement, in the class docstring and here.

**Why this layer carries the thesis.** `V_w` and `V_lv` are both measured well
clinically, so they are pinned per virtual patient and never inferred. Only the Layer 1
material parameters and the drug clearance are hidden.

---

## 4.4 Layer 3: the closed circulation

**Implemented.** Three compartments (left ventricle, systemic arterial, venous return),
with cavity volume and arterial pressure integrated and the venous compartment closing
the balance algebraically:

```
V_ven = V_tot - V_lv - C_art * p_art
```

so total stressed volume is conserved to machine precision by construction rather than to
integrator tolerance by luck.

**Valves.** A softplus of width `VALVE_SMOOTH_MMHG` replaces `max(0, dp)`. A hard
rectifier has a kink that stiff solvers dislike and gradient-based methods cannot
differentiate through. Cost: a small backwards leak while the valve is shut, bounded by
`width * log(2) / R` and asserted small against stroke volume in `tests/test_layers.py`.

**The important correction: which compartment fills the ventricle.** An earlier version
set `C_ven` to a systemic venous compliance of about 100 mL/mmHg. That pins filling
pressure to a near-constant, because the reservoir absorbs any volume the ventricle
declines to accept at a cost of hundredths of a mmHg. The model then produced an HCM
ventricle whose end-diastolic pressure was **insensitive to its own passive stiffness**:
stiffening the tissue simply lowered end-diastolic volume until the pressure returned to
where it started. Since elevated filling pressure is the entire clinical problem in HCM,
that was fatal rather than cosmetic.

The compartment upstream of the mitral valve is the pulmonary venous bed and the left
atrium, whose combined compliance is roughly an order of magnitude smaller. With that
value a stiff ventricle refuses volume, the volume backs up, and filling pressure rises,
which is both the correct mechanism and the reason the disease is symptomatic.

**Cost.** The systemic veins, the right heart and the pulmonary circulation are absorbed
into one return compartment. This is the standard single-ventricle closed-loop
simplification. It means the model cannot represent pulmonary hypertension, right heart
failure, or ventricular interdependence, and that `V_tot` is a *stressed* volume, not
total blood volume.

---

## 4.5 Outflow obstruction

**Implemented.**

```
A_lvot = A0 * clip((V_lv / (V_w * c_ref))^p, A_min_frac, 1)
R_lvot = k_obs / A_lvot^2 * |Q_av|
Q_av   = positive root of  (k_obs / A_lvot^2) Q^2 + R_av Q - dp = 0
```

**`k_obs` is not a fitted parameter.** It is the simplified Bernoulli relation clinicians
use at the bedside, `gradient = 4 v^2` for `v` in m/s, rewritten for a flow in mL/s
through an area in cm^2: `k_obs = 4e-4 mmHg s^2 cm^4 / mL^2`. `tests/test_layers.py`
asserts the round trip.

**The implicit solve is exact.** `R_lvot` depends on `Q_av` and vice versa, but
substituting gives a plain quadratic whose positive root is closed-form. No inner Newton
loop, no convergence failure mode, no loss of smoothness.

**Two deviations from the specification's form, both forced.**

*Crowding as a ratio rather than an absolute volume.* Written in terms of `V_lv` alone,
the law cannot distinguish a small normal ventricle from an obstructed hypertrophic one,
and it gave a 20 mmHg resting gradient in a structurally normal heart. What narrows the
tract is a thick septum crowding a small cavity, which is `V_lv / V_w`.

*A power law rather than a linear collapse.* Peak gradient is `k_obs Q^2 / A^2`, so it
needs a small area *while flow is still high*, which is mid-ejection. With a linear
collapse the area only becomes small near the end of ejection when flow has nearly
stopped, and the model produced a 15 mmHg peak gradient for a patient it was meant to make
severely obstructive. No linear law fixes this: a healthy ventricle at end-systole reaches
a cavity-to-wall ratio of 0.28 while an HCM ventricle at end-*diastole* is already at 0.33,
so the two overlap and no threshold separates them. A cubic law discriminates on how
*fast* the area falls instead, which is the clinically correct distinction: obstruction is
mid-systolic, and the late-peaking dagger-shaped Doppler envelope is a consequence.

**Costs.** Systolic anterior motion of the mitral valve is not represented as a mechanism,
only its consequence. The area law is phenomenological: `A0`, `c_ref` and the exponent are
`assumed`, and they are the least well grounded constants in the model.

---

## 4.6 Drug

```
C_ss    = dose / CL
phi_eff = phi_baseline * (1 - E_max * C_ss / (EC50 + C_ss))
```

Steady state only. Costs are set out in `02_drug_and_dosing.md` section 2.8; the sharpest
is that the model can say *whether* a maintained dose takes a patient below the floor but
not *when*, while the clinical protocol is fundamentally about timing.

`DRUG_CL_L_PER_H` is **back-calculated** from the documented 350-700 ng/mL target band at
the 5 mg starting dose [@wang2024jaha_titration], not taken from a reported clearance. It
is labelled `calibrated` and the agreement between the model's 401 ng/mL at 5 mg and that
band is therefore *not* evidence of anything. `DRUG_CL_PM_FRACTION` is the one drug
constant that is `measured`: a 72% reduction in apparent clearance in CYP2C19 poor
metabolisers [@popPK2024].

`DRUG_E_MAX` and `DRUG_EC50_NG_PER_ML` are `assumed`. They were **not** tuned against any
published dose-to-ejection-fraction curve, which is what makes the comparison in
`05_validation_targets.md` section 5.4 an independent prediction.

---

## 4.7 Numerics

Fixed-step RK4 in normalised beat phase `tau = t / T`. Two reasons: every heart rate then
uses the same step count, so a cohort with mixed rates advances in lockstep and a
five-thousand-patient study is minutes rather than hours; and the beat boundary lands
exactly on a step boundary, so the steady-state comparison is exact rather than
interpolated.

The specification asks for `solve_ivp` with `Radau` or `BDF`. The system turns out to be
only mildly stiff (the fastest time constant is the ~10 ms arterial-valve product), the
fixed step is far inside the stability limit, and an adaptive per-patient solver cannot
be vectorised across a cohort. `STEPS_PER_BEAT = 400` was chosen by convergence study, not
by feel: every reported observable matches a 2400-step run to five significant figures and
the peak gradient, the most step-sensitive quantity, to 0.05%.
`tests/test_model.py::test_step_count_is_converged` re-runs that comparison.

---

## 4.8 Provenance table

Every constant in `src/hcmtwin/defaults.py`. Values in this table are checked against the
code by CI.

| Parameter | Symbol | Value | Units | Source | Confidence |
|---|---|---|---|---|---|
| `CA_DIAST_UM` | Ca_diast | 0.10 | uM | Conventional diastolic free calcium in ventricular myocytes; @campbell2018recruitment operating range | assumed |
| `CA_PEAK_UM` | Ca_peak | 1.00 | uM | Conventional peak systolic free calcium in ventricular myocytes | assumed |
| `CA_TAU_R_S` | tau_r | 0.110 | s | Calibrated to systolic duration and peak aortic flow, not to the calcium literature; see section 4.0 | calibrated |
| `PHI_HEALTHY` | phi | 0.35 | dimensionless | Calibrated to the healthy resting gates in 05; SRX/DRX direction from @anderson2018srx | calibrated |
| `PHI_HCM` | phi | 0.55 | dimensionless | Calibrated to the HCM emergence gate; SRX depopulation in HCM from @anderson2018srx, @toepfer2020hcm | calibrated |
| `K_PARK_TOT_PER_S` | k_park_tot | 30.0 | 1/s | Assumed: fast enough to equilibrate within a beat, slow enough to retain within-beat dynamics | assumed |
| `K_ATT_PER_S` | k_att | 48.0 | 1/s | Effective whole-cell attachment rate, calibrated to peak attached fraction and ejection | calibrated |
| `K_DET_PER_S` | k_det | 25.0 | 1/s | Effective detachment rate; reciprocal sets the 40 ms intrinsic relaxation time | calibrated |
| `HILL_N` | h | 3.0 | dimensionless | Hill coefficient of the force-calcium relation in intact myocardium | assumed |
| `CA50_REF_UM` | Ca50_ref | 0.60 | uM | Calibrated so peak activation sits below saturation at peak calcium | calibrated |
| `BETA_LEN` | beta_len | 1.50 | dimensionless | Length sensitivity of calcium affinity; magnitude consistent with @campbell2018recruitment length-dependent activation | calibrated |
| `BETA_OVERLAP` | beta_overlap | 4.00 | dimensionless | Ascending limb of the sarcomere force-length relation, about 4 per unit stretch at a 2.0 um reference; see section 4.2.1 | assumed |
| `OVERLAP_MAX` | overlap_max | 1.30 | dimensionless | Plateau of the force-length relation | assumed |
| `K_FORCE_PER_KPA` | k_force | 0.045 | 1/kPa | Force-dependent recruitment, mechanism from @campbell2018recruitment; magnitude calibrated to a monotone Frank-Starling relation | calibrated |
| `K_XB_PER_S` | k_xb | 40.0 | 1/s | Cross-bridge distortion relaxation rate; with XB_HALF sets v_max | assumed |
| `XB_HALF` | xb_half | 0.055 | dimensionless | Gives an unloaded shortening velocity of 2.2 fiber lengths per second, in the range reported for slow beta-cardiac myosin | calibrated |
| `XB_MAX_GAIN` | xb_max_gain | 0.30 | dimensionless | Cap on force enhancement during stretch | assumed |
| `T_REF_KPA` | T_ref | 130.0 | kPa | Calibrated so realised peak active stress is 40-50 kPa and the healthy gates pass | calibrated |
| `A_PAS_KPA` | a_pas | 0.90 | kPa | Calibrated to the 5-12 mmHg healthy end-diastolic pressure range; see 05 section 5.3 for the [GAP] on human fiber-level constants | calibrated |
| `B_PAS` | b_pas | 10.0 | dimensionless | Calibrated with A_PAS_KPA; consistent with exponential fits in @sommer2015biomech, @hollander2010passivestiffness | calibrated |
| `A_PAS_HCM_KPA` | a_pas | 4.00 | kPa | Calibrated to the HCM filling-pressure gate; several-fold elevation consistent with @villemain2019stiffness | calibrated |
| `B_PAS_HCM` | b_pas | 10.0 | dimensionless | Left at the healthy value; the reference patient's elevated filling pressure is carried by the scale alone | calibrated |
| `MYOCARDIUM_DENSITY_G_PER_ML` | rho | 1.05 | g/mL | Conventional value used throughout the echocardiographic literature, e.g. @lang2015chamber mass formulae | measured |
| `V_W_HEALTHY_ML` | V_w | 140.0 | mL | 147 g, well inside the 115 g/m^2 normal limit at 1.9 m^2 | measured |
| `V_W_HCM_ML` | V_w | 245.0 | mL | 257 g; gives a 1.59 cm wall, above the 1.5 cm diagnostic threshold @ommen2024guideline | calibrated |
| `V_LV_REF_HEALTHY_ML` | V_lv_ref | 60.0 | mL | Calibrated so end-diastolic volume lands in the @lang2015chamber normal range | calibrated |
| `BSA_M2` | BSA | 1.90 | m^2 | Representative adult body surface area, used only to index volumes @lang2015chamber | assumed |
| `V_LV_REF_HCM_ML` | V_lv_ref | 60.0 | mL | Deliberately equal to the healthy value: the small HCM cavity must emerge from stiff tissue, not be assumed | assumed |
| `C_ART_ML_PER_MMHG` | C_art | 1.70 | mL/mmHg | Calibrated to a normal pulse pressure at the reference stroke volume | calibrated |
| `C_VEN_ML_PER_MMHG` | C_ven | 15.0 | mL/mmHg | Pulmonary venous plus left atrial compliance, not systemic venous; see section 4.4 | assumed |
| `R_SYS_MMHG_S_PER_ML` | R_sys | 1.04 | mmHg*s/mL | Calibrated to a mean arterial pressure of 85-95 mmHg at 4.5-5.5 L/min | calibrated |
| `R_AV_MMHG_S_PER_ML` | R_av | 0.006 | mmHg*s/mL | Assumed small non-obstructive valve loss; the dominant aortic loss is the Bernoulli term | assumed |
| `R_MV_MMHG_S_PER_ML` | R_mv | 0.006 | mmHg*s/mL | Assumed, as for R_av | assumed |
| `V_TOT_ML` | V_tot | 394.0 | mL | Stressed volume, calibrated to the healthy filling pressure and end-diastolic volume | calibrated |
| `VALVE_SMOOTH_MMHG` | eps | 0.30 | mmHg | Assumed softplus width; leak bounded and tested against stroke volume | assumed |
| `HR_BPM` | HR | 65.0 | bpm | Resting adult heart rate, mid the 60-70 bpm band used in 05 | assumed |
| `K_OBS_MMHG_S2_CM4_PER_ML2` | k_obs | 0.0004 | mmHg*s^2*cm^4/mL^2 | Exact restatement of the simplified Bernoulli relation gradient = 4 v^2 in the model's units | measured |
| `A0_LVOT_CM2` | A0 | 4.50 | cm^2 | Calibrated so a normal ventricle's peak outflow velocity is 0.7-1.4 m/s | calibrated |
| `LVOT_EXPONENT` | p | 3.00 | dimensionless | Assumed cubic collapse; a linear law cannot separate normal from obstructed, see section 4.5 | assumed |
| `CROWDING_REF` | c_ref | 0.30 | dimensionless | Assumed: above the healthy end-systolic ratio of 0.28, below the HCM mid-ejection ratio of 0.18 | assumed |
| `A_MIN_FRAC_LVOT` | A_min | 0.0005 | dimensionless | Numerical floor only; the physical choke is the area itself going to zero | assumed |
| `DRUG_E_MAX` | E_max | 0.70 | dimensionless | Assumed ceiling on the achievable shift in availability; NOT fitted to any trial curve | assumed |
| `DRUG_EC50_NG_PER_ML` | EC50 | 350.0 | ng/mL | Assumed at the lower edge of the 350-700 ng/mL therapeutic band @wang2024jaha_titration; NOT fitted to an outcome | assumed |
| `DRUG_CL_L_PER_H` | CL | 0.52 | L/h | Back-calculated from the 5 mg dose and the target concentration band @wang2024jaha_titration; not a reported clearance | calibrated |
| `DRUG_CL_PM_FRACTION` | - | 0.28 | dimensionless | Apparent clearance reduced by 72% in CYP2C19 poor metabolisers @popPK2024 | measured |
| `DOSE_MID_MG_PER_DAY` | - | 10.0 | mg/day | An approved maintenance dose two escalation steps above the 5 mg start @camzyos_label | measured |
| `STEPS_PER_BEAT` | - | 400 | dimensionless | Chosen by convergence study against 2400 steps; see section 4.7 | assumed |
| `MAX_BEATS` | - | 40 | dimensionless | Assumed beat cap; reaching it is reported as non-convergence, never hidden | assumed |
| `STEADY_TOL_ML` | - | 0.05 | mL | Assumed steady-state tolerance; tightening it by 500x does not move the answer, per test | assumed |
| `STEADY_TOL_MMHG` | - | 0.05 | mmHg | Assumed, as for STEADY_TOL_ML | assumed |
| `INIT_CAVITY_OFFSET_ML` | - | 55.0 | mL | Convergence aid only; a test asserts the converged beat is independent of it | assumed |
| `INIT_ARTERIAL_PRESSURE_MMHG` | - | 85.0 | mmHg | Convergence aid only, as above | assumed |
| `ANNULUS_LENGTH_CM` | L | 9.00 | cm | Assumed scale converting fiber lengthening rate to an e'-like velocity; gives about 9 cm/s in health, where tissue Doppler puts a normal septal e' @nagueh2016diastolic | assumed |
| `PROVOCATION_PRELOAD_FACTOR` | - | 0.75 | dimensionless | Assumed Valsalva analogue; effect size reported in clinical units by the tie-breaker table | assumed |
| `PROVOCATION_TACHYCARDIA_FACTOR` | - | 1.50 | dimensionless | Assumed: about 65 to about 98 bpm | assumed |
| `PROVOCATION_AFTERLOAD_FACTOR` | - | 1.25 | dimensionless | Assumed handgrip analogue | assumed |
| `PROVOCATION_EXERCISE_HR_FACTOR` | - | 1.85 | dimensionless | Assumed: about 65 to about 120 bpm, a submaximal stress-echo workload | assumed |
| `PROVOCATION_EXERCISE_VOLUME_FACTOR` | - | 1.10 | dimensionless | Assumed muscle-pump recruitment of unstressed volume | assumed |
| `PROVOCATION_EXERCISE_RESISTANCE_FACTOR` | - | 0.70 | dimensionless | Assumed exercise vasodilatation | assumed |
| `EF_INTERRUPTION_THRESHOLD` | - | 0.50 | dimensionless | Label instruction to interrupt if ejection fraction falls below 50% @camzyos_label | measured |
| `LVOT_OBSTRUCTIVE_THRESHOLD_MMHG` | - | 30.0 | mmHg | Guideline definition of obstruction @ommen2024guideline | measured |
| `TRIAL_MIN_EF` | - | 0.55 | dimensionless | EXPLORER-HCM enrolment criterion @olivotto2020explorer | measured |
| `TRIAL_MIN_WALL_THICKNESS_CM` | - | 1.30 | cm | Diagnostic threshold with family history or positive genotype @ommen2024guideline | measured |
| `TRIAL_MIN_RESTING_GRADIENT_MMHG` | - | 30.0 | mmHg | Obstruction threshold used as an enrolment criterion @ommen2024guideline | measured |
| `TRIAL_MIN_PROVOKED_GRADIENT_MMHG` | - | 50.0 | mmHg | Provoked gradient enrolment criterion @olivotto2020explorer | measured |

### Counting the confidence labels

Of the 65 constants: 10 are `measured`, 22 are `calibrated`, and 33 are `assumed`.

That distribution is itself a result and it should be read plainly. **Half the model is
assumed.** Most of the assumed constants are numerics, convergence aids, provocation
magnitudes and the obstruction area law, and none of the assumed constants is a hidden
parameter the analysis claims to recover. But it does mean the model's absolute outputs
should be treated as illustrative and its *relative* and *directional* outputs as the
substantive ones. The sensitivity analysis exists partly to show which of the assumed
constants the conclusions are actually exposed to.
