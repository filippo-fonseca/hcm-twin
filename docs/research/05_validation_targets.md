# 05. Validation targets

Every range asserted in `tests/test_validation.py` appears here with its source and the
population it describes. Where sources disagree, both are recorded and the one the tests
use is marked.

A ceiling on what this can establish: matching resting haemodynamics is necessary, not
sufficient. A model can reproduce a normal pressure-volume loop and still be wrong about
everything that matters under perturbation. The Frank-Starling, afterload, diastolic and
drug gates exist because they test *responses*, which are much harder to get right by
accident than a single operating point.

---

## 5.1 Normal resting haemodynamics

| Quantity | Accepted range | Population | Source | Notes |
|---|---|---|---|---|
| Ejection fraction | 0.52-0.72 (men), 0.54-0.74 (women) | Healthy adults | @lang2015chamber | Tests use **0.55-0.70**, a deliberately narrower band inside the guideline range, so the healthy reference sits mid-normal rather than at an edge. |
| End-diastolic volume | Upper limit 74 mL/m^2 (men), 61 mL/m^2 (women) | Healthy adults, 2D echo | @lang2015chamber | At the reference 1.9 m^2 body surface area the male upper limit is about 141 mL. Tests use **110-130 mL**. |
| End-systolic volume | Upper limit 31 mL/m^2 (men), 24 mL/m^2 (women) | Healthy adults, 2D echo | @lang2015chamber | About 59 mL at 1.9 m^2. Not gated directly; constrained through ejection fraction and stroke volume. |
| Stroke volume | 65-75 mL | Healthy adults at rest | Derived from cardiac output and heart rate below | Not an independently cited range; it is the arithmetic consequence of a 4.5-5.5 L/min output at 60-70 bpm, and is recorded as derived rather than measured. |
| Peak LV pressure | 110-130 mmHg | Healthy adults at rest | Follows from a normal systolic arterial pressure with a small outflow gradient | Derived, not independently cited. |
| End-diastolic pressure | 5-12 mmHg | Healthy adults at rest | Standard invasive range; also the pressure at which the Klotz normalised relation sits near normal filling volumes [@klotz2006edpvr] | Tests use **5-12 mmHg**. |
| Mean arterial pressure | 85-95 mmHg | Healthy adults at rest | Standard | Derived from a normal 120/80 brachial pressure. |
| Cardiac output | 4.5-5.5 L/min | Healthy adults at rest | Standard | Tests use this at 60-70 bpm. |
| Heart rate | 60-70 bpm | Healthy adults at rest | Standard | The reference is 65 bpm. |

`[GAP]` Several rows above are marked derived rather than cited. Standard textbook resting
haemodynamics turn out to be surprisingly hard to attach a single primary citation to,
because they are consensus rather than the result of one study. What would be needed is a
large contemporary normative haemodynamic series with invasive pressures. We have not
substituted a fabricated citation; the rows say derived and the arithmetic that produced
them is stated.

### Chamber geometry

| Quantity | Accepted range | Population | Source | Notes |
|---|---|---|---|---|
| LV mass index | Upper limit 115 g/m^2 (men), 95 g/m^2 (women) | Healthy adults | @lang2015chamber | About 219 g at 1.9 m^2 for men. The healthy reference wall volume of 140 mL is 147 g, comfortably normal. |
| Septal / posterior wall thickness | Abnormal above 1.0 cm (men), 0.9 cm (women) | Healthy adults | @lang2015chamber | The model returns 0.92 cm for the healthy reference. Tests gate **0.6-1.1 cm**. |
| Myocardial density | 1.05 g/mL | Human myocardium | Conventional value used throughout the echocardiographic literature to convert LV volume to mass | Recorded as conventional. |

---

## 5.2 Typical HCM geometry and function

| Quantity | Accepted range | Population | Source | Notes |
|---|---|---|---|---|
| Maximal wall thickness for diagnosis | >= 1.5 cm, or >= 1.3 cm with family history or positive genotype | HCM | @ommen2024guideline | The virtual cohort's eligibility filter uses 1.3 cm so that genotype-positive patients with milder hypertrophy are included. |
| Outflow gradient defining obstruction | >= 30 mmHg | HCM | @ommen2024guideline | Used as `LVOT_OBSTRUCTIVE_THRESHOLD_MMHG`. |
| Enrolment gradient | >= 50 mmHg at rest or with provocation | EXPLORER-HCM | @olivotto2020explorer | The virtual eligibility filter uses >= 30 resting **or** >= 50 provoked, which is the more permissive and more commonly stated form. |
| Baseline ejection fraction | >= 55% required for enrolment | EXPLORER-HCM | @olivotto2020explorer | Used as `TRIAL_MIN_EF`. |
| Baseline resting outflow gradient | About 55 mmHg | SEQUOIA-HCM | @hegde2024aficamten_echo | Model's HCM reference: 76 mmHg, i.e. towards the severe end of the trial distribution. |
| Baseline Valsalva gradient | About 86 mmHg | SEQUOIA-HCM | @hegde2024aficamten_echo | |
| Ejection fraction in HCM | Normal or supranormal while symptomatic | HCM | @olivotto2020explorer | The HCM gate requires >= 0.70 and above the healthy reference. |
| Global longitudinal strain | Reduced in HCM despite preserved ejection fraction | HCM | @maron2024valor_strain | The model's strain surrogate must fall from the healthy reference; it does, 0.209 to 0.154. |

`[GAP]` We did not obtain a normative distribution of end-diastolic *volume* in HCM from a
source we could check, only the qualitative statement that the cavity is small. What would
be needed is a cardiac-MRI cohort reporting end-diastolic volume index in obstructive HCM
with dispersion. The HCM gate therefore does not assert an end-diastolic volume range; it
asserts the *relative* facts (stroke volume below the healthy reference, filling pressure
above it), which is the weaker but supportable claim.

---

## 5.3 Passive pressure-volume relation

The independent check on the passive side is Klotz and colleagues' finding that
volume-normalised end-diastolic pressure-volume relations from 80 human hearts of varied
aetiology collapse onto a single curve, `EDP = 28.2 * V_n^2.79` mmHg [@klotz2006edpvr].

The test suite does not fit to this curve. It asserts the two structural properties the
curve has: the passive relation is monotone in volume, and it is convex. Both are checked
directly on the relation at fixed volume rather than through the coupled loop, so the
result is about the constitutive law rather than about the circulation around it.

Disputed point, recorded: reported exponential stiffness constants for passive myocardium
vary widely between preparations and between fitting conventions, and human ventricular
data are usually reported as orthotropic tissue-level constants rather than as the single
representative-fiber pair a one-fiber model needs [@sommer2015biomech;
@hollander2010passivestiffness]. `[GAP]` What would be needed is a representative-fiber
`(a, b)` pair fitted to human ventricular myocardium under the exact constitutive form
this model uses. Absent that, `A_PAS_KPA` and `B_PAS` are labelled `calibrated`, chosen so
the coupled model lands in the end-diastolic pressure range above, and then varied across
a wide prior so no conclusion depends on the specific pair.

---

## 5.4 Drug effect

| Quantity | Value | Population | Source | Notes |
|---|---|---|---|---|
| Placebo-corrected ejection-fraction change | -4.8 points (95% CI -6.4 to -3.3) | SEQUOIA-HCM, aficamten | @maron2024sequoia | **Independent comparison, not a target.** Model prediction at the mid dose: -5.3 points. |
| Resting gradient reduction | About 55 to about 20 mmHg | SEQUOIA-HCM | @hegde2024aficamten_echo | Model: 76 to 37 mmHg at the mid dose. |
| Ejection fraction below 50% | 5.2% (interim), 8.7% (extended) | MAVA-LTE | @rader2024mavalte; @garciapavia2024longterm | Model, on the trial-eligible virtual cohort: about 4%. |
| Ejection fraction below 50% | 3.5% on drug vs 0.7% on placebo | SEQUOIA-HCM | @maron2024sequoia | |
| Target plasma concentration | 350-700 ng/mL | Mavacamten titration | @wang2024jaha_titration | Model at the 5 mg starting dose: about 401 ng/mL. |

The three model figures in that table are **predictions**. No parameter was fitted to any
of them. `04_model_provenance.md` records the confidence label on every drug parameter and
none is `calibrated` against a trial outcome.

---

## 5.5 Response gates, which are the demanding ones

These have no numeric target; they assert the sign and monotonicity of a response, which
is what distinguishes a model from a curve.

| Gate | Requirement | Rationale |
|---|---|---|
| Frank-Starling | Raising stressed volume raises both end-diastolic volume and stroke volume, monotonically | The defining property of cardiac muscle in situ |
| Afterload | Raising systemic resistance lowers stroke volume and raises end-systolic volume | Standard pressure-volume physiology |
| Loop shape | Closes, traverses counter-clockwise, shows isovolumic phases | A loop that does not close is not a steady state |
| Diastolic | Passive relation monotone and convex; raising `a_pas` raises filling pressure at fixed volume | The constitutive law, checked directly |
| HCM emergence | Raising availability and stiffness with a thicker wall yields supranormal ejection fraction, reduced stroke volume, elevated filling pressure, elevated energy cost, reduced strain | The disease must be a consequence, not an input |
| Drug direction | Increasing dose lowers ejection fraction and lowers peak gradient, monotonically | The therapeutic bargain |

---

## 5.6 Ejection realism, added after the first calibration pass

Not in the specification, added because the first calibrated model passed every
haemodynamic gate while ejecting its entire stroke volume in 154 ms at a peak aortic flow
of about 1000 mL/s. Both are roughly twice physiological, and both were invisible to the
gates as originally written. This is recorded because it is a good illustration of how a
model can satisfy its checks and still be wrong.

| Quantity | Target | Basis |
|---|---|---|
| Peak LVOT velocity | 0.7-1.4 m/s | Normal Doppler LVOT velocity; a normal tract does not exceed about 1.5 m/s |
| Ejection duration | 250-340 ms | Normal at a resting heart rate |
| Peak aortic flow | 350-600 mL/s | Normal peak aortic flow |

`[GAP]` These three are stated from standard echocardiographic practice rather than from a
single citable normative series we checked. What would be needed is a normative Doppler
series reporting all three with dispersion. They are used as *calibration* targets and not
as pass/fail gates, and the constants they influenced are labelled `calibrated`.
