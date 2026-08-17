# 06. Measurement noise

**These numbers decide the answer.** An identifiability result is not a statement about a
model; it is a statement about what a model plus a set of error bars permits. Halve the
error bars and confounded parameters separate; double them and identifiable parameters
stop being identifiable. So this file is the load-bearing one, and every figure in it is
either sourced or explicitly marked as an assumption with its direction of conservatism
stated.

The values here are mirrored in `hcmtwin.observables.SPECS`, and
`tests/test_observables.py` asserts that every observable has an entry with both levels
populated.

---

## 6.1 Two noise levels, deliberately

The whole analysis runs twice.

**Realistic** is routine two-dimensional echocardiography in ordinary clinical practice,
read once, by one reader, on one visit. This is what most patients actually get.

**Optimistic** is three-dimensional echocardiography, or a core-laboratory read, or
averaged repeat acquisitions. This is what a trial gets, and it is the best case a
proposed clinical protocol could plausibly ask for.

Reporting both is not hedging. If a tie-breaker only works at the optimistic level, that
is a finding with a concrete implication: the proposal requires a core lab, and should say
so.

A caution about which quantity is being described. **Test-retest** variability repeats the
entire acquisition *and* analysis and is the relevant figure here, because a follow-up
study is a new acquisition. **Inter-observer** variability re-reads a single dataset and
is substantially smaller. Quoting the second where the first is meant would make every
conclusion in this project too optimistic, and it is a common enough slip to be worth
naming [@thavendiranathan2013reproducibility].

---

## 6.2 The table

| Observable | Realistic | Optimistic | Kind | Source | Confidence |
|---|---|---|---|---|---|
| Ejection fraction | 0.050 | 0.030 | absolute | @thavendiranathan2013reproducibility | sourced |
| End-diastolic volume | 8% | 5% | relative | @thavendiranathan2013reproducibility; @jenkins20043de | sourced |
| End-systolic volume | 10% | 6% | relative | @thavendiranathan2013reproducibility; @jenkins20043de | sourced |
| Wall thickness | 0.10 cm | 0.06 cm | absolute | assumption, see 6.4 | assumed |
| LV mass | 10% | 6% | relative | @jenkins20043de | sourced |
| Peak LVOT gradient | 35% | 20% | relative | @geske2009gradient; @kizilbash1998spontaneous | sourced, see 6.3 |
| End-diastolic pressure (invasive) | 2.0 mmHg | 1.0 mmHg | absolute | assumption, see 6.4 | assumed |
| E/e' | 15% | 9% | relative | @nagueh2016diastolic and the component variabilities below | sourced |
| Peak strain amplitude (GLS surrogate) | 8% | 5% | relative | @karlsen2019gls; @salte2023dlstrain | sourced |
| Mean arterial pressure | 5.0 mmHg | 3.0 mmHg | absolute | assumption, see 6.4 | assumed |
| Cardiac output | 12% | 8% | relative | assumption, see 6.4 | assumed |
| Heart rate | 3.0 bpm | 2.0 bpm | absolute | assumption, see 6.4 | assumed |
| Stroke volume (derived) | 12% | 7% | relative | propagated | derived |
| Thickness-to-cavity ratio (derived) | 12% | 7% | relative | propagated | derived |
| Stroke volume index (derived) | 12% | 7% | relative | propagated | derived |
| Stroke work (invasive) | 15% | 10% | relative | assumption | assumed |
| ATP cost per unit stroke work (research) | 20% | 12% | relative | assumption | assumed |

Fields marked *derived* are algebraic functions of others and are **excluded from the
likelihood** in the identifiability analysis. Including them would count one measurement
several times under different names and shrink the posterior for free. They are retained
for the sensitivity analysis, where no such double counting occurs.

---

## 6.3 The one that matters most: the outflow gradient

The outflow gradient is the obvious tie-breaker candidate. It is the observable most
reactive to provocation, it is the primary efficacy measurement clinically, and it is the
thing a Valsalva maneuver was invented to change.

It is also, by some distance, the noisiest thing in this project.

Geske and colleagues reported a coefficient of variation of **0.52** for repeated
measurement of the resting gradient and **0.46** for the provoked gradient, and found that
half of the patients studied changed obstruction status within a single haemodynamic
evaluation [@geske2009gradient]. Kizilbash and colleagues had already documented
substantial spontaneous variability with no intervention at all
[@kizilbash1998spontaneous].

Note what that variability *is*. It is not mostly reader error. It is real
beat-to-beat and hour-to-hour physiological variation in a load-dependent quantity: the
gradient genuinely differs between two honest measurements of the same patient. For the
purposes of an inference it is noise regardless of its origin, because it corrupts the map
from measurement to parameter identically.

We use 35% (realistic) and 20% (optimistic) rather than the reported 46-52%. That is a
**deliberate departure towards optimism**, on the grounds that the cited figures include
spontaneous variability over hours whereas a protocolised same-session rest-and-provocation
pair would see less of it. The direction matters: if the analysis concludes that the
gradient is *not* a useful tie-breaker even at 35%, that conclusion is safe, because the
true figure is likely worse. If it concludes the gradient *is* useful, the conclusion is
fragile and must be reported with the caveat that at the literature's own variability it
may not survive. This asymmetry is stated in the writeup.

---

## 6.4 The assumptions, each with its direction

Five entries above are marked `assumed`. Each is recorded here with what would resolve it.

**Wall thickness, 0.10 cm / 0.06 cm.** `[GAP]` We did not locate a test-retest study
reporting absolute reproducibility of maximal wall thickness in HCM, which is a
surprisingly specific measurement (the maximum over a segmented wall, not a single
parasternal reading). What would be needed is a repeat-acquisition study in an HCM cohort
reporting limits of agreement in cm. The assumed 0.10 cm is about 7% of a typical HCM wall
and is chosen to be neither trivially small nor implausibly large. It matters less than it
might, because wall thickness is a *measured* quantity in this project's framing rather
than a target of inference.

**End-diastolic pressure, 2.0 mmHg / 1.0 mmHg.** Catheter-derived, so instrument error is
small and the dominant variability is physiological and respirophasic. `[GAP]` No
test-retest series located. Assumed. This observable is excluded from the default
non-invasive feature set anyway, so it affects only the supplementary invasive analysis.

**Mean arterial pressure, 5.0 mmHg / 3.0 mmHg.** Brachial cuff. `[GAP]` Assumed;
the figure is chosen to reflect that a cuff reading is a single-visit snapshot of a
quantity with real short-term variability.

**Cardiac output, 12% / 8%.** Doppler LVOT velocity-time integral times area times heart
rate. The dominant error is the squared dependence on LVOT diameter. `[GAP]` Assumed;
a test-retest series for Doppler stroke volume would resolve it.

**Heart rate, 3.0 bpm / 2.0 bpm.** Electrocardiographic measurement error is negligible;
this figure represents genuine variation in resting rate between visits. Assumed.

**Stroke work and ATP cost.** Both are research measurements (pressure-volume
catheterisation, 31P magnetic resonance spectroscopy). `[GAP]` Assumed, deliberately
generous, and both are outside the routine set so neither enters the default analysis.

---

## 6.5 On E/e'

The component variabilities are good: a correlation coefficient of 0.94 with a coefficient
of variation of about 6% for the E wave, and 0.98 with about 7% for septal e'. The
inter-observer intraclass correlation for the ratio itself is more modest, around 0.70.
Combining the component figures in quadrature gives about 9%, which is our optimistic
level; the realistic level of 15% reflects the weaker agreement reported for the ratio.

A separate and larger caveat applies here, and it is not about noise. E/e' is a
*surrogate* for filling pressure and a fairly loose one: meta-analysis of its diagnostic
accuracy against invasively measured filling pressure found the relationship considerably
weaker than its clinical ubiquity suggests [@sharifov2016eeprime]. And the quantity this
model computes is a surrogate for that surrogate: the model has no atrium, so no A wave,
and no explicit long-axis motion, so no true annular velocity. What is computed is the
Doppler velocity implied by the peak atrioventricular pressure gradient, over the peak
myocardial lengthening rate scaled to a velocity.

Its *direction* is trustworthy and its absolute value is not comparable to a clinical
E/e'. Any conclusion resting on it should be read as "a filling-pressure surrogate helps"
rather than "E/e' of such-and-such helps".

---

## 6.6 The noise model

Additive Gaussian, independent across observables, with the standard deviations above.
Both simplifications are worth naming.

**Independence is false.** End-diastolic and end-systolic volume come from the same traced
image and their errors are correlated; so do stroke volume and ejection fraction. The
project mitigates the worst of it by excluding derived fields from the likelihood, but
residual correlation remains, and it makes the posterior slightly *tighter* than reality.
So the identifiability results are, in this specific respect, optimistic. Stated in the
writeup. `[GAP]` Resolving it properly would need a measurement-error covariance matrix
from a repeat-acquisition study reporting joint variability, which we did not locate.

**Gaussian is approximately true and its tails are not.** A misread apical view produces a
gross outlier, not a two-sigma deviation. A heavier-tailed likelihood would be more honest
about real practice; a Gaussian is what the analysis uses, and it will make the posterior
slightly narrower than a robust treatment would.

**Proportional versus absolute** follows the measurement. Volumes and gradients scale
with the thing measured; ejection fraction, pressures and heart rate do not. The kind is
recorded per observable in `SPECS`.
