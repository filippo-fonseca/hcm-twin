# 07. The gap: what is known, what is not, and why this framing answers it

Written after the rest of the dossier. This becomes the introduction of the writeup.

---

## 7.1 The clinical situation, stated precisely

Picture two patients on the day before myosin-inhibitor treatment starts. Same symptoms,
same obstruction, same wall thickness, **same ejection fraction**. On the measurement that
governs their safety, they are one data point, not two.

Escalate the same dose in both and they diverge. One settles at comfortable support with a
relieved gradient. The other falls through the 50% ejection-fraction floor and dosing must
be paused. In the long-term extension of the pivotal mavacamten programme this happened to
5.2% of patients at interim analysis and to 8.7% over 739 patient-years, with nadir
ejection fractions between 30% and 48%; all recovered on interruption
[@rader2024mavalte; @garciapavia2024longterm]. In SEQUOIA-HCM the corresponding aficamten
figure was 3.5% [@maron2024sequoia].

So the question is not "is this drug safe" — the answer is that it is, with monitoring.
The question is: **what property of these two hearts, invisible at baseline, decides the
slope of their dose-response curve?**

---

## 7.2 Three explanations, consistent with identical baselines, implying different actions

**Less contractile reserve.** The crasher had fewer parked myosin heads to begin with. When
you start parking more, there is less spare capacity to absorb it. Mechanistically this is
a high resting `phi`, and it is real in this model rather than a metaphor, because
thick-filament mechanosensing means a depleted parked pool cannot be recruited when load
rises [@campbell2018recruitment].

**Stiffer tissue.** The reassuring ejection fraction was masking a small, poorly filling
cavity. A high ejection fraction computed on a small end-diastolic volume is not the same
reserve as the same fraction on a normal one, and diastolic myocardial stiffness is
substantially higher in HCM [@villemain2019stiffness].

**Slow clearance.** They got roughly twice the exposure from the same pill. CYP2C19 poor
metabolisers have apparent clearance reduced by about 72% and a terminal half-life of
about 533 hours against 72 hours in extensive metabolisers [@popPK2024; @wu2024cyp2c19].

The first two say the heart is fragile. The third says lower the dose and carry on. Today
these are distinguished by dosing the patient and repeating the echocardiogram, which is a
controlled experiment run on a person.

---

## 7.3 What is known

**The molecular mechanism.** The parked/available myosin equilibrium is characterised, the
super-relaxed state has a functional definition (basal ATPase 0.002-0.004 per second), HCM
mutations destabilise it, and mavacamten restabilises it [@anderson2018srx].

**The causal direction.** Hypertrophy is downstream. Early myosin inhibition prevented
hypertrophy, fibrosis and disarray from developing in genetic mouse models, and later
treatment produced partial regression [@green2016myk461].

**How to titrate safely.** Population pharmacokinetic and exposure-response modelling
produced the starting dose, the permitted maintenance doses, the assessment intervals and
the concentration bands now in the label [@merali2024posology; @popPK2024;
@wang2024jaha_titration; @camzyos_label].

**What the drug does on average.** Placebo-corrected ejection-fraction change of -4.8
points against a resting gradient falling from about 55 to about 20 mmHg
[@maron2024sequoia; @hegde2024aficamten_echo].

**How to model a ventricle.** The one-fiber relation [@arts1991onefiber], force-dependent
recruitment [@campbell2018recruitment; @campbell2020espvr], and the normalised human
end-diastolic pressure-volume relation [@klotz2006edpvr] between them supply a defensible
zero-dimensional chamber.

---

## 7.4 What is not known

**Which patient will over-respond, before dosing them.** Titration is reactive by
construction: dose, measure, adjust. Nothing in the current toolkit converts a baseline
study into a predicted slope.

**Whether the distinguishing property is measurable at all.** This is the deeper question
and it is prior to the first. Before asking "which measurement predicts over-response", one
should ask "is the underlying property recoverable from *any* non-invasive measurement
set, and if not, which combinations of properties are indistinguishable?" That question
has not been asked for this system.

**Whether stressing the patient helps.** Diastolic stress testing is an established idea
in a neighbouring problem: resting echocardiography can miss elevated filling pressures
that appear on exercise [@obokata2015eeprime]. Whether an analogous maneuver separates
contractile reserve from tissue stiffness from drug clearance in HCM is unasked.

---

## 7.5 Why the existing approaches structurally cannot answer it

**Exposure-response models have no heart in them.** They map concentration to outcome with
a fitted function and random effects. "Too much drug" and "a ventricle with no reserve"
both appear as a steeper individual slope, and the model has no vocabulary for the
difference. This is not a deficiency of execution; it is what a statistical model is. And
because the random effects are estimated on the enrolled cohort, the model cannot leave
that cohort's covariate distribution.

**Mechanistic models have not been asked the inverse question.** They are run forward:
given parameters, predict behaviour. Running them backwards, from noisy clinical
measurements to tissue parameters, requires committing to what a clinic actually measures
and to how badly it measures it. That commitment is uncomfortable, because it can produce
the answer "you cannot get there from here". It is also the only way to find out.

---

## 7.6 Why the identifiability framing works, and the trick that makes it tractable

Frame it as an inverse problem: hidden tissue parameters, a forward model, noisy
observations, and a posterior. Naively this is hopeless. A ventricle has geometry *and*
material, and with six or more unknowns and a handful of correlated measurements the
posterior is a ridge, not a point.

The trick is a division of labour that clinical imaging already performs.

**Imaging measures shape accurately and material not at all.** Wall thickness, cavity
volume and mass are what echocardiography and cardiac MRI are good at. Tissue stiffness
and the fraction of myosin heads available are what they cannot see.

So pin the shape and infer only the material. In this model that reduces the unknowns to
five: myosin availability, two passive stiffness parameters, calcium sensitivity, and drug
clearance. Five unknowns against a dozen correlated observables is a solvable inverse
problem. Six or more, with geometry floating, is not.

That reduction is the specific reason this project can produce a real answer rather than a
plausible-looking one, and it is enforced in the code as a type distinction rather than a
convention.

---

## 7.7 What the project can and cannot claim

**Can claim.** Which hidden parameters the model's observables are sensitive to, and how
much. Which pairs are confounded under realistic and optimistic measurement error. Whether
adding a specific provocation maneuver breaks a specific confound, and whether the
discriminating signal exceeds documented measurement variability. All of that is a
statement about *this model plus these error bars*, which is a real and falsifiable class
of statement.

**Cannot claim.** That a named variant maps to a numerical parameter: the mapping is
literature-derived and approximate, and claims are about regions of parameter space, never
about named patients. That the model's absolute numbers are right: a third of its constants
are `assumed` (see `04_model_provenance.md`), so directional and relative results are the
substantive ones. That anything here is validated against a patient, because it is not.

**A caution about the direction of the error.** Several modelling choices make the
posterior *tighter* than reality: the noise model treats observables as independent when
their errors are correlated, it is Gaussian where real error has heavier tails, and the
gradient noise used (35%) is deliberately below the 46-52% coefficient of variation
reported in the literature [@geske2009gradient]. So a conclusion of the form "these two
parameters remain confounded" is safe, and one of the form "this maneuver separates them"
is fragile and must be reported with that asymmetry attached.

---

## 7.8 Why every outcome is a result

**If the parameters separate cleanly at rest,** then a resting echocardiogram in principle
contains the information needed to predict the dose-response slope, and the task becomes
building the estimator. That is worth knowing and it is good news.

**If they do not separate, and a maneuver fixes it,** the project has produced a testable
clinical proposal: resting measurements cannot separate these two mechanisms, but adding
this specific maneuver can, and here is the expected effect size in clinical units against
documented measurement variability. That is the strongest possible outcome, and the
tie-breaker table is where it lives.

**If they do not separate, and no maneuver fixes it,** the project has found a real limit
and named it. That is a *stronger* finding than the first, because it says that no amount
of additional model sophistication will convert a non-invasive study into a dose
recommendation, and it redirects effort towards the measurement that would be needed.

There is no version of a correctly-executed run of this project that produces nothing. The
negative results are reported with the same prominence as the positive ones, and the
writeup is structured so that they have to be.
