# 00. The argument in one page

*This file is written last and is the shortest path into the project. For the full case
see `07_gap_statement.md`; for anything about a specific number see
`04_model_provenance.md`.*

---

## The question

Two patients with hypertrophic cardiomyopathy have the **same ejection fraction** the day
before myosin-inhibitor treatment starts. Escalate the same dose in both. One settles at
comfortable support. The other falls through the 50% ejection-fraction floor and dosing is
paused. About 5% of patients in the mavacamten long-term extension crossed that floor
transiently [@rader2024mavalte].

What property of these two hearts, invisible at baseline, decides the slope of their
dose-response curve?

Three explanations fit identical baselines and imply opposite management: the crasher had
**less contractile reserve**, or **stiffer tissue** masking a small poorly-filling cavity,
or they simply **clear the drug slowly** and got twice the exposure. The first two mean the
heart is fragile. The third means lower the dose and carry on.

---

## Why the existing tools cannot answer it

The published industry models are statistical: concentration in, ejection fraction out,
fitted to trial data. They are the right tool for designing a titration schedule and they
produced the one in current use [@merali2024posology; @wang2024jaha_titration]. But there
is no heart between the input and the output, so "too much drug" and "a ventricle with no
reserve" both show up as a steeper individual slope with no way to tell them apart.

Mechanistic models of HCM exist and are good [@margara2022mechanism]. They have not been
asked the inverse question: which of their parameters could a clinic actually recover?

---

## What this project does

Builds a three-layer mechanistic ventricle (myosin population, one-fiber chamber, closed
circulation), takes the molecular shift as the input, and lets the phenotype emerge.
Nothing writes down an ejection fraction.

Then asks the inverse question properly: sample a virtual population, generate the
measurements a clinic would obtain *with the error bars those measurements actually carry*,
and ask which hidden tissue parameters can be recovered and which combinations are
invisible.

**The trick that makes it tractable.** Clinical imaging measures shape accurately and
material not at all. So pin the shape (wall volume, cavity volume, mass) and infer only
the material (myosin availability, two passive stiffness parameters, calcium sensitivity,
drug clearance). Five unknowns against a dozen correlated observables is a solvable
inverse problem; with geometry floating it is not. The split is enforced as a type
distinction in the code, not as a convention.

---

## What has emerged so far

**The phenotype is not imposed.** Raising myosin availability and passive stiffness and
thickening the wall yields, as consequences: supranormal ejection fraction (0.75 against a
0.66 healthy reference), reduced stroke volume, elevated filling pressure, a raised
filling-pressure surrogate, reduced longitudinal strain despite preserved ejection
fraction, and roughly twice the ATP cost per unit of external work.

**Thick wall alone raises ejection fraction.** Give the HCM geometry *healthy* material and
ejection fraction is 0.758, slightly above the diseased reference. This is correct
physiology, and it is the sharpest statement of why the project exists: in a thick-walled
ventricle, a reassuring ejection fraction is close to uninformative. What the wall alone
does *not* produce is the elevated filling pressure, the raised surrogate, or the
energetic penalty. Those need the material.

**Two independent quantitative predictions land.** Applying the pivotal trial's own
enrolment criteria to the sampled cohort gives an over-responder rate near 4%, against a
published 5.2% [@rader2024mavalte]. The reference patient loses 5.3 ejection-fraction
points at the mid dose, against a placebo-corrected -4.8 points reported for aficamten in
SEQUOIA-HCM [@maron2024sequoia]. **Neither was fitted.** No parameter in the model was
calibrated against a trial outcome, and `04_model_provenance.md` records the confidence
label on every constant so that claim can be checked rather than trusted.

**The most attractive tie-breaker is also the noisiest thing in the problem.** The outflow
gradient is the most provocation-reactive observable and the primary efficacy measurement
clinically. It also has a measurement-to-measurement coefficient of variation around 0.5,
with half of patients changing obstruction status within a single haemodynamic study
[@geske2009gradient]. Any proposal that rests on it has to clear that bar, and the
tie-breaker table reports whether it does.

---

## The honest frame

Every outcome is a result. If the parameters separate at rest, a resting echocardiogram
contains the information and the task is to build the estimator. If they do not but a
maneuver fixes it, the project yields a testable clinical proposal with an effect size in
clinical units. If nothing fixes it, the project has found a real limit and named the
measurement that would be needed, which is the strongest finding of the three because it
redirects effort.

A third of the model's constants are `assumed`. Its absolute numbers are illustrative and
its directional and relative results are the substantive ones. Several modelling choices
make the posterior tighter than reality, so "these parameters remain confounded" is a safe
conclusion here and "this maneuver separates them" is a fragile one. Both are labelled as
such wherever they appear.
