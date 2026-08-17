# 02. Myosin inhibitors, their trials, and the safety threshold

---

## 2.1 Mechanism

Mavacamten is a small-molecule allosteric inhibitor of cardiac myosin. It shifts the
parked/available equilibrium towards parked, and the evidence that this is the operative
mechanism rather than a general poisoning of the motor is direct: mavacamten produced
roughly a twenty-fold reduction in the basal ATPase rate of a 25-heptad heavy meromyosin
construct, and in porcine and human cardiac fibers it increased the super-relaxed
population while lowering tension [@anderson2018srx].

So the drug and the disease act on the same axis in opposite directions. That is the fact
the model is built around, and it is why one parameter, `phi`, carries both.

Aficamten is a structurally distinct cardiac myosin inhibitor with the same broad
intent [@aficamten2024natcardio]. Its shorter half-life and different exposure-response
profile make it the interesting comparator for anything this project concludes about
clearance variability, and its trial results are used below as an independent anchor.

The model represents inhibition as a saturating effect on `phi`:

```
phi_eff = phi_baseline * (1 - E_max * C_ss / (EC50 + C_ss))
```

Saturating rather than linear because the drug shifts an equilibrium: even at very high
exposure some heads remain unparked, so the achievable reduction has a ceiling.

---

## 2.2 The interruption threshold and why it exists

The therapeutic point of a myosin inhibitor in HCM is to reduce contractility, because in
HCM contractility is too high. But reduction is a dial, not a switch, and pushed far
enough it converts an over-contracting heart into an under-contracting one.

The line sits at an ejection fraction below 50%. The approved labelling instructs that
treatment be interrupted if ejection fraction falls below 50%, and that treatment not be
initiated or up-titrated when it is below 55% [@camzyos_label]. In the trials this was a
prespecified interruption criterion assessed on the site-read echocardiogram.

Three things about that threshold are worth stating plainly, because they motivate the
whole project.

It is a *systolic* threshold applied to a *diastolic* disease. Ejection fraction is the
measurement being protected, and it is the one measurement HCM patients tend to have
plenty of.

It is a threshold on a noisy quantity. Test-retest variability in ejection fraction by
two-dimensional echocardiography exceeds 0.10, and is about 0.06 even for non-contrast
three-dimensional echocardiography [@thavendiranathan2013reproducibility]. A patient
measured at 0.52 might be at 0.47.

It is applied *after* the fact. The patient is dosed, then measured, then the dose is
paused. That is a controlled experiment run on a person, and the question this project
asks is whether some measurement taken beforehand could have substituted for it.

---

## 2.3 How often is it crossed, and by how much

In the interim analysis of the EXPLORER-LTE cohort of MAVA-LTE, 12 patients (5.2%)
developed transient reductions in ejection fraction below 50% leading to temporary
treatment interruption, and all recovered [@rader2024mavalte].

With longer follow-up, over 739 patient-years of exposure, 20 patients (8.7%) experienced
22 such episodes, an exposure-adjusted incidence of 2.77 per 100 patient-years. Ejection
fraction at the nadir ranged from 30% to 48%, 6 patients (2.6%) fell below 40%, and every
one recovered to at least 50% after interruption [@garciapavia2024longterm].

In the pivotal trial itself, a fall in ejection fraction below 50% was seen in about 7% of
mavacamten-treated patients and about 2% of placebo-treated patients
[@olivotto2020explorer].

For aficamten in SEQUOIA-HCM the corresponding figures were lower: core-laboratory
ejection fraction below 50% in 5 of 142 patients (3.5%) on aficamten versus 1 (0.7%) on
placebo, with no treatment interruptions for low ejection fraction [@maron2024sequoia].

**These are the numbers the model's over-responder rate should be compared against**, and
the comparison is only meaningful on a cohort constructed with the same enrolment
criteria. `population.py` applies them explicitly.

---

## 2.4 Magnitude of the ejection-fraction effect

SEQUOIA-HCM reported a placebo-corrected change in ejection fraction of **-4.8 percentage
points** (95% CI -6.4 to -3.3) alongside a large fall in obstruction: resting outflow
gradient from about 55 to about 20 mmHg and Valsalva gradient from about 86 to about
35 mmHg [@maron2024sequoia; @hegde2024aficamten_echo].

That pairing (a modest systolic cost for a large obstructive benefit) is the therapeutic
bargain, and reproducing its *ratio* is a far more demanding test of a mechanistic model
than reproducing either number alone.

**This project uses that figure as an independent comparison and not as a calibration
target.** No parameter in `defaults.py` was fitted to it. See
`04_model_provenance.md` for the confidence label on every drug parameter and
`tests/test_validation.py::test_exposure_response_direction_and_magnitude` for how loosely
the model is held to it.

---

## 2.5 The titration schedule, and where it came from

Mavacamten starts at 5 mg once daily. Permitted maintenance doses are 2.5, 5, 10 and
15 mg once daily, with 15 mg the maximum [@camzyos_label]. The model's dose ladder is
exactly this list plus zero.

Dose decisions in EXPLORER-HCM were made at weeks 8 and 14 on the basis of the Valsalva
outflow gradient, the plasma concentration, and the ejection fraction; the target plasma
concentration band is 350 to 700 ng/mL, and concentrations above 1000 ng/mL were initially
hypothesised to be excessive [@wang2024jaha_titration]. Assessment intervals are long
because the drug approaches steady state slowly, and the schedule was designed so that
most patients would be near steady state before the next escalation opportunity
[@wang2024jaha_titration].

Two implications for the model.

The steady-state-only pharmacokinetics is a real limitation with a real consequence: the
model can say *whether* a maintained dose takes a patient below the floor, and cannot say
*when*. Since the clinical protocol is fundamentally about timing, this bounds what the
project can claim.

The mid dose used for the over-responder label is 10 mg/day. It is a real maintenance dose
and sits two escalation steps above the 5 mg start.

---

## 2.6 Pharmacokinetics, and why clearance is a hidden parameter with a wide prior

Mavacamten is metabolised predominantly by CYP2C19 (about 74%), with CYP3A4 (about 18%)
and CYP2C9 (about 8%) accounting for the rest [@cyp_fractions].

CYP2C19 phenotype is the most influential covariate on exposure. Median apparent clearance
is reduced by about 72% in poor metabolisers relative to normal metabolisers
[@popPK2024]. The half-life difference is dramatic: about 72 hours in extensive
metabolisers, about 150 hours in intermediate metabolisers, and about 533 hours in poor
metabolisers [@wu2024cyp2c19].

So two patients handed the same pill can sit several-fold apart in exposure. "They simply
clear the drug slowly" is therefore a live competing explanation for over-response, and it
competes only if the model lets clearance vary as widely as people do. The prior on
`clearance_l_per_h` runs from 0.13 to 1.10 L/h, an eightfold span, and clearance is one of
the five hidden parameters the identifiability analysis tries to recover.

The distinction matters clinically because the three explanations imply different actions.
A patient with little contractile reserve or very stiff tissue has a fragile heart. A slow
metaboliser just needs a lower dose.

### Does genotyping help?

The natural response to the above is to genotype prospectively. In-silico dosing work
comparing strategies with and without prospective CYP2C19 genotyping has been reported
[@genotype2025jacc], as has exposure-response modelling for co-administration with CYP3A4
and CYP2C19 inhibitors [@merali2025cypinhibitors].

`[GAP]` We were not able to confirm the headline conclusion of the genotyping comparison
against its full text; the record we located is a conference abstract in a JACC supplement
rather than a full paper. What would be needed is the primary publication, reporting the
simulated rate of ejection-fraction excursions below 50% under genotype-guided versus
echo-guided titration with an uncertainty interval. We therefore make **no claim** in
either direction about whether prospective genotyping improves outcomes over echo-guided
titration, and the writeup does not rest on one. What we do say, from the pharmacokinetics
alone, is that genotype explains a large share of exposure variability and therefore
cannot explain the share of over-response variability attributable to the heart itself.
That last clause is exactly the gap this project addresses.

---

## 2.7 Real-world data

A one-year observational study of mavacamten in routine practice has been published
[@maron2025realworld], and longer real-world follow-up to 108 weeks exists. `[GAP]` We
have not confirmed the ejection-fraction excursion rate in the real-world cohorts against
their full texts, and real-world rates are the more relevant comparator for a model of an
unselected population than trial rates are. What would be needed is the excursion rate
below 50% per patient-year in a consecutive real-world series with a stated echo protocol.
Until then, the model's over-responder rate is compared only against the trial and
extension figures in section 2.3, which are the ones we have checked.

---

## 2.8 What the model represents, and what it does not

Represented: a maintained daily dose, a steady-state plasma concentration set by a
per-patient hidden clearance, and a saturating reduction in myosin availability.

Not represented, all of which bound the conclusions:

- Absorption and the multi-week approach to steady state, so no statement about timing.
- Titration as a feedback process. The model doses open-loop; the clinic does not.
- Drug-drug interactions with CYP inhibitors and inducers.
- Any direct drug effect on relaxation rate or calcium handling separate from the change
  in head availability. This one shows up as a discrepancy: the trials report improved
  filling parameters on treatment, while the model's end-diastolic pressure is essentially
  flat or very slightly higher. Reported in the writeup rather than smoothed over.
