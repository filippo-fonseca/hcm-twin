# 03. What has already been modelled, and what has not

Two traditions have grown up around this problem and they barely talk to each other. The
gap this project occupies is between them, so it is worth keeping them clearly apart.

---

## 3.1 The statistical tradition: population pharmacokinetics and exposure-response

### What these models contain

A population pharmacokinetic model describes how plasma concentration evolves from a dose
across a population, with between-subject variability partitioned into covariate effects
and unexplained random effects. An exposure-response model then maps concentration onto an
outcome, typically the outflow gradient for efficacy and the ejection fraction for safety,
usually through a saturating function with random effects on its parameters.

For mavacamten the published examples are substantial and were regulatory-grade. Merali
and colleagues used population pharmacokinetic and exposure-response analyses to recommend
the posology [@merali2024posology]. A dedicated pharmacokinetic characterisation in
patients with HCM supported the dose-titration scheme and identified CYP2C19 phenotype as
the dominant covariate on exposure, with clearance reduced by about 72% in poor
metabolisers [@popPK2024]. Later work extended the exposure-response models to
co-administration with CYP3A4 and CYP2C19 inhibitors [@merali2025cypinhibitors], and a
simulation study compared dosing with and without prospective CYP2C19 genotyping
[@genotype2025jacc].

### What they were used to decide

Real decisions, well made: the starting dose, the permitted maintenance doses, the
timing of assessments, the plasma concentration bands that trigger a dose change, and the
guidance for patients on interacting drugs [@wang2024jaha_titration; @camzyos_label].
These are the right tool for that job and they work.

### Their structural limits

Three, and none is a criticism of execution.

**There is no heart between the input and the output.** The map from concentration to
ejection fraction is fitted, not derived. Two patients with the same fitted random effect
are the same patient as far as the model is concerned, however different their ventricles
are. So the model cannot separate "this patient received too much drug" from "this patient
had a ventricle with no reserve": both appear as a steeper individual exposure-response
slope, and the model has no vocabulary for the difference.

**They cannot leave the covariate distribution of the trial population.** The random
effects are estimated on the enrolled cohort. Asked about a patient whose combination of
wall thickness, cavity size and tissue stiffness did not occur in the trial, the model
interpolates over a distribution rather than computing a consequence.

**They cannot answer a counterfactual about measurement.** "Would a Valsalva-provoked
gradient measured before the first dose have identified this patient?" is not a question a
concentration-to-outcome model can be asked, because the maneuver is not in it.

---

## 3.2 The mechanistic tradition: multiscale cardiac modelling

### Cell-level electromechanics of HCM variants under myosin modulators

Margara and colleagues built human electromechanical models of HCM variants and simulated
their response to myosin modulation, framing the work as mechanism-based and personalised
therapy [@margara2022mechanism]. Related work has produced metabolite-sensitive,
thermodynamically constrained models of cross-bridge cycling and electromechanics in
human induced pluripotent stem cell derived cardiomyocytes, which is the natural way to
represent the energetic side of the disease [@hipsc_metabolite2022].

These models are considerably richer than the one built here: they have ion channels,
explicit calcium handling, and thermodynamically consistent cross-bridge cycles. They are
also mostly cell-scale, and where they reach tissue scale they do not close a circulation
around the ventricle.

### Myofilament and cross-bridge models

Campbell, Janssen and Campbell's force-dependent recruitment model is the direct ancestor
of this project's Layer 1 [@campbell2018recruitment]. Its central finding, that the
off-to-on transition rate rises with force, is imported here essentially unchanged. The
follow-up work carrying that mechanism up to a whole ventricle and showing that it
steepens the end-systolic pressure-volume relationship [@campbell2020espvr] turned out to
be directly load-bearing: this project independently hit the failure that paper's
mechanism prevents.

Multiscale modelling of cardiomyopathy mutations, bridging molecular simulation and
whole-heart function, has been reviewed [@multiscale2017].

### Ventricular and circulatory mechanics

The one-fiber approach originates with Arts and colleagues, who showed that under
rotational symmetry and homogeneous wall loading the ratio of fiber stress to cavity
pressure depends mainly on the cavity-to-wall volume ratio and is largely independent of
other geometric detail [@arts1991onefiber]. That result is what makes a zero-dimensional
model defensible rather than merely convenient, and it is the reason this project can pin
the shape and infer only the material. The CircAdapt framework builds a full closed-loop
circulation on the same foundation [@circadapt2025], and open lumped-parameter solvers
exist for the circulatory side [@svzerodsolver2025]. The wider field is reviewed in
[@cardiacmechanics_review2025] (preprint).

For the passive side, Klotz and colleagues showed that volume-normalised end-diastolic
pressure-volume relations from 80 human hearts collapse onto a single curve,
`EDP = 28.2 * V_norm^2.79` [@klotz2006edpvr]. This project uses that as an independent
check on the shape of its passive relation rather than as a calibration target.

### What the mechanistic tradition has not done

It has largely not asked the inverse question. These models are used forward: given
parameters, predict behaviour. The question of which parameters a *clinical measurement
set* could recover, and which combinations are invisible to it, is asked far less often,
and to our knowledge has not been asked for HCM under myosin inhibition.

---

## 3.3 The intersection, stated explicitly

**Mechanistic models of HCM exist.** They are detailed, well validated at the cell scale,
and can simulate myosin modulation.

**Exposure-response models of myosin inhibitors exist.** They are regulatory-grade and
they set the dosing schedule in current use.

**The identifiability question between them has not been answered.** Nobody has taken a
mechanistic ventricle, given it hidden tissue-level parameters, generated the measurements
a clinic would actually obtain with the error bars those measurements actually carry, and
asked which of the hidden parameters can be recovered and which are confounded.

That question is the one that decides whether a mechanistic model of HCM can ever be
*personalised* from clinical data, as opposed to merely being illustrative. If the
parameters that determine over-response are not identifiable from a resting echo, then no
amount of model sophistication will convert a resting echo into a dose recommendation, and
the honest response is to say which additional measurement would be needed. If they are
identifiable, that is a concrete and testable clinical proposal.

Either answer is a result. That is the gap, and `07_gap_statement.md` develops it.

---

## 3.4 What this project deliberately does *not* try to out-do

It is worth being clear about where the prior work is simply better.

- **Electrophysiology.** This model prescribes a calcium transient and simulates no ion
  channels at all. The cell-level models cited above are far more faithful.
- **Cross-bridge thermodynamics.** The ATP accounting here is one ATP per attachment,
  which is a caricature next to a thermodynamically constrained cycle.
- **Spatial detail.** Zero-dimensional means no fiber disarray, no regional fibrosis, no
  systolic anterior motion of the mitral valve as an explicit mechanism.
- **Population pharmacokinetics.** Steady state only, one compartment implied, no
  absorption.

The trade is deliberate and it buys one thing: a steady-state beat in about 37
milliseconds, which makes a five-thousand-patient study with thirteen conditions each and
a Markov-chain posterior for fifty of them a laptop-scale computation. An identifiability
analysis is a question about many thousands of solves, not about one very good one.
