# 01. What hypertrophic cardiomyopathy is, mechanically

Everything in this file is restated in our own words from the sources cited. Numbers carry
a citation; `[GAP]` marks a place where the literature does not supply what the model
needs, together with what we did instead.

---

## 1.1 The parked/available equilibrium

A myosin head in cardiac muscle has somewhere to be other than reaching for actin. A large
fraction of heads fold back against the thick filament backbone in a sequestered
conformation, structurally recognisable as the interacting-heads motif and functionally
recognisable by an extremely low ATP turnover. Anderson and colleagues fixed the
functional definition quantitatively: basal ATPase rates in the range 0.002 to 0.004 per
second identify the super-relaxed population, against a disordered-relaxed population that
turns over roughly an order of magnitude faster [@anderson2018srx].

A parked head is not merely inactive. It is inactive *and cheap*. That combination is the
reason the equilibrium matters clinically rather than only biophysically: it sets both how
hard the muscle can pull and what the pull costs.

The ratio between parked and available is what a healthy heart tunes and what HCM
mistunes. The model calls it `phi`, the fraction of unattached heads that are unparked,
and derives both parking rate constants from it plus a fixed total turnover so that
disease and drug move one interpretable number in opposite directions.

### What stabilises the parked state

Three things, and all three are places a mutation can act.

*The interacting-heads motif itself.* The two heads of a myosin dimer dock against each
other and against the filament backbone. Mutations at the interfaces destabilise the
docked arrangement directly [@anderson2018srx].

*Thick-filament mechanosensing.* The filament releases heads in proportion to the load it
is bearing. Campbell, Janssen and Campbell showed that a model in which the off-to-on
transition rate rises linearly with force reproduces the length-dependent behaviour of
permeabilised myocardium significantly better than one with a constant rate (F-test,
p < 0.001) [@campbell2018recruitment]. This is not a detail: it is the dominant molecular
route to length-dependent activation, and therefore to the Frank-Starling relation.

*Regulatory light chain phosphorylation and myosin-binding protein C.* Both modulate the
equilibrium. `[GAP]` We did not locate a source giving a quantitative mapping from
phosphorylation state to a shift in the parked fraction in human myocardium of the kind
the model would need to represent it as a separate parameter. What would be needed is a
titration of super-relaxed occupancy against regulatory light chain phosphorylation
stoichiometry in human ventricular tissue. In the absence of that, the model folds all
stabilising influences into the single parameter `phi` and treats variation in it as
variation in "how HCM this tissue is", which is a coarser claim than a mechanistic one and
is described that way throughout.

### How much does HCM move it?

Directionally the answer is settled. HCM mutations destabilise the parked state and
mavacamten restabilises it; the R403Q mutation sharply reduces super-relaxed occupancy in
cardiac fibers, and mavacamten increases it while lowering tension
[@anderson2018srx]. Toepfer and colleagues demonstrated depopulation of the super-relaxed
state in mouse left ventricular tissue carrying R403Q, V606M and R719W, and tied it to
cardiomyocyte energetics [@toepfer2020hcm].

`[GAP]` We could not obtain a defensible pair of numbers for the parked fraction in
*healthy human* versus *HCM human* ventricular myocardium. What is available is
mutation-specific, largely murine or reconstituted-protein, and reported with assay
conventions that do not translate cleanly into a single occupancy figure. What would be
needed is single-nucleotide-turnover measurements on human ventricular tissue from control
and genotyped HCM donors, reported as a percentage occupancy with confidence intervals.
Instead, the model's `phi = 0.35` healthy and `phi = 0.55` HCM are **calibrated**: chosen
so that the coupled model reproduces the resting haemodynamics and the HCM phenotype
recorded in `05_validation_targets.md`, and then varied across a wide prior
(0.28 to 0.62) so that no conclusion rests on the specific pair. They are labelled
`calibrated` in the provenance table and must not be quoted as measurements.

---

## 1.2 Three consequences of one shift

Too many heads available at once produces three things at once, and the model has to let
all three follow from the same parameter rather than imposing them.

**Hypercontractility.** More heads reachable means more attached at peak calcium, so more
force per beat. In the model this is the direct consequence of raising `phi`: the
steady-state attached fraction rises, active stress rises, and ejection fraction rises.

**Slower relaxation and a stiff diastole.** Relaxation requires heads to let go, and there
are more of them engaged. Separately, HCM myocardium is passively stiffer: non-invasive
shear-wave measurements put diastolic myocardial stiffness substantially higher in HCM than
in healthy adults [@villemain2019stiffness]. The model represents the passive component
explicitly as `a_pas` and `b_pas`, and the residual active tone at end-diastole falls out
of the kinetics.

**A higher ATP bill.** Every attachment cycle costs one ATP; a parked head costs nothing.
Moving heads out of the parked state therefore raises the energetic cost of a beat
directly, which Toepfer and colleagues connected to cardiomyocyte energetics and
metabolism [@toepfer2020hcm]. The model tracks the time integral of the attachment flux
over the beat, so cost per unit of external work is an output, not an input, and the
validation gate requires it to rise in the HCM reference. It does, by a factor of about
two.

---

## 1.3 Thick filament versus thin filament: two different parameters

Sarcomeric HCM is not one disease mechanically. Mutations in the beta-myosin heavy chain
and in myosin-binding protein C act principally on head availability. Mutations in the
thin filament (troponin T, troponin I, tropomyosin, actin) act principally on the calcium
sensitivity of the regulatory apparatus.

The model keeps them separate, and this is deliberate rather than decorative:
`phi_baseline` is the thick-filament axis and `ca50_ref_um` is the thin-filament axis.
Whether a clinical measurement set can tell the two apart is one of the questions the
identifiability analysis is built to answer, and it could not even be posed if they shared
a parameter.

A myosin inhibitor acts on the thick-filament axis. Whether that predicts a difference in
response between the two mutation classes is a model output; note the honesty constraint
in `07_gap_statement.md` that the mapping from a *named variant* to a numerical parameter
is literature-derived and approximate, so claims here are about regions of parameter
space, never about named patients.

---

## 1.4 Why hypertrophy is downstream

The thickened septum is the recognisable feature on imaging, and it is the *consequence*,
not the cause. The strongest evidence is interventional and comes from mice. Green and
colleagues showed that early treatment with the myosin inhibitor MYK-461 (now mavacamten)
prevented the development of hypertrophy, fibrosis and myocyte disarray in genetic mouse
models of HCM, and that treatment begun after hypertrophy was established produced partial
regression [@green2016myk461]. If hypertrophy were the primary lesion, a drug that acts
only on sarcomere contractility could not prevent it.

The reading is that the molecular shift raises wall stress and energy demand, and that
years of that provoke the remodelling response. The remodelling then feeds back: a thicker
wall around a smaller cavity generates more pressure for the same fiber stress, which in
the obstructive form narrows the outflow tract further.

Two consequences for the model.

It takes the molecular shift as an input and lets the mechanics emerge. Nothing writes down
an ejection fraction.

It does **not** simulate the remodelling itself. Wall volume is sampled per virtual
patient rather than grown over simulated months. That is a real limitation and is stated
as one: the model can say what a given geometry-plus-material combination does today, and
cannot say how that combination arose or where it is heading. Growth and remodelling is
listed in the specification as beyond the first three weeks and it has not been attempted.

---

## 1.5 The clinical shape of the disease

Two features of HCM drive every design choice downstream.

**Ejection fraction is normal or high while the patient is symptomatic.** The squeeze is
fine; the failure is in filling. EXPLORER-HCM required an ejection fraction of at least
55% for enrolment and its participants were symptomatic at NYHA class II or III
[@olivotto2020explorer]. HCM is a diastolic disease wearing a systolic disease's clothes,
and that is precisely why an ejection-fraction-based safety threshold is a blunt
instrument.

**Obstruction is dynamic and mid-systolic.** The gradient depends on loading and varies
substantially between measurements even with no intervention at all: repeated measurement
of the resting gradient has a coefficient of variation of about 0.52, and of the provoked
gradient about 0.46, with half of studied patients changing obstruction status within a
single haemodynamic evaluation [@geske2009gradient; @kizilbash1998spontaneous]. This is
the single most consequential noise figure in the project, because the outflow gradient is
otherwise the most attractive tie-breaker candidate.

---

## 1.6 The contrast with dilated cardiomyopathy, briefly

Dilated cardiomyopathy is the mirror image and the contrast sharpens what HCM is. There
the sarcomere generates too little force, the ventricle dilates rather than thickens,
ejection fraction falls, and the therapeutic goal is to increase contractility rather than
to reduce it. Consistently with the mechanism above, a dilated-cardiomyopathy mutation can
act by *stabilising* the interacting-heads motif and the super-relaxed state, parking heads
that the heart needs [@anderson2018srx, and the E525K work cited therein].

The same axis, in the opposite direction, with the opposite treatment. That is the
strongest available argument that the axis is the right one to model.
