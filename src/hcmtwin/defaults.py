"""Every physiological constant in the model, in one place.

Hard rule (enforced by ``tests/test_provenance.py``): every module-level constant defined
here must have a row in the provenance table at the end of
``docs/research/04_model_provenance.md``, giving its value, units, source, and a
confidence label of ``measured``, ``calibrated``, or ``assumed``. A constant with no row
fails CI. Unit conversions are *not* physiological constants and live in
:mod:`hcmtwin.units`.

Confidence labels mean:

``measured``
    A direct experimental or clinical-cohort measurement, or an exact consequence of one
    (e.g. the Bernoulli coefficient).
``calibrated``
    A value fitted inside a cited model, or back-calculated here from cited clinical
    observations. It is not an independent measurement and must not be presented as one.
``assumed``
    Chosen by us. The rationale is recorded in the provenance table. Every ``assumed``
    entry is a candidate for the sensitivity analysis, not a fact.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# Calcium transient (hcmtwin.calcium)
# --------------------------------------------------------------------------------------

CA_DIAST_UM: float = 0.10
"""Diastolic intracellular free [Ca2+], micromolar."""

CA_PEAK_UM: float = 1.00
"""Systolic peak intracellular free [Ca2+], micromolar."""

CA_TAU_R_S: float = 0.110
"""Time constant of the prescribed transient, seconds. The waveform
``(t/tau) * exp(1 - t/tau)`` peaks exactly at ``t = tau``, so this is also the time to
peak calcium.

Longer than the measured time-to-peak of a human ventricular calcium transient, which is
nearer 40-60 ms, and the discrepancy is deliberate. The prescribed waveform has a single
time constant governing both the upstroke and the decay, so it cannot match both the fast
rise and the roughly 300 ms total duration. Duration was prioritised, because what the
mechanics actually depend on is how long active stress is available: at the measured
upstroke value the model ejected its whole stroke volume in 154 ms at a peak aortic flow
of 1000 mL/s, roughly twice physiological. Calibrated to systolic duration, not to the
calcium literature, and labelled accordingly in the provenance table."""

# --------------------------------------------------------------------------------------
# Layer 1: sarcomere (hcmtwin.sarcomere)
# --------------------------------------------------------------------------------------

PHI_HEALTHY: float = 0.35
"""Resting availability ``phi = k_park_off / (k_park_off + k_park_on)``, dimensionless:
the fraction of *unattached* myosin heads that are unparked and able to reach actin.
This single number is what "HCM" means in this model. Healthy is low, HCM is high, and
the drug lowers it."""

PHI_HCM: float = 0.55
"""Representative availability in untreated HCM myocardium, dimensionless."""

K_PARK_TOT_PER_S: float = 30.0
"""Total parked <-> available turnover rate ``k_park_off + k_park_on``, per second.
Fixed across the population so that ``phi`` alone sets the equilibrium; disease and drug
then move one interpretable quantity in opposite directions."""

K_ATT_PER_S: float = 48.0
"""Maximal actin attachment rate of an available head at full thin-filament activation,
per second."""

K_DET_PER_S: float = 25.0
"""Cross-bridge detachment rate, per second. Its reciprocal (40 ms) sets the intrinsic
mechanical relaxation time once calcium has fallen."""

HILL_N: float = 3.0
"""Hill coefficient of the steady-state force-calcium relation, dimensionless."""

CA50_REF_UM: float = 0.60
"""Calcium concentration for half-maximal thin-filament activation at reference
sarcomere length (``lambda = 1``), micromolar."""

BETA_LEN: float = 1.50
"""Length sensitivity of calcium affinity, dimensionless, in
``Ca50(lam) = Ca50_ref * (1 - beta_len * (lam - 1))``. Positive means a stretched
sarcomere binds calcium more avidly. This term is what produces Frank-Starling behaviour
at the whole-organ level."""

BETA_OVERLAP: float = 4.00
"""Slope of the active force-length relation on its ascending limb, dimensionless.

Cardiac force rises from roughly 20% to 100% of maximum as the sarcomere lengthens from
about 1.7 to 2.1 micrometres, which at a 2.0 micrometre reference length is a slope of
about 4 per unit stretch."""

OVERLAP_MAX: float = 1.30
"""Plateau of the active force-length relation, dimensionless: overlap stops improving
once the filaments are fully engaged, at a stretch of about 1.075 for the slope above."""

K_FORCE_PER_KPA: float = 0.045
"""Force sensitivity of the parked-to-available transition, per kilopascal.

Thick-filament mechanosensing: the rate at which myosin heads leave the parked state rises
with the stress the filament is bearing (Campbell, Janssen & Campbell 2018). This is the
dominant source of length-dependent activation and therefore of the Frank-Starling
relation, and it is also what gives "contractile reserve" a mechanistic meaning: a
ventricle whose parked pool is already depleted cannot recruit when load rises."""

K_XB_PER_S: float = 40.0
"""Rate at which cross-bridge distortion relaxes back to zero, per second. With
``XB_HALF`` it sets the maximum fiber shortening velocity,
``v_max = K_XB_PER_S * XB_HALF``."""

XB_HALF: float = 0.055
"""Cross-bridge distortion, in fiber-strain units, at which active stress falls to zero.
Together with ``K_XB_PER_S`` this puts unloaded shortening velocity at 2.2 fiber lengths
per second, which is in the range reported for slow beta-cardiac myosin."""

XB_MAX_GAIN: float = 0.30
"""Cap on force enhancement during lengthening, dimensionless: active stress may not
exceed 1.3 times its isometric value however fast the fiber is stretched."""

T_REF_KPA: float = 130.0
"""Isometric fiber stress produced when every myosin head is attached (``A = 1``),
kilopascal. Peak physiological ``A`` is well below 1, so realised peak active stress is
roughly 40-50 kPa."""

A_PAS_KPA: float = 0.90
"""Scale of the exponential passive fiber stress, kilopascal, in
``sigma_p = a_pas * (exp(b_pas * (lam - 1)) - 1)``."""

B_PAS: float = 10.0
"""Exponent of the passive fiber stress relation, dimensionless. Larger means a stiffer,
more sharply upcurving diastolic pressure-volume relation."""

A_PAS_HCM_KPA: float = 4.00
"""Representative passive stiffness scale in HCM myocardium, kilopascal."""

B_PAS_HCM: float = 10.0
"""Representative passive stiffness exponent in HCM myocardium, dimensionless.

Equal to the healthy value for the reference patient: in this calibration the elevated
filling pressure is carried entirely by the scale ``A_PAS_HCM_KPA``. The exponent is still
a per-patient hidden parameter, because both arms of the passive law can move in real
disease and because the identifiability analysis needs to ask whether a measurement can
tell the two arms apart. It largely cannot, which is one of the findings."""

# --------------------------------------------------------------------------------------
# Layer 2: chamber (hcmtwin.chamber)
# --------------------------------------------------------------------------------------

MYOCARDIUM_DENSITY_G_PER_ML: float = 1.05
"""Density of myocardium, g/mL. Converts the model's wall *volume* into the wall *mass*
an echocardiogram reports."""

V_W_HEALTHY_ML: float = 140.0
"""Representative healthy left-ventricular wall volume, mL (about 147 g of myocardium)."""

V_W_HCM_ML: float = 245.0
"""Representative wall volume in established HCM, mL (about 263 g)."""

V_LV_REF_HEALTHY_ML: float = 60.0
"""Reference (zero-fiber-strain) cavity volume of a healthy left ventricle, mL."""

BSA_M2: float = 1.90
"""Representative adult body surface area, m^2, used to index volumes."""

V_LV_REF_HCM_ML: float = 60.0
"""Reference (unloaded) cavity volume in established HCM, mL.

Deliberately close to the healthy value. The small cavity that characterises HCM is an
*end-diastolic* finding, and in this model it has to emerge from the stiff tissue
refusing to fill rather than from an unloaded cavity that was assumed small to begin
with. Assuming it away would be assuming the answer: the reference patient below reaches
an 82 mL end-diastolic volume from the *same* 60 mL unloaded volume as the healthy
reference, purely because the stiff tissue stops filling early."""

# --------------------------------------------------------------------------------------
# Layer 3: circulation (hcmtwin.circulation)
# --------------------------------------------------------------------------------------

C_ART_ML_PER_MMHG: float = 1.70
"""Total systemic arterial compliance, mL/mmHg."""

C_VEN_ML_PER_MMHG: float = 15.0
"""Compliance of the compartment that actually fills the left ventricle, mL/mmHg:
the pulmonary veins plus the left atrium, not the systemic veins.

This distinction is load-bearing and an earlier version got it wrong. Setting this to a
systemic venous compliance of about 100 mL/mmHg pins ventricular filling pressure to a
near-constant value, because the reservoir absorbs any volume the ventricle declines to
accept at a cost of hundredths of a millimetre of mercury. The model then produced an HCM
ventricle whose end-diastolic pressure was *insensitive to its own passive stiffness*:
stiffening the tissue simply lowered end-diastolic volume until the pressure came back to
where it started. Since elevated filling pressure is the entire clinical problem in HCM,
that is not a small error.

The compartment upstream of the mitral valve is the pulmonary venous bed and the left
atrium, whose combined compliance is roughly an order of magnitude smaller. With that
value a stiff ventricle refuses volume, the volume backs up, and the filling pressure
rises -- which is both the correct mechanism and the reason the disease is symptomatic.
The systemic veins and the right heart are absorbed into this same return compartment,
which is the standard single-ventricle closed-loop simplification and is recorded in the
provenance document."""

R_SYS_MMHG_S_PER_ML: float = 1.04
"""Systemic vascular resistance, mmHg*s/mL. This is the afterload knob."""

R_AV_MMHG_S_PER_ML: float = 0.006
"""Aortic valve resistance, mmHg*s/mL: a small, non-obstructive loss term."""

R_MV_MMHG_S_PER_ML: float = 0.006
"""Mitral valve resistance, mmHg*s/mL."""

V_TOT_ML: float = 394.0
"""Total *stressed* blood volume distributed between left ventricle, arteries and veins,
mL. This is the preload knob. It is not total blood volume: only the stressed fraction
generates pressure."""

VALVE_SMOOTH_MMHG: float = 0.30
"""Width of the smooth approximation to the valve diode, mmHg. Replacing ``max(0, dp)``
with a softplus of this width keeps the right-hand side differentiable and the system
tractable for stiff solvers, at the cost of a small non-physical regurgitant leak."""

HR_BPM: float = 65.0
"""Resting heart rate, beats per minute."""

# --------------------------------------------------------------------------------------
# Outflow obstruction (hcmtwin.obstruction)
# --------------------------------------------------------------------------------------

K_OBS_MMHG_S2_CM4_PER_ML2: float = 4.0e-4
"""Coefficient of the convective pressure loss through the outflow tract, in
mmHg*s^2*cm^4/mL^2. This is not a free parameter: it is the simplified Bernoulli relation
``gradient = 4 v^2`` (v in m/s) rewritten for flow in mL/s through an area in cm^2."""

A0_LVOT_CM2: float = 4.50
"""Unobstructed left-ventricular outflow tract cross-sectional area, cm^2."""

LVOT_EXPONENT: float = 3.00
"""Steepness of the outflow-tract collapse, dimensionless.

A cubic law. See :mod:`hcmtwin.obstruction` for why a linear law cannot work: a healthy
ventricle at end systole and an HCM ventricle at end diastole occupy the same
cavity-to-wall ratio, so no threshold separates them, and only the *rate* at which the
area falls distinguishes a chamber that obstructs mid-ejection from one that does not."""

CROWDING_REF: float = 0.30
"""Cavity-to-wall volume ratio at or above which the outflow tract is fully open,
dimensionless. A healthy ventricle is above it through the whole of mid-ejection
(0.54 at peak flow) and only narrows once flow has nearly stopped; an HCM ventricle is at
0.18 at peak flow, where the cubic law cuts its area to about a fifth."""

A_MIN_FRAC_LVOT: float = 5.0e-4
"""Floor on the outflow area as a fraction of ``A0_LVOT_CM2``, dimensionless.

Deliberately negligible. An earlier version used a floor of 0.12, which left a wide-open
channel at cavity obliteration and let a hypercontractile ventricle eject past zero
volume. Letting the area close instead makes the flow choke itself off: the Bernoulli
loss scales as ``1/A^2``, so ``Q -> A sqrt(dp / k_obs) -> 0``, smoothly and with no extra
machinery. The residual floor exists only to keep a division defined, and it caps the
attainable end-systolic volume at ``CROWDING_CRIT * V_w``, which for a 250 mL HCM wall is
about 10 mL: near-obliteration, which is what a severely obstructed ventricle actually
does."""

# --------------------------------------------------------------------------------------
# Drug (hcmtwin.drug)
# --------------------------------------------------------------------------------------

DRUG_E_MAX: float = 0.70
"""Maximum fractional reduction in ``phi`` achievable by the myosin inhibitor,
dimensionless. Saturating below 1 encodes that the drug shifts an equilibrium rather than
parking every head."""

DRUG_EC50_NG_PER_ML: float = 350.0
"""Steady-state plasma concentration producing half the maximal shift in ``phi``,
ng/mL."""

DRUG_CL_L_PER_H: float = 0.52
"""Population-typical apparent clearance CL/F, L/h, for a CYP2C19 normal metaboliser."""

DRUG_CL_PM_FRACTION: float = 0.28
"""Apparent clearance in a CYP2C19 poor metaboliser as a fraction of the normal-metaboliser
value, dimensionless: a 72% reduction, hence roughly 3.6-fold higher exposure from the
same dose."""

DOSE_MID_MG_PER_DAY: float = 10.0
"""The "mid dose" used to define the over-responder label, mg/day."""

# --------------------------------------------------------------------------------------
# Solver (hcmtwin.model)
# --------------------------------------------------------------------------------------

STEPS_PER_BEAT: int = 400
"""Fixed integration steps per cardiac cycle, dimensionless.

Integration is in normalised beat phase, so this is the same count at every heart rate.
Chosen by convergence rather than by feel: at 400 steps every reported observable agrees
with a 2400-step run to five significant figures, and the peak outflow gradient -- the
most step-sensitive quantity, because it depends on the square of a flow -- agrees to
0.05%. ``tests/test_model.py`` re-runs that comparison so the choice cannot silently
decay."""

MAX_BEATS: int = 40
"""Beat cap for the steady-state search, dimensionless. Reaching it is reported as
non-convergence, never silently."""

STEADY_TOL_ML: float = 0.05
"""Convergence tolerance on beat-to-beat change in end-diastolic volume, mL."""

STEADY_TOL_MMHG: float = 0.05
"""Convergence tolerance on beat-to-beat change in mean arterial pressure, mmHg."""

INIT_CAVITY_OFFSET_ML: float = 55.0
"""Starting cavity volume for the beat iteration, expressed as an offset above the
reference cavity volume, mL. Affects only how many beats the steady-state search needs;
``tests/test_model.py`` starts from deliberately wrong initial conditions and asserts the
converged beat is unchanged."""

INIT_ARTERIAL_PRESSURE_MMHG: float = 85.0
"""Starting arterial pressure for the beat iteration, mmHg. Same status as
``INIT_CAVITY_OFFSET_ML``: a convergence aid, not a result."""

# --------------------------------------------------------------------------------------
# Observable surrogates (hcmtwin.observables)
# --------------------------------------------------------------------------------------

ANNULUS_LENGTH_CM: float = 9.0
"""Nominal long-axis length, cm, converting the model's fiber lengthening rate (per
second) into an e'-like annular velocity (cm/s). It sets the absolute scale of the e'
surrogate and nothing else; with the default value a healthy ventricle returns about
9 cm/s, which is where tissue Doppler puts a normal septal e'."""

# --------------------------------------------------------------------------------------
# Provocation maneuvers (hcmtwin.provocation)
# --------------------------------------------------------------------------------------

PROVOCATION_PRELOAD_FACTOR: float = 0.75
"""Stressed blood volume during the preload-reduction maneuver, as a fraction of resting.
The Valsalva analogue."""

PROVOCATION_TACHYCARDIA_FACTOR: float = 1.50
"""Heart rate during the tachycardia maneuver, as a multiple of resting: about 65 to
about 98 beats per minute."""

PROVOCATION_AFTERLOAD_FACTOR: float = 1.25
"""Systemic vascular resistance during the handgrip analogue, as a multiple of resting."""

PROVOCATION_EXERCISE_HR_FACTOR: float = 1.85
"""Heart rate during the combined exercise maneuver, as a multiple of resting: about 65
to about 120 beats per minute, a submaximal stress-echo workload."""

PROVOCATION_EXERCISE_VOLUME_FACTOR: float = 1.10
"""Stressed blood volume during exercise, as a multiple of resting: the muscle pump
recruits volume from the unstressed reservoir."""

PROVOCATION_EXERCISE_RESISTANCE_FACTOR: float = 0.70
"""Systemic vascular resistance during exercise, as a multiple of resting: exercising
muscle beds vasodilate."""

# --------------------------------------------------------------------------------------
# Clinical thresholds
# --------------------------------------------------------------------------------------

EF_INTERRUPTION_THRESHOLD: float = 0.50
"""Ejection fraction below which trial protocols and the approved label interrupt dosing,
dimensionless."""

LVOT_OBSTRUCTIVE_THRESHOLD_MMHG: float = 30.0
"""Peak outflow gradient defining obstructive HCM, mmHg."""

TRIAL_MIN_EF: float = 0.55
"""Minimum ejection fraction for trial eligibility, dimensionless."""

TRIAL_MIN_WALL_THICKNESS_CM: float = 1.30
"""Minimum maximal wall thickness for a diagnosis of hypertrophic cardiomyopathy, cm.

The guideline threshold is 1.5 cm in isolation and 1.3 cm with a family history or a
positive genotype; the lower figure is used here so that genotype-positive patients with
milder hypertrophy are not excluded from the virtual cohort."""

TRIAL_MIN_RESTING_GRADIENT_MMHG: float = 30.0
"""Minimum resting peak outflow gradient for eligibility, mmHg."""

TRIAL_MIN_PROVOKED_GRADIENT_MMHG: float = 50.0
"""Minimum provoked peak outflow gradient for eligibility, mmHg, when the resting
gradient is below threshold."""
