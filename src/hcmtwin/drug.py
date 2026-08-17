"""Dose to effective myosin availability.

Steady-state exposure only. No absorption phase, no accumulation transient, no
titration dynamics: a dose maps to a steady plasma concentration, and that concentration
maps to a shift in ``phi``. Real mavacamten takes weeks to reach steady state and the
approved titration schedule is built around that delay, so this model can say nothing
about *when* a patient crosses the ejection-fraction floor, only *whether* they do at a
given maintained dose. That limit is stated in the writeup.

``C_ss    = dose / CL``

``phi_eff = phi_baseline * (1 - E_max * C_ss / (EC50 + C_ss))``

Clearance is a hidden, per-patient parameter with a deliberately wide prior. It has to
be: CYP2C19 handles the majority of mavacamten metabolism, apparent clearance in a poor
metaboliser is roughly a quarter of that in a normal metaboliser, and so two patients on
the same pill can sit several-fold apart in exposure. "They simply clear the drug slowly"
is one of the three competing explanations for over-response that this project exists to
try to tell apart, and it only competes if the model lets clearance vary as widely as
people do.
"""

from __future__ import annotations

from .backend import SCALAR, Backend, Numeric
from .units import HOURS_PER_DAY, ML_PER_L, NG_PER_MG


def steady_state_concentration_ng_per_ml(
    dose_mg_per_day: Numeric,
    clearance_l_per_h: Numeric,
) -> Numeric:
    """Average steady-state plasma concentration, ng/mL.

    ``C_ss = dose_rate / CL``, with the unit chain written out:
    ``mg/day * 1e6 ng/mg / (L/h * 24 h/day * 1e3 mL/L)``.

    At the 5 mg/day starting dose and the population-typical clearance this returns about
    400 ng/mL, which sits inside the 350-700 ng/mL band the titration literature treats as
    the therapeutic target. That agreement is not a validation: the clearance constant was
    back-calculated from that same band, and it is labelled ``calibrated`` in the
    provenance table for precisely this reason.
    """
    return dose_mg_per_day * NG_PER_MG / (clearance_l_per_h * HOURS_PER_DAY * ML_PER_L)


def effective_phi(
    phi_baseline: Numeric,
    concentration_ng_per_ml: Numeric,
    e_max: float,
    ec50_ng_per_ml: float,
) -> Numeric:
    """Availability after drug, dimensionless.

    A saturating ``E_max`` model rather than a linear one, because a myosin inhibitor
    shifts an equilibrium: even at very high exposure some heads remain unparked, so the
    achievable reduction has a ceiling.
    """
    return phi_baseline * (
        1.0 - e_max * concentration_ng_per_ml / (ec50_ng_per_ml + concentration_ng_per_ml)
    )


def phi_after_dose(
    phi_baseline: Numeric,
    dose_mg_per_day: Numeric,
    clearance_l_per_h: Numeric,
    e_max: float,
    ec50_ng_per_ml: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Convenience composition of the two steps above."""
    del xp  # No transcendental operations needed; kept for signature symmetry.
    c_ss = steady_state_concentration_ng_per_ml(dose_mg_per_day, clearance_l_per_h)
    return effective_phi(phi_baseline, c_ss, e_max, ec50_ng_per_ml)


APPROVED_DOSE_LADDER_MG_PER_DAY: tuple[float, ...] = (0.0, 2.5, 5.0, 10.0, 15.0)
"""The dose levels simulated for every virtual patient.

Untreated plus the four maintenance doses the approved label permits. Not a physiological
constant, so it lives here rather than in :mod:`hcmtwin.defaults`; the label itself is
cited in ``docs/research/02_drug_and_dosing.md``."""
