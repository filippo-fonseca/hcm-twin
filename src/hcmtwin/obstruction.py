"""Dynamic left-ventricular outflow tract obstruction.

The clinically important nonlinearity in obstructive HCM is a positive feedback loop: as
the cavity empties, the thickened septum and the mitral apparatus crowd the outflow
channel; the narrower channel accelerates the blood passing through it; and the faster
flow both raises the pressure loss and, in the real heart, drags the mitral leaflet
further into the way. This module captures the first two links of that loop and leaves
the third (systolic anterior motion proper) out, which is stated plainly in the
provenance document.

``A_lvot = A0 * clip((c / c_ref)^p, A_min_frac, 1)``,  ``c = V_lv / V_w``

``R_lvot = k_obs / A_lvot^2 * |Q_av|``

``Q_av   = max(0, (p_lv - p_art) / (R_av + R_lvot))``

Solving it without iterating
----------------------------

``R_lvot`` depends on ``Q_av`` and ``Q_av`` depends on ``R_lvot``, so the flow is defined
implicitly. Substituting one into the other gives a plain quadratic in ``Q_av``,

``b Q^2 + R_av Q - dp = 0``   with   ``b = k_obs / A_lvot^2``

whose positive root is available in closed form. So the "implicit solve" the specification
asks for is done exactly and at the cost of one square root, with no inner Newton loop, no
convergence failure mode, and no loss of smoothness.

Why ``k_obs`` is not a fitted parameter
---------------------------------------

``k_obs = 4e-4 mmHg s^2 cm^4 / mL^2`` is the simplified Bernoulli relation clinicians
already use at the bedside -- ``gradient = 4 v^2`` for ``v`` in m/s -- rewritten for a flow
in mL/s through an area in cm^2. It is a unit conversion of an accepted clinical formula,
not a knob. The genuinely assumed parameters here are the three describing how the
*area* collapses.

Why crowding is measured as a ratio, and why the law is a power law
-------------------------------------------------------------------

The specification writes the collapse in terms of absolute cavity volume. Doing that makes
a small *normal* ventricle indistinguishable from an obstructed hypertrophic one: a
healthy heart at a 40 mL end-systolic volume would be handed the same outflow area as an
HCM heart at 40 mL, and the model reported a 20 mmHg resting gradient in a structurally
normal heart. What actually narrows the tract is a thick septum crowding a small cavity,
which is the dimensionless ratio ``c = V_lv / V_w``, not either volume alone.

The linear form of the specification then fails for a second, subtler reason. Timing.
Peak gradient is ``k_obs Q^2 / A^2``, so it needs a small area *while flow is still high*,
which is mid-ejection, not end-systole. With a linear collapse the area only becomes small
near the end of ejection, by which point flow has nearly stopped, and the model produced a
peak gradient of 15 mmHg for a patient it was supposed to make severely obstructive. There
is no linear law that fixes this: a healthy ventricle at end-systole reaches ``c = 0.28``
while an HCM ventricle at end-*diastole* is already at ``c = 0.33``, so the two overlap and
no single threshold separates them.

A power law does separate them, because it discriminates on how *fast* the area falls
rather than on where a threshold sits. With ``p = 3`` and ``c_ref = 0.30``, a healthy
ventricle is fully open through the whole of mid-ejection (``c = 0.54`` at peak flow) and
only narrows late when flow is negligible, while an HCM ventricle is at ``c = 0.18`` at
peak flow and sees its area cut to about a fifth. That is the clinical picture: obstruction
is a mid-systolic phenomenon in a small, thick-walled chamber, and the dagger-shaped
Doppler envelope it produces is late-peaking for exactly this reason.

Recorded as a deviation in ``docs/research/04_model_provenance.md``.
"""

from __future__ import annotations

from .backend import SCALAR, Backend, Numeric


def lvot_area_cm2(
    cavity_volume_ml: Numeric,
    wall_volume_ml: Numeric,
    a0_cm2: float,
    crowding_ref: float,
    exponent: float,
    a_min_frac: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Instantaneous outflow tract cross-sectional area, cm^2.

    Args:
        cavity_volume_ml: Instantaneous cavity volume V_lv, mL.
        wall_volume_ml: Wall volume V_w, mL.
        a0_cm2: Unobstructed outflow area.
        crowding_ref: Cavity-to-wall ratio at or above which the tract is fully open.
        exponent: Steepness of the collapse below that ratio.
        a_min_frac: Floor on the open fraction, for numerical safety only.
        xp: Numeric backend.
    """
    crowding = cavity_volume_ml / wall_volume_ml
    normalised = xp.maximum(crowding / crowding_ref, 0.0)
    return a0_cm2 * xp.clip(normalised**exponent, a_min_frac, 1.0)


def aortic_flow_ml_per_s(
    pressure_drop_mmhg: Numeric,
    lvot_area: Numeric,
    r_av_mmhg_s_per_ml: float,
    k_obs: float,
    smooth_mmhg: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Forward aortic flow through valve resistance plus outflow obstruction, mL/s.

    The positive root of ``b Q^2 + R_av Q - dp = 0``, with the driving pressure passed
    through the same softplus rectifier the non-obstructed valves use so that the
    right-hand side stays smooth at valve opening.
    """
    dp = xp.softplus(pressure_drop_mmhg, smooth_mmhg)
    b = k_obs / (lvot_area * lvot_area)
    discriminant = r_av_mmhg_s_per_ml * r_av_mmhg_s_per_ml + 4.0 * b * dp
    return (xp.sqrt(discriminant) - r_av_mmhg_s_per_ml) / (2.0 * b)


def lvot_gradient_mmhg(
    aortic_flow: Numeric,
    lvot_area: Numeric,
    k_obs: float,
) -> Numeric:
    """Instantaneous pressure gradient across the outflow tract alone, mmHg.

    ``k_obs * Q^2 / A^2``. This is the convective loss only -- it deliberately excludes
    the small resistive loss across the valve itself -- because it is the quantity
    continuous-wave Doppler estimates from peak velocity via ``4 v^2``, and reporting
    anything else under the name "gradient" would make the comparison to clinical data
    dishonest.
    """
    return k_obs * aortic_flow * aortic_flow / (lvot_area * lvot_area)
