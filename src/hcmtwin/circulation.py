"""Layer 3: a closed circulatory loop that conserves blood volume exactly.

Three compartments: the left ventricle, a lumped systemic arterial compartment, and a
lumped systemic venous compartment. The right heart and pulmonary circulation are absorbed
into the venous compartment, which is a real simplification and is recorded as one.

Volume bookkeeping
------------------

Rather than integrating three pressures and hoping their volumes add up, this module
integrates cavity volume and arterial pressure and then *closes* the balance:

``V_ven = V_tot - V_lv - C_art * p_art``

so total stressed volume is conserved to machine precision by construction rather than to
integrator tolerance by luck. ``tests/test_properties.py`` still checks it, because a
conservation law you have not tested is a conservation law you have assumed.

Preload is controlled by ``V_tot`` and afterload by ``R_sys``. Those two, plus heart rate,
are the only things a provocation maneuver turns.
"""

from __future__ import annotations

from .backend import SCALAR, Backend, Numeric


def venous_pressure_mmhg(
    total_volume_ml: Numeric,
    cavity_volume_ml: Numeric,
    arterial_pressure_mmhg: Numeric,
    c_art_ml_per_mmhg: float,
    c_ven_ml_per_mmhg: float,
) -> Numeric:
    """Venous pressure that closes the volume balance, mmHg."""
    venous_volume_ml = (
        total_volume_ml - cavity_volume_ml - c_art_ml_per_mmhg * arterial_pressure_mmhg
    )
    return venous_volume_ml / c_ven_ml_per_mmhg


def valve_flow_ml_per_s(
    pressure_drop_mmhg: Numeric,
    resistance_mmhg_s_per_ml: float,
    smooth_mmhg: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Forward flow through a resistive one-way valve, mL/s.

    A hard ``max(0, dp) / R`` has a kink at ``dp = 0`` that every stiff solver hates and
    that no gradient-based method can differentiate through. Replacing the rectifier with
    a softplus of width ``smooth_mmhg`` keeps the loop tractable now and differentiable
    later. The cost is a small backwards leak while the valve is shut, bounded by
    ``smooth_mmhg * log(2) / R`` and checked against the stroke volume in
    ``tests/test_circulation.py``.
    """
    return xp.softplus(pressure_drop_mmhg, smooth_mmhg) / resistance_mmhg_s_per_ml


def systemic_flow_ml_per_s(
    arterial_pressure_mmhg: Numeric,
    venous_pressure_mmhg_value: Numeric,
    r_sys_mmhg_s_per_ml: Numeric,
) -> Numeric:
    """Flow through the systemic bed, mL/s. Bidirectional: no valve here."""
    return (arterial_pressure_mmhg - venous_pressure_mmhg_value) / r_sys_mmhg_s_per_ml
