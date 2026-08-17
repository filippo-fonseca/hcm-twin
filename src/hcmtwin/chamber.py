"""Layer 2: the one-fiber chamber, which is where the thesis lives.

The one-fiber approach (Arts et al., 1991) says that if the ventricular wall is loaded
roughly homogeneously, a *single* representative fiber stress and strain suffice to
describe the whole wall, and the map between them and cavity pressure and volume depends
on almost nothing except the dimensionless ratio of cavity volume to wall volume:

``eps_f = (1/3) log((V_lv + V_w/3) / (V_lv_ref + V_w/3))``

``sigma_f = p_lv * (1 + 3 V_lv / V_w)``

Why this layer carries the thesis
---------------------------------

Both ``V_lv`` and ``V_w`` are measured directly and well by clinical imaging. In this
project they are pinned per virtual patient and never inferred. Only the *material*
parameters of Layer 1 -- ``phi``, ``a_pas``, ``b_pas``, ``Ca50_ref`` -- and the drug
clearance are hidden. Six unknowns become five, of which only a few are strongly coupled,
and an unidentifiable mess becomes a solvable inverse problem. That reduction is the
specific reason this project can produce a real answer rather than a plausible-looking
one, and it happens here.
"""

from __future__ import annotations

import math

from .backend import SCALAR, Backend, Numeric
from .units import MMHG_PER_KPA


def fiber_strain(
    cavity_volume_ml: Numeric,
    wall_volume_ml: float,
    ref_cavity_volume_ml: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Representative natural fiber strain, dimensionless.

    Zero when the cavity sits at its reference volume, positive when the chamber is
    filled beyond it.
    """
    third_wall = wall_volume_ml / 3.0
    ratio = (cavity_volume_ml + third_wall) / (ref_cavity_volume_ml + third_wall)
    # log1p keeps precision when the chamber is near its reference volume, where the
    # ratio is close to 1 and the difference is the whole signal.
    return xp.log1p(ratio - 1.0) / 3.0


def stretch_from_volume(
    cavity_volume_ml: Numeric,
    wall_volume_ml: float,
    ref_cavity_volume_ml: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Sarcomere stretch ``lam = exp(eps_f)``, dimensionless."""
    return xp.exp(fiber_strain(cavity_volume_ml, wall_volume_ml, ref_cavity_volume_ml, xp))


def cavity_pressure_mmhg(
    fiber_stress_kpa: Numeric,
    cavity_volume_ml: Numeric,
    wall_volume_ml: float,
) -> Numeric:
    """Cavity pressure implied by a representative fiber stress, mmHg.

    ``p_lv = sigma_f / (1 + 3 V_lv / V_w)``, converted from kPa to mmHg at this single
    boundary. The denominator is the geometric amplification factor: a thick wall around
    a small cavity turns modest fiber stress into high cavity pressure, which is exactly
    the HCM situation and is not something the model has to be told.
    """
    return MMHG_PER_KPA * fiber_stress_kpa / (1.0 + 3.0 * cavity_volume_ml / wall_volume_ml)


def fiber_stress_from_pressure_kpa(
    cavity_pressure_mmhg_value: Numeric,
    cavity_volume_ml: Numeric,
    wall_volume_ml: float,
) -> Numeric:
    """Inverse of :func:`cavity_pressure_mmhg`, kPa. Used by tests and by calibration."""
    return (
        cavity_pressure_mmhg_value
        * (1.0 + 3.0 * cavity_volume_ml / wall_volume_ml)
        / MMHG_PER_KPA
    )


def wall_thickness_cm(cavity_volume_ml: float, wall_volume_ml: float) -> float:
    """Wall thickness under a thick-walled spherical assumption, cm.

    Reported rather than the raw wall volume so that the model emits the same quantity a
    sonographer measures. A healthy 140 mL wall around a 120 mL cavity gives about
    0.92 cm; a 250 mL HCM wall around an 80 mL cavity gives about 1.6 cm. Neither number
    was tuned to land there.
    """
    r_inner_cm = (3.0 * cavity_volume_ml / (4.0 * math.pi)) ** (1.0 / 3.0)
    r_outer_cm = (3.0 * (cavity_volume_ml + wall_volume_ml) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return r_outer_cm - r_inner_cm
