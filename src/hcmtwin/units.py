"""Unit conventions and exact conversion factors.

Every quantity in ``hcmtwin`` carries its unit in the variable name (``_ml``, ``_mmhg``,
``_kpa``, ``_s``, ``_um``, ``_ng_per_ml``, ...). This module holds the *definitional*
conversions between them. Nothing here is a physiological constant: physiological
constants live in :mod:`hcmtwin.defaults` and each one has a provenance row in
``docs/research/04_model_provenance.md``.

The canonical internal units are:

=====================  ==========================
Quantity               Unit
=====================  ==========================
time                   second (s)
volume                 millilitre (mL)
pressure               millimetre of mercury (mmHg)
stress                 kilopascal (kPa)
flow                   mL / s
resistance             mmHg * s / mL
compliance             mL / mmHg
area                   cm^2
length                 cm
calcium concentration  micromolar (uM)
drug concentration     ng / mL
dose                   mg / day
clearance              L / h
mass                   gram (g)
=====================  ==========================

Rationale for the mixed system: pressures and volumes are reported in mmHg and mL by
every clinical source the model is validated against, while myocardial stress is reported
in kPa by every mechanics source it is built from. Converting at the single boundary
between :mod:`hcmtwin.sarcomere` (kPa) and :mod:`hcmtwin.chamber` (mmHg) is less
error-prone than converting inside every citation.
"""

from __future__ import annotations

# 1 mmHg == 133.322387415 Pa exactly, by the SI definition of the conventional millimetre
# of mercury (BIPM: rho_Hg = 13595.1 kg/m^3, g_n = 9.80665 m/s^2).
PA_PER_MMHG: float = 133.322387415
MMHG_PER_KPA: float = 1000.0 / PA_PER_MMHG  # ~7.500617
KPA_PER_MMHG: float = PA_PER_MMHG / 1000.0  # ~0.133322

SECONDS_PER_MINUTE: float = 60.0
MINUTES_PER_HOUR: float = 60.0
ML_PER_L: float = 1000.0
NG_PER_MG: float = 1.0e6
HOURS_PER_DAY: float = 24.0


def kpa_to_mmhg(value_kpa: float) -> float:
    """Convert a stress or pressure from kPa to mmHg."""
    return value_kpa * MMHG_PER_KPA


def mmhg_to_kpa(value_mmhg: float) -> float:
    """Convert a stress or pressure from mmHg to kPa."""
    return value_mmhg * KPA_PER_MMHG


def bpm_to_period_s(heart_rate_bpm: float) -> float:
    """Beat period T = 60 / HR, in seconds."""
    return SECONDS_PER_MINUTE / heart_rate_bpm


def ml_per_s_to_l_per_min(flow_ml_per_s: float) -> float:
    """Convert a flow from mL/s to L/min (the unit cardiac output is reported in)."""
    return flow_ml_per_s * SECONDS_PER_MINUTE / ML_PER_L
