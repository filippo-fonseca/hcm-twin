"""hcmtwin: a mechanistic digital twin of a hypertrophic cardiomyopathy ventricle.

The question the package exists to answer: two patients share an ejection fraction the day
before myosin-inhibitor treatment starts, and under the same dose escalation one settles
at comfortable support while the other falls through the 50% floor. What measurement,
taken beforehand, could have told them apart?

Start with :func:`hcmtwin.model.simulate` and :func:`hcmtwin.observables.observe`.
"""

from __future__ import annotations

from .model import BeatResult, BeatTrace, simulate, simulate_cohort
from .observables import HiddenTruth, Observables, hidden_truth, observe
from .parameters import (
    HCM_GEOMETRY,
    HCM_MATERIAL,
    HEALTHY_GEOMETRY,
    HEALTHY_MATERIAL,
    RESTING_LOADING,
    HiddenMaterial,
    Loading,
    MeasuredGeometry,
    ModelConstants,
)

__version__ = "0.1.0"

__all__ = [
    "HCM_GEOMETRY",
    "HCM_MATERIAL",
    "HEALTHY_GEOMETRY",
    "HEALTHY_MATERIAL",
    "RESTING_LOADING",
    "BeatResult",
    "BeatTrace",
    "HiddenMaterial",
    "HiddenTruth",
    "Loading",
    "MeasuredGeometry",
    "ModelConstants",
    "Observables",
    "hidden_truth",
    "observe",
    "simulate",
    "simulate_cohort",
]
