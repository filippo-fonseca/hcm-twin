"""Provocation maneuvers, as modifiers on loading and nothing else.

Every maneuver here returns a new :class:`~hcmtwin.parameters.Loading`. None of them can
touch :class:`~hcmtwin.parameters.MeasuredGeometry` or
:class:`~hcmtwin.parameters.HiddenMaterial`, which is enforced by the type signature
rather than by discipline. That matters more than it sounds: the whole point of the
tie-breaker search is to ask whether stressing a patient reveals a hidden *tissue*
property, and it would be circular if a maneuver could quietly alter the tissue.

The magnitudes are stated as fractional changes and are ``assumed`` in the provenance
sense: they are chosen to be recognisable analogues of bedside maneuvers rather than
fitted to any particular protocol. The tie-breaker analysis reports the effect size each
one produces in clinical units, so a reader can rescale.

What these maneuvers are *not*
------------------------------

They are steady states at altered loading, not transients. A real Valsalva has four
phases and the interesting one lasts a few seconds; a real exercise test involves
catecholamines that change calcium handling and relaxation rate, neither of which this
model has. So "tachycardia" here captures the shortening of filling time and nothing else,
which is the mechanically dominant effect in a stiff ventricle but is not the whole of
exercise. Any tie-breaker that depends on the tachycardia maneuver has to be read with
that limit attached, and the writeup says so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import defaults as d
from .parameters import Loading


@dataclass(frozen=True, slots=True)
class Provocation:
    """A named modifier on loading."""

    name: str
    description: str
    clinical_analogue: str
    heart_rate_factor: float = 1.0
    blood_volume_factor: float = 1.0
    resistance_factor: float = 1.0

    def apply(self, loading: Loading) -> Loading:
        """Return the loading state this maneuver produces."""
        return loading.scaled(
            heart_rate_factor=self.heart_rate_factor,
            blood_volume_factor=self.blood_volume_factor,
            resistance_factor=self.resistance_factor,
        )


REST = Provocation(
    name="rest",
    description="Baseline resting conditions; no modification.",
    clinical_analogue="supine resting echocardiogram",
)

PRELOAD_REDUCTION = Provocation(
    name="preload_reduction",
    description="Stressed blood volume reduced, shrinking end-diastolic volume.",
    clinical_analogue="Valsalva strain phase (or standing from squat, or amyl nitrite)",
    blood_volume_factor=d.PROVOCATION_PRELOAD_FACTOR,
)

TACHYCARDIA = Provocation(
    name="tachycardia",
    description="Heart rate raised, shortening the time available for filling.",
    clinical_analogue="atrial pacing, or the chronotropic component of exercise",
    heart_rate_factor=d.PROVOCATION_TACHYCARDIA_FACTOR,
)

AFTERLOAD_INCREASE = Provocation(
    name="afterload_increase",
    description="Systemic vascular resistance raised.",
    clinical_analogue="sustained isometric handgrip",
    resistance_factor=d.PROVOCATION_AFTERLOAD_FACTOR,
)

EXERCISE = Provocation(
    name="exercise",
    description=(
        "Combined: heart rate up, systemic resistance down as muscle beds dilate, venous "
        "return up as the muscle pump recruits stressed volume."
    ),
    clinical_analogue="symptom-limited exercise stress echocardiogram",
    heart_rate_factor=d.PROVOCATION_EXERCISE_HR_FACTOR,
    blood_volume_factor=d.PROVOCATION_EXERCISE_VOLUME_FACTOR,
    resistance_factor=d.PROVOCATION_EXERCISE_RESISTANCE_FACTOR,
)

ALL_PROVOCATIONS: tuple[Provocation, ...] = (
    REST,
    PRELOAD_REDUCTION,
    TACHYCARDIA,
    AFTERLOAD_INCREASE,
    EXERCISE,
)
"""Every maneuver the tie-breaker search may propose, ``REST`` included as the control."""

STRESS_PROVOCATIONS: tuple[Provocation, ...] = ALL_PROVOCATIONS[1:]
"""The four maneuvers that actually change something."""

BY_NAME: dict[str, Provocation] = {p.name: p for p in ALL_PROVOCATIONS}


def apply_named(name: str, loading: Loading) -> Loading:
    """Apply a maneuver by name, failing loudly on a typo."""
    try:
        return BY_NAME[name].apply(loading)
    except KeyError:
        raise KeyError(f"unknown provocation {name!r}; known: {sorted(BY_NAME)}") from None


ProvocationFn = Callable[[Loading], Loading]
