"""The prescribed calcium transient.

This model does not simulate ion channels. The activator calcium waveform is imposed
analytically and identically on every beat, which is the single largest simplification in
the package and is documented as such in ``docs/research/04_model_provenance.md``.

What that buys: the whole cell layer collapses to three myosin states, a steady-state
beat solves in milliseconds, and every difference between two virtual patients is
attributable to myofilament material properties rather than to electrophysiology.

What that costs: no rate-dependent calcium handling, so the tachycardia maneuver captures
only the shortening of filling time and not the positive staircase or the calcium-load
changes that accompany real exercise. Any conclusion that leans on the tachycardia
maneuver has to be read with that in mind, and the writeup says so.
"""

from __future__ import annotations

from .backend import SCALAR, Backend, Numeric


def calcium_um(
    time_in_beat_s: Numeric,
    ca_diast_um: float,
    ca_peak_um: float,
    tau_r_s: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Intracellular free calcium at a given time within the beat, in uM.

    ``Ca(t) = Ca_diast + (Ca_peak - Ca_diast) * (t / tau_r) * exp(1 - t / tau_r)``

    The waveform rises to exactly ``Ca_peak`` at ``t = tau_r`` and decays smoothly
    thereafter, reaching about 9% of its amplitude by ``t = 5 tau_r``. With the default
    80 ms time constant that puts the calcium transient essentially over by 400 ms, which
    is what bounds systole in this model.

    Args:
        time_in_beat_s: Time since the start of the beat, seconds. Must be >= 0.
        ca_diast_um: Diastolic calcium, uM.
        ca_peak_um: Peak calcium, uM.
        tau_r_s: Time to peak, seconds.
        xp: Numeric backend.

    Returns:
        Calcium concentration, uM.
    """
    u = time_in_beat_s / tau_r_s
    return ca_diast_um + (ca_peak_um - ca_diast_um) * u * xp.exp(1.0 - u)


def beat_period_s(heart_rate_bpm: float) -> float:
    """Beat period T = 60 / HR, seconds.

    Heart rate is a top-level input rather than a derived quantity because tachycardia is
    one of the provocation maneuvers, and shortening T is precisely how a maneuver
    shortens filling time.
    """
    return 60.0 / heart_rate_bpm
