"""Analysis: sensitivity, identifiability, and the tie-breaker search.

Three questions, in order, each depending on the last.

:mod:`~hcmtwin.analysis.sensitivity` (D3)
    Which hidden parameters move which measurements at all? A parameter no measurement
    responds to cannot be recovered from any of them, whatever the inference method.

:mod:`~hcmtwin.analysis.identifiability` (D4)
    Of the parameters that do move measurements, which *combinations* move them the same
    way? Those combinations are invisible, and the parameters in them are confounded.

:mod:`~hcmtwin.analysis.tiebreaker` (D5)
    For each confounded pair, does any provocation maneuver break the tie, and is the
    discriminating signal bigger than the measurement error?
"""

from __future__ import annotations

__all__ = ["identifiability", "sensitivity", "tiebreaker"]
