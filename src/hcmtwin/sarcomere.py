"""Layer 1: the three-state myosin population and the stress it generates.

The state variables are fractions of the total myosin head population:

``S``
    **parked** -- folded back against the thick filament, unable to reach actin,
    consuming negligible ATP.
``D``
    **available** -- unparked and unbound, able to attach.
``A``
    **attached** -- bound to actin and generating force.

with ``S + D + A = 1`` at all times. This is enforced structurally by integrating all
three and checking the constraint, not by eliminating one of them, so that
``tests/test_properties.py`` is actually testing the integrator and the right-hand side
rather than testing an algebraic identity.

The availability knob
---------------------

The resting parked/available equilibrium is

``phi = k_park_off / (k_park_off + k_park_on)``

which is the fraction of *unattached* heads that are unparked. ``phi`` is the single
number that means "HCM" in this model. Healthy myocardium has a low ``phi``; HCM
mutations that destabilise the folded state raise it; a myosin inhibitor lowers it. The
two rate constants are derived from ``phi`` and a fixed total turnover rate, so disease
and treatment move one interpretable quantity in opposite directions rather than two
correlated ones in unclear directions.

Everything downstream -- hypercontractility, slowed relaxation, elevated energy cost -- is
a consequence of that one shift plus the passive stiffness, never an imposed phenotype.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .backend import SCALAR, Backend, Numeric

CA50_FLOOR_UM: float = 0.05
"""Lower clamp on the length-adjusted Ca50, uM.

Not a physiological constant: a numerical guard so that an extreme stretch in a corner of
the sampled parameter space cannot drive ``Ca50(lam)`` to zero or negative and produce a
divide-by-zero. It sits far below any calcium sensitivity the population sampler
generates; ``tests/test_sarcomere.py`` asserts the clamp is never active in the
physiological range."""


class MyosinRates(NamedTuple):
    """The two parking rate constants implied by an availability ``phi``."""

    k_park_off_per_s: float
    """Parked -> available, per second."""

    k_park_on_per_s: float
    """Available -> parked, per second."""


def rates_from_phi(phi: float, k_park_tot_per_s: float) -> MyosinRates:
    """Split a fixed total parking turnover into the two directional rates.

    ``k_park_off = phi * k_tot`` and ``k_park_on = (1 - phi) * k_tot``, so that
    ``phi = k_park_off / (k_park_off + k_park_on)`` by construction and the total
    turnover -- how fast the equilibrium is reached -- is held constant while the
    equilibrium itself moves.
    """
    if not 0.0 < phi < 1.0:
        raise ValueError(f"phi must lie in (0, 1), got {phi}")
    return MyosinRates(phi * k_park_tot_per_s, (1.0 - phi) * k_park_tot_per_s)


def ca50_um(
    stretch: Numeric,
    ca50_ref_um: float,
    beta_len: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Length-dependent calcium sensitivity, uM.

    ``Ca50(lam) = Ca50_ref * (1 - beta_len * (lam - 1))``

    With ``beta_len > 0`` a stretched sarcomere becomes *more* calcium-sensitive, so it
    develops more force at the same calcium. That is the cell-level origin of
    Frank-Starling behaviour in this model. Removing this term does not merely weaken the
    Frank-Starling response, it abolishes most of it, which is why the term is not
    optional.
    """
    return xp.maximum(ca50_ref_um * (1.0 - beta_len * (stretch - 1.0)), CA50_FLOOR_UM)


def thin_filament_activation(
    calcium_um_value: Numeric,
    stretch: Numeric,
    ca50_ref_um: float,
    hill_n: float,
    beta_len: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Fraction of thin-filament sites available for attachment, dimensionless in (0, 1).

    ``n(Ca, lam) = Ca^h / (Ca^h + Ca50(lam)^h)``
    """
    half = ca50_um(stretch, ca50_ref_um, beta_len, xp)
    ca_h = calcium_um_value**hill_n
    return ca_h / (ca_h + half**hill_n)


class SarcomereDerivatives(NamedTuple):
    """Time derivatives of the myosin populations plus the instantaneous ATP flux."""

    ds_dt: Numeric
    dd_dt: Numeric
    da_dt: Numeric
    atp_flux_per_s: Numeric
    """Rate of ATP consumption per myosin head, per second.

    One ATP is charged per attachment event, so the flux is the attachment flux
    ``k_att * n * D``. Parked heads are charged nothing, which is the whole point: the
    energetic penalty of HCM in this model comes from having moved heads out of the
    parked state, not from any separately imposed inefficiency."""


def force_recruitment_factor(
    fiber_stress_kpa: Numeric,
    k_force_per_kpa: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Multiplier on the parked-to-available rate from thick-filament mechanosensing.

    ``1 + k_force * max(sigma_f, 0)``

    Campbell, Janssen and Campbell (2018) found that a model in which the off-to-on
    transition rate rises linearly with force reproduces the length-dependent behaviour of
    permeabilised myocardium significantly better than one with a constant rate: the thick
    filament senses the load it is bearing and releases more heads. In the intact ventricle
    this is the dominant contributor to length-dependent activation, which is to say it is
    the origin of the Frank-Starling relation.

    Two reasons this matters more here than as a refinement.

    First, the specification's Layer 1 gets its Frank-Starling behaviour entirely from
    ``beta_len``, the length sensitivity of calcium affinity. That term alone is not
    enough: at a physiologically defensible ``beta_len`` the coupled model returned a
    stroke volume that was flat, then falling, as preload rose, because the geometric
    factor in the one-fiber relation works against it. Pushing ``beta_len`` high enough to
    compensate would have taken it far outside the range the myofilament literature
    supports.

    Second, it is what makes contractile reserve a real quantity rather than a label. A
    ventricle whose resting availability is already high has few parked heads left to
    recruit when load rises, so it responds to a stress maneuver weakly. That is precisely
    one of the three competing explanations for over-response the project exists to try to
    separate, and without this term the model could not express it.

    The feedback is positive but self-limiting: recruitment draws from a finite parked
    pool, so it saturates on its own as ``S`` empties.
    """
    return 1.0 + k_force_per_kpa * xp.maximum(fiber_stress_kpa, 0.0)


def derivatives(
    parked: Numeric,
    available: Numeric,
    attached: Numeric,
    calcium_um_value: Numeric,
    stretch: Numeric,
    fiber_stress_kpa: Numeric,
    k_park_off_per_s: Numeric,
    k_park_on_per_s: Numeric,
    k_att_per_s: float,
    k_det_per_s: float,
    k_force_per_kpa: float,
    ca50_ref_um: Numeric,
    hill_n: float,
    beta_len: float,
    xp: Backend = SCALAR,
) -> SarcomereDerivatives:
    """Right-hand side of the three-state myosin scheme.

    ``dS/dt = -k_park_off(sigma) * S + k_park_on * D``

    ``dD/dt =  k_park_off(sigma) * S - k_park_on * D - k_att * n(Ca, lam) * D + k_det * A``

    ``dA/dt =  k_att * n(Ca, lam) * D - k_det * A``

    with ``k_park_off(sigma) = k_park_off * (1 + k_force * sigma_f)``. At zero load the
    scheme reduces exactly to the specification's, so ``phi`` retains its clean reading as
    the unloaded resting availability.

    The three derivatives sum to zero identically, which is what keeps ``S + D + A = 1``
    through the integration.
    """
    n = thin_filament_activation(calcium_um_value, stretch, ca50_ref_um, hill_n, beta_len, xp)
    k_off_effective = k_park_off_per_s * force_recruitment_factor(
        fiber_stress_kpa, k_force_per_kpa, xp
    )
    attach_flux = k_att_per_s * n * available
    detach_flux = k_det_per_s * attached
    unpark_flux = k_off_effective * parked
    repark_flux = k_park_on_per_s * available

    ds_dt = repark_flux - unpark_flux
    da_dt = attach_flux - detach_flux
    dd_dt = unpark_flux - repark_flux - attach_flux + detach_flux
    return SarcomereDerivatives(ds_dt, dd_dt, da_dt, attach_flux)


def distortion_derivative(
    distortion: Numeric,
    fiber_strain_rate_per_s: Numeric,
    k_xb_per_s: float,
) -> Numeric:
    """Time derivative of mean cross-bridge distortion, per second.

    ``dx/dt = d(eps_f)/dt - k_xb * x``

    Shortening the fiber drags every attached head backwards along its power stroke, so
    ``x`` falls; cycling replaces distorted heads with freshly attached undistorted ones,
    so ``x`` decays towards zero at the cycling rate. Holding the fiber at fixed length
    therefore recovers ``x = 0`` and full isometric force.

    Crucially the right-hand side of the whole model stays explicit: stress is a function
    of the *state* ``x``, and ``dx/dt`` uses the strain rate that has already been computed
    from the flows. No implicit solve, no circular dependency.
    """
    return fiber_strain_rate_per_s - k_xb_per_s * distortion


def overlap_factor(
    stretch: Numeric,
    beta_overlap: float,
    overlap_max: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Fraction of maximal force available from thick-thin filament overlap.

    ``clip(1 + beta_overlap * (lam - 1), 0, overlap_max)``

    The classic sarcomere force-length relation, linearised on its ascending limb and
    plateaued above it: force climbs steeply as the sarcomere lengthens from about
    1.7 micrometres to about 2.1, then flattens.

    This term is absent from the specification's Layer 1 and its absence is not cosmetic.
    Cavity pressure in the one-fiber relation is ``sigma_f / (1 + 3 V/V_w)``, so at a fixed
    fiber stress a *smaller* cavity produces a *higher* pressure. With no length dependence
    in the active stress, the chamber's end-systolic pressure-generating capacity therefore
    rises monotonically as it empties: 141 mmHg at 100 mL, 224 at 42 mL, 314 at 13 mL for
    the healthy reference. That is a *negative* end-systolic elastance, and its consequence
    is that the ventricle never meets a force balance during ejection and empties until
    something else stops it. In the first coupled version something else did: the
    force-velocity term, which capped how far the chamber could travel in one systole and
    so pinned stroke volume at a constant, destroying the Frank-Starling relation the model
    is required to reproduce.

    Restoring the overlap term restores a positive end-systolic elastance, ejection
    terminates on a force balance the way it does in a real ventricle, and stroke volume
    rises with preload for the physiological reason rather than by accident.
    """
    return xp.clip(1.0 + beta_overlap * (stretch - 1.0), 0.0, overlap_max)


def active_stress_kpa(
    attached: Numeric,
    distortion: Numeric,
    stretch: Numeric,
    t_ref_kpa: float,
    beta_overlap: float,
    overlap_max: float,
    xb_half: float,
    xb_max_gain: float,
    xp: Backend = SCALAR,
) -> Numeric:
    """Active fiber stress, kPa. ``sigma_a = T_ref * A * overlap(lam) * g(x)``.

    The specification writes ``sigma_a = T_ref * A``, with no velocity dependence. That
    form is fine for an isometric preparation and untenable for a ventricle: a muscle with
    no force-velocity relation has no upper bound on shortening speed, and coupling one to
    a circulation makes a hypercontractile chamber eject straight past zero volume. This
    was not a theoretical worry; it is what the first coupled run of this model did.

    The gain ``g(x) = clip(1 + x / xb_half, 0, 1 + xb_max_gain)`` is the standard
    distortion form. At constant shortening velocity ``v`` the distortion settles at
    ``x = -v / k_xb``, so ``g`` falls linearly with velocity and reaches zero at
    ``v_max = k_xb * xb_half`` -- a Hill-type force-velocity relation, linearised. The
    upper cap represents force enhancement during stretch, bounded because a real
    cross-bridge detaches rather than bearing unlimited load.

    Recorded in ``docs/research/04_model_provenance.md`` as an addition to the
    specification's Layer 1, with its consequence: peak stroke volume and peak outflow
    velocity are both self-limiting, and the reference contractility ``T_ref`` had to be
    recalibrated upward to compensate for the loss during ejection.
    """
    gain = xp.clip(1.0 + distortion / xb_half, 0.0, 1.0 + xb_max_gain)
    return t_ref_kpa * attached * overlap_factor(stretch, beta_overlap, overlap_max, xp) * gain


def passive_stress_kpa(
    stretch: Numeric,
    a_pas_kpa: Numeric,
    b_pas: Numeric,
    xp: Backend = SCALAR,
) -> Numeric:
    """Passive fiber stress, kPa, stiff in both directions.

    For ``lam >= 1`` this is exactly the specification's relation,
    ``sigma_p = a_pas * (exp(b_pas * (lam - 1)) - 1)``: the titin-and-collagen tension
    that resists filling and sets the diastolic pressure-volume curve.

    For ``lam < 1`` the same relation is mirrored,
    ``sigma_p = -a_pas * (exp(b_pas * (1 - lam)) - 1)``, giving a compressive branch of
    equal steepness. The specification's single exponential is bounded below by
    ``-a_pas`` no matter how far the chamber is squeezed, which means it supplies at most
    about 1 kPa of restoring stress against 50 kPa of active stress and cannot stop cavity
    obliteration. Myocardium is a nearly incompressible solid folding in on itself at
    small cavity volumes and resists that at least as stiffly as it resists stretch, so
    the mirrored branch is both the physically defensible choice and the one that keeps
    end-systolic volume positive.

    The branch used in every diastolic validation gate is the unmodified ``lam >= 1`` one.
    """
    excess = stretch - 1.0
    tension = a_pas_kpa * (xp.exp(b_pas * excess) - 1.0)
    compression = -a_pas_kpa * (xp.exp(-b_pas * excess) - 1.0)
    return _select_branch(excess, tension, compression)


def _select_branch(excess: Numeric, if_stretched: Numeric, if_compressed: Numeric) -> Numeric:
    if isinstance(excess, np.ndarray):
        return np.where(excess >= 0.0, if_stretched, if_compressed)
    return if_stretched if excess >= 0.0 else if_compressed


def total_stress_kpa(
    attached: Numeric,
    distortion: Numeric,
    stretch: Numeric,
    t_ref_kpa: float,
    beta_overlap: float,
    overlap_max: float,
    xb_half: float,
    xb_max_gain: float,
    a_pas_kpa: Numeric,
    b_pas: Numeric,
    xp: Backend = SCALAR,
) -> Numeric:
    """Total fiber stress, kPa: active plus passive."""
    return active_stress_kpa(
        attached, distortion, stretch, t_ref_kpa, beta_overlap, overlap_max, xb_half,
        xb_max_gain, xp,
    ) + passive_stress_kpa(stretch, a_pas_kpa, b_pas, xp)


def resting_populations(phi: float, k_att_per_s: float, k_det_per_s: float, n_rest: float) -> tuple[
    float, float, float
]:
    """Analytic steady state of the three-state scheme at a fixed activation ``n_rest``.

    Used only to start the integration close to where it will end up, which cuts several
    beats off every steady-state search. Solving

    ``S = ((1 - phi) / phi) D``,  ``A = (k_att n / k_det) D``,  ``S + D + A = 1``

    gives ``D = 1 / (1/phi + k_att n / k_det)``.
    """
    ratio_attached = k_att_per_s * n_rest / k_det_per_s
    available = 1.0 / (1.0 / phi + ratio_attached)
    parked = ((1.0 - phi) / phi) * available
    attached = ratio_attached * available
    return parked, available, attached
