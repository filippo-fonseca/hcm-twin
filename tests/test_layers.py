"""Unit tests for each layer in isolation, before any of them are coupled."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hcmtwin import chamber, circulation, drug, obstruction, sarcomere
from hcmtwin import defaults as d
from hcmtwin.backend import ARRAY, SCALAR
from hcmtwin.calcium import beat_period_s, calcium_um
from hcmtwin.units import MMHG_PER_KPA, bpm_to_period_s, kpa_to_mmhg, mmhg_to_kpa

# ======================================================================================
# Units
# ======================================================================================


def test_pressure_conversions_round_trip() -> None:
    for value in (0.0, 1.0, 16.0, 133.0):
        assert mmhg_to_kpa(kpa_to_mmhg(value)) == pytest.approx(value, rel=1e-12)


def test_one_atmosphere_is_760_mmhg() -> None:
    assert kpa_to_mmhg(101.325) == pytest.approx(760.0, rel=1e-6)


def test_beat_period() -> None:
    assert bpm_to_period_s(60.0) == pytest.approx(1.0)
    assert beat_period_s(120.0) == pytest.approx(0.5)


def test_a_physiological_pressure_is_a_physiological_stress() -> None:
    """120 mmHg is 16 kPa. If this ever fails, every stress in the model is wrong."""
    assert mmhg_to_kpa(120.0) == pytest.approx(16.0, abs=0.01)
    assert MMHG_PER_KPA == pytest.approx(7.5006, abs=1e-3)


# ======================================================================================
# Calcium
# ======================================================================================


def test_calcium_peaks_at_tau_and_returns_to_diastolic() -> None:
    tau = d.CA_TAU_R_S
    peak = calcium_um(tau, d.CA_DIAST_UM, d.CA_PEAK_UM, tau)
    assert peak == pytest.approx(d.CA_PEAK_UM, rel=1e-12)
    assert calcium_um(0.0, d.CA_DIAST_UM, d.CA_PEAK_UM, tau) == pytest.approx(d.CA_DIAST_UM)
    late = calcium_um(12.0 * tau, d.CA_DIAST_UM, d.CA_PEAK_UM, tau)
    assert d.CA_DIAST_UM < late < d.CA_DIAST_UM + 0.02


def test_calcium_never_dips_below_diastolic() -> None:
    times = np.linspace(0.0, 1.5, 400)
    values = calcium_um(times, d.CA_DIAST_UM, d.CA_PEAK_UM, d.CA_TAU_R_S, ARRAY)
    assert np.all(values >= d.CA_DIAST_UM - 1e-12)


# ======================================================================================
# Sarcomere
# ======================================================================================


def test_phi_recovers_from_its_rate_constants() -> None:
    for phi in (0.1, 0.35, 0.55, 0.9):
        rates = sarcomere.rates_from_phi(phi, d.K_PARK_TOT_PER_S)
        recovered = rates.k_park_off_per_s / (rates.k_park_off_per_s + rates.k_park_on_per_s)
        assert recovered == pytest.approx(phi, rel=1e-12)
        total = rates.k_park_off_per_s + rates.k_park_on_per_s
        assert total == pytest.approx(d.K_PARK_TOT_PER_S, rel=1e-12)


@pytest.mark.parametrize("phi", [0.0, 1.0, -0.1, 1.5])
def test_phi_outside_the_open_interval_is_rejected(phi: float) -> None:
    with pytest.raises(ValueError):
        sarcomere.rates_from_phi(phi, d.K_PARK_TOT_PER_S)


def test_derivatives_sum_to_zero() -> None:
    derivs = sarcomere.derivatives(
        0.5, 0.3, 0.2, 0.7, 1.05, 30.0,
        *sarcomere.rates_from_phi(0.4, d.K_PARK_TOT_PER_S),
        d.K_ATT_PER_S, d.K_DET_PER_S, d.K_FORCE_PER_KPA,
        d.CA50_REF_UM, d.HILL_N, d.BETA_LEN,
    )
    assert derivs.ds_dt + derivs.dd_dt + derivs.da_dt == pytest.approx(0.0, abs=1e-12)


def test_activation_is_sigmoidal_in_calcium() -> None:
    values = [
        sarcomere.thin_filament_activation(ca, 1.0, d.CA50_REF_UM, d.HILL_N, d.BETA_LEN)
        for ca in (0.05, 0.2, 0.6, 1.5, 5.0)
    ]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))
    assert values[2] == pytest.approx(0.5, abs=1e-9), "Ca50 must be the half-activation point"
    assert values[0] < 0.05 and values[-1] > 0.95


def test_stretch_raises_calcium_sensitivity() -> None:
    """Positive beta_len means a stretched sarcomere binds calcium more avidly."""
    slack = sarcomere.ca50_um(0.95, d.CA50_REF_UM, d.BETA_LEN)
    reference = sarcomere.ca50_um(1.00, d.CA50_REF_UM, d.BETA_LEN)
    stretched = sarcomere.ca50_um(1.12, d.CA50_REF_UM, d.BETA_LEN)
    assert stretched < reference < slack


def test_ca50_floor_is_never_active_in_the_physiological_range() -> None:
    for stretch in np.linspace(0.80, 1.25, 60):
        value = sarcomere.ca50_um(float(stretch), 0.45, d.BETA_LEN)
        assert value > sarcomere.CA50_FLOOR_UM * 1.5


def test_overlap_factor_rises_then_plateaus() -> None:
    low = sarcomere.overlap_factor(0.85, d.BETA_OVERLAP, d.OVERLAP_MAX)
    mid = sarcomere.overlap_factor(1.00, d.BETA_OVERLAP, d.OVERLAP_MAX)
    high = sarcomere.overlap_factor(1.30, d.BETA_OVERLAP, d.OVERLAP_MAX)
    assert low < mid < high
    assert mid == pytest.approx(1.0)
    assert high == pytest.approx(d.OVERLAP_MAX)
    assert sarcomere.overlap_factor(0.5, d.BETA_OVERLAP, d.OVERLAP_MAX) == pytest.approx(0.0)


def test_passive_stress_is_stiff_in_both_directions() -> None:
    stretched = sarcomere.passive_stress_kpa(1.15, d.A_PAS_KPA, d.B_PAS)
    neutral = sarcomere.passive_stress_kpa(1.0, d.A_PAS_KPA, d.B_PAS)
    compressed = sarcomere.passive_stress_kpa(0.85, d.A_PAS_KPA, d.B_PAS)
    assert neutral == pytest.approx(0.0, abs=1e-12)
    assert stretched > 0.0
    assert compressed < 0.0
    assert abs(compressed) == pytest.approx(stretched, rel=1e-9), (
        "the compressive branch mirrors the tensile one, so cavity obliteration is "
        "resisted as stiffly as filling"
    )


def test_force_velocity_reduces_force_during_shortening() -> None:
    isometric = sarcomere.active_stress_kpa(
        0.4, 0.0, 1.0, d.T_REF_KPA, d.BETA_OVERLAP, d.OVERLAP_MAX, d.XB_HALF, d.XB_MAX_GAIN
    )
    shortening = sarcomere.active_stress_kpa(
        0.4, -0.03, 1.0, d.T_REF_KPA, d.BETA_OVERLAP, d.OVERLAP_MAX, d.XB_HALF, d.XB_MAX_GAIN
    )
    lengthening = sarcomere.active_stress_kpa(
        0.4, 0.03, 1.0, d.T_REF_KPA, d.BETA_OVERLAP, d.OVERLAP_MAX, d.XB_HALF, d.XB_MAX_GAIN
    )
    assert shortening < isometric < lengthening
    assert shortening >= 0.0


def test_unloaded_shortening_velocity_matches_its_definition() -> None:
    v_max = d.K_XB_PER_S * d.XB_HALF
    distortion = -v_max / d.K_XB_PER_S
    stress = sarcomere.active_stress_kpa(
        0.4, distortion, 1.0, d.T_REF_KPA, d.BETA_OVERLAP, d.OVERLAP_MAX, d.XB_HALF,
        d.XB_MAX_GAIN,
    )
    assert stress == pytest.approx(0.0, abs=1e-12)


def test_force_recruitment_is_monotone_and_neutral_at_zero_load() -> None:
    assert sarcomere.force_recruitment_factor(0.0, d.K_FORCE_PER_KPA) == pytest.approx(1.0)
    assert sarcomere.force_recruitment_factor(-5.0, d.K_FORCE_PER_KPA) == pytest.approx(1.0)
    values = [
        sarcomere.force_recruitment_factor(s, d.K_FORCE_PER_KPA) for s in (0.0, 10.0, 30.0, 60.0)
    ]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))


def test_resting_populations_satisfy_the_constraint() -> None:
    parked, available, attached = sarcomere.resting_populations(0.35, d.K_ATT_PER_S, d.K_DET_PER_S, 0.01)
    assert parked + available + attached == pytest.approx(1.0, rel=1e-12)
    assert available / (parked + available) == pytest.approx(0.35, rel=1e-9)


# ======================================================================================
# Chamber
# ======================================================================================


def test_fiber_strain_is_zero_at_the_reference_volume() -> None:
    assert chamber.fiber_strain(60.0, 140.0, 60.0) == pytest.approx(0.0, abs=1e-15)
    assert chamber.stretch_from_volume(60.0, 140.0, 60.0) == pytest.approx(1.0, rel=1e-15)


def test_fiber_strain_matches_the_arts_relation() -> None:
    volume, wall, reference = 118.0, 140.0, 60.0
    expected = math.log((volume + wall / 3.0) / (reference + wall / 3.0)) / 3.0
    assert chamber.fiber_strain(volume, wall, reference) == pytest.approx(expected, rel=1e-12)


def test_cavity_pressure_inverts() -> None:
    stress = 16.0
    pressure = chamber.cavity_pressure_mmhg(stress, 110.0, 140.0)
    assert chamber.fiber_stress_from_pressure_kpa(pressure, 110.0, 140.0) == pytest.approx(
        stress, rel=1e-12
    )


def test_thick_wall_amplifies_pressure_for_the_same_fiber_stress() -> None:
    """Laplace: at fixed fiber stress a thicker wall around a smaller cavity makes more
    pressure. This is the geometric half of HCM hypercontractility."""
    thin = chamber.cavity_pressure_mmhg(40.0, 110.0, 140.0)
    thick = chamber.cavity_pressure_mmhg(40.0, 80.0, 245.0)
    assert thick > thin


def test_wall_thickness_matches_clinical_expectation() -> None:
    healthy = chamber.wall_thickness_cm(118.0, 140.0)
    hypertrophied = chamber.wall_thickness_cm(80.0, 245.0)
    assert 0.80 <= healthy <= 1.00, healthy
    assert 1.40 <= hypertrophied <= 1.80, hypertrophied


def test_wall_thickness_is_consistent_with_the_spherical_volumes() -> None:
    cavity, wall = 100.0, 180.0
    thickness = chamber.wall_thickness_cm(cavity, wall)
    r_inner = (3.0 * cavity / (4.0 * math.pi)) ** (1.0 / 3.0)
    r_outer = r_inner + thickness
    reconstructed = 4.0 / 3.0 * math.pi * (r_outer**3 - r_inner**3)
    assert reconstructed == pytest.approx(wall, rel=1e-9)


# ======================================================================================
# Circulation
# ======================================================================================


def test_valve_flow_is_forward_only_and_smooth() -> None:
    forward = circulation.valve_flow_ml_per_s(20.0, 0.006, 0.30)
    shut = circulation.valve_flow_ml_per_s(-20.0, 0.006, 0.30)
    assert forward == pytest.approx(20.0 / 0.006, rel=1e-6)
    assert 0.0 <= shut < 1e-6


def test_valve_leak_is_bounded_and_small() -> None:
    """The softplus costs a small backwards leak; it must stay negligible next to stroke volume."""
    leak = circulation.valve_flow_ml_per_s(0.0, d.R_AV_MMHG_S_PER_ML, d.VALVE_SMOOTH_MMHG)
    bound = d.VALVE_SMOOTH_MMHG * math.log(2.0) / d.R_AV_MMHG_S_PER_ML
    assert leak == pytest.approx(bound, rel=1e-9)
    assert leak * 0.3 < 15.0, "leak over a systole is a substantial fraction of stroke volume"


def test_valve_flow_has_no_kink() -> None:
    grid = np.linspace(-3.0, 3.0, 601)
    flows = circulation.valve_flow_ml_per_s(grid, 0.006, 0.30, ARRAY)
    second = np.diff(flows, 2)
    assert np.all(np.isfinite(second))
    assert np.max(np.abs(np.diff(second))) < 1.0e3


def test_systemic_flow_is_bidirectional() -> None:
    assert circulation.systemic_flow_ml_per_s(100.0, 8.0, 1.0) > 0.0
    assert circulation.systemic_flow_ml_per_s(5.0, 8.0, 1.0) < 0.0


# ======================================================================================
# Obstruction
# ======================================================================================


def test_lvot_area_shrinks_as_the_cavity_crowds() -> None:
    areas = [
        obstruction.lvot_area_cm2(
            v, 245.0, d.A0_LVOT_CM2, d.CROWDING_REF, d.LVOT_EXPONENT, d.A_MIN_FRAC_LVOT
        )
        for v in (90.0, 60.0, 40.0, 20.0)
    ]
    assert all(b < a for a, b in zip(areas, areas[1:], strict=False)), areas
    assert areas[0] == pytest.approx(d.A0_LVOT_CM2), "a well-filled cavity must be wide open"


def test_healthy_geometry_never_narrows_during_high_flow() -> None:
    """A structurally normal ventricle stays open through mid-ejection."""
    area = obstruction.lvot_area_cm2(
        75.0, d.V_W_HEALTHY_ML, d.A0_LVOT_CM2, d.CROWDING_REF, d.LVOT_EXPONENT,
        d.A_MIN_FRAC_LVOT,
    )
    assert area == pytest.approx(d.A0_LVOT_CM2)


def test_aortic_flow_solves_the_quadratic_exactly() -> None:
    """The closed-form root must satisfy the implicit equation it replaced."""
    for dp in (5.0, 25.0, 90.0):
        for area in (0.6, 1.5, 4.5):
            q = obstruction.aortic_flow_ml_per_s(
                dp, area, d.R_AV_MMHG_S_PER_ML, d.K_OBS_MMHG_S2_CM4_PER_ML2, 1e-9
            )
            r_lvot = d.K_OBS_MMHG_S2_CM4_PER_ML2 / area**2 * abs(q)
            assert q * (d.R_AV_MMHG_S_PER_ML + r_lvot) == pytest.approx(dp, rel=1e-9)


def test_aortic_flow_is_zero_when_the_valve_is_shut() -> None:
    q = obstruction.aortic_flow_ml_per_s(
        -30.0, 2.0, d.R_AV_MMHG_S_PER_ML, d.K_OBS_MMHG_S2_CM4_PER_ML2, d.VALVE_SMOOTH_MMHG
    )
    assert 0.0 <= q < 1.0


def test_gradient_is_the_bernoulli_relation() -> None:
    """gradient = 4 v^2 with v in m/s. If this drifts, the Doppler comparison is void."""
    area_cm2 = 1.2
    velocity_m_per_s = 3.0
    flow_ml_per_s = velocity_m_per_s * 100.0 * area_cm2
    gradient = obstruction.lvot_gradient_mmhg(
        flow_ml_per_s, area_cm2, d.K_OBS_MMHG_S2_CM4_PER_ML2
    )
    assert gradient == pytest.approx(4.0 * velocity_m_per_s**2, rel=1e-9)


# ======================================================================================
# Drug
# ======================================================================================


def test_steady_state_concentration_units() -> None:
    """5 mg/day at the population-typical clearance lands in the documented target band."""
    c_ss = drug.steady_state_concentration_ng_per_ml(5.0, d.DRUG_CL_L_PER_H)
    assert 350.0 <= c_ss <= 700.0, c_ss


def test_slow_metabolisers_get_more_exposure() -> None:
    normal = drug.steady_state_concentration_ng_per_ml(5.0, d.DRUG_CL_L_PER_H)
    poor = drug.steady_state_concentration_ng_per_ml(
        5.0, d.DRUG_CL_L_PER_H * d.DRUG_CL_PM_FRACTION
    )
    assert poor / normal == pytest.approx(1.0 / d.DRUG_CL_PM_FRACTION, rel=1e-9)
    assert poor / normal > 3.0


def test_effective_phi_saturates_below_baseline() -> None:
    phi0 = 0.55
    for dose in (0.0, 2.5, 5.0, 15.0, 1000.0):
        phi = drug.phi_after_dose(
            phi0, dose, d.DRUG_CL_L_PER_H, d.DRUG_E_MAX, d.DRUG_EC50_NG_PER_ML
        )
        assert 0.0 < phi <= phi0
    saturated = drug.phi_after_dose(
        phi0, 1.0e9, d.DRUG_CL_L_PER_H, d.DRUG_E_MAX, d.DRUG_EC50_NG_PER_ML
    )
    assert saturated == pytest.approx(phi0 * (1.0 - d.DRUG_E_MAX), rel=1e-6)


def test_zero_dose_leaves_phi_untouched() -> None:
    assert drug.phi_after_dose(0.42, 0.0, 0.5, d.DRUG_E_MAX, d.DRUG_EC50_NG_PER_ML) == 0.42


# ======================================================================================
# Backend agreement
# ======================================================================================


def test_scalar_and_array_backends_agree() -> None:
    for value in (-4.0, -0.4, 0.0, 0.4, 4.0, 40.0):
        assert SCALAR.softplus(value, 0.3) == pytest.approx(
            float(ARRAY.softplus(np.array([value]), 0.3)[0]), rel=1e-12, abs=1e-15
        )
        assert SCALAR.exp(value) == pytest.approx(float(ARRAY.exp(np.array([value]))[0]))
        assert SCALAR.clip(value, -1.0, 1.0) == pytest.approx(
            float(ARRAY.clip(np.array([value]), -1.0, 1.0)[0])
        )


def test_softplus_does_not_overflow() -> None:
    assert math.isfinite(SCALAR.softplus(1.0e6, 0.3))
    assert np.all(np.isfinite(ARRAY.softplus(np.array([1.0e6, -1.0e6]), 0.3)))
