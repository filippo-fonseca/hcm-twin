"""The parameter containers, and the split the whole project rests on.

Clinical imaging measures **shape** with high accuracy and **material** not at all. So
this model pins the shape per virtual patient and infers only the material. That split is
not a comment, it is a type: :class:`MeasuredGeometry` holds what an echocardiogram
returns, :class:`HiddenMaterial` holds what it cannot, and no function in the package
accepts a merged dictionary of the two. Any analysis that wants to treat a geometric
quantity as unknown has to say so explicitly by constructing a different object, which is
exactly the friction we want.

:class:`Loading` is a third, separate thing: the operating conditions the ventricle is
placed under. Provocation maneuvers are modifiers on ``Loading`` and nothing else, so a
maneuver can never accidentally change the patient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from . import defaults as d


@dataclass(frozen=True, slots=True)
class MeasuredGeometry:
    """Chamber shape. Treated as known exactly, because clinically it nearly is.

    Wall volume and cavity volume are what echocardiography and cardiac MRI measure well;
    wall thickness and LV mass are algebraic consequences of them under the thick-walled
    spherical assumption, computed here so the model reports the same quantity a
    sonographer would rather than a different one with the same name.
    """

    wall_volume_ml: float
    """Myocardial wall volume V_w, mL. Clinically: LV mass / myocardial density."""

    ref_cavity_volume_ml: float
    """Reference cavity volume V_lv_ref, mL: the cavity volume at which representative
    fiber strain is zero. Clinically this is not measured directly; it is treated as
    measured here because it is tightly determined by the imaged unloaded cavity size,
    and because letting it float would smuggle a geometric unknown into the material
    inference the project is about. Recorded as a modelling choice, not a measurement."""

    body_surface_area_m2: float = d.BSA_M2
    """Body surface area, m^2, from height and weight. Genuinely measured, and needed
    because clinical volumes are reported both raw and indexed to it."""

    def __post_init__(self) -> None:
        if self.body_surface_area_m2 <= 0.0:
            raise ValueError(
                f"body_surface_area_m2 must be positive, got {self.body_surface_area_m2}"
            )
        if self.wall_volume_ml <= 0.0:
            raise ValueError(f"wall_volume_ml must be positive, got {self.wall_volume_ml}")
        if self.ref_cavity_volume_ml <= 0.0:
            raise ValueError(
                f"ref_cavity_volume_ml must be positive, got {self.ref_cavity_volume_ml}"
            )

    @property
    def lv_mass_g(self) -> float:
        """Left-ventricular mass, g. The echo report line, not the model's state."""
        return self.wall_volume_ml * d.MYOCARDIUM_DENSITY_G_PER_ML

    def wall_thickness_cm(self, cavity_volume_ml: float) -> float:
        """Wall thickness, cm, at a given cavity volume, as a thick-walled sphere.

        Both the cavity and the epicardial surface are treated as spheres of the same
        centre; the thickness is the difference of their radii. At a healthy 140 mL wall
        and 120 mL cavity this returns about 0.92 cm, and at an HCM 250 mL wall and 80 mL
        cavity about 1.6 cm, which is the range echocardiography reports for the two
        groups without any of it having been imposed.
        """
        r_inner_cm = (3.0 * cavity_volume_ml / (4.0 * math.pi)) ** (1.0 / 3.0)
        r_outer_cm = (3.0 * (cavity_volume_ml + self.wall_volume_ml) / (4.0 * math.pi)) ** (
            1.0 / 3.0
        )
        return r_outer_cm - r_inner_cm


@dataclass(frozen=True, slots=True)
class HiddenMaterial:
    """Tissue material properties. Treated as unknown, because clinically they are.

    These five numbers are the target of the whole identifiability analysis: which of
    them can a non-invasive measurement set recover, and which combinations of them are
    invisible?
    """

    phi_baseline: float
    """Untreated myosin availability, dimensionless in (0, 1). The single number that
    means "HCM" in this model."""

    a_pas_kpa: float
    """Passive stiffness scale, kPa."""

    b_pas: float
    """Passive stiffness exponent, dimensionless."""

    ca50_ref_um: float
    """Calcium sensitivity of the thin filament at reference length, uM. Thin-filament
    HCM mutations act here; thick-filament mutations act on ``phi_baseline``. Keeping the
    two separate is why the model can distinguish the two mutation classes at all."""

    clearance_l_per_h: float
    """Apparent drug clearance CL/F, L/h. Hidden, wide prior, and the whole reason
    "this patient got twice the exposure" is a live competing explanation for
    over-response."""

    def __post_init__(self) -> None:
        if not 0.0 < self.phi_baseline < 1.0:
            raise ValueError(f"phi_baseline must lie in (0, 1), got {self.phi_baseline}")
        for name in ("a_pas_kpa", "b_pas", "ca50_ref_um", "clearance_l_per_h"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class Loading:
    """Operating conditions. The only thing a provocation maneuver is allowed to touch."""

    heart_rate_bpm: float
    """Heart rate, beats per minute. Raising it is the tachycardia/exercise maneuver, and
    it shortens filling time, which is the mechanism that matters in a stiff ventricle."""

    total_blood_volume_ml: float
    """Total *stressed* blood volume, mL. Preload. Lowering it is the Valsalva
    maneuver."""

    systemic_resistance_mmhg_s_per_ml: float
    """Systemic vascular resistance, mmHg*s/mL. Afterload. Raising it is handgrip."""

    def __post_init__(self) -> None:
        for name in (
            "heart_rate_bpm",
            "total_blood_volume_ml",
            "systemic_resistance_mmhg_s_per_ml",
        ):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")

    def scaled(
        self,
        heart_rate_factor: float = 1.0,
        blood_volume_factor: float = 1.0,
        resistance_factor: float = 1.0,
    ) -> Loading:
        """Return a copy with each field multiplied by the given factor."""
        return replace(
            self,
            heart_rate_bpm=self.heart_rate_bpm * heart_rate_factor,
            total_blood_volume_ml=self.total_blood_volume_ml * blood_volume_factor,
            systemic_resistance_mmhg_s_per_ml=(
                self.systemic_resistance_mmhg_s_per_ml * resistance_factor
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelConstants:
    """Everything held fixed across the virtual population.

    Separated from :class:`HiddenMaterial` because these are not patient properties in
    this model: they are the shared physiology and the shared numerics. Tests override
    fields here rather than writing magic numbers into solver code.
    """

    # Calcium
    ca_diast_um: float = d.CA_DIAST_UM
    ca_peak_um: float = d.CA_PEAK_UM
    ca_tau_r_s: float = d.CA_TAU_R_S

    # Sarcomere
    k_park_tot_per_s: float = d.K_PARK_TOT_PER_S
    k_att_per_s: float = d.K_ATT_PER_S
    k_det_per_s: float = d.K_DET_PER_S
    beta_overlap: float = d.BETA_OVERLAP
    overlap_max: float = d.OVERLAP_MAX
    k_force_per_kpa: float = d.K_FORCE_PER_KPA
    hill_n: float = d.HILL_N
    beta_len: float = d.BETA_LEN
    t_ref_kpa: float = d.T_REF_KPA
    k_xb_per_s: float = d.K_XB_PER_S
    xb_half: float = d.XB_HALF
    xb_max_gain: float = d.XB_MAX_GAIN

    # Circulation
    c_art_ml_per_mmhg: float = d.C_ART_ML_PER_MMHG
    c_ven_ml_per_mmhg: float = d.C_VEN_ML_PER_MMHG
    r_av_mmhg_s_per_ml: float = d.R_AV_MMHG_S_PER_ML
    r_mv_mmhg_s_per_ml: float = d.R_MV_MMHG_S_PER_ML
    valve_smooth_mmhg: float = d.VALVE_SMOOTH_MMHG

    # Obstruction
    obstruction_enabled: bool = True
    k_obs_mmhg_s2_cm4_per_ml2: float = d.K_OBS_MMHG_S2_CM4_PER_ML2
    a0_lvot_cm2: float = d.A0_LVOT_CM2
    crowding_ref: float = d.CROWDING_REF
    lvot_exponent: float = d.LVOT_EXPONENT
    a_min_frac_lvot: float = d.A_MIN_FRAC_LVOT

    # Drug
    drug_e_max: float = d.DRUG_E_MAX
    drug_ec50_ng_per_ml: float = d.DRUG_EC50_NG_PER_ML

    # Solver
    steps_per_beat: int = d.STEPS_PER_BEAT
    max_beats: int = d.MAX_BEATS
    steady_tol_ml: float = d.STEADY_TOL_ML
    steady_tol_mmhg: float = d.STEADY_TOL_MMHG


HEALTHY_GEOMETRY = MeasuredGeometry(
    wall_volume_ml=d.V_W_HEALTHY_ML,
    ref_cavity_volume_ml=d.V_LV_REF_HEALTHY_ML,
)
"""Representative healthy chamber shape."""

HCM_GEOMETRY = MeasuredGeometry(
    wall_volume_ml=d.V_W_HCM_ML,
    ref_cavity_volume_ml=d.V_LV_REF_HCM_ML,
)
"""Representative HCM chamber shape: thick wall, small cavity."""

HEALTHY_MATERIAL = HiddenMaterial(
    phi_baseline=d.PHI_HEALTHY,
    a_pas_kpa=d.A_PAS_KPA,
    b_pas=d.B_PAS,
    ca50_ref_um=d.CA50_REF_UM,
    clearance_l_per_h=d.DRUG_CL_L_PER_H,
)
"""Representative healthy tissue material."""

HCM_MATERIAL = HiddenMaterial(
    phi_baseline=d.PHI_HCM,
    a_pas_kpa=d.A_PAS_HCM_KPA,
    b_pas=d.B_PAS_HCM,
    ca50_ref_um=d.CA50_REF_UM,
    clearance_l_per_h=d.DRUG_CL_L_PER_H,
)
"""Representative HCM tissue material: more heads available, stiffer passive tissue.

Note what is *not* here: no imposed ejection fraction, no imposed stroke volume, no
imposed diastolic pressure. Those have to emerge."""

RESTING_LOADING = Loading(
    heart_rate_bpm=d.HR_BPM,
    total_blood_volume_ml=d.V_TOT_ML,
    systemic_resistance_mmhg_s_per_ml=d.R_SYS_MMHG_S_PER_ML,
)
"""Resting operating conditions."""
