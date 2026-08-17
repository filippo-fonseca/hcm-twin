"""The three analysis figures: sensitivity matrix, confounding map, tie-breaker table.

Every function writes the CSV behind its figure next to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..observables import SPECS
from . import style

PRETTY: dict[str, str] = {
    "phi_baseline": "myosin availability",
    "a_pas_kpa": "stiffness scale",
    "b_pas": "stiffness exponent",
    "ca50_ref_um": "calcium sensitivity",
    "clearance_l_per_h": "drug clearance",
    "wall_volume_ml": "wall volume",
    "ref_cavity_volume_ml": "unloaded cavity",
    "body_surface_area_m2": "body surface area",
    "heart_rate_bpm": "heart rate",
    "total_blood_volume_ml": "stressed volume",
    "systemic_resistance_mmhg_s_per_ml": "vascular resistance",
    "edv_ml": "end-diastolic volume",
    "esv_ml": "end-systolic volume",
    "stroke_volume_ml": "stroke volume",
    "ejection_fraction": "ejection fraction",
    "wall_thickness_cm": "wall thickness",
    "lv_mass_g": "LV mass",
    "peak_lvot_gradient_mmhg": "outflow gradient",
    "end_diastolic_pressure_mmhg": "filling pressure (invasive)",
    "e_over_e_prime": "E/e' surrogate",
    "peak_strain_amplitude": "strain amplitude",
    "mean_arterial_pressure_mmhg": "mean arterial pressure",
    "cardiac_output_l_per_min": "cardiac output",
    "thickness_to_cavity_ratio": "thickness/cavity",
    "stroke_volume_index_ml_per_m2": "stroke volume index",
    "stroke_work_j": "stroke work (invasive)",
    "atp_cost_per_stroke_work": "ATP per unit work",
    "ef_at_mid_dose": "EF at mid dose",
    "ef_drop_at_mid_dose": "EF drop at mid dose",
    "ef_slope_per_mg": "EF slope per mg",
    "preload_reduction": "preload reduction (Valsalva)",
    "tachycardia": "tachycardia",
    "afterload_increase": "afterload (handgrip)",
    "exercise": "exercise",
}


def _label(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


def plot_sensitivity_matrix(
    matrix: pd.DataFrame,
    output: Path,
    title: str = "Total-order Sobol indices",
    subtitle: str = "How much of each measurement's variance each parameter explains",
) -> pd.DataFrame:
    """Heatmap of total-order indices: observables down, parameters across.

    Sequential single-hue ramp, because the quantity is a magnitude in [0, 1] with no
    meaningful zero-crossing. Every cell above a readable threshold is direct-labelled, so
    the figure does not rely on colour discrimination to be read, and the CSV beside it is
    the table view.
    """
    import matplotlib.pyplot as plt

    style.apply()
    values = matrix.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    figure, ax = plt.subplots(figsize=(0.62 * n_cols + 4.0, 0.30 * n_rows + 2.2))

    image = ax.imshow(
        values,
        cmap=style.sequential_cmap(),
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(values))),
        aspect="auto",
    )
    ax.set_xticks(range(n_cols), [_label(c) for c in matrix.columns], rotation=38, ha="right")
    ax.set_yticks(range(n_rows), [_label(r) for r in matrix.index])
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    # A 2 px surface-coloured gap between cells, so adjacent fills never touch.
    ax.grid(which="minor", color=style.SURFACE, linewidth=2.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    threshold = 0.5 * float(np.nanmax(values)) if np.isfinite(values).any() else 0.5
    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            if not np.isfinite(value) or value < 0.02:
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}".lstrip("0"),
                ha="center",
                va="center",
                fontsize=6.8,
                color=style.SURFACE if value > threshold else style.TEXT_SECONDARY,
            )

    ax.set_title(title)
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=style.TEXT_SECONDARY,
    )
    bar = figure.colorbar(image, ax=ax, fraction=0.022, pad=0.015)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=7.5, colors=style.TEXT_MUTED)
    bar.set_label("total-order index", fontsize=8, color=style.TEXT_SECONDARY)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    matrix.to_csv(output.with_suffix(".csv"))
    return matrix


def plot_confounding_map(
    confounding: pd.DataFrame,
    output: Path,
    parameter_order: tuple[str, ...],
    noise_level: str = "realistic",
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Posterior correlation between hidden parameters, as a lower-triangular map.

    Diverging ramp with a neutral midpoint, because correlation has a genuine zero and a
    sign. Lower triangle only: a symmetric matrix drawn in full asks the reader to check
    that it is symmetric.
    """
    import matplotlib.pyplot as plt

    style.apply()
    subset = confounding[confounding["noise_level"] == noise_level]
    n = len(parameter_order)
    grid = np.full((n, n), np.nan)
    for _, row in subset.iterrows():
        i = parameter_order.index(row["parameter_a"])
        j = parameter_order.index(row["parameter_b"])
        grid[max(i, j), min(i, j)] = row["mean_correlation"]

    figure, ax = plt.subplots(figsize=(5.6, 4.6))
    image = ax.imshow(grid, cmap=style.diverging_cmap(), vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n), [_label(p) for p in parameter_order], rotation=38, ha="right")
    ax.set_yticks(range(n), [_label(p) for p in parameter_order])
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=2.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            value = grid[i, j]
            if not np.isfinite(value):
                continue
            confounded = abs(value) > threshold
            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold" if confounded else "normal",
                color=style.SURFACE if abs(value) > 0.55 else style.TEXT_PRIMARY,
            )
            if confounded:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=style.TEXT_PRIMARY,
                        linewidth=1.4,
                    )
                )

    ax.set_title(f"Posterior correlation between hidden parameters ({noise_level} noise)")
    ax.text(
        0.0,
        1.015,
        f"Outlined cells exceed |r| = {threshold:.2f} and are called confounded",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=style.TEXT_SECONDARY,
    )
    bar = figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=7.5, colors=style.TEXT_MUTED)
    bar.set_label("correlation", fontsize=8, color=style.TEXT_SECONDARY)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    subset.to_csv(output.with_suffix(".csv"), index=False)
    return subset


def plot_recovery(
    recovery_summary: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    """How wide the posterior is for each hidden parameter, at both noise levels.

    A relative credible-interval width near or above 2 means the interval spans the whole
    prior box: the measurements said nothing about that parameter.
    """
    import matplotlib.pyplot as plt

    style.apply()
    levels = list(dict.fromkeys(recovery_summary["noise_level"]))
    order = recovery_summary.groupby("parameter")["median_ci90_width"].max().sort_values().index
    figure, ax = plt.subplots(figsize=(6.6, 0.52 * len(order) + 1.8))

    bar_height = 0.34
    gap = 0.04
    for level_index, level in enumerate(levels):
        subset = recovery_summary[recovery_summary["noise_level"] == level].set_index("parameter")
        colour = style.SERIES[level_index]
        for row_index, parameter in enumerate(order):
            if parameter not in subset.index:
                continue
            width = float(subset.loc[parameter, "median_ci90_width"])
            offset = (level_index - (len(levels) - 1) / 2) * (bar_height + gap)
            style.rounded_barh(ax, row_index + offset, width, bar_height, colour)
            ax.text(
                width + 0.04,
                row_index + offset,
                f"{width:.2f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color=style.TEXT_SECONDARY,
            )
        ax.plot([], [], color=colour, linewidth=6, label=f"{level} noise")

    ax.axvline(0.60, color=style.TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(
        0.60,
        len(order) - 0.35,
        " recoverable",
        fontsize=7.5,
        color=style.TEXT_SECONDARY,
        ha="left",
        va="center",
    )
    ax.set_yticks(range(len(order)), [_label(p) for p in order])
    ax.set_xlabel("Width of the 90% credible interval, relative to the true value")
    ax.set_xlim(0, max(2.2, float(recovery_summary["median_ci90_width"].max()) * 1.18))
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("How much of each hidden parameter the measurements recover")
    ax.legend(loc="lower right")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    recovery_summary.to_csv(output.with_suffix(".csv"), index=False)
    return recovery_summary


def plot_tiebreaker(
    detail: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    """Per confounded pair: does each maneuver narrow the invisible direction, and is it real?

    Two panels sharing a row order. Left, how wide the posterior is along the direction the
    resting study could not resolve, before and after adding the maneuver. Right, the
    discriminating signal in units of that observable's measurement error, which is what
    decides whether the proposal is a test or a hope.

    The left panel deliberately does *not* plot posterior correlation. Correlation
    describes the shape of the uncertainty and not its size, and adding information can
    raise it while genuinely improving the inference; ranking maneuvers by correlation drop
    would mislabel a helpful maneuver as harmful.
    """
    import matplotlib.pyplot as plt

    style.apply()
    usable = detail[detail["usable"]]
    if usable.empty:
        raise ValueError("no usable tie-breaker rows to plot")

    grouped = (
        usable.groupby(["pair", "maneuver"])
        .agg(
            before=("ridge_width_before", "median"),
            after=("ridge_width_after", "median"),
            snr=("best_signal_to_noise", "median"),
        )
        .reset_index()
        .sort_values(["pair", "after"])
    )
    labels = [f"{row['pair']}\n{_label(row['maneuver'])}" for _, row in grouped.iterrows()]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 0.46 * len(grouped) + 2.2),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    bar_height = 0.34
    for row_index, (_, row) in enumerate(grouped.iterrows()):
        style.rounded_barh(
            axes[0],
            row_index + (bar_height + 0.04) / 2,
            float(row["before"]),
            bar_height,
            style.SERIES[0],
        )
        style.rounded_barh(
            axes[0],
            row_index - (bar_height + 0.04) / 2,
            float(row["after"]),
            bar_height,
            style.SERIES[1],
        )
        axes[0].text(
            float(row["before"]) + 0.015,
            row_index + (bar_height + 0.04) / 2,
            f"{row['before']:.3f}",
            va="center",
            fontsize=7.2,
            color=style.TEXT_SECONDARY,
        )
        axes[0].text(
            float(row["after"]) + 0.015,
            row_index - (bar_height + 0.04) / 2,
            f"{row['after']:.3f}",
            va="center",
            fontsize=7.2,
            color=style.TEXT_SECONDARY,
        )
    axes[0].plot([], [], color=style.SERIES[0], linewidth=6, label="baseline alone")
    axes[0].plot([], [], color=style.SERIES[1], linewidth=6, label="baseline + maneuver")
    # Narrower is better here, unlike every other bar chart in the deliverables, so say so.
    axes[0].text(
        0.995,
        -0.14,
        "narrower is better",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=style.TEXT_MUTED,
    )
    axes[0].set_yticks(range(len(grouped)), labels)
    axes[0].set_xlabel("Posterior width along the direction the resting study could not resolve")
    axes[0].set_xlim(0, float(grouped[["before", "after"]].to_numpy().max()) * 1.22)
    axes[0].set_ylim(-0.7, len(grouped) - 0.3)
    axes[0].grid(axis="y", visible=False)
    axes[0].set_title("Does the maneuver narrow the invisible direction?")
    axes[0].legend(loc="lower right")

    for row_index, (_, row) in enumerate(grouped.iterrows()):
        snr = float(row["snr"])
        colour = style.STATUS_GOOD if snr >= 1.0 else style.TEXT_MUTED
        style.rounded_barh(axes[1], row_index, snr, 0.5, colour)
        axes[1].text(
            snr + 0.05,
            row_index,
            f"{snr:.2f}",
            va="center",
            fontsize=7.2,
            color=style.TEXT_SECONDARY,
        )
    axes[1].axvline(1.0, color=style.TEXT_PRIMARY, linewidth=1.1, linestyle=(0, (4, 3)))
    axes[1].text(
        1.0,
        len(grouped) - 0.35,
        " one measurement error",
        fontsize=7.5,
        color=style.TEXT_SECONDARY,
        ha="left",
        va="center",
    )
    axes[1].set_yticks(range(len(grouped)), [""] * len(grouped))
    axes[1].set_xlabel("Discriminating signal, in units of measurement error")
    axes[1].set_ylim(-0.7, len(grouped) - 0.3)
    axes[1].grid(axis="y", visible=False)
    axes[1].set_title("Is the signal bigger than the error bar?")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    grouped.to_csv(output.with_suffix(".csv"), index=False)
    return grouped


def plot_over_responder_separation(
    labelled: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    """The premise, drawn: do over-responders look different at baseline?

    Baseline ejection fraction against the eventual drop, coloured by outcome. If the two
    groups overlap on the horizontal axis, a baseline ejection fraction does not identify
    the over-responders, which is the entire motivation.
    """
    import matplotlib.pyplot as plt

    style.apply()
    eligible = labelled[labelled["trial_eligible"]].copy()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    safe = eligible[~eligible["over_responder"]]
    crash = eligible[eligible["over_responder"]]

    axes[0].scatter(
        safe["ejection_fraction"],
        safe["ef_drop_at_mid_dose"],
        s=13,
        color=style.SERIES[0],
        alpha=0.55,
        linewidths=0,
        label="tolerated",
    )
    axes[0].scatter(
        crash["ejection_fraction"],
        crash["ef_drop_at_mid_dose"],
        s=26,
        color=style.SERIES[1],
        alpha=0.9,
        linewidths=0.8,
        edgecolors=style.SURFACE,
        label="crossed the floor",
    )
    axes[0].set_xlabel("Baseline ejection fraction")
    axes[0].set_ylabel("Ejection-fraction drop at the mid dose")
    axes[0].set_title("Baseline does not separate them")
    axes[0].legend(loc="upper left")

    bins = np.linspace(
        float(eligible["ejection_fraction"].min()),
        float(eligible["ejection_fraction"].max()),
        26,
    )
    # Only plot a group that has members: a density histogram of an empty series divides
    # by zero and silently paints nothing, which reads as "no over-responders" rather
    # than "this cohort was too small to contain one".
    groups = [(safe, "tolerated", style.SERIES[0]), (crash, "crossed the floor", style.SERIES[1])]
    present = [(g, label, colour) for g, label, colour in groups if len(g) > 0]
    axes[1].hist(
        [g["ejection_fraction"] for g, _, _ in present],
        bins=bins,
        density=True,
        color=[c for _, _, c in present],
        label=[label for _, label, _ in present],
        histtype="stepfilled",
        alpha=0.72,
        linewidth=0,
    )
    axes[1].set_xlabel("Baseline ejection fraction")
    axes[1].set_ylabel("Density")
    axes[1].set_title("The same picture as distributions")
    axes[1].legend(loc="upper right")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)

    columns = [
        "patient_id",
        "ejection_fraction",
        "ef_drop_at_mid_dose",
        "over_responder",
        "wall_thickness_cm",
        "peak_lvot_gradient_mmhg",
        "e_over_e_prime",
    ]
    data = eligible[columns]
    data.to_csv(output.with_suffix(".csv"), index=False)
    return data


def plot_observable_noise_context(output: Path) -> pd.DataFrame:
    """Measurement error per observable, at both levels. The figure the result rests on."""
    import matplotlib.pyplot as plt

    style.apply()
    rows = [
        {
            "observable": name,
            "units": spec.units,
            "kind": spec.noise_kind,
            "realistic": spec.noise_realistic,
            "optimistic": spec.noise_optimistic,
            "invasive": spec.invasive,
            "routine": spec.routine,
        }
        for name, spec in SPECS.items()
        if spec.noise_kind == "relative"
    ]
    frame = pd.DataFrame(rows).sort_values("realistic")

    figure, ax = plt.subplots(figsize=(6.8, 0.42 * len(frame) + 1.8))
    for row_index, (_, row) in enumerate(frame.iterrows()):
        style.rounded_barh(ax, row_index + 0.19, float(row["realistic"]), 0.34, style.SERIES[0])
        style.rounded_barh(ax, row_index - 0.19, float(row["optimistic"]), 0.34, style.SERIES[1])
        ax.text(
            float(row["realistic"]) + 0.006,
            row_index + 0.19,
            f"{100 * row['realistic']:.0f}%",
            va="center",
            fontsize=7.2,
            color=style.TEXT_SECONDARY,
        )
    ax.plot([], [], color=style.SERIES[0], linewidth=6, label="realistic (routine 2D echo)")
    ax.plot([], [], color=style.SERIES[1], linewidth=6, label="optimistic (3D or core lab)")
    ax.set_yticks(range(len(frame)), [_label(o) for o in frame["observable"]])
    ax.set_xlabel("Relative measurement error (standard deviation as a fraction)")
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("The error bars the identifiability result rests on")
    ax.legend(loc="lower right")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    return frame
