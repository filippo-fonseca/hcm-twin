"""Pressure-volume loops and the beat waveforms behind them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..model import BeatResult
from . import style


def loop_frame(result: BeatResult, label: str) -> pd.DataFrame:
    """The data behind a pressure-volume loop, so the figure ships with its table."""
    trace = result.trace
    if trace is None:
        raise ValueError("simulate(..., record_trace=True) is needed to draw a loop")
    return pd.DataFrame(
        {
            "label": label,
            "time_s": trace.time_s,
            "cavity_volume_ml": trace.cavity_volume_ml,
            "lv_pressure_mmhg": trace.lv_pressure_mmhg,
            "arterial_pressure_mmhg": trace.arterial_pressure_mmhg,
            "venous_pressure_mmhg": trace.venous_pressure_mmhg,
            "aortic_flow_ml_per_s": trace.aortic_flow_ml_per_s,
            "mitral_flow_ml_per_s": trace.mitral_flow_ml_per_s,
            "lvot_gradient_mmhg": trace.lvot_gradient_mmhg,
            "attached_fraction": trace.attached,
            "parked_fraction": trace.parked,
            "calcium_um": trace.calcium_um,
        }
    )


def plot_loops(
    results: dict[str, BeatResult],
    output: Path,
    title: str = "Pressure-volume loops",
    subtitle: str | None = None,
) -> pd.DataFrame:
    """Draw up to three loops on shared axes, direct-labelled.

    Direct labels rather than a legend box alone, because the aqua slot sits below the
    contrast floor on a light surface and identity must never be colour-only.
    """
    import matplotlib.pyplot as plt

    style.apply()
    if len(results) > len(style.SERIES):
        raise ValueError(
            f"at most {len(style.SERIES)} loops per axes; fold the rest into a small "
            "multiple rather than cycling hues"
        )

    figure, ax = plt.subplots(figsize=(5.4, 4.2))
    frames: list[pd.DataFrame] = []
    for index, (label, result) in enumerate(results.items()):
        frame = loop_frame(result, label)
        frames.append(frame)
        colour = style.SERIES[index]
        volume = np.append(frame["cavity_volume_ml"].to_numpy(), frame["cavity_volume_ml"].iloc[0])
        pressure = np.append(
            frame["lv_pressure_mmhg"].to_numpy(), frame["lv_pressure_mmhg"].iloc[0]
        )
        ax.plot(volume, pressure, color=colour, linewidth=2.0, solid_joinstyle="round")
        peak = int(np.argmax(pressure))
        ax.annotate(
            label,
            xy=(volume[peak], pressure[peak]),
            xytext=(6, 6),
            textcoords="offset points",
            color=colour,
            fontsize=8.5,
            fontweight="600",
        )

    ax.set_xlabel("Left-ventricular volume (mL)")
    ax.set_ylabel("Left-ventricular pressure (mmHg)")
    ax.set_title(title)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=style.TEXT_SECONDARY,
        )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(axis="both")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output.with_suffix(".csv"), index=False)
    return data


def plot_dose_response(
    doses: np.ndarray,
    ejection_fraction: np.ndarray,
    gradient_mmhg: np.ndarray,
    output: Path,
    ef_threshold: float = 0.50,
) -> pd.DataFrame:
    """Dose against ejection fraction and against outflow gradient, as two panels.

    Two panels rather than one chart with two y-axes. A dual-axis chart lets the author
    choose the apparent relationship between two series by scaling, and there is no
    honest way to read one.
    """
    import matplotlib.pyplot as plt

    style.apply()
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    axes[0].plot(doses, ejection_fraction, color=style.SERIES[0], marker="o", markersize=5)
    axes[0].axhline(ef_threshold, color=style.STATUS_CRITICAL, linewidth=1.2, linestyle=(0, (4, 3)))
    axes[0].text(
        doses[-1],
        ef_threshold,
        "  interruption threshold",
        va="center",
        ha="left",
        fontsize=7.5,
        color=style.STATUS_CRITICAL,
    )
    axes[0].set_ylabel("Ejection fraction")
    axes[0].set_xlabel("Maintained dose (mg/day)")
    axes[0].set_title("Systolic cost")
    axes[0].set_ylim(min(0.45, float(np.min(ejection_fraction)) - 0.03), None)

    axes[1].plot(doses, gradient_mmhg, color=style.SERIES[1], marker="o", markersize=5)
    axes[1].axhline(30.0, color=style.TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    axes[1].text(
        doses[-1],
        30.0,
        "  obstruction threshold",
        va="center",
        ha="left",
        fontsize=7.5,
        color=style.TEXT_SECONDARY,
    )
    axes[1].set_ylabel("Peak outflow gradient (mmHg)")
    axes[1].set_xlabel("Maintained dose (mg/day)")
    axes[1].set_title("Obstructive benefit")
    axes[1].set_ylim(bottom=0)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)

    data = pd.DataFrame(
        {
            "dose_mg_per_day": doses,
            "ejection_fraction": ejection_fraction,
            "peak_lvot_gradient_mmhg": gradient_mmhg,
        }
    )
    data.to_csv(output.with_suffix(".csv"), index=False)
    return data
