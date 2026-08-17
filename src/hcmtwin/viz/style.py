"""Shared visual language: one palette, one set of rules, applied everywhere.

The palette is fixed and validated rather than chosen per figure. Categorical hues are
assigned in a fixed order and never cycled; a chart that would need a fourth series folds
to "other" or becomes small multiples instead. The sequential ramp is a single hue,
light to dark. The diverging ramp is two hues with a neutral gray midpoint, used only for
quantities that genuinely have a zero-crossing (correlations), never for magnitudes.

Accessibility is not a post-hoc pass. The categorical slots below clear colour-vision
separation on every pair, not merely on adjacent pairs, which is what scatter and matrix
forms require. The aqua slot sits below a 3:1 contrast ratio against the light surface, so
every figure that uses it also carries direct labels: identity is never conveyed by colour
alone, and every figure ships with the CSV behind it as a table view.
"""

from __future__ import annotations

from typing import Any

# --- Categorical: fixed order, capped at three for all-pairs safety -------------------
SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_DARK: tuple[str, ...] = ("#3987e5", "#d95926", "#199e70")

# --- Sequential: one hue, light to dark ----------------------------------------------
SEQUENTIAL: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

# --- Diverging: two poles, neutral gray midpoint --------------------------------------
DIVERGING_LOW: str = "#0d366b"
DIVERGING_MID: str = "#f0efec"
DIVERGING_HIGH: str = "#8f1d1c"

SURFACE: str = "#fcfcfb"
TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
TEXT_MUTED: str = "#8a8880"
GRID: str = "#e6e5e1"

STATUS_GOOD: str = "#008300"
STATUS_CRITICAL: str = "#e34948"


def sequential_cmap():  # type: ignore[no-untyped-def]
    """Single-hue light-to-dark colormap for magnitudes."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("hcm_sequential", list(SEQUENTIAL))


def diverging_cmap():  # type: ignore[no-untyped-def]
    """Two-pole colormap with a neutral midpoint, for signed quantities only."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "hcm_diverging", [DIVERGING_LOW, "#6da7ec", DIVERGING_MID, "#e06f6e", DIVERGING_HIGH]
    )


def apply() -> None:
    """Set the matplotlib defaults every figure in this project inherits.

    Recessive grid and axes, thin marks, generous whitespace. The point of doing this
    once is that no individual figure gets to have its own opinion, so the deliverables
    read as one document rather than as eight.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0,
            "text.color": TEXT_PRIMARY,
        }
    )


def titled(ax: Any, title: str, subtitle: str | None = None) -> None:
    """Set a title, and a subtitle above the axes without the two colliding.

    matplotlib places the title a fixed pad above the axes, and anything drawn at an axes
    y-coordinate just over 1 lands in the same place. The first version of these figures
    overprinted every subtitle on its own title. The fix is to reserve the space rather
    than to nudge either of them.
    """
    if subtitle:
        ax.set_title(title, pad=24)
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )
    else:
        ax.set_title(title)


def rounded_barh(ax: Any, y: float, width: float, height: float, color: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """A horizontal bar rounded at the value end and square at the baseline.

    Square where it meets the axis so the bar reads as growing *from* the baseline, rounded
    at the far end so the data end is soft. Rounding both ends turns a bar into a pill,
    which floats and reads as an interval rather than a magnitude.

    The corner radius is a fraction of the bar height rather than half of it, for the same
    reason: at half the height the two arcs meet and the shape is a pill again.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    magnitude = abs(width)
    if magnitude <= 0.0:
        return None
    radius = min(0.32 * height, 0.45 * magnitude)
    body = FancyBboxPatch(
        (0.0, y - height / 2.0),
        max(magnitude - radius, 1e-12),
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0,
        facecolor=color,
        mutation_aspect=1.0,
        **kwargs,
    )
    ax.add_patch(body)
    # Square off the baseline end by covering the left arc.
    ax.add_patch(
        Rectangle(
            (0.0, y - height / 2.0),
            min(radius, magnitude),
            height,
            linewidth=0,
            facecolor=color,
        )
    )
    return body


def annotate_illustrative(ax: Any) -> None:
    """Stamp a figure as illustrative.

    Required by the project's honesty constraints: nothing schematic appears beside real
    results without a label saying which it is.
    """
    ax.text(
        0.995,
        1.02,
        "ILLUSTRATIVE",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=STATUS_CRITICAL,
        fontweight="bold",
    )
