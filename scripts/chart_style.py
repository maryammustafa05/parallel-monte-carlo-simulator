"""Shared look for every chart (used by the plot scripts and the dashboard).

Colours come from a colour-blind-safe reference palette: blue is the main data
series, orange marks the risk threshold, and text stays in neutral greys
(never in a data colour) so it is always readable.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"         # chart background
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"            # hairline grid, one step off the surface
BLUE = "#2a78d6"            # main series (median, distribution)
BLUE_LIGHT = "#b7d3f6"      # outer band (P5-P95)
BLUE_MID = "#86b6ef"        # inner band (P25-P75)
ORANGE = "#eb6834"          # risk: worst 5 % tail, VaR threshold

FIGSIZE = (10, 5.6)
DPI = 150


def money(value: float, _pos: object = None) -> str:
    """Axis/label formatter: 8,245 under ten thousand, 12.5K / 1.2M above."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:,.0f}"


def style_axes(ax: Axes) -> None:
    """Recessive axes: light hairline grid, no top/right frame, grey tick labels."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10, length=0, pad=6)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.set_major_formatter(FuncFormatter(money))


def add_titles(ax: Axes, title: str, subtitle: str) -> None:
    """Left-aligned title with a smaller grey subtitle underneath."""
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold",
                 color=TEXT_PRIMARY, pad=28)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10,
            color=TEXT_SECONDARY, va="bottom")


def describe_params(params: dict) -> str:
    """One-line summary of the run's inputs, e.g. for chart subtitles."""
    return (f"Start {money(params['initial'])} · return {params['mu']:.1%}/yr · "
            f"volatility {params['sigma']:.1%}/yr · {params['trials']:,} simulated futures")
