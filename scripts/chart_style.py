"""Shared look for every chart (used by the plot scripts and the dashboard).

Colours come from a colour-blind-safe reference palette: blue is the main data
series, orange marks the risk threshold, and text stays in neutral greys
(never in a data colour) so it is always readable.

There are two themes. LIGHT is used for the PNG files in docs/images. DARK is
used by the dashboard when the browser/Streamlit is in dark mode: it is not just
"inverted" — each colour is a step of the same hue chosen to read well on a dark
background.
"""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True)
class Theme:
    surface: str          # chart background
    text_primary: str     # titles, labels, values
    text_secondary: str   # axis labels, subtitles, notes
    grid: str             # hairline grid, one step off the surface
    blue: str             # main series (median line, distribution outline)
    band_outer: str       # P5-P95 band
    band_inner: str       # P25-P75 band
    orange: str           # risk: worst tail, VaR threshold, mean line


LIGHT = Theme(
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e4e3df",
    blue="#2a78d6",
    band_outer="#b7d3f6",
    band_inner="#86b6ef",
    orange="#eb6834",
)

DARK = Theme(
    surface="#0e1117",    # Streamlit's dark page colour, so the chart blends in
    text_primary="#f0efec",
    text_secondary="#c3c2b7",
    grid="#2c2f36",
    blue="#86b6ef",       # lighter step of the same blue: stands out on the dark bands
    band_outer="#184f95",
    band_inner="#256abf",
    orange="#d95926",
)

FIGSIZE = (10, 5.6)
DPI = 150


def money(value: float, _pos: object = None) -> str:
    """Axis/label formatter: 8,245 under ten thousand, 12.5K / 1.2M above."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}K"
    if 0 < abs(value) < 1:
        return f"{value:.2g}"  # tiny values (near-total losses) would otherwise show as "0"
    return f"{value:,.0f}"


def amount(value: float) -> str:
    """Exact-looking amount for labels: 8,245 or 17,983; tiny values as 0.033."""
    if 0 < abs(value) < 1:
        return f"{value:.2g}"
    return f"{value:,.0f}"


def percent(fraction: float) -> str:
    """0.95 -> '95 %', 0.975 -> '97.5 %'."""
    return f"{100 * fraction:g} %"


def style_axes(ax: Axes, theme: Theme) -> None:
    """Recessive axes: light hairline grid, no top/right frame, grey tick labels."""
    ax.set_facecolor(theme.surface)
    ax.figure.set_facecolor(theme.surface)
    ax.grid(True, axis="y", color=theme.grid, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    ax.tick_params(colors=theme.text_secondary, labelsize=10, length=0, pad=6)
    ax.xaxis.label.set_color(theme.text_secondary)
    ax.yaxis.label.set_color(theme.text_secondary)
    ax.yaxis.set_major_formatter(FuncFormatter(money))


def add_titles(ax: Axes, title: str, subtitle: str, theme: Theme) -> None:
    """Left-aligned title with a smaller grey subtitle underneath."""
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold",
                 color=theme.text_primary, pad=28)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10,
            color=theme.text_secondary, va="bottom")


def add_legend(ax: Axes, loc: str, theme: Theme) -> None:
    """Frameless legend with text in the theme's text colour (never a data colour)."""
    legend = ax.legend(loc=loc, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme.text_primary)


def describe_params(params: dict) -> str:
    """One-line summary of the run's inputs, e.g. for chart subtitles."""
    return (f"Start {money(params['initial'])} · return {params['mu']:.1%}/yr · "
            f"volatility {params['sigma']:.1%}/yr · {params['trials']:,} simulated futures")
