"""Fan chart: how the range of possible portfolio values widens over time.

Reads  <input>/fan_chart.csv and <input>/summary.json  (written by mc_serial / mc_hybrid)
Writes a PNG.

    python scripts/plot_fan_chart.py --input results/run --output docs/images/fan_chart.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

import chart_style as cs


def make_fan_chart(fan: pd.DataFrame, summary: dict, theme: cs.Theme = cs.LIGHT) -> Figure:
    """Builds the fan chart figure: P5-P95 band, P25-P75 band, median and mean lines."""
    params = summary["params"]
    years = fan["month"] / 12.0

    # Figure() instead of pyplot: no global state, safe inside the Streamlit dashboard.
    fig = Figure(figsize=cs.FIGSIZE, dpi=cs.DPI)
    ax = fig.add_subplot()
    cs.style_axes(ax, theme)

    ax.fill_between(years, fan["p05"], fan["p95"], color=theme.band_outer, alpha=0.55,
                    linewidth=0, label="90 % of futures (P5 – P95)")
    ax.fill_between(years, fan["p25"], fan["p75"], color=theme.band_inner, alpha=0.55,
                    linewidth=0, label="50 % of futures (P25 – P75)")
    ax.plot(years, fan["median"], color=theme.blue, linewidth=2, solid_capstyle="round",
            label="Median (typical outcome)")
    ax.plot(years, fan["mean"], color=theme.orange, linewidth=2, solid_capstyle="round",
            label="Mean (average outcome)")
    ax.axhline(params["initial"], color=theme.text_secondary, linewidth=1)
    # Label the starting-amount line at the right end, where it is clear of the median/mean.
    ax.text(0.985 * years.iloc[-1], params["initial"], f"Initial {params['initial']:,.0f}",
            color=theme.text_secondary, fontsize=9, ha="right", va="bottom")

    # A few labelled sample points on the median: halfway and at the end. Labels go
    # BELOW the median because the mean line runs just above it.
    last = len(fan) - 1
    for i in sorted({last // 2, last}):
        x, y = years.iloc[i], fan["median"].iloc[i]
        ax.plot(x, y, "o", markersize=8, color=theme.blue, markeredgecolor=theme.surface,
                markeredgewidth=2, zorder=5)
        ax.annotate(f"{y:,.0f}", (x, y), xytext=(-6, -8), textcoords="offset points",
                    ha="right", va="top", fontsize=10, color=theme.text_primary,
                    fontweight="bold")

    # Direct labels for the band edges at the right end.
    for column, name in (("p95", "P95"), ("p05", "P5")):
        y = fan[column].iloc[-1]
        ax.annotate(f"{name} {cs.money(y)}", (years.iloc[-1], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=theme.text_secondary)

    ax.set_xlim(0, years.iloc[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Years from today")
    ax.set_ylabel("Portfolio value")
    cs.add_titles(ax, "Range of possible portfolio values over time", cs.describe_params(params),
                  theme)
    cs.add_legend(ax, "upper left", theme)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the fan chart of a simulation run.")
    parser.add_argument("--input", type=Path, default=Path("results/run"),
                        help="run directory containing fan_chart.csv and summary.json")
    parser.add_argument("--output", type=Path, default=Path("docs/images/fan_chart.png"))
    parser.add_argument("--theme", choices=("light", "dark"), default="light",
                        help="colour theme (light for documents)")
    args = parser.parse_args()

    fan = pd.read_csv(args.input / "fan_chart.csv")
    summary = json.loads((args.input / "summary.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    theme = cs.DARK if args.theme == "dark" else cs.LIGHT
    make_fan_chart(fan, summary, theme).savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
