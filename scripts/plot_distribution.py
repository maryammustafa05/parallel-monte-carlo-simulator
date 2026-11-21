"""Distribution of final portfolio values, with the VaR threshold marked.

Reads  <input>/final_hist.csv and <input>/summary.json
Writes a PNG.

    python scripts/plot_distribution.py --input results/run --output docs/images/distribution.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import chart_style as cs

# The C++ histogram has 2048 fine bins; groups of this many are merged so the
# shape is smooth enough to read (2048 / 16 = 128 bars).
MERGE_BINS = 16
# The far right tail is very long and thin; showing up to the 99.5th percentile
# keeps the interesting part readable.
SHOW_UP_TO_QUANTILE = 0.995


def coarsen(hist: pd.DataFrame, group: int) -> pd.DataFrame:
    """Merges every `group` adjacent bins into one (adding their counts)."""
    keys = np.arange(len(hist)) // group
    return hist.groupby(keys).agg(bin_lower=("bin_lower", "first"),
                                  bin_upper=("bin_upper", "last"),
                                  count=("count", "sum")).reset_index(drop=True)


def histogram_quantile(hist: pd.DataFrame, q: float) -> float:
    """Value below which a fraction q of the futures end, read from the histogram
    (interpolating inside the bin in log space, like the C++ engine does)."""
    counts = hist["count"].to_numpy(dtype=float)
    cumulative = np.cumsum(counts)
    target = q * cumulative[-1]
    b = int(np.searchsorted(cumulative, target))
    before = cumulative[b - 1] if b > 0 else 0.0
    fraction = (target - before) / counts[b] if counts[b] > 0 else 0.0
    lo, hi = np.log(hist["bin_lower"].iloc[b]), np.log(hist["bin_upper"].iloc[b])
    return float(np.exp(lo + fraction * (hi - lo)))


def make_distribution_chart(hist: pd.DataFrame, summary: dict,
                            theme: cs.Theme = cs.LIGHT) -> Figure:
    """Builds the distribution figure: probability density, worst tail, V0 and VaR lines."""
    params = summary["params"]
    risk = summary["risk"]
    initial = params["initial"]
    confidence = risk["confidence"]
    tail_share = cs.percent(1 - confidence)        # e.g. "5 %" at 95 % confidence
    # The VaR threshold is the (1 - confidence) quantile of the final value.
    threshold = histogram_quantile(hist, 1 - confidence)

    bins = coarsen(hist, MERGE_BINS)
    total = bins["count"].sum()
    # Bins are spaced in log space, so they have different widths. Dividing by the
    # width gives a density, so wide bins don't look taller just for being wide.
    density = bins["count"] / total / (bins["bin_upper"] - bins["bin_lower"]) * 1000
    cumulative = bins["count"].cumsum() / total
    first = int(np.argmax(bins["count"].to_numpy() > 0))
    last = int(np.searchsorted(cumulative.to_numpy(), SHOW_UP_TO_QUANTILE))
    bins, density = bins.iloc[first:last + 1], density.iloc[first:last + 1]

    edges = np.append(bins["bin_lower"].to_numpy(), bins["bin_upper"].iloc[-1])
    heights = np.append(density.to_numpy(), density.iloc[-1])

    fig = Figure(figsize=cs.FIGSIZE, dpi=cs.DPI)
    ax = fig.add_subplot()
    cs.style_axes(ax, theme)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:g} %")

    # Whole distribution as a soft area with a 2px outline.
    ax.fill_between(edges, heights * 100, step="post", color=theme.blue, alpha=0.12, linewidth=0)
    ax.step(edges, heights * 100, where="post", color=theme.blue, linewidth=2,
            label="Share of futures ending at this value")
    # Worst 5 % of futures, below the VaR threshold, highlighted in orange.
    tail = edges <= threshold
    ax.fill_between(edges, heights * 100, where=tail, step="post", color=theme.orange,
                    alpha=0.35, linewidth=0, label=f"Worst {tail_share} of futures")

    top = (heights * 100).max()
    ax.axvline(initial, color=theme.text_primary, linewidth=1)
    ax.text(initial, top * 1.04, f" Initial {initial:,.0f}", color=theme.text_primary,
            fontsize=9, ha="left", va="bottom")
    # The threshold line is named in the legend (not next to the line), so its label
    # can never collide with the axis, whatever the confidence level.
    ax.axvline(threshold, color=theme.orange, linewidth=2,
               label=f"Worst-{tail_share} threshold: {threshold:,.0f}")

    ax.text(0.99, 0.74,
            f"Value at Risk ({cs.percent(confidence)}): {risk['var']:,.0f}\n"
            f"Chance of ending below the start: {risk['prob_loss']:.1%}\n"
            f"Median final value: {summary['final_value']['median']:,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=theme.text_primary)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, top * 1.15)
    ax.xaxis.set_major_formatter(cs.money)
    ax.set_xlabel(f"Portfolio value after {params['months'] // 12} years"
                  if params["months"] % 12 == 0 else
                  f"Portfolio value after {params['months']} months")
    ax.set_ylabel("Share of futures per 1,000 of value")
    cs.add_titles(ax, "Where the portfolio could end up", cs.describe_params(params), theme)
    cs.add_legend(ax, "upper right", theme)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the final-value distribution of a run.")
    parser.add_argument("--input", type=Path, default=Path("results/run"),
                        help="run directory containing final_hist.csv and summary.json")
    parser.add_argument("--output", type=Path, default=Path("docs/images/distribution.png"))
    parser.add_argument("--theme", choices=("light", "dark"), default="light",
                        help="colour theme (light for documents)")
    args = parser.parse_args()

    hist = pd.read_csv(args.input / "final_hist.csv")
    summary = json.loads((args.input / "summary.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    theme = cs.DARK if args.theme == "dark" else cs.LIGHT
    make_distribution_chart(hist, summary, theme).savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
