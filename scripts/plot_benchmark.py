"""Benchmark analysis: median times, speedup, efficiency, and the charts.

Reads results/benchmark.csv (strong scaling) and results/benchmark_weak.csv
(weak scaling) written by scripts/run_benchmark.sh, and writes:
  <output-dir>/benchmark_time.png        median wall time vs number of workers
  <output-dir>/benchmark_speedup.png     speedup vs workers, with the ideal line
  <output-dir>/benchmark_efficiency.png  efficiency heat-map over processes x threads
  <output-dir>/benchmark_weak.png        weak scaling: time as the problem grows with workers
  <summary>                               table of medians, speedup, efficiency, Karp-Flatt

Definitions (D5):
  speedup    S = T_serial / T_parallel          (T = median of the repeated runs)
  efficiency E = S / workers                    (1.0 = perfect use of every worker)
  Karp-Flatt e = (1/S - 1/p) / (1 - 1/p)        (measured "serial fraction"; for p > 1)

    python scripts/plot_benchmark.py --input results/benchmark.csv --weak results/benchmark_weak.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import chart_style as cs

AQUA = "#1baf7a"  # third categorical colour (blue, orange, aqua are colour-blind safe together)


def secs(x: float) -> str:
    """Seconds with sensible precision: 0.27 s, 4.83 s, 23.4 s."""
    return f"{x:.2f} s" if x < 10 else f"{x:.1f} s"


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration with median times, speedup, efficiency and Karp-Flatt."""
    keys = ["mode", "processes", "threads", "total_workers", "oversubscribed"]
    table = (raw.groupby(keys, as_index=False)
                .agg(runs=("run", "count"), total_s=("total_s", "median"),
                     min_s=("total_s", "min"), max_s=("total_s", "max"),
                     compute_s=("compute_s", "median"), reduce_s=("reduce_s", "median")))
    serial = table.loc[table["mode"] == "serial", "total_s"].iloc[0]
    table["speedup"] = serial / table["total_s"]
    # Range of speedups implied by the slowest / fastest repeat (shown as whiskers).
    table["speedup_lo"] = serial / table["max_s"]
    table["speedup_hi"] = serial / table["min_s"]
    table["efficiency"] = table["speedup"] / table["total_workers"]
    p = table["total_workers"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kf = (1 / table["speedup"] - 1 / p) / (1 - 1 / p)
    table["karp_flatt"] = np.where(p > 1, kf, np.nan)
    table["spread_pct"] = 100 * (table["max_s"] - table["min_s"]) / table["total_s"]
    return table.sort_values(["mode", "total_workers", "processes"], ascending=[False, True, True])


def layout_of(row: pd.Series) -> str:
    """Which kind of parallelism a configuration uses."""
    if row["mode"] == "serial":
        return "serial"
    if row["threads"] == 1:
        return "processes"
    if row["processes"] == 1:
        return "threads"
    return "mixed"


SERIES = {  # layout -> (legend label, colour, marker)
    "processes": ("MPI processes only (P × 1)", cs.LIGHT.blue, "o"),
    "threads": ("OpenMP threads only (1 × T)", cs.LIGHT.orange, "s"),
    "mixed": ("Mixed processes × threads", AQUA, "D"),
}


def _base_figure(theme: cs.Theme) -> tuple[Figure, object]:
    fig = Figure(figsize=cs.FIGSIZE, dpi=cs.DPI)
    ax = fig.add_subplot()
    cs.style_axes(ax, theme)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:g}")
    return fig, ax


def _shade_oversubscribed(ax, cores: int, x_max: float, theme: cs.Theme) -> None:
    """Grey area for 'more workers than cores', with a label."""
    ax.axvspan(cores + 0.5, x_max, color=theme.grid, alpha=0.6, linewidth=0, zorder=0)
    ax.text(cores + 0.7, 0.03, f"more workers\nthan {cores} cores", transform=ax.get_xaxis_transform(),
            va="bottom", fontsize=9, color=theme.text_secondary)


def _scatter_layouts(ax, par: pd.DataFrame, column: str, low: str, high: str,
                     theme: cs.Theme) -> None:
    """One coloured series per layout, joined by a line where it is a single curve.
    Thin vertical whiskers show the fastest-to-slowest range of the repeated runs."""
    for layout, (label, colour, marker) in SERIES.items():
        rows = par[par["layout"] == layout].sort_values("total_workers")
        if rows.empty:
            continue
        ax.vlines(rows["total_workers"], rows[low], rows[high], color=colour, linewidth=1,
                  alpha=0.6, zorder=2)
        # Mixed layouts are separate points (e.g. 2x2 and 5x2), not one curve.
        style = "-" if layout != "mixed" else "none"
        ax.plot(rows["total_workers"], rows[column], linestyle=style, marker=marker, color=colour,
                linewidth=2, markersize=8, markeredgecolor=theme.surface, markeredgewidth=2,
                label=label, zorder=3)


def best_pinned(par: pd.DataFrame, column: str, largest: bool) -> pd.Series:
    """Best configuration among those with one worker per core (not oversubscribed):
    oversubscribed runs are too noisy to headline."""
    pinned = par[par["oversubscribed"] == 0]
    return pinned.loc[pinned[column].idxmax() if largest else pinned[column].idxmin()]


def plot_time(table: pd.DataFrame, cores: int, trials: int, theme: cs.Theme) -> Figure:
    serial = table[table["mode"] == "serial"].iloc[0]
    par = table[table["mode"] == "hybrid"]
    fig, ax = _base_figure(theme)
    x_max = par["total_workers"].max() + 0.8
    _shade_oversubscribed(ax, cores, x_max, theme)
    ax.axhline(serial["total_s"], color=theme.text_secondary, linewidth=1)
    ax.text(2.3, serial["total_s"], f"serial baseline {secs(serial['total_s'])}",
            ha="left", va="bottom", fontsize=9, color=theme.text_secondary)
    _scatter_layouts(ax, par, "total_s", "min_s", "max_s", theme)
    best = best_pinned(par, "total_s", largest=False)
    ax.annotate(f"fastest pinned: {best['processes']:.0f} × {best['threads']:.0f}, "
                f"{secs(best['total_s'])}", (best["total_workers"], best["total_s"]),
                xytext=(0, -18), textcoords="offset points", ha="center", va="top",
                fontsize=9, color=theme.text_primary, fontweight="bold")
    ax.set_xlim(0.5, x_max)
    # Headroom above the baseline holds the legend, clear of every data point.
    ax.set_ylim(0, max(serial["total_s"], par["max_s"].max()) * 1.4)
    ax.set_xticks(sorted(par["total_workers"].unique()))
    ax.set_xlabel("Workers (processes × threads)")
    ax.set_ylabel("Median wall time (s)")
    cs.add_titles(ax, "Wall time drops as workers are added",
                  f"{trials:,} trials × 120 months · dots: median of repeated runs, "
                  "whiskers: fastest to slowest run", theme)
    cs.add_legend(ax, "upper right", theme)
    fig.tight_layout()
    return fig


def plot_speedup(table: pd.DataFrame, cores: int, trials: int, theme: cs.Theme) -> Figure:
    par = table[table["mode"] == "hybrid"]
    fig, ax = _base_figure(theme)
    x_max = par["total_workers"].max() + 0.8
    _shade_oversubscribed(ax, cores, x_max, theme)
    ideal = np.array([1, cores])
    ax.plot(ideal, ideal, color=theme.text_secondary, linewidth=1)
    # Label the ideal line halfway up, just left of it (clear of the legend).
    mid = (1 + cores) / 2
    ax.annotate("ideal: speedup = workers", (mid, mid), xytext=(-8, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=9,
                color=theme.text_secondary)
    _scatter_layouts(ax, par, "speedup", "speedup_lo", "speedup_hi", theme)
    best = best_pinned(par, "speedup", largest=True)
    ax.annotate(f"{best['speedup']:.2f}×", (best["total_workers"], best["speedup"]),
                xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10,
                color=theme.text_primary, fontweight="bold")
    ax.set_xlim(0.5, x_max)
    ax.set_ylim(0, max(cores, par["speedup_hi"].max()) * 1.1)
    ax.set_xticks(sorted(par["total_workers"].unique()))
    ax.set_xlabel("Workers (processes × threads)")
    ax.set_ylabel("Speedup vs serial (×)")
    cs.add_titles(ax, "Speedup grows, then flattens near the physical core count",
                  f"Strong scaling: fixed {trials:,} trials · speedup = T_serial / T_parallel"
                  " · whiskers: range over repeated runs",
                  theme)
    cs.add_legend(ax, "upper left", theme)
    fig.tight_layout()
    return fig


def plot_efficiency(table: pd.DataFrame, theme: cs.Theme) -> Figure:
    """Heat-map: rows = processes, columns = threads, cell = efficiency."""
    from matplotlib.colors import LinearSegmentedColormap

    par = table[table["mode"] == "hybrid"]
    procs = sorted(par["processes"].unique())
    threads = sorted(par["threads"].unique())
    grid = np.full((len(procs), len(threads)), np.nan)
    for _, row in par.iterrows():
        grid[procs.index(row["processes"]), threads.index(row["threads"])] = row["efficiency"]

    # One-hue sequential ramp (light = low efficiency, dark = high).
    ramp = LinearSegmentedColormap.from_list("blues", ["#e8f1fc", "#86b6ef", "#2a78d6", "#104281"])
    fig = Figure(figsize=(8, 5.6), dpi=cs.DPI)
    ax = fig.add_subplot()
    ax.set_facecolor(theme.surface)
    fig.set_facecolor(theme.surface)
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=ramp, vmin=0, vmax=1, aspect="auto")
    for i in range(len(procs)):
        for j in range(len(threads)):
            if np.isnan(grid[i, j]):
                continue
            value = grid[i, j]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if value > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(threads)), [str(t) for t in threads])
    ax.set_yticks(range(len(procs)), [str(p) for p in procs])
    ax.set_xlabel("OpenMP threads per process")
    ax.set_ylabel("MPI processes")
    ax.tick_params(colors=theme.text_secondary, length=0)
    ax.xaxis.label.set_color(theme.text_secondary)
    ax.yaxis.label.set_color(theme.text_secondary)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    bar.ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    bar.ax.tick_params(colors=theme.text_secondary, length=0)
    bar.outline.set_visible(False)
    cs.add_titles(ax, "Parallel efficiency for each processes × threads layout",
                  "Efficiency = speedup ÷ workers · empty cells were not measured", theme)
    fig.tight_layout()
    return fig


def plot_weak(weak: pd.DataFrame, per_worker: int, theme: cs.Theme) -> Figure:
    med = weak.groupby("processes", as_index=False)["total_s"].median()
    t1 = med.loc[med["processes"] == 1, "total_s"].iloc[0]
    med["efficiency"] = t1 / med["total_s"]
    fig, ax = _base_figure(theme)
    ax.axhline(t1, color=theme.text_secondary, linewidth=1)
    ax.text(med["processes"].max(), t1, "ideal: time stays flat ", ha="right", va="bottom",
            fontsize=9, color=theme.text_secondary)
    ax.plot(med["processes"], med["total_s"], "-o", color=theme.blue, linewidth=2, markersize=8,
            markeredgecolor=theme.surface, markeredgewidth=2, zorder=3)
    for _, row in med.iterrows():
        ax.annotate(f"{secs(row['total_s'])}\n({row['efficiency']:.0%})",
                    (row["processes"], row["total_s"]), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    color=theme.text_primary)
    ax.set_xticks(med["processes"])
    ax.set_ylim(0, med["total_s"].max() * 1.35)
    ax.set_xlabel("MPI processes (1 thread each)")
    ax.set_ylabel("Median wall time (s)")
    cs.add_titles(ax, "Weak scaling: more workers, proportionally more work",
                  f"{per_worker:,} trials per process · in brackets: weak-scaling efficiency "
                  "T(1) / T(P)", theme)
    fig.tight_layout()
    return fig


def markdown_table(table: pd.DataFrame) -> str:
    """The summary as a Markdown table (pasted into docs/benchmark-report.md)."""
    lines = ["| Layout | P × T | Workers | Median time (s) | Spread | Compute (s) | Reduce (ms) "
             "| Speedup | Efficiency | Karp–Flatt e |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        layout = "serial" if r["mode"] == "serial" else (
            f"{layout_of(r)}{' (oversubscribed)' if r['oversubscribed'] else ''}")
        kf = "—" if np.isnan(r["karp_flatt"]) else f"{r['karp_flatt']:.3f}"
        lines.append(
            f"| {layout} | {r['processes']:.0f} × {r['threads']:.0f} | {r['total_workers']:.0f} "
            f"| {r['total_s']:.2f} | ±{r['spread_pct'] / 2:.0f} % | {r['compute_s']:.2f} "
            f"| {1000 * r['reduce_s']:.1f} | {r['speedup']:.2f} | {r['efficiency']:.0%} | {kf} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse and plot benchmark results.")
    parser.add_argument("--input", type=Path, default=Path("results/benchmark.csv"))
    parser.add_argument("--weak", type=Path, default=Path("results/benchmark_weak.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    parser.add_argument("--summary", type=Path, default=Path("results/benchmark_summary.csv"))
    parser.add_argument("--cores", type=int, default=None,
                        help="cores of the benchmark machine (default: largest non-oversubscribed P×T)")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    table = summarise(raw)
    table["layout"] = table.apply(layout_of, axis=1)
    cores = args.cores or int(table.loc[table["oversubscribed"] == 0, "total_workers"].max())
    trials = int(raw["trials"].iloc[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    table.drop(columns="layout").to_csv(args.summary, index=False, float_format="%.6g")
    theme = cs.LIGHT
    outputs = {
        "benchmark_time.png": plot_time(table, cores, trials, theme),
        "benchmark_speedup.png": plot_speedup(table, cores, trials, theme),
        "benchmark_efficiency.png": plot_efficiency(table, theme),
    }
    if args.weak.exists():
        weak = pd.read_csv(args.weak)
        per_worker = int(weak.loc[weak["processes"] == 1, "trials"].iloc[0])
        outputs["benchmark_weak.png"] = plot_weak(weak, per_worker, theme)
    for name, fig in outputs.items():
        fig.savefig(args.output_dir / name)
        print(f"wrote {args.output_dir / name}")
    print(f"wrote {args.summary}\n")
    print(markdown_table(table))


if __name__ == "__main__":
    main()
