"""Streamlit dashboard for the Monte Carlo simulator.

Pick the investment settings and the parallel layout in the sidebar, press
"Run simulation", and the dashboard runs the real C++ program through `mpirun`,
then shows the fan chart, the final-value distribution, VaR and timings.

Start it from the repository root (after `make`):
    streamlit run dashboard/app.py
then open http://localhost:8501 (VS Code forwards the port from the VM).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from plot_distribution import make_distribution_chart  # noqa: E402
from plot_fan_chart import make_fan_chart  # noqa: E402

MC_HYBRID = REPO / "build" / "mc_hybrid"
RUNS_DIR = REPO / "results" / "dashboard"
CORES = os.cpu_count() or 1
TRIAL_CHOICES = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]


def build_command(settings: dict, out_dir: Path) -> list[str]:
    """The mpirun command line for these settings."""
    cmd = ["mpirun", "--bind-to", "none"]
    if settings["processes"] > CORES:
        cmd.append("--oversubscribe")  # more processes than cores: Open MPI refuses otherwise
    cmd += ["-np", str(settings["processes"]), str(MC_HYBRID)]
    for flag in ("initial", "mu", "sigma", "months", "trials", "confidence", "seed", "threads"):
        cmd += [f"--{flag}", str(settings[flag])]
    cmd += ["--out", str(out_dir)]
    return cmd


def run_simulation(settings: dict) -> dict:
    """Runs the C++ engine and loads its outputs. Raises RuntimeError on failure."""
    out_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    cmd = build_command(settings, out_dir)
    # HWLOC_COMPONENTS=-gl hides a harmless hwloc display warning on VMs (see README).
    env = {**os.environ, "HWLOC_COMPONENTS": "-gl"}
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit code {proc.returncode}):\n"
                           f"{' '.join(cmd)}\n\n{proc.stderr or proc.stdout}")
    return {
        "summary": json.loads((out_dir / "summary.json").read_text()),
        "fan": pd.read_csv(out_dir / "fan_chart.csv"),
        "hist": pd.read_csv(out_dir / "final_hist.csv"),
        "command": " ".join(cmd),
        "stdout": proc.stdout,
        "wall_s": wall,
        "out_dir": out_dir,
    }


def sidebar() -> tuple[dict, bool]:
    """Draws the input controls; returns the settings and whether Run was pressed."""
    st.sidebar.header("Investment")
    initial = st.sidebar.number_input("Initial investment", min_value=100.0, max_value=1e9,
                                      value=10_000.0, step=1_000.0, format="%.0f")
    mu_pct = st.sidebar.slider("Expected annual return (%)", -10.0, 25.0, 7.0, 0.5)
    sigma_pct = st.sidebar.slider("Annual volatility (%)", 0.0, 60.0, 15.0, 0.5)
    years = st.sidebar.slider("Horizon (years)", 1, 40, 10)
    confidence_pct = st.sidebar.select_slider("VaR confidence (%)", [90, 95, 97.5, 99], 95)

    st.sidebar.header("Simulation")
    trials = st.sidebar.select_slider("Number of simulated futures", TRIAL_CHOICES,
                                      value=1_000_000, format_func=lambda n: f"{n:,}")
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    st.sidebar.header("Parallelism")
    processes = st.sidebar.slider("MPI processes", 1, max(CORES, 2) * 2, min(4, CORES))
    threads = st.sidebar.slider("OpenMP threads per process", 1, max(CORES, 2) * 2, 1)
    workers = processes * threads
    st.sidebar.caption(f"{workers} worker(s) in total on {CORES} CPU core(s).")
    if workers > CORES:
        st.sidebar.warning("More workers than cores: they will take turns, so it "
                           "won't run faster (results stay identical).")

    run = st.sidebar.button("Run simulation", type="primary", width="stretch")
    settings = {
        "initial": initial, "mu": mu_pct / 100, "sigma": sigma_pct / 100,
        "months": years * 12, "trials": trials, "confidence": confidence_pct / 100,
        "seed": int(seed), "processes": processes, "threads": threads,
    }
    return settings, run


def show_results(result: dict) -> None:
    """Headline numbers, both charts, a table view and the timing breakdown."""
    s = result["summary"]
    final, risk, timing = s["final_value"], s["risk"], s["timing"]
    initial = s["params"]["initial"]
    conf = risk["confidence"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median final value", f"{final['median']:,.0f}",
              f"{final['median'] / initial - 1:+.1%} vs start", delta_color="off")
    c2.metric(f"Value at Risk ({conf:.0%})", f"{risk['var']:,.0f}",
              help=f"With {conf:.0%} confidence, the loss over the whole horizon "
                   f"will not exceed this amount.")
    c3.metric("Chance of a loss", f"{risk['prob_loss']:.1%}",
              help="Share of simulated futures that end below the initial investment.")
    c4.metric("Compute time", f"{timing['total_s']:.2f} s",
              f"{s['parallel']['mpi_processes']} proc × {s['parallel']['omp_threads']} thr",
              delta_color="off")

    tab_fan, tab_dist, tab_table, tab_run = st.tabs(
        ["Fan chart", "Final-value distribution", "Table", "Run details"])
    with tab_fan:
        st.pyplot(make_fan_chart(result["fan"], s))
    with tab_dist:
        st.pyplot(make_distribution_chart(result["hist"], s))
    with tab_table:
        table = result["fan"].copy()
        table.insert(1, "year", table["month"] / 12)
        st.dataframe(table.round(2), hide_index=True, width="stretch")
    with tab_run:
        st.write(f"**Simulation:** {timing['compute_s']:.3f} s computing, "
                 f"{timing['reduce_s']:.3f} s combining results across processes "
                 f"(MPI_Reduce); {result['wall_s']:.2f} s including program start-up.")
        st.write(f"**Histogram checksum:** `{s['histogram']['checksum']}`. It is the "
                 "same for any number of processes/threads with the same seed.")
        st.write(f"**Output folder:** `{result['out_dir'].relative_to(REPO)}`")
        st.code(result["command"], language="bash")
        st.code(result["stdout"], language="text")


def main() -> None:
    st.set_page_config(page_title="Monte Carlo Risk Simulator", layout="wide")
    st.title("Parallel Monte Carlo Investment & Risk Simulator")
    st.caption("Simulates many possible futures of an investment with a hybrid "
               "MPI + OpenMP C++ engine and summarises the spread of outcomes.")

    settings, run = sidebar()

    if not MC_HYBRID.exists():
        st.error(f"`{MC_HYBRID.relative_to(REPO)}` not found. Build it first: run `make` "
                 "in the repository root, then reload this page.")
        return

    if run:
        with st.spinner(f"Simulating {settings['trials']:,} futures on "
                        f"{settings['processes']} process(es) × {settings['threads']} "
                        f"thread(s)…"):
            try:
                st.session_state["result"] = run_simulation(settings)
            except RuntimeError as err:
                st.error(str(err))
                return

    if "result" in st.session_state:
        show_results(st.session_state["result"])
    else:
        st.info("Choose settings in the sidebar and press **Run simulation**.")


main()
