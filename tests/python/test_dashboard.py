"""Dashboard smoke test: load the real Streamlit app, press "Run simulation" and
check that a run completes and the results appear (no browser needed).

Uses Streamlit's own testing tool (streamlit.testing.v1.AppTest), which runs the
app script exactly as the server would and lets us click widgets from Python.
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "dashboard" / "app.py"


def test_dashboard_runs_a_simulation() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert "Run simulation" in at.info[0].value  # waiting for the user

    # Small, quick run on 2 processes x 2 threads.
    trials = next(w for w in at.select_slider if w.label == "Number of simulated futures")
    trials.set_value(100_000)
    next(w for w in at.slider if w.label == "MPI processes").set_value(2)
    next(w for w in at.slider if w.label == "OpenMP threads per process").set_value(2)
    at.button[0].click()
    at.run()

    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert "Median final value" in labels
    assert "Value at Risk (95%)" in labels

    # The run really happened with the chosen layout, and the shown VaR is the file's VaR.
    result = at.session_state["result"]
    summary = result["summary"]
    assert summary["params"]["trials"] == 100_000
    assert summary["parallel"] == {"mpi_processes": 2, "omp_threads": 2}
    shown_var = next(m for m in at.metric if m.label.startswith("Value at Risk")).value
    assert shown_var == f"{summary['risk']['var']:,.0f}"
    assert json.loads((result["out_dir"] / "summary.json").read_text()) == summary
    print(f"\n  dashboard run: {result['command']}\n"
          f"  median {summary['final_value']['median']:,.0f}, VaR {shown_var}, "
          f"checksum {summary['histogram']['checksum']}", end="")
