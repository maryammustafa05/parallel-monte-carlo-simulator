"""C++ engine vs the independent NumPy reference (Phase 4).

Both simulate the same model with DIFFERENT random numbers, so their answers
differ by Monte Carlo noise. "Agree" therefore means: the difference is within
4 standard errors (4 SE). For a percentile the standard error comes from the
classic formula
    SE(q-quantile) = sqrt(q (1 - q) / N) / f(x_q),
where f(x_q) is the probability density at that point (estimated from the
reference samples). Two independent runs of size N each -> SE of the
difference = sqrt(SE_cpp^2 + SE_ref^2).

Run: .venv/bin/python -m pytest -s tests/python    (after `make`)
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_numpy  # noqa: E402  (needs the path set above)

REPO = Path(__file__).resolve().parents[2]
MC_SERIAL = REPO / "build" / "mc_serial"

PARAMS = {"initial": 10000.0, "mu": 0.07, "sigma": 0.15, "months": 120, "confidence": 0.95}
N = 200_000
CHECK_MONTHS = (1, 12, 60, 120)
NUM_SE = 4.0
# The fan chart comes from histograms. --exact runs show the histogram error is
# about 0.002 %, so 0.05 % is a generous allowance on top of statistical noise.
HISTOGRAM_REL_ALLOWANCE = 5e-4


def quantile_se(samples: np.ndarray, q: float, n: int) -> float:
    """Standard error of the q-quantile of n samples, with the density at the
    quantile estimated from `samples` by a small finite difference."""
    h = 0.01
    lo, hi = np.quantile(samples, [q - h, q + h])
    density = (2 * h) / (hi - lo)
    return math.sqrt(q * (1 - q) / n) / density


@pytest.fixture(scope="module")
def cpp(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Runs mc_serial once (with --exact) and loads its outputs."""
    if not MC_SERIAL.exists():
        pytest.fail(f"{MC_SERIAL} not found: run `make` first")
    out = tmp_path_factory.mktemp("cpp_run")
    cmd = [str(MC_SERIAL), "--trials", str(N), "--seed", "42", "--exact", "--out", str(out)]
    for key, value in PARAMS.items():
        cmd += [f"--{key}", str(value)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    with open(out / "summary.json") as f:
        summary = json.load(f)
    with open(out / "fan_chart.csv") as f:
        fan = {int(row["month"]): {k: float(v) for k, v in row.items()} for row in csv.DictReader(f)}
    return {"summary": summary, "fan": fan}


@pytest.fixture(scope="module")
def ref() -> dict:
    """Runs the NumPy reference once with a different seed and generator."""
    p = PARAMS
    return reference_numpy.simulate(
        p["initial"], p["mu"], p["sigma"], p["months"], N, seed=12345, keep_months=CHECK_MONTHS
    )


def test_histogram_matches_exact(cpp: dict) -> None:
    """Histogram percentiles and VaR are within 1 % of the exact ones (same samples)."""
    exact = cpp["summary"]["exact"]
    for key in ("rel_diff_median", "rel_diff_p05", "rel_diff_p95", "rel_diff_var"):
        print(f"\n  histogram vs exact {key[9:]:<6}: {100 * exact[key]:.4f} %", end="")
        assert exact[key] < 0.01


def test_reference_matches_formula(ref: dict) -> None:
    """Sanity check of the reference itself: mean of V_T vs V0 * (1 + mu/12)^T."""
    final = ref["samples"][PARAMS["months"]]
    expected = PARAMS["initial"] * (1 + PARAMS["mu"] / 12) ** PARAMS["months"]
    se = final.std() / math.sqrt(N)
    print(f"\n  reference mean {final.mean():.2f} vs formula {expected:.2f} "
          f"({(final.mean() - expected) / se:+.2f} SE)", end="")
    assert abs(final.mean() - expected) < NUM_SE * se


def test_final_percentiles_agree(cpp: dict, ref: dict) -> None:
    """C++ exact median/P5/P95 of V_T vs NumPy, within 4 SE."""
    final = ref["samples"][PARAMS["months"]]
    exact = cpp["summary"]["exact"]
    for name, q in (("p05", 0.05), ("median", 0.50), ("p95", 0.95)):
        ours, theirs = exact[name], float(np.quantile(final, q))
        se_diff = math.sqrt(2) * quantile_se(final, q, N)
        print(f"\n  final {name:<6}: C++ {ours:10.2f}  NumPy {theirs:10.2f}  "
              f"diff {(ours - theirs) / se_diff:+.2f} SE", end="")
        assert abs(ours - theirs) < NUM_SE * se_diff


def test_fan_chart_agrees(cpp: dict, ref: dict) -> None:
    """C++ histogram fan chart (P5, median, P95) vs NumPy at several months."""
    columns = {"p05": (0.05, 1), "median": (0.50, 3), "p95": (0.95, 5)}
    for month in CHECK_MONTHS:
        samples = ref["samples"][month]
        for name, (q, col) in columns.items():
            ours = cpp["fan"][month][name]
            theirs = ref["fan"][month][col]
            se_diff = math.sqrt(2) * quantile_se(samples, q, N)
            tol = NUM_SE * se_diff + HISTOGRAM_REL_ALLOWANCE * abs(theirs)
            print(f"\n  month {month:3d} {name:<6}: C++ {ours:10.2f}  NumPy {theirs:10.2f}  "
                  f"diff {(ours - theirs) / se_diff:+.2f} SE", end="")
            assert abs(ours - theirs) < tol


def test_mean_var_and_prob_loss_agree(cpp: dict, ref: dict) -> None:
    """Final mean, VaR and P(loss) agree within 4 SE."""
    final = ref["samples"][PARAMS["months"]]
    summary = cpp["summary"]
    V0, c = PARAMS["initial"], PARAMS["confidence"]

    # Mean: SE = std / sqrt(N) for each run.
    se_mean = math.sqrt(summary["final_value"]["std"] ** 2 / N + final.var() / N)
    mean_diff = summary["final_value"]["mean"] - final.mean()
    print(f"\n  mean     : C++ {summary['final_value']['mean']:10.2f}  NumPy {final.mean():10.2f}  "
          f"diff {mean_diff / se_mean:+.2f} SE", end="")
    assert abs(mean_diff) < NUM_SE * se_mean

    # VaR = V0 - P5, so it has the same SE as the 5th percentile.
    ref_var = reference_numpy.value_at_risk(V0, final, c)
    se_var = math.sqrt(2) * quantile_se(final, 1 - c, N)
    var_diff = summary["risk"]["var"] - ref_var
    print(f"\n  VaR      : C++ {summary['risk']['var']:10.2f}  NumPy {ref_var:10.2f}  "
          f"diff {var_diff / se_var:+.2f} SE", end="")
    assert abs(var_diff) < NUM_SE * se_var

    # P(loss) is a proportion: binomial SE = sqrt(p (1 - p) / N) for each run.
    p_ref = float(np.mean(final < V0))
    p_cpp = summary["risk"]["prob_loss"]
    se_p = math.sqrt(2 * p_ref * (1 - p_ref) / N)
    print(f"\n  P(loss)  : C++ {p_cpp:10.4f}  NumPy {p_ref:10.4f}  "
          f"diff {(p_cpp - p_ref) / se_p:+.2f} SE", end="")
    assert abs(p_cpp - p_ref) < NUM_SE * se_p
