"""Chart scripts must work for ordinary AND extreme runs, in both themes.

Before Phase 7 the distribution chart crashed for sigma = 0 (all futures equal)
and was unreadable for very wide outcomes. This test draws every chart for a set
of runs that cover those cases and saves them (saving is where matplotlib fails).
It also checks the benchmark maths on a small hand-made table.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import chart_style as cs  # noqa: E402
from plot_benchmark import summarise  # noqa: E402
from plot_distribution import make_distribution_chart  # noqa: E402
from plot_fan_chart import make_fan_chart  # noqa: E402

RUNS = {
    "normal": ["--trials", "20000"],
    "flat": ["--sigma", "0", "--trials", "2000"],            # every future identical
    "wiped_out": ["--sigma", "3", "--trials", "5000"],      # (almost) everyone loses everything
    "very_wide": ["--sigma", "1", "--trials", "20000"],     # outcomes span ~10 orders of magnitude
    "one_month_99": ["--months", "1", "--confidence", "0.99", "--trials", "20000"],
}


@pytest.fixture(scope="module", params=list(RUNS))
def run_dir(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp(request.param)
    subprocess.run([str(REPO / "build" / "mc_serial"), *RUNS[request.param], "--out", str(out)],
                   check=True, capture_output=True)
    return out


@pytest.mark.parametrize("theme", [cs.LIGHT, cs.DARK], ids=["light", "dark"])
def test_charts_render(run_dir: Path, theme: cs.Theme, tmp_path: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    fan = pd.read_csv(run_dir / "fan_chart.csv")
    hist = pd.read_csv(run_dir / "final_hist.csv")
    for name, fig in (("fan", make_fan_chart(fan, summary, theme)),
                      ("dist", make_distribution_chart(hist, summary, theme))):
        path = tmp_path / f"{name}.png"
        fig.savefig(path)
        assert path.stat().st_size > 10_000, f"{name} chart looks empty"


def test_benchmark_maths() -> None:
    """Serial 10 s; 2 workers 6 s; 4 workers 4 s (medians of the runs below)."""
    rows = [("serial", 1, 1, 10.0), ("serial", 1, 1, 11.0), ("serial", 1, 1, 9.0),
            ("hybrid", 2, 1, 6.0), ("hybrid", 2, 1, 5.0), ("hybrid", 2, 1, 7.0),
            ("hybrid", 2, 2, 4.0), ("hybrid", 2, 2, 4.0), ("hybrid", 2, 2, 8.0)]
    raw = pd.DataFrame(
        [{"mode": m, "processes": p, "threads": t, "total_workers": p * t, "trials": 100,
          "run": i % 3 + 1, "total_s": s, "compute_s": s, "reduce_s": 0.0, "oversubscribed": 0}
         for i, (m, p, t, s) in enumerate(rows)])
    table = summarise(raw).set_index("total_workers")
    assert table.loc[1, "total_s"] == 10.0                      # median, not mean
    assert table.loc[2, "speedup"] == pytest.approx(10 / 6)
    assert table.loc[4, "speedup"] == pytest.approx(2.5)
    assert table.loc[4, "efficiency"] == pytest.approx(2.5 / 4)
    # Karp-Flatt: e = (1/S - 1/p) / (1 - 1/p) = (0.4 - 0.25) / 0.75 = 0.2
    assert table.loc[4, "karp_flatt"] == pytest.approx(0.2)
    assert math.isnan(table.loc[1, "karp_flatt"])
