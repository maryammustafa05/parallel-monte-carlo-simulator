"""Independent NumPy implementation of the Monte Carlo model (CLAUDE.md section 4).

Why this exists: if the C++ engine had a bug in the model (wrong monthly
conversion, wrong clamp, wrong percentile), its own tests might share the same
mistake. This file implements the same model from scratch, in a different
language, with a different random number generator (NumPy's PCG64 instead of
our xoshiro256**) and with exact percentiles (no histograms). The C++ results
must agree with it within statistical error: see test_reference.py.

It keeps all N values of one month in memory at a time, so use it for
N up to a few million.

Command-line use (writes a fan chart in the same CSV format as the C++ program):
    python tests/python/reference_numpy.py --trials 200000 --output build/ref_fan.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PERCENTILES = (5, 25, 50, 75, 95)


def simulate(
    initial: float,
    mu: float,
    sigma: float,
    months: int,
    trials: int,
    seed: int,
    keep_months: tuple[int, ...] = (),
) -> dict:
    """Simulates `trials` paths month by month (all trials at once, vectorised).

    Returns a dict with:
      "fan":     array of shape (months + 1, 7): month, p05, p25, median, p75, p95, mean
      "samples": {month: copy of all V_t values} for each month in keep_months
                 (the final month is always included)
    """
    rng = np.random.default_rng(seed)
    mu_m = mu / 12.0
    sigma_m = sigma / np.sqrt(12.0)

    values = np.full(trials, float(initial))
    fan = [[0, *([float(initial)] * len(PERCENTILES)), float(initial)]]
    samples: dict[int, np.ndarray] = {}
    wanted = set(keep_months) | {months}

    for t in range(1, months + 1):
        returns = mu_m + sigma_m * rng.standard_normal(trials)
        returns = np.maximum(returns, -1.0)  # a loss cannot exceed 100 %
        values *= 1.0 + returns
        fan.append([t, *np.percentile(values, PERCENTILES), values.mean()])
        if t in wanted:
            samples[t] = values.copy()

    return {"fan": np.array(fan), "samples": samples}


def value_at_risk(initial: float, final_values: np.ndarray, confidence: float) -> float:
    """VaR = max(0, V0 - (1 - confidence) quantile of the final values)."""
    worst_case = np.percentile(final_values, 100.0 * (1.0 - confidence))
    return max(0.0, initial - float(worst_case))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--initial", type=float, default=10000.0)
    parser.add_argument("--mu", type=float, default=0.07)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--trials", type=int, default=200000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, required=True, help="fan chart CSV to write")
    args = parser.parse_args()

    result = simulate(args.initial, args.mu, args.sigma, args.months, args.trials, args.seed)
    final = result["samples"][args.months]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output,
        result["fan"],
        delimiter=",",
        header="month,p05,p25,median,p75,p95,mean",
        comments="",
        fmt=["%d"] + ["%.10g"] * 6,
    )
    print(f"reference: {args.trials} trials x {args.months} months -> {args.output}")
    print(
        f"  final: mean {final.mean():.2f}  median {np.median(final):.2f}  "
        f"VaR({args.confidence:.0%}) {value_at_risk(args.initial, final, args.confidence):.2f}  "
        f"P(loss) {np.mean(final < args.initial):.4f}"
    )


if __name__ == "__main__":
    main()
