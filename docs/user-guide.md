# User guide

Everything you need to install, run and read the results of the simulator.
For a 2-minute start, see the [README's Quick start](../README.md#quick-start).

## 1. Install

Tested on Ubuntu 24.04 (native, VirtualBox VM, or WSL2). You need a C++ compiler, Open MPI
and Python 3:

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3-venv python3-pip git
git clone https://github.com/maryammustafa05/parallel-monte-carlo-simulator.git
cd parallel-monte-carlo-simulator
bash scripts/setup_env.sh          # checks the tools, creates .venv, installs Python packages (~1 min)
```

`setup_env.sh` prints `[ok]` for each tool, or the exact `apt` command for anything missing
(`bash scripts/setup_env.sh --install` runs that command for you). It also warns if the
machine has 2 or fewer cores. In VirtualBox, give the VM more cores under
*Settings → System → Processor* while it's powered off.

## 2. Build and test

```bash
make              # builds build/mc_serial, build/mc_hybrid, test programs; no warnings expected
make test         # all tests, ~1 minute; ends with "ALL TESTS PASSED"
```

| Command | What it does |
|---|---|
| `make` | builds everything |
| `make serial` / `make hybrid` | builds only `build/mc_serial` / `build/mc_hybrid` |
| `make test` | builds and runs every test (needs `.venv`; or `PYTHON=/path/to/python make test`) |
| `make bench` | full benchmark + charts, 10–20 minutes |
| `make clean` | deletes `build/` |

## 3. Run a simulation

**Serial** (one core, the baseline):

```bash
./build/mc_serial --trials 1000000
```

**Parallel**: `-np` MPI processes, each with `--threads` OpenMP threads:

```bash
mpirun --bind-to none -np 4 ./build/mc_hybrid --trials 1000000 --threads 1
```

- Keep processes × threads ≤ the number of cores (`nproc`). More doesn't run faster.
- `--bind-to none` stops Open MPI from pinning a whole process to one core, which would
  squeeze all of its threads onto that core.
- More *processes* than cores needs `--oversubscribe`. Extra *threads* don't.

Both programs print a short report and write three files into `--out` (default `results/run`):

```
mc_hybrid: 1000000 trials x 120 months, 4 process(es) x 1 thread(s)
  time: total 2.843 s (compute 2.837 s, reduce 0.007 s)
  final value: mean 20093.63  median 17983.26  P5 8245.17  P95 39049.85
  VaR(95%): 1754.83   P(loss): 0.1075   histogram overflow: low 0, high 0
  checksum 8d550353ce5046de   wrote results/demo/{fan_chart.csv, final_hist.csv, summary.json}
```

The same seed gives the same results and checksum for any number of processes and threads.

### All options

| Flag | Meaning | Default | Allowed |
|---|---|---|---|
| `--initial X` | starting investment V0 | 10000 | > 0 and ≤ 10¹² |
| `--mu X` | expected **annual** return (0.07 = 7 %) | 0.07 | > −1 and ≤ 10 |
| `--sigma X` | **annual** volatility (0.15 = 15 %) | 0.15 | 0 to 10 |
| `--months N` | horizon in months | 120 | 1 to 1200 |
| `--trials N` | number of simulated futures (`1e6` is accepted) | 1000000 | ≥ 1 |
| `--confidence X` | VaR confidence level | 0.95 | strictly between 0 and 1 |
| `--seed N` | random seed (any 64-bit unsigned number) | 42 | 0 to 2⁶⁴−1 |
| `--threads N` | OpenMP threads per process (`mc_hybrid` only) | 1 | 1 to 256 |
| `--bins N` | histogram bins per month (accuracy vs memory) | 2048 | 2 to 65536 |
| `--out DIR` | output folder (created if missing) | `results/run` | |
| `--exact` | also compute exact percentiles from all final values | off | needs `--trials` ≤ 10⁷ |
| `--help` | show the options | | |

Combinations that could overflow a double, such as `--sigma 10 --months 1200`, are
rejected with an explanation. Bad input prints one `error:` line and exits with code 1.

**The model in one line:** each month, return r = max(−1, μ/12 + σ/√12 · Z) with Z ~ Normal(0,1),
and V ← V·(1 + r). A month with r = −1 wipes the portfolio out, and it stays at 0.

## 4. Read the results

**`fan_chart.csv`**: one row per month, 0 to T:

```
month,p05,p25,median,p75,p95,mean
0,10000,10000,10000,10000,10000,10000
1,9346.208637,9764.635924,10058.0708,10349.0693,10769.27655,10057.78767
```

`p05` = 5 % of futures are below this value; `median` = the typical future; `mean` = the
average, which is pulled up by a few very good futures.

**`summary.json`**: everything else ([real example](../results/sample/summary.json)):

| Field | Meaning |
|---|---|
| `final_value.median`, `p05`, `p95` | typical, bad (worst 5 %) and good (best 5 %) final value |
| `final_value.mean`, `std`, `min`, `max` | average, spread, extremes of the final value |
| `risk.var` | **Value at Risk**: with `confidence` (e.g. 95 %) you won't lose more than this over the horizon. `var = max(0, initial − threshold)` |
| `risk.threshold` | the (1 − confidence) quantile of the final value, e.g. the 5th percentile |
| `risk.prob_loss` | share of futures that end below the starting amount |
| `risk.prob_total_loss` | share of futures wiped out completely |
| `histogram.overflow_low/high` | values outside the histogram range (should be 0 or tiny) |
| `histogram.checksum` | fingerprint of all counts: equal checksums = identical results |
| `timing.total_s`, `compute_s`, `reduce_s` | wall time, simulation time, MPI communication time (slowest process) |
| `exact` (with `--exact`) | true median/P5/P95/VaR and the histogram's relative difference from them |

**`final_hist.csv`**: `bin_lower,bin_upper,count` of the final month, used for the
distribution chart.

## 5. Charts

```bash
bash scripts/run_demo.sh                 # build + 1M-trial run on up to 4 processes + both charts
bash scripts/run_demo.sh --sigma 0.25    # extra flags go to mc_hybrid
```

Or for any run folder:

```bash
.venv/bin/python scripts/plot_fan_chart.py    --input results/run --output fan.png
.venv/bin/python scripts/plot_distribution.py --input results/run --output dist.png [--theme dark]
```

- **Fan chart:** shaded bands for the middle 50 % and 90 % of futures, the median and mean
  paths, and the starting amount.
- **Distribution:** where the portfolio ends. The worst (1 − confidence) share is shaded
  orange; the orange line is the VaR threshold. For very wide outcomes it switches to a log
  axis, and if every future is identical it says so in words.

## 6. Dashboard

```bash
.venv/bin/streamlit run dashboard/app.py
```

Open **http://localhost:8501**. In the sidebar, set the investment, the number of futures,
and the MPI processes and threads, then press **Run simulation**. You get the headline numbers,
both charts, a table, and a *Run details* tab with the exact `mpirun` command, timing and checksum.
Runs are saved in `results/dashboard/<date-time>/`.

**Working in a VM through VS Code Remote-SSH:** VS Code forwards the port automatically
when you start Streamlit in its terminal. If the browser says it can't connect, open the
**Ports** tab, click *Forward a Port*, enter `8501`, and use the address shown under
*Forwarded Address*.

## 7. Benchmark

```bash
make bench
```

This runs `scripts/run_benchmark.sh` (serial baseline + every processes × threads layout up to
the core count, plus 2× oversubscribed layouts, 3 interleaved repeats, a weak-scaling series) and
then `scripts/plot_benchmark.py` (medians, speedup, efficiency, Karp–Flatt, 4 charts in
`docs/images/`). Data: `results/benchmark.csv`, `results/benchmark_weak.csv`,
`results/benchmark_summary.csv`, `results/benchmark_env.txt`.

Options: `bash scripts/run_benchmark.sh --trials N --weak-trials N --repeats R`, or `--quick` for
a 20-second smoke run into `build/bench_quick/`. Pick `--trials` so the serial run takes
20–60 s on your machine. Close other programs while it runs, and on a laptop keep it plugged
in: the [benchmark report](benchmark-report.md) shows how much host activity changes timings.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Authorization required, but no authorization protocol specified` (several times) | Harmless: Open MPI's hardware library tries to reach the VM's display. Hide it with `export HWLOC_COMPONENTS=-gl` (add to `~/.bashrc`). |
| `There are not enough slots available ...` | More processes than cores. Use fewer `-np`, or add `--oversubscribe` (slower, same results). |
| `mpirun: command not found` / `mpicxx: not found` | Open MPI is missing: `sudo apt install -y openmpi-bin libopenmpi-dev`. |
| `make test` says `numpy/pytest not found` | The Python environment is missing: `bash scripts/setup_env.sh`. |
| Dashboard: "build/mc_hybrid not found" | Run `make` first, then reload the page. |
| Browser can't reach `localhost:8501` | See the port-forwarding note in section 6. |
| Timings vary a lot between runs | Normal on a VM or a laptop on battery. Repeat and use the median; plug in and close other programs. |
| `error: ...` and exit code 1 | An input is out of range; the message names the flag and the allowed range. |
