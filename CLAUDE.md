# CLAUDE.md — Parallel Monte Carlo Investment & Risk Simulator (MPI + OpenMP)

> This file is the master brief for Claude Code. Read it fully before touching any code.
> Source of truth for *what* to build: the project proposal (`docs/proposal/proposal.pdf`).
> Source of truth for *how* to build it: this file. Keep both in sync with the code.

---

## 0. How to work on this project (rules for Claude Code)

1. **One phase at a time.** Phases are in section 9. Finish a phase, then STOP and report. Do not start the next phase until the user says "continue".
2. **Before each phase:** state in 2–3 plain sentences what you are about to build and why.
3. **After each phase** (all five, every time):
   - build with no warnings (`make`), 
   - run the tests (`make test`) and show the result,
   - tick the matching boxes in the README roadmap and update any README section that changed,
   - write a short plain-language summary (what was built, what was decided, what is next),
   - suggest a git commit message (do **not** commit or push unless asked).
4. **Never claim something works unless you ran it.** Paste real command output, never invented numbers. Benchmark numbers in the README come only from real runs on the user's machine.
5. **Explain decisions in simple language.** The user is a BS Software Engineering student learning parallel computing. Prefer a clear 10-line function over a clever 3-line one. Comment the *why*, not the *what*.
6. **Avoid unnecessary complexity.** No external C++ libraries (no Boost, Eigen, nlohmann/json). No frameworks beyond what section 3 lists. If you want to add something not listed, ask first.
7. **If a tool is missing** (`mpicxx`, `mpirun`, `matplotlib`, …) do not work around it silently. Tell the user the exact install command (section 11) and wait.
8. **Ask, don't guess,** when a requirement is ambiguous. Two short questions at most, with a recommended default.

---

## 1. Project summary

Investors face uncertain future returns. A **Monte Carlo simulation** runs a very large number of random "possible futures" of an investment and studies the spread of outcomes. Doing 1 million – 1 billion trials on one CPU core is slow, so this project builds a **hybrid parallel engine**:

- **MPI** splits the trials across several *processes*.
- **OpenMP** splits each process's trials across several *threads*.
- Results are merged: threads first (OpenMP), then processes (MPI).
- The merged data produces a **fan chart** (median, 5th and 95th percentile per month), the **95 % Value at Risk (VaR)**, a **performance benchmark** (time, speedup, efficiency, scalability), and a simple **chart/dashboard**.

Course: Parallel & Distributed Computing (PDC), BS SE, FAST NUCES Karachi, Fall 2026.
Team: Maryam Mustafa (23k-2046), Umaima Fatima (23k-3043).

---

## 2. Proposal objectives → where they are satisfied

Every objective must be demonstrably done. This table is the project's checklist of truth.

| # | Proposal objective | Module(s) | Proof / deliverable |
|---|---|---|---|
| O1 | Monte Carlo engine, month-by-month growth under random returns | `src/core/simulation.*`, `rng.hpp`, `config.*` | `mc_serial` runs; analytic-moment test passes |
| O2 | Hybrid MPI + OpenMP parallelization (processes → threads) | `src/hybrid/main_hybrid.cpp`, `accumulator.*` | `mc_hybrid` runs with `-np P` and `--threads T`; parallel result == serial result test |
| O3a | Median projected value over time | `analytics.*` | `fan_chart.csv` column `median` |
| O3b | Best/worst-case percentile bands (P5 / P95) | `analytics.*` | `fan_chart.csv` columns `p05`, `p95` |
| O3c | Value at Risk at a confidence level | `analytics.*` | `summary.json` field `var` |
| O4 | Benchmark: execution time, speedup, scalability for varying processes/threads | `scripts/run_benchmark.sh`, `plot_benchmark.py` | `results/benchmark.csv` + 3 plots + `docs/benchmark-report.md` |
| O5 | Simple, clear output (chart and/or basic dashboard) | `scripts/plot_*.py`, `dashboard/app.py` | PNG charts + Streamlit dashboard |

---

## 3. Technology stack (fixed by the proposal; extras justified)

| Component | Choice | Why (simple language) |
|---|---|---|
| Language | **C++17** | Proposal says C/C++. C++ gives us `std::vector`, `std::string`, and classes while staying as fast as C. |
| Distributed parallelism | **MPI** (OpenMPI or MPICH), compiled with `mpicxx` | Standard way to split work across processes. |
| Shared-memory parallelism | **OpenMP** (`-fopenmp`) | Split a loop across CPU cores with one `#pragma`. |
| Build | **Makefile** | Simple, no extra tools. Targets: `all`, `serial`, `hybrid`, `test`, `bench`, `clean`. |
| Output data | **CSV + JSON**, written by hand | No JSON library needed; the output is small and flat. |
| Plots | **Python 3 + Matplotlib** (+ NumPy, pandas) | Proposal choice. |
| Dashboard | **Streamlit** (small, optional extra) | Sliders → runs the C++ binary → shows charts. Fallback if unavailable: static HTML report made by Python. |
| Tests | Tiny C++ assert helper (`tests/cpp/test_util.hpp`) + `pytest` | No GoogleTest dependency. |
| Environment | Linux or **WSL2 (Ubuntu)** on a normal laptop | No special hardware. |

Compiler flags: `-std=c++17 -O3 -march=native -Wall -Wextra -fopenmp`. Use `-O2` without `-march=native` if the user wants binaries portable between machines.

---

## 4. The simulation model (exact specification)

### 4.1 Inputs (command-line flags)

| Flag | Meaning | Default |
|---|---|---|
| `--initial` | Initial investment amount V0 | 10000 |
| `--mu` | Expected **annual** return (e.g. 0.07 = 7 %) | 0.07 |
| `--sigma` | **Annual** volatility (e.g. 0.15 = 15 %) | 0.15 |
| `--months` | Horizon T in months | 120 |
| `--trials` | Number of simulated futures N | 1000000 |
| `--confidence` | VaR confidence level | 0.95 |
| `--seed` | Global RNG seed | 42 |
| `--threads` | OpenMP threads per process (hybrid only) | 1 |
| `--bins` | Histogram bins per month | 2048 |
| `--out` | Output directory | `results/run` |
| `--exact` | Also keep all final values (validation, small N only) | off |

Validation: `initial>0`, `months≥1`, `trials≥1`, `sigma≥0`, `0<confidence<1`, `threads≥1`. Print a clear error and exit non-zero on bad input.

### 4.2 One trial

Convert annual → monthly (simple, documented convention):

```
mu_m    = mu / 12
sigma_m = sigma / sqrt(12)
```

For month t = 1..T:

```
Z   ~ Normal(0, 1)
r_t = max(-1.0, mu_m + sigma_m * Z)     // a loss can't exceed 100 %
V_t = V_{t-1} * (1 + r_t)               // V_0 = initial
```

Keep the model this simple on purpose: it matches the proposal ("randomly generated monthly return based on average return and volatility").

### 4.3 Metrics

From all N trials:

- **Per month t:** median (P50), P05, P95, and mean of `V_t`. (P05/P95 are the "worst/best-case bands".)
- **Final value `V_T`:** mean, std-dev, min, max.
- **VaR at confidence c** (default 95 %): `VaR = max(0, V0 − P_{(1−c)}(V_T))`. Meaning: "with 95 % confidence you will not lose more than VaR over the horizon." Also report `prob_loss = P(V_T < V0)`.
- Stretch (only if time): CVaR (average loss in the worst 5 %).

### 4.4 Analytic sanity checks (used in tests)

Because monthly returns are independent (ignoring the −100 % clamp, negligible for normal parameters):

```
E[V_T]   = V0 * (1 + mu_m)^T
E[V_T^2] = V0^2 * ((1 + mu_m)^2 + sigma_m^2)^T
```

The simulated mean and variance must match these within Monte Carlo error (≈ 4 standard errors).

---

## 5. Key design decisions (read before coding)

### D1. Never store every path — use per-month histograms
1 billion trials × 120 months is far too much memory. Instead, each month has a **fixed-size histogram** (e.g. 2048 bins) of values. Each trial just does `hist[t][bin(V_t)]++`. Histograms from threads and processes are merged by **adding counts**, which is exactly what OpenMP/MPI reductions do. Percentiles are then read from the cumulative counts.
- Memory: `T × bins × 8 bytes` = 120 × 2048 × 8 ≈ 2 MB per thread. Independent of N.
- Accuracy: error ≤ one bin width (well under 1 % with the ranges below).
- Bin ranges are chosen **per month, in log space**, centred on the analytic mean of `ln V_t` with ±6 standard deviations, so resolution stays good as the fan widens. Values outside are clamped into the end bins and counted (`overflow_low`, `overflow_high`) and reported in `summary.json`.
- `--exact` mode (small N) keeps all final values to compare against histogram percentiles.

### D2. Results must not depend on the number of processes/threads
The workload is cut into fixed **chunks of 4096 trials**. Chunk `c` has its **own RNG seed** = hash(global seed, c). It does not matter *which* process/thread runs chunk `c`; it produces identical numbers. Therefore:
- Integer histogram counts are **bit-identical** for any (P, T), and identical to the serial run.
- This gives a powerful test: `mc_serial` vs `mc_hybrid -np 4 --threads 4` must match exactly (floating sums to ~1e-9 relative).

### D3. Own RNG, not `std::` distributions
`std::normal_distribution` output differs between compilers/standard libraries. Use **xoshiro256\*\*** (fast, good quality) seeded through **splitmix64**, and **Box–Muller** for normals (use both values of each pair). This also avoids sharing RNG state between threads (no locks, no false sharing).

### D4. Work split
- **MPI:** `num_chunks = ceil(N / 4096)`. Rank `r` gets a contiguous block of chunks: `base = nc / P`, first `nc % P` ranks get one extra. Handles N not divisible by P.
- **OpenMP:** inside a rank, `#pragma omp parallel` → each thread owns a **private accumulator** (histograms + moment sums) → `#pragma omp for schedule(static)` over the rank's chunks → after the loop, threads merge into the rank accumulator inside `#pragma omp critical` (done once per thread, cheap).
- **MPI reduction:** `MPI_Reduce` with `MPI_SUM` on the histogram (`uint64`) and sums (`double`); `MPI_MIN`/`MPI_MAX` for min/max. Only rank 0 computes analytics and writes files.
- Initialise MPI with `MPI_Init_thread(..., MPI_THREAD_FUNNELED)` (only the main thread calls MPI).

### D5. Fair benchmarking
- **Baseline = `mc_serial`**: a separate binary built **without** MPI and OpenMP, sharing only `src/core`. Speedup is `T_serial / T_parallel`.
- Time with `MPI_Wtime()` after `MPI_Barrier`; report the **max over ranks**; also record `t_compute` and `t_reduce` separately (communication overhead).
- Run each configuration **3 times**, report the **median**.
- Strong scaling: fixed N, vary (P, T). Weak scaling: N grows with P·T.
- Compute: `speedup = T1/Tp`, `efficiency = speedup/(P·T)`, optional Karp–Flatt serial fraction.
- Honest reporting: on one laptop all MPI ranks share memory and cores, and hyper-threads don't give full cores — expect speedup to flatten near the number of *physical* cores. Explain this in the report instead of hiding it.

---

## 6. Architecture

```
            ┌──────────────────────────────────────────────┐
 CLI flags →│ config  (parse + validate)                   │
            └──────────────┬───────────────────────────────┘
                           │
        ┌──────────────────▼───────────────────┐
        │ mc_hybrid (MPI)  │  mc_serial        │
        │ rank r → chunks  │  all chunks       │
        │   OpenMP threads │                   │
        │     simulate_chunk(c) ← core engine  │   core = rng + simulation + accumulator
        └──────────────────┬───────────────────┘
                           │ thread merge → MPI_Reduce
                  ┌────────▼─────────┐
                  │ analytics        │ percentiles, mean, VaR
                  └────────┬─────────┘
                  ┌────────▼─────────┐
                  │ io               │ fan_chart.csv, summary.json, timing.json
                  └────────┬─────────┘
          ┌────────────────┴─────────────────┐
   scripts/plot_*.py (Matplotlib)     dashboard/app.py (Streamlit)
          └── benchmark.csv ← scripts/run_benchmark.sh ──► plot_benchmark.py
```

Layering rule: `src/core/` must contain **no MPI and no OpenMP** code. Parallelism lives only in `src/hybrid/`. This keeps the serial baseline honest and the core easy to test.

---

## 7. Repository structure (create exactly this)

```
parallel-monte-carlo-simulator/
├── CLAUDE.md                     # this file
├── README.md                     # project documentation (maintained every phase)
├── LICENSE                       # MIT
├── Makefile
├── requirements.txt              # numpy, pandas, matplotlib, streamlit, pytest
├── .gitignore                    # build/, results/*, !results/sample/, __pycache__/, .venv/
├── .github/workflows/ci.yml      # optional: install OpenMPI, make, make test
├── src/
│   ├── core/
│   │   ├── config.hpp / .cpp         # Config struct, CLI parsing, validation
│   │   ├── rng.hpp                   # splitmix64, xoshiro256**, Box–Muller
│   │   ├── accumulator.hpp / .cpp    # histograms + moment sums, merge()
│   │   ├── simulation.hpp / .cpp     # bin layout, simulate_chunk()
│   │   ├── analytics.hpp / .cpp      # percentiles from histograms, VaR
│   │   └── io.hpp / .cpp             # CSV / JSON writers
│   ├── serial/main_serial.cpp        # baseline driver
│   └── hybrid/main_hybrid.cpp        # MPI + OpenMP driver
├── tests/
│   ├── cpp/
│   │   ├── test_util.hpp
│   │   ├── test_rng.cpp              # determinism, mean≈0, std≈1
│   │   ├── test_accumulator.cpp      # merge correctness
│   │   ├── test_analytics.cpp        # percentiles/VaR on known histograms
│   │   └── test_analytic_moments.cpp # E[V_T], E[V_T^2] vs formulas
│   ├── python/
│   │   ├── reference_numpy.py        # independent NumPy implementation
│   │   ├── test_reference.py         # C++ vs NumPy statistical agreement
│   │   └── test_dashboard.py         # Streamlit AppTest: a run completes in the UI
│   ├── consistency_test.sh           # serial == hybrid(P,T) for several (P,T)
│   └── run_tests.sh
├── scripts/
│   ├── setup_env.sh                  # checks/installs toolchain on Ubuntu/WSL2
│   ├── run_demo.sh                   # one default run + plots
│   ├── run_benchmark.sh              # full (P,T) matrix, 3 repeats → benchmark.csv
│   ├── chart_style.py                # shared colours/fonts for all charts
│   ├── plot_fan_chart.py
│   ├── plot_distribution.py          # final-value histogram + VaR line
│   └── plot_benchmark.py             # time, speedup, efficiency (+ weak scaling)
├── dashboard/app.py                  # Streamlit UI
├── docs/
│   ├── proposal/proposal.pdf         # copy of the submitted proposal
│   ├── architecture.md
│   ├── design-decisions.md           # D1–D5 in student-friendly language
│   ├── user-guide.md
│   ├── benchmark-report.md           # filled with real numbers in Phase 6
│   └── images/                       # charts used by README
└── results/
    └── sample/                       # one small committed example run
```

---

## 8. Output formats (stable contract between C++ and Python)

`results/<run>/fan_chart.csv`
```
month,p05,p25,median,p75,p95,mean
0,10000,10000,10000,10000,10000,10000
1,...
```

`results/<run>/summary.json`
```json
{
  "params": {"initial":10000,"mu":0.07,"sigma":0.15,"months":120,"trials":1000000,"confidence":0.95,"seed":42},
  "parallel": {"mpi_processes":4,"omp_threads":2},
  "final_value": {"mean":0,"std":0,"min":0,"max":0,"median":0,"p05":0,"p95":0},
  "risk": {"var":0,"prob_loss":0,"confidence":0.95},
  "histogram": {"bins":2048,"overflow_low":0,"overflow_high":0,"checksum":"0123456789abcdef"},
  "timing": {"total_s":0,"compute_s":0,"reduce_s":0}
}
```

`checksum` (added in Phase 2) is a 16-hex-digit FNV-1a hash of all integer histogram counts; equal checksums prove serial == parallel.

`results/<run>/final_hist.csv` (for the distribution plot): `bin_lower,bin_upper,count` for month T.

With `--exact` (≤ 10⁷ trials), `summary.json` also gets `"exact": {"median","p05","p95","var","rel_diff_median","rel_diff_p05","rel_diff_p95","rel_diff_var"}` (exact values from all final values, and the histogram's relative difference from them).

`results/benchmark.csv`
```
mode,processes,threads,total_workers,trials,run,total_s,compute_s,reduce_s
```
(`mode` is `serial` or `hybrid`.) `plot_benchmark.py` derives median, speedup, efficiency.

---

## 9. Development phases (do in order; stop after each)

### Phase 0 — Repository & environment
Tasks: create the folder tree from section 7 (empty placeholder files are fine), `.gitignore`, `LICENSE`, `requirements.txt`, `Makefile` skeleton, `scripts/setup_env.sh`; copy the proposal PDF to `docs/proposal/`; verify toolchain (`g++ --version`, `mpicxx --version`, `mpirun --version`, `python3 -c "import matplotlib"`); write a 10-line "hello hybrid" program printing rank/thread IDs to prove MPI + OpenMP work, then delete it (or keep under `docs/`).
Done when: `mpirun -np 2 ./hello` with `OMP_NUM_THREADS=2` prints 4 distinct (rank, thread) lines.

### Phase 1 — Serial engine (Objective O1)
Tasks: `config`, `rng`, `accumulator`, `simulation`, `analytics`, `io`, `main_serial`. Implement D1–D3. Tests: `test_rng`, `test_accumulator`, `test_analytics`, `test_analytic_moments`.
Done when: `./build/mc_serial --trials 1000000` writes `fan_chart.csv` and `summary.json`; the simulated mean matches `V0·(1+mu_m)^T` within 4 standard errors; all unit tests pass.

### Phase 2 — OpenMP (first half of O2)
Tasks: `main_hybrid.cpp` with MPI temporarily compiled out or `P=1`; private accumulators per thread, merge in `critical`.
Done when: `--threads 1,2,4` give **identical** histogram counts to the serial run (`tests/consistency_test.sh`), and wall time drops with threads.

### Phase 3 — MPI (second half of O2)
Tasks: `MPI_Init_thread`, chunk-block split (D4), `MPI_Reduce`, rank-0-only output, timing breakdown.
Done when: `mpirun -np {1,2,3,4} ./build/mc_hybrid --threads {1,2,4}` all equal serial exactly (including a case where N is not divisible by P); `-np 3` proves uneven split works.

### Phase 4 — Analytics & outputs (Objective O3)
Tasks: finish `analytics` (P05/P50/P95 per month, VaR, prob_loss, final stats), `--exact` validation mode, `final_hist.csv`.
Done when: histogram VaR and exact VaR (N ≤ 5·10⁶) differ by < 1 %; `tests/python/test_reference.py` shows C++ and NumPy medians/P5/P95 agree within statistical tolerance.

### Phase 5 — Visualization & dashboard (Objective O5)
Tasks: `plot_fan_chart.py` (shaded P5–P95 band, median line, a few labelled sample points), `plot_distribution.py` (final-value histogram + VaR line + initial-investment line), `dashboard/app.py` (sliders for initial/mu/sigma/months/trials/processes/threads; "Run" button calls `mpirun` via `subprocess`; shows fan chart, VaR box, timing).
Done when: PNGs saved to `docs/images/`; dashboard launches with `streamlit run dashboard/app.py` and a run completes in the UI.

### Phase 6 — Benchmark & scalability (Objective O4)
Tasks: `run_benchmark.sh` (strong-scaling matrix, 3 repeats, serial baseline first, thread pinning via `OMP_PLACES=cores OMP_PROC_BIND=close`, `--oversubscribe` only if needed and flagged in CSV), weak-scaling run, `plot_benchmark.py` (time, speedup vs ideal line, efficiency heat-map), `docs/benchmark-report.md` with real tables and an honest discussion (Amdahl's law, hyper-threading, shared laptop memory, reduction cost).
Calibrate N so the serial run takes ~20–60 s. Only use 10⁹ trials for one optional "headline" run.
Done when: `results/benchmark.csv` and 3 plots exist and the report explains the observed curve.

### Phase 7 — Testing & hardening
Tasks: edge cases (`trials=1`, `trials<P·T`, `sigma=0` → deterministic growth, `months=1`, large `sigma` causing clamps), input-validation tests, `make test` runs everything, optional GitHub Actions workflow.
Done when: `make test` is green from a clean clone.

### Phase 8 — Documentation & final report
Tasks: finalise README (real results + images), `docs/architecture.md`, `docs/design-decisions.md`, `docs/user-guide.md`, demo script, a short "Objectives achieved" table with evidence links, 5-minute demo outline for the viva.
Done when: a classmate can clone the repo and reproduce a run and the benchmark by following only the README.

---

## 10. Testing strategy (summary)

| Level | What | How |
|---|---|---|
| Unit | RNG determinism & distribution; accumulator merge; percentile/VaR maths | `tests/cpp/*` |
| Analytic | Mean and second moment vs closed-form | `test_analytic_moments.cpp` |
| Consistency | serial == hybrid for many (P, T), incl. uneven N | `tests/consistency_test.sh` |
| Independent reference | C++ vs NumPy implementation (statistical) | `tests/python/test_reference.py` |
| Edge cases | tiny N, sigma=0, bad input | `tests/cpp` + shell |
| Performance | Speedup/efficiency sanity (not a pass/fail gate) | `scripts/run_benchmark.sh` |

---

## 11. Environment setup (Ubuntu / WSL2)

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3-venv python3-pip git
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# verify
g++ --version && mpicxx --version && mpirun --version
nproc                       # logical cores available
```

OpenMPI notes: running more ranks than cores needs `mpirun --oversubscribe`. To avoid MPI pinning ranks to one core when also using threads, use `mpirun --bind-to none -np P ...` and set `OMP_PLACES=cores OMP_PROC_BIND=close`.

---

## 12. Coding conventions

- C++17, 4-space indent, `snake_case` functions/variables, `PascalCase` types, `UPPER_CASE` constants.
- Every header has `#pragma once`. Every public function has a one-line comment saying what it does and any units.
- No global mutable state. No `using namespace std;` in headers.
- Use `uint64_t` for counts and trial indices (N can reach 10⁹).
- Compile clean with `-Wall -Wextra`. Treat warnings as bugs.
- Python: PEP 8, type hints on functions, scripts take `--input` / `--output` flags, never hard-code paths.
- Commits (when the user asks): imperative present tense, e.g. `Add OpenMP chunk loop with private accumulators`.

---

## 13. Risks & how we handle them

| Risk | Mitigation |
|---|---|
| Storing all paths exhausts memory | Histogram design (D1) |
| Results change with core count (can't validate) | Per-chunk seeding (D2) |
| Speedup disappoints on a laptop | Honest benchmark report; vary N; separate compute vs reduce time; explain hyper-threading/memory bandwidth |
| Histogram percentile error | Per-month log-space bins, `--exact` cross-check, overflow counters |
| MPI not installed / WSL2 quirks | `scripts/setup_env.sh`, section 11 notes |
| Scope creep | Out of scope unless user asks: monthly contributions, real market data, GPU, fat-tail models, web hosting |

---

## 14. Definition of done (whole project)

- [ ] O1–O5 each have a working demo and a test or artifact listed in section 2.
- [ ] `make && make test` passes on a clean clone.
- [ ] README has real benchmark results, charts, and up-to-date roadmap.
- [ ] `docs/benchmark-report.md` explains the speedup curve honestly.
- [ ] Proposal PDF stored in `docs/proposal/`.
- [ ] Team can run the 5-minute demo from the README alone.

---

## 15. Start here

When the user says "start", begin **Phase 0** immediately. Do not re-ask the questions already answered in section 16.

---

## 16. Confirmed environment and decisions (already answered by the user)

- **Machine:** Ubuntu 24.04 LTS inside a **VirtualBox VM**. Linux user `maryam`. The user works from **VS Code on Windows via Remote-SSH** (host alias `pdc-vm`), so every command you run executes inside the VM.
- **Toolchain already installed:** `build-essential`, `openmpi-bin`, `libopenmpi-dev`, `python3-venv`, `python3-pip`, `git`, `curl`. Verify with `g++ --version`, `mpicxx --version`, `mpirun --version` but do not reinstall unless missing.
- **Core count:** run `nproc` yourself and record it in `docs/benchmark-report.md`. This is a VM, so the speedup ceiling is the number of cores given to the VM, not the host's. If `nproc` is 1 or 2, tell the user to raise the VM's CPU count in VirtualBox (Settings > System > Processor, with the VM powered off) before Phase 6.
- **Output choice:** Matplotlib images **and** the Streamlit dashboard (both wanted).
- **Dashboard access:** the dashboard runs in the VM. VS Code forwards the port automatically, so the user opens `http://localhost:8501` in the Windows browser (see the Ports tab).
- **Repository:** this folder is a clone of the user's GitHub repo `parallel-monte-carlo-simulator`. Do not run `git commit` or `git push` unless asked; suggest commit messages at the end of each phase.
- **Python:** use a virtual environment at `.venv` and `pip install -r requirements.txt`. The VM has internet access.
- **OpenMPI in a VM:** use `mpirun --bind-to none` and add `--oversubscribe` only when the number of ranks exceeds the cores. If OpenMPI warns about running as root or about the network interface, tell the user rather than silently disabling checks.
- **Files already present:** `CLAUDE.md`, `README.md`, and the proposal PDF in `docs/proposal/` (if the PDF is missing, mention it and continue).
- **Language of explanations:** plain English, for a BS Software Engineering student new to parallel computing.
