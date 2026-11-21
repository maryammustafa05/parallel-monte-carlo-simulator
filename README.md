# Parallel Monte Carlo Investment & Risk Simulator (MPI + OpenMP)

A hybrid-parallel C++ engine that simulates millions of possible futures of an
investment and reports the median path, the 5th–95th percentile "fan", and the
95 % Value at Risk (VaR). **MPI** splits the trials across processes, and
**OpenMP** splits each process's share across threads.

Course project for Parallel & Distributed Computing (PDC), BS Software Engineering,
FAST NUCES Karachi, Fall 2026.
Team: Maryam Mustafa (23k-2046), Umaima Fatima (23k-3043).

> **Status:** Phase 1 done: the serial (single-core) simulation engine works and is tested.
> MPI and OpenMP parallelism come next (Phases 2–3).

---

## Requirements

Ubuntu 24.04 (native, VM or WSL2) with:

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3-venv python3-pip git
```

Then, from the repository root:

```bash
bash scripts/setup_env.sh        # checks the tools, creates .venv, installs Python packages
source .venv/bin/activate
```

Verified on the development VM: g++ 13.3.0, Open MPI 4.1.6, Python 3.12.3, 5 logical cores.

## Quick check: MPI + OpenMP "hello"

```bash
make                                                    # builds build/hello
OMP_NUM_THREADS=2 mpirun --bind-to none -np 2 ./build/hello
```

Expected output: four lines, one per (rank, thread) pair. The order varies from run to run.

```
hello from rank 0 of 2, thread 0 of 2
hello from rank 0 of 2, thread 1 of 2
hello from rank 1 of 2, thread 0 of 2
hello from rank 1 of 2, thread 1 of 2
```

### Known harmless message: "Authorization required, but no authorization protocol specified"

On a VM with a desktop, Open MPI's hardware-detection library (hwloc) tries to
contact the display server to look for GPUs and prints this line several times.
It does not affect results. To silence it, disable hwloc's graphics plugin:

```bash
export HWLOC_COMPONENTS=-gl      # add to ~/.bashrc to make it permanent
```

## Run the serial simulator

```bash
make                                   # builds build/mc_serial (and the tests)
./build/mc_serial --trials 1000000     # 1 million futures, 120 months, defaults below
./build/mc_serial --help               # all options
```

Real output on the development VM (seed 42, defaults: V0 = 10 000, mu = 7 %/year,
sigma = 15 %/year, 120 months):

```
mc_serial: 1000000 trials x 120 months in 30.919 s (compute 30.913 s)
  final value: mean 20093.63  median 17983.26  P5 8245.17  P95 39049.85
  VaR(95%): 1754.83   P(loss): 0.1075   histogram overflow: low 0, high 0
  wrote results/run/fan_chart.csv and results/run/summary.json
```

In words: after 10 years the typical (median) outcome is about 17 980, the average is
about 20 090, and with 95 % confidence you will not lose more than about 1 755 of
your 10 000. About 10.8 % of the futures end below the starting amount.
The simulated mean (20 093.63) matches the exact formula V0·(1 + mu/12)^120 = 20 096.61
to within 0.3 standard errors.

Outputs (format details in [CLAUDE.md](CLAUDE.md) section 8):

| File | Contents |
|---|---|
| `results/run/fan_chart.csv` | per month: P5, P25, median, P75, P95, mean |
| `results/run/summary.json` | parameters, final-value stats, VaR, overflow counters, timing |

**Note on speed:** the VM runs math functions roughly 5–10× slower than the same CPU
natively would (likely because VirtualBox runs on top of Windows Hyper-V). This changes
absolute times, not correctness, and speedup ratios stay meaningful.

## Build targets

| Command | What it does |
|---|---|
| `make` | Builds everything available: `build/hello`, `build/mc_serial`, test binaries |
| `make serial` | Serial baseline `build/mc_serial` |
| `make hybrid` | MPI + OpenMP engine `mc_hybrid` (Phases 2–3) |
| `make test` | Builds and runs all tests (C++ unit tests + end-to-end checks) |
| `make bench` | Runs the scalability benchmark (Phase 6) |
| `make clean` | Deletes `build/` |

## Repository layout

```
src/core/      simulation engine shared by both programs (no MPI / OpenMP here)
src/serial/    serial baseline driver
src/hybrid/    MPI + OpenMP driver
tests/         C++ unit tests, Python reference check, serial-vs-parallel consistency
scripts/       setup, demo, benchmark and plotting scripts
dashboard/     Streamlit dashboard
docs/          proposal, design notes, benchmark report, images
results/       run outputs (only results/sample/ is committed)
```

See [CLAUDE.md](CLAUDE.md) for the full specification and design decisions.

## Roadmap

- [x] **Phase 0:** repository skeleton, toolchain check, MPI + OpenMP hello test
- [x] **Phase 1:** serial Monte Carlo engine (O1)
- [ ] **Phase 2:** OpenMP threads (O2, part 1)
- [ ] **Phase 3:** MPI processes (O2, part 2)
- [ ] **Phase 4:** analytics: percentiles, VaR, exact-mode validation (O3)
- [ ] **Phase 5:** Matplotlib charts and Streamlit dashboard (O5)
- [ ] **Phase 6:** benchmark and scalability report (O4)
- [ ] **Phase 7:** edge-case tests and hardening
- [ ] **Phase 8:** final documentation and demo

## License

MIT. See [LICENSE](LICENSE).
