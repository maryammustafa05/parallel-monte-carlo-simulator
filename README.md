# Parallel Monte Carlo Investment & Risk Simulator (MPI + OpenMP)

A hybrid-parallel C++ engine that simulates millions of possible futures of an
investment and reports the median path, the 5th–95th percentile "fan", and the
95 % Value at Risk (VaR). **MPI** splits the trials across processes, and
**OpenMP** splits each process's share across threads.

Course project for Parallel & Distributed Computing (PDC), BS Software Engineering,
FAST NUCES Karachi, Fall 2026.
Team: Maryam Mustafa (23k-2046), Umaima Fatima (23k-3043).

> **Status:** Phase 0 done (repository and environment). The simulator itself is not written yet.

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

## Build targets

| Command | What it does |
|---|---|
| `make` | Builds everything available (today: `build/hello`) |
| `make serial` | Serial baseline `mc_serial` (Phase 1) |
| `make hybrid` | MPI + OpenMP engine `mc_hybrid` (Phases 2–3) |
| `make test` | Runs all tests |
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
- [ ] **Phase 1:** serial Monte Carlo engine (O1)
- [ ] **Phase 2:** OpenMP threads (O2, part 1)
- [ ] **Phase 3:** MPI processes (O2, part 2)
- [ ] **Phase 4:** analytics: percentiles, VaR, exact-mode validation (O3)
- [ ] **Phase 5:** Matplotlib charts and Streamlit dashboard (O5)
- [ ] **Phase 6:** benchmark and scalability report (O4)
- [ ] **Phase 7:** edge-case tests and hardening
- [ ] **Phase 8:** final documentation and demo

## License

MIT. See [LICENSE](LICENSE).
