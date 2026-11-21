# Design decisions

The important choices behind the simulator, in plain language. Each one says what problem
it solves, what we chose, what else we could have done, and how we know it works. D1–D5
were planned before coding; D6–D10 came up while building and testing.

---

## D1. Never store every path: use per-month histograms

**Problem.** One simulated future is 120 monthly values. A billion futures would be 120
billion numbers, about 960 GB. No laptop has that much memory, and the parallel processes
would have to send all of it to each other.

**Decision.** For each month we keep a **histogram**: a fixed row of 2048 counters, each
covering a small range of values. A trial that reaches value V in month t just adds 1 to
the right counter. Percentiles (median, P5, P95) are read off the running totals of the
counters, interpolating inside a bin.

- **Memory:** 120 months × 2048 bins × 8 bytes ≈ **2 MB per worker**, whether we run a
  thousand trials or a billion.
- **Merging is just adding.** Two histograms combine by adding counter by counter. That is
  exactly what an OpenMP merge or `MPI_Reduce(..., MPI_SUM)` does, so the parallel design
  becomes simple.
- **Bins are spaced in log space, per month.** Investment values spread out multiplicatively
  (a 10 % gain on 20,000 is twice a 10 % gain on 10,000), so equal steps in ln(V) give
  even resolution. Each month's range is centred on the expected ln(V_t) ± 6 standard
  deviations, so it widens with the fan.
- **Out-of-range values** are put in the end bins and counted (`overflow_low`,
  `overflow_high` in `summary.json`), so we would notice if the range were too narrow.

**Alternatives rejected.** Storing all final values and sorting them is exact, but needs
memory proportional to N and an expensive gather between processes. We kept it only as a
validation mode (`--exact`, D6).

**Evidence.** Histogram percentiles and VaR differ from the exact ones by at most
**0.005 %** at 5 million trials (the target was < 1 %). See [README](../README.md#correctness).

---

## D2. Results must not depend on how many processes or threads run

**Problem.** If 4 workers gave a slightly different answer from 1 worker, we couldn't
tell a parallel bug from normal randomness, and we couldn't test the parallel code at all.

**Decision.** The trials are cut into fixed **chunks of 4096**. Chunk *c* gets its own random
seed, computed only from the global seed and *c*. So chunk 7 always produces exactly the same
numbers, whether process 0, process 3, or thread 2 runs it. The histograms are integer
counts, and integer addition gives the same result in any order. So:

> `mc_serial` and `mpirun -np P mc_hybrid --threads T` produce **bit-identical** histogram
> counts for every P and T.

Floating-point sums (used for the mean) are added in a different order, so they can differ
in the last few digits (about 10⁻¹⁵ relative); the tests allow 10⁻⁹.

**How we prove it.** Every run prints a **checksum** (FNV-1a hash) of all integer counts.
The test suite compares serial with 16 different layouts, including uneven splits and
processes that get no work at all. Every checksum matches.

---

## D3. Our own random number generator

**Problem.** `std::normal_distribution` is allowed to be implemented differently by each
compiler's standard library, so the same seed can give different numbers on different
machines. A shared generator would also need locks between threads.

**Decision.** Three small, well-known algorithms in [rng.hpp](../src/core/rng.hpp):

- **splitmix64** turns a seed (and a chunk number) into a well-mixed starting state;
- **xoshiro256\*\*** generates fast, high-quality uniform random numbers;
- **Box–Muller** turns two uniforms into two Normal(0, 1) values. We use *both* values
  instead of throwing one away.

Each chunk creates its own generator, so threads never share random state: no locks and no
cache-line fights.

**Evidence.** Unit tests check determinism and that one million samples have mean ≈ 0 and
variance ≈ 1 within 4 standard errors. The independent NumPy implementation, which uses a
completely different generator, agrees with every statistic (D7).

---

## D4. How the work is split

**MPI level (processes).** With `nc` chunks and `P` processes, each process gets a
**contiguous block** of `nc / P` chunks. The first `nc % P` processes take one extra, so
13 chunks on 3 processes become 5 + 4 + 4. This handles any N and any P, including more
processes than chunks (some then get none).

**OpenMP level (threads).** Inside a process:

```cpp
#pragma omp parallel                 // start T threads
{
    Accumulator local(...);          // each thread has its OWN histograms: no sharing
    #pragma omp for schedule(static) // split this process's chunks among the threads
    for (chunk ...) simulate_chunk(..., local);
    #pragma omp critical             // one thread at a time...
    total.merge(local);              // ...adds its results (once per thread: cheap)
}
```

Private accumulators mean threads never write to the same memory while simulating. There
are no locks in the hot loop and no race conditions. The only synchronised step is the
final merge, which happens once per thread.

**Combining processes.** `MPI_Reduce` with `MPI_SUM` adds all histograms and sums onto
process 0, and `MPI_MIN` / `MPI_MAX` handle the smallest and largest final value. Only
process 0 analyses and writes files. MPI is started with `MPI_THREAD_FUNNELED`: threads
exist, but only the main thread calls MPI.

**Why `schedule(static)`.** Every chunk costs the same (4096 trials × T months), so a
fixed split is balanced and has no scheduling overhead. `dynamic` would only add cost.

---

## D5. Fair benchmarking

- **The baseline is a separate program.** `mc_serial` is built **without** MPI and OpenMP
  and shares only the core engine. Speedup = T_serial / T_parallel against this honest
  baseline, not against "the parallel program on 1 core".
- **Timing** uses `MPI_Wtime()` after an `MPI_Barrier`, and reports the **slowest**
  process, because a parallel run is finished only when its last worker is. Compute and
  reduce (communication) are timed separately.
- **3 repeats, median**, **interleaved in rounds**, so a slow spell on the host affects all
  configurations equally (D10).
- **Pinning.** Each process gets its own block of cores
  (`mpirun --map-by slot:PE=T --bind-to core`) and its threads are pinned inside it
  (`OMP_PLACES=cores OMP_PROC_BIND=close`), so no two workers share a core.
- **Honest reporting.** The report explains why the speedup is 2.56× and not 5×, using
  measurements (weak scaling, a cache-size experiment), not excuses. See
  [benchmark-report.md](benchmark-report.md).

---

## D6. An `--exact` mode for validation

Histograms are an approximation, so we need a way to check them. With `--exact`, every
final value is also kept: per thread, merged per process, and collected on process 0 with
`MPI_Gatherv`. They are then sorted to compute the true percentiles and VaR, and
`summary.json` reports how far the histogram answers are from them. It's limited to 10⁷
trials (80 MB), and the gathering happens **after** the timer stops, so validation never
distorts benchmarks.

## D7. An independent reference implementation in NumPy

A bug in the *model* (say, a wrong annual-to-monthly conversion) would fool the C++ unit
tests, because they were written by the same people with the same understanding.
[reference_numpy.py](../tests/python/reference_numpy.py) implements the model again from
scratch, in another language, with another random generator. The two must agree within
4 standard errors on 19 statistics. If the reference used an 8 % return instead of 7 %, the
median would differ by about 50 standard errors, so the test really is sensitive.

## D8. Core engine without parallel code, compiled once

`src/core/` contains **no MPI and no OpenMP**. It is compiled once with plain `g++` and the
same object files are linked into `mc_serial`, `mc_hybrid` and the tests. This keeps the
serial baseline honest (D5), makes the core easy to unit-test, and guarantees that every
program runs byte-for-byte the same simulation code, which D2's exact equality relies on.

## D9. Wiped-out portfolios and input limits

A month with a return of −100 % or worse leaves the portfolio at exactly 0, and it stays
there. 0 has no logarithm, so it can't go in a log-spaced histogram. Such trials are
**counted separately per month**, and percentiles treat them as the smallest values. So with
extreme volatility the median correctly becomes 0 (not 2.6·10⁻⁴⁰, as it was before this fix),
and `summary.json` reports `prob_total_loss`.

Inputs are range-checked (for example `--months` ≤ 1200, `--mu` > −1), including a check
that the chosen return, volatility and horizon can't overflow a double. Before Phase 7, some
inputs crashed the program or wrote `inf` into the results. Each limit has a unit test.

## D10. Benchmark repeats are interleaved

The development machine is a VM on a Windows laptop, and its speed changes with the host's
power state: the same serial run took between 6.5 s and 31 s in different sessions.
Running "3 × config A, then 3 × config B" would mix that drift into the comparison.
Instead the benchmark runs **round 1 of every configuration, then round 2, then round 3**,
and reports medians, with whiskers on the charts showing the spread.
