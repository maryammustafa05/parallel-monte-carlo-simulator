# 5-minute demo outline (viva)

A timed plan for presenting the project live, with the commands to type and what to say.
Every number below comes from real runs on the development VM. Your timings will vary a
little; the results and checksums will not.

## Before the viva (10 minutes, not part of the 5)

- [ ] Laptop **plugged in**, Windows power mode **Best performance**, other apps closed.
- [ ] VM running; in the terminal: `cd ~/parallel-monte-carlo-simulator && export HWLOC_COMPONENTS=-gl`
- [ ] `make && make test` → `ALL TESTS PASSED` (proves the build is fresh).
- [ ] Start the dashboard in a **second** terminal: `.venv/bin/streamlit run dashboard/app.py`,
      open http://localhost:8501, and press **Run simulation** once so it's warmed up.
- [ ] Open in tabs: `docs/images/fan_chart.png`, `docs/images/benchmark_speedup.png`,
      `docs/benchmark-report.md`.
- [ ] Terminal font large enough for the room.

## The 5 minutes

### 0:00–0:45 · The problem (talk, no commands)

> "An investor doesn't know future returns. A Monte Carlo simulation answers 'what could
> happen?' by simulating a million possible futures, month by month, with random returns,
> and studying the spread: the typical outcome, the bad and good cases, and the Value at
> Risk. One core is slow, so we split the futures across **MPI processes** and each
> process's share across **OpenMP threads**."

Show the **fan chart** PNG: the median path, the 50 % and 90 % bands, the starting line.

### 0:45–2:00 · It runs, in parallel, and gives the same answer

```bash
./build/mc_serial --trials 300000 --out results/viva_serial
mpirun --bind-to none -np 4 ./build/mc_hybrid --trials 300000 --threads 1 --out results/viva_p4
mpirun --bind-to none -np 2 ./build/mc_hybrid --trials 300000 --threads 2 --out results/viva_p2t2
```

Point at the **checksum** line: it is identical in all three.

> "The checksum is a fingerprint of every histogram count. Serial, 4 processes, and
> 2 processes × 2 threads give bit-identical results. That's by design: the trials are cut
> into chunks of 4096, and each chunk has its own random seed, so it doesn't matter who runs
> it. That's what makes the parallel code testable."

Read one result aloud: *"Median after 10 years about 18,000; with 95 % confidence you won't
lose more than about 1,750 of the 10,000; about 11 % of futures end with a loss."*
(Real run: median 17,953, VaR 1,758, P(loss) 10.8 %, checksum `7133731103a03a3c` for all three.)

### 2:00–3:00 · The dashboard

Switch to the browser. Change **volatility** from 15 % to 30 %, press **Run simulation**.

> "Higher volatility gives a wider fan, a higher VaR and a higher chance of loss, but the
> median *falls*. That's volatility drag: losses hurt more than equal gains help."

(Real run at 30 %: median 12,853 against 17,983 at 15 %, VaR 7,342, P(loss) 39.6 %; the
mean stays about 20,100.)

Open the *Final-value distribution* tab (orange tail = worst 5 %) and the *Run details*
tab (the `mpirun` command it ran, compute vs MPI-reduce time).

### 3:00–4:15 · Performance, honestly

Show `benchmark_speedup.png`.

> "3 million trials: serial 19.5 s, best parallel 7.6 s on 5 processes, a **2.56× speedup**.
> Not 5×, and we can explain why with measurements. Communication isn't the problem: the
> MPI reduce takes 2–8 ms, under 0.1 %. Amdahl's law with our serial part predicts 4.99×.
> The weak-scaling test shows each core gets *slower* when the others are busy. We traced
> much of that to the shared 8 MB CPU cache: each worker's 2 MB of histograms stop fitting.
> With smaller histograms, weak-scaling efficiency went from 74 % to 92 %. The rest is a 15 W
> laptop chip with only 4 physical cores."

### 4:15–5:00 · How we know it's right, and wrap-up

> "Three independent checks. One: the simulated mean matches the exact formula
> V0·(1 + μ/12)^T. Two: `--exact` mode sorts all final values, and the histogram
> percentiles are within 0.005 %. Three: a completely separate NumPy implementation with a
> different random generator agrees on 19 statistics within 4 standard errors. Plus 16
> serial-vs-parallel layouts and edge cases like zero volatility and wiped-out portfolios.
> `make test` runs all of it in about a minute."

Close: *"Objectives O1–O5 are each met, with evidence linked in the README."*

## Likely questions and short answers

| Question | Answer |
|---|---|
| Why not store every path? | 10⁹ trials × 120 months ≈ 960 GB. Histograms need 2 MB per worker, whatever N is, and merge by simple addition, which is exactly what `MPI_Reduce` with `MPI_SUM` does. |
| How accurate are the histogram percentiles? | Measured against exact sorting: ≤ 0.005 % at 5 M trials (`--exact`). Bins are log-spaced and centred per month, ±6 standard deviations. |
| Why do processes and threads give the same speed here? | On one machine they share the same cores, cache and memory. MPI's real advantage, more machines, can't show on one laptop. |
| What's in `#pragma omp critical`, and isn't it a bottleneck? | Only the final merge of each thread's private histograms, once per thread. The simulation loop itself has no locks. |
| Why `MPI_THREAD_FUNNELED`? | Threads exist, but only the main thread calls MPI, after the parallel region. |
| How do you handle N not divisible by P? | Chunks are split into contiguous blocks; the first `nc % P` processes get one extra chunk (13 chunks on 3 processes → 5, 4, 4). Tested, including processes with no chunks. |
| Why your own RNG? | `std::normal_distribution` can differ between compilers. xoshiro256** + splitmix64 + Box–Muller are reproducible everywhere, fast, and each chunk has its own generator, so no shared state between threads. |
| Is the speedup measurement fair? | Baseline is a separate binary without MPI/OpenMP; 3 interleaved repeats, median; pinned cores; slowest process counts. |
| What is Karp–Flatt telling you? | Our measured "serial fraction" grows with workers (0.19 → 0.24–0.41). A fixed serial part would give a constant, so the loss is hardware contention, not serial code. |
| What would you improve? | Smaller histogram memory (fewer bins or 32-bit per-thread counts) to fit in cache; run on a real multi-node cluster; richer models (fat tails, contributions). |
| What happens with extreme inputs? | Validated ranges and an overflow check; wiped-out portfolios are counted separately, so the median correctly becomes 0. Each case has a test. |

## If something goes wrong during the demo

- **Dashboard won't load** → VS Code **Ports** tab → *Forward a Port* → `8501`.
- **"not enough slots"** → use a smaller `-np` (the VM has 5 cores).
- **A run is unusually slow** → say so honestly: the VM shares the laptop with Windows; the
  benchmark used medians of 3 interleaved runs for exactly this reason.
- **Anything else** → show the pre-made PNGs and `results/sample/` instead of live runs.
