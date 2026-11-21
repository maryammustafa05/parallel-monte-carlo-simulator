// mc_hybrid: the parallel driver.
//
// Phase 2 (this version): OpenMP threads inside ONE process.
// Phase 3 adds MPI on top: each process will run this same thread code on its
// own block of chunks, and the processes' results are then reduced to rank 0.
//
// How the threads cooperate (design decision D4):
//   1. `#pragma omp parallel` starts a team of threads.
//   2. Each thread creates its OWN private Accumulator, so threads never write
//      to the same memory while simulating (no locks, no race conditions).
//   3. `#pragma omp for schedule(static)` hands each thread a fixed share of the
//      chunks. Each chunk has its own seed, so which thread runs it doesn't
//      change its numbers (design decision D2).
//   4. After the loop, each thread adds its accumulator into the shared total
//      inside `#pragma omp critical` (one thread at a time). This happens once
//      per thread, so it costs almost nothing.

#include <omp.h>

#include <chrono>
#include <iostream>
#include <string>

#include "core/accumulator.hpp"
#include "core/config.hpp"
#include "core/io.hpp"
#include "core/simulation.hpp"

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;

    mc::Config cfg;
    std::string error;
    switch (mc::parse_args(argc, argv, cfg, error)) {
        case mc::ParseResult::Help:
            std::cout << mc::usage(argv[0]);
            return 0;
        case mc::ParseResult::Error:
            std::cerr << "error: " << error << "\n(run with --help to see all options)\n";
            return 1;
        case mc::ParseResult::Ok:
            break;
    }
    if (cfg.exact) {
        std::cerr << "note: --exact is not implemented yet (Phase 4); ignored\n";
    }

    const auto start = clock::now();

    const mc::BinLayout layout = mc::make_bin_layout(cfg);
    mc::Accumulator total(cfg.months, cfg.bins);
    const uint64_t chunks = mc::num_chunks(cfg.trials);
    int threads_used = 1;

    #pragma omp parallel num_threads(cfg.threads)
    {
        // Private to this thread. Created inside the parallel region so its
        // memory is allocated (and first touched) by the thread that uses it.
        mc::Accumulator local(cfg.months, cfg.bins);

        #pragma omp for schedule(static)
        for (uint64_t c = 0; c < chunks; ++c) {
            mc::simulate_chunk(cfg, layout, c, local);
        }
        // (implicit barrier here: every thread has finished its chunks)

        #pragma omp critical
        total.merge(local);

        #pragma omp single
        threads_used = omp_get_num_threads();  // the OS/runtime may give fewer than asked
    }

    mc::Timing timing;
    timing.compute_s = std::chrono::duration<double>(clock::now() - start).count();
    timing.reduce_s = 0.0;  // no MPI yet (Phase 3)
    timing.total_s = timing.compute_s + timing.reduce_s;

    mc::ParallelInfo parallel;
    parallel.mpi_processes = 1;
    parallel.omp_threads = threads_used;
    if (!mc::finish_run("mc_hybrid", cfg, layout, total, parallel, timing, error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }
    return 0;
}
