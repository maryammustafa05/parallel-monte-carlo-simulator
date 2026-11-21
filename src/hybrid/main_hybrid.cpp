// mc_hybrid: the MPI + OpenMP parallel driver.
//
// Two levels of parallelism (design decision D4):
//
//   Level 1 — MPI processes ("ranks"), started by `mpirun -np P`.
//     The chunks are cut into P contiguous blocks; rank r runs block r.
//     Processes don't share memory, so at the end their results are sent to
//     rank 0 and added together with MPI_Reduce. Only rank 0 writes files.
//
//   Level 2 — OpenMP threads inside each process (`--threads T`).
//     `#pragma omp for` shares the rank's block of chunks among T threads.
//     Each thread fills its OWN private Accumulator (no locks, no races), and
//     afterwards adds it into the rank's total inside `#pragma omp critical`.
//
// Each chunk has its own random seed (design decision D2), so who runs a chunk
// doesn't matter: every (P, T) gives exactly the same histogram counts as mc_serial.

#include <mpi.h>
#include <omp.h>

#include <climits>
#include <iostream>
#include <limits>
#include <string>

#include "core/accumulator.hpp"
#include "core/config.hpp"
#include "core/io.hpp"
#include "core/simulation.hpp"

namespace {

// Runs this rank's chunks [first, first + count) on cfg.threads OpenMP threads,
// adding everything into `total`. Returns the number of threads actually used.
int simulate_block(const mc::Config& cfg, const mc::BinLayout& layout, uint64_t first,
                   uint64_t count, mc::Accumulator& total) {
    int threads_used = 1;
    #pragma omp parallel num_threads(cfg.threads)
    {
        // Private to this thread. Created inside the parallel region so its
        // memory is allocated (and first touched) by the thread that uses it.
        mc::Accumulator local(cfg.months, cfg.bins);

        #pragma omp for schedule(static)
        for (uint64_t c = first; c < first + count; ++c) {
            mc::simulate_chunk(cfg, layout, c, local);
        }
        // (implicit barrier here: every thread has finished its chunks)

        #pragma omp critical
        total.merge(local);

        #pragma omp single
        threads_used = omp_get_num_threads();  // the runtime may give fewer than asked
    }
    return threads_used;
}

// Adds every rank's accumulator into rank 0's (the result is only valid on rank 0).
// Counts and sums use MPI_SUM; the smallest/largest final value use MPI_MIN/MPI_MAX.
void reduce_to_root(mc::Accumulator& acc, int rank) {
    const bool root = (rank == 0);

    // Histograms: one big array of uint64 counts. On rank 0 we reduce "in place"
    // (its own counts are both input and output), so no second 2 MB buffer is needed.
    MPI_Reduce(root ? MPI_IN_PLACE : acc.hist.data(), acc.hist.data(),
               static_cast<int>(acc.hist.size()), MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(root ? MPI_IN_PLACE : acc.month_sum.data(), acc.month_sum.data(),
               static_cast<int>(acc.month_sum.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Small scalars, packed into arrays so we need only a few messages.
    uint64_t counts[4] = {acc.trials, acc.loss_count, acc.overflow_low, acc.overflow_high};
    double sums[2] = {acc.final_sum, acc.final_sum_sq};
    MPI_Reduce(root ? MPI_IN_PLACE : counts, counts, 4, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(root ? MPI_IN_PLACE : sums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // A rank with no trials (possible when trials < P * CHUNK_SIZE) has no
    // valid min/max, so it sends +inf / -inf, which never wins.
    const double inf = std::numeric_limits<double>::infinity();
    double min_value = (acc.trials > 0) ? acc.final_min : inf;
    double max_value = (acc.trials > 0) ? acc.final_max : -inf;
    MPI_Reduce(root ? MPI_IN_PLACE : &min_value, &min_value, 1, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(root ? MPI_IN_PLACE : &max_value, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (root) {
        acc.trials = counts[0];
        acc.loss_count = counts[1];
        acc.overflow_low = counts[2];
        acc.overflow_high = counts[3];
        acc.final_sum = sums[0];
        acc.final_sum_sq = sums[1];
        acc.final_min = min_value;
        acc.final_max = max_value;
    }
}

// Largest value of x over all ranks (result on rank 0). Used for timing:
// a parallel run is only as fast as its slowest process.
double max_over_ranks(double x) {
    double result = 0.0;
    MPI_Reduce(&x, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return result;
}

// Shuts MPI down and returns `code`, so main can `return finish(code);`.
int finish(int code) {
    MPI_Finalize();
    return code;
}

}  // namespace

int main(int argc, char** argv) {
    // FUNNELED: several threads exist, but only the main thread calls MPI.
    // (All our MPI calls are outside the OpenMP parallel region.)
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const bool root = (rank == 0);

    if (provided < MPI_THREAD_FUNNELED) {
        if (root) std::cerr << "error: this MPI library does not support MPI_THREAD_FUNNELED\n";
        return finish(1);
    }

    // Every rank parses the same command line, so they all agree on the settings
    // (and all exit together on bad input). Only rank 0 prints.
    mc::Config cfg;
    std::string error;
    switch (mc::parse_args(argc, argv, cfg, error)) {
        case mc::ParseResult::Help:
            if (root) std::cout << mc::usage(argv[0]);
            return finish(0);
        case mc::ParseResult::Error:
            if (root) std::cerr << "error: " << error << "\n(run with --help to see all options)\n";
            return finish(1);
        case mc::ParseResult::Ok:
            break;
    }
    // MPI message sizes are `int`, so the histogram must have fewer than INT_MAX entries.
    if (static_cast<long long>(cfg.months) * cfg.bins > INT_MAX) {
        if (root) std::cerr << "error: --months x --bins is too large for one MPI message\n";
        return finish(1);
    }
    if (cfg.exact && root) {
        std::cerr << "note: --exact is not implemented yet (Phase 4); ignored\n";
    }

    // Start the clock together on all ranks (design decision D5).
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    // --- Compute: this rank's block of chunks, on OpenMP threads ---
    const mc::BinLayout layout = mc::make_bin_layout(cfg);
    mc::Accumulator acc(cfg.months, cfg.bins);
    uint64_t first = 0, count = 0;
    mc::chunk_block(mc::num_chunks(cfg.trials), ranks, rank, first, count);
    const int threads_used = simulate_block(cfg, layout, first, count, acc);
    const double compute_s = MPI_Wtime() - start;

    // Wait for the slowest rank, so reduce_s measures only communication,
    // not ranks waiting for each other.
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Reduce: add all ranks' results on rank 0 ---
    const double reduce_start = MPI_Wtime();
    reduce_to_root(acc, rank);
    const double reduce_s = MPI_Wtime() - reduce_start;
    const double total_s = MPI_Wtime() - start;

    mc::Timing timing;
    timing.compute_s = max_over_ranks(compute_s);
    timing.reduce_s = max_over_ranks(reduce_s);
    timing.total_s = max_over_ranks(total_s);

    // --- Output: rank 0 only ---
    int exit_code = 0;
    if (root) {
        mc::ParallelInfo parallel;
        parallel.mpi_processes = ranks;
        parallel.omp_threads = threads_used;
        if (!mc::finish_run("mc_hybrid", cfg, layout, acc, parallel, timing, error)) {
            std::cerr << "error: " << error << "\n";
            exit_code = 1;
        }
    }
    return finish(exit_code);
}
