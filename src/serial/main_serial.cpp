// mc_serial: the single-core baseline (design decision D5).
// Built WITHOUT MPI and OpenMP so that speedup = T_serial / T_parallel is fair.

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
    if (cfg.threads != 1) {
        std::cerr << "note: mc_serial always uses 1 thread; --threads is ignored\n";
    }

    const auto start = clock::now();

    // Every chunk, one after another, into one accumulator.
    const mc::BinLayout layout = mc::make_bin_layout(cfg);
    mc::Accumulator acc(cfg.months, cfg.bins);
    acc.keep_final_values = cfg.exact;
    const uint64_t chunks = mc::num_chunks(cfg.trials);
    for (uint64_t c = 0; c < chunks; ++c) {
        mc::simulate_chunk(cfg, layout, c, acc);
    }

    mc::Timing timing;
    timing.compute_s = std::chrono::duration<double>(clock::now() - start).count();
    timing.reduce_s = 0.0;               // nothing to merge in the serial program
    timing.total_s = timing.compute_s;   // total = compute + reduce (file writing not timed)

    mc::ParallelInfo parallel;  // 1 process, 1 thread
    if (!mc::finish_run("mc_serial", cfg, layout, acc, parallel, timing, error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }
    return 0;
}
