// mc_serial: the single-core baseline (design decision D5).
// Built WITHOUT MPI and OpenMP so that speedup = T_serial / T_parallel is fair.

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#include "core/accumulator.hpp"
#include "core/analytics.hpp"
#include "core/config.hpp"
#include "core/io.hpp"
#include "core/simulation.hpp"

namespace {

// Seconds elapsed between two steady_clock time points.
double seconds_between(std::chrono::steady_clock::time_point a,
                       std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

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
    if (cfg.exact) {
        std::cerr << "note: --exact is not implemented yet (Phase 4); ignored\n";
    }

    const auto start = clock::now();

    // --- Simulate: every chunk, one after another, into one accumulator ---
    const mc::BinLayout layout = mc::make_bin_layout(cfg);
    mc::Accumulator acc(cfg.months, cfg.bins);
    const uint64_t chunks = mc::num_chunks(cfg.trials);
    for (uint64_t c = 0; c < chunks; ++c) {
        mc::simulate_chunk(cfg, layout, c, acc);
    }
    const auto compute_done = clock::now();

    // --- Analyse and write results ---
    const auto fan = mc::compute_fan_chart(acc, layout, cfg.initial);
    const auto final_stats = mc::compute_final_stats(acc, layout);
    const auto risk = mc::compute_risk(acc, layout, cfg.initial, cfg.confidence);

    if (!mc::ensure_directory(cfg.out, error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }
    const std::string fan_path = cfg.out + "/fan_chart.csv";
    if (!mc::write_fan_chart_csv(fan_path, fan)) {
        std::cerr << "error: cannot write " << fan_path << "\n";
        return 1;
    }

    mc::Timing timing;
    timing.compute_s = seconds_between(start, compute_done);
    timing.reduce_s = 0.0;  // nothing to merge in the serial program
    timing.total_s = seconds_between(start, clock::now());

    const std::string summary_path = cfg.out + "/summary.json";
    if (!mc::write_summary_json(summary_path, cfg, mc::ParallelInfo{}, final_stats, risk, acc,
                                timing)) {
        std::cerr << "error: cannot write " << summary_path << "\n";
        return 1;
    }

    // --- Short human-readable report ---
    std::printf("mc_serial: %llu trials x %d months in %.3f s (compute %.3f s)\n",
                static_cast<unsigned long long>(cfg.trials), cfg.months, timing.total_s,
                timing.compute_s);
    std::printf("  final value: mean %.2f  median %.2f  P5 %.2f  P95 %.2f\n", final_stats.mean,
                final_stats.median, final_stats.p05, final_stats.p95);
    std::printf("  VaR(%.0f%%): %.2f   P(loss): %.4f   histogram overflow: low %llu, high %llu\n",
                cfg.confidence * 100.0, risk.var, risk.prob_loss,
                static_cast<unsigned long long>(acc.overflow_low),
                static_cast<unsigned long long>(acc.overflow_high));
    std::printf("  wrote %s and %s\n", fan_path.c_str(), summary_path.c_str());
    return 0;
}
