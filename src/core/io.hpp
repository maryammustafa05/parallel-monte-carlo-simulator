#pragma once
// Writes results to disk as CSV and JSON (hand-written; no JSON library needed).
// The formats are the contract with the Python scripts: see CLAUDE.md section 8.

#include <string>
#include <vector>

#include "accumulator.hpp"
#include "analytics.hpp"
#include "config.hpp"

namespace mc {

// How the run was parallelised (1 and 1 for the serial program).
struct ParallelInfo {
    int mpi_processes = 1;
    int omp_threads = 1;
};

// Wall-clock times in seconds.
struct Timing {
    double total_s = 0;    // whole run
    double compute_s = 0;  // simulation only
    double reduce_s = 0;   // merging results between processes (0 for serial)
};

// Creates directory `path` (and parents) if needed. Returns false + message on failure.
bool ensure_directory(const std::string& path, std::string& error);

// Writes fan_chart.csv: month,p05,p25,median,p75,p95,mean. Returns false on failure.
bool write_fan_chart_csv(const std::string& path, const std::vector<FanRow>& rows);

// Writes final_hist.csv: bin_lower,bin_upper,count of the final month's histogram.
bool write_final_hist_csv(const std::string& path, const Accumulator& acc,
                          const BinLayout& layout);

// Writes summary.json with parameters, final stats, risk, histogram info and timing.
// `exact` is nullptr unless --exact was given; then an "exact" section is added.
bool write_summary_json(const std::string& path, const Config& cfg,
                        const ParallelInfo& parallel, const FinalStats& final_stats,
                        const RiskStats& risk, const Accumulator& acc, const Timing& timing,
                        const ExactStats* exact);

// Shared last step of mc_serial and mc_hybrid: computes the analytics from acc,
// writes fan_chart.csv, final_hist.csv and summary.json into cfg.out and prints a
// short report.
// Returns false + message on failure.
bool finish_run(const std::string& program, const Config& cfg, const BinLayout& layout,
                const Accumulator& acc, const ParallelInfo& parallel, const Timing& timing,
                std::string& error);

}  // namespace mc
