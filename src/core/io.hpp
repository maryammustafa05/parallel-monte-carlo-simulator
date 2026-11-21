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

// Writes summary.json with parameters, final stats, risk, histogram info and timing.
bool write_summary_json(const std::string& path, const Config& cfg,
                        const ParallelInfo& parallel, const FinalStats& final_stats,
                        const RiskStats& risk, const Accumulator& acc, const Timing& timing);

}  // namespace mc
