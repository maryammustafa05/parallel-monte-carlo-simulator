#pragma once
// Simulation settings: defaults, command-line parsing and validation.

#include <cstdint>
#include <string>

namespace mc {

// All inputs of one simulation run (see CLAUDE.md section 4.1).
struct Config {
    double initial = 10000.0;       // initial investment V0 (currency units)
    double mu = 0.07;               // expected ANNUAL return (0.07 = 7 %)
    double sigma = 0.15;            // ANNUAL volatility (0.15 = 15 %)
    int months = 120;               // horizon T in months
    uint64_t trials = 1000000;      // number of simulated futures N
    double confidence = 0.95;       // VaR confidence level
    uint64_t seed = 42;             // global RNG seed
    int threads = 1;                // OpenMP threads per process (hybrid only)
    int bins = 2048;                // histogram bins per month
    std::string out = "results/run";  // output directory
    bool exact = false;             // also keep all final values (Phase 4)
};

// Outcome of parsing the command line.
enum class ParseResult { Ok, Help, Error };

// Reads flags like "--trials 1000000" into cfg, then validates it.
// On Error, `error` holds a human-readable message.
ParseResult parse_args(int argc, char** argv, Config& cfg, std::string& error);

// Checks that every setting is in its allowed range. Returns false + message if not.
bool validate(const Config& cfg, std::string& error);

// Returns the --help text for a program called `program_name`.
std::string usage(const std::string& program_name);

}  // namespace mc
