#include "io.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace mc {

namespace {

// Formats a number with up to 10 significant digits ("10000", "0.07", "12345.6789").
std::string num(double x) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.10g", x);
    return buffer;
}

std::string num(uint64_t x) { return std::to_string(x); }

}  // namespace

bool ensure_directory(const std::string& path, std::string& error) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        error = "cannot create directory '" + path + "': " + ec.message();
        return false;
    }
    return true;
}

bool write_fan_chart_csv(const std::string& path, const std::vector<FanRow>& rows) {
    std::ofstream file(path);
    if (!file) return false;
    file << "month,p05,p25,median,p75,p95,mean\n";
    for (const FanRow& r : rows) {
        file << r.month << ',' << num(r.p05) << ',' << num(r.p25) << ',' << num(r.median)
             << ',' << num(r.p75) << ',' << num(r.p95) << ',' << num(r.mean) << '\n';
    }
    return static_cast<bool>(file);
}

bool write_summary_json(const std::string& path, const Config& cfg,
                        const ParallelInfo& parallel, const FinalStats& f,
                        const RiskStats& risk, const Accumulator& acc, const Timing& timing) {
    std::ofstream file(path);
    if (!file) return false;
    file << "{\n"
         << "  \"params\": {\"initial\": " << num(cfg.initial)
         << ", \"mu\": " << num(cfg.mu)
         << ", \"sigma\": " << num(cfg.sigma)
         << ", \"months\": " << cfg.months
         << ", \"trials\": " << num(cfg.trials)
         << ", \"confidence\": " << num(cfg.confidence)
         << ", \"seed\": " << num(cfg.seed) << "},\n"
         << "  \"parallel\": {\"mpi_processes\": " << parallel.mpi_processes
         << ", \"omp_threads\": " << parallel.omp_threads << "},\n"
         << "  \"final_value\": {\"mean\": " << num(f.mean)
         << ", \"std\": " << num(f.std)
         << ", \"min\": " << num(f.min)
         << ", \"max\": " << num(f.max)
         << ", \"median\": " << num(f.median)
         << ", \"p05\": " << num(f.p05)
         << ", \"p95\": " << num(f.p95) << "},\n"
         << "  \"risk\": {\"var\": " << num(risk.var)
         << ", \"prob_loss\": " << num(risk.prob_loss)
         << ", \"confidence\": " << num(risk.confidence) << "},\n"
         << "  \"histogram\": {\"bins\": " << cfg.bins
         << ", \"overflow_low\": " << num(acc.overflow_low)
         << ", \"overflow_high\": " << num(acc.overflow_high) << "},\n"
         << "  \"timing\": {\"total_s\": " << num(timing.total_s)
         << ", \"compute_s\": " << num(timing.compute_s)
         << ", \"reduce_s\": " << num(timing.reduce_s) << "}\n"
         << "}\n";
    return static_cast<bool>(file);
}

}  // namespace mc
