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

// Formats a 64-bit checksum as 16 hex digits (a string, because JSON numbers
// are often read as doubles, which cannot hold all 64 bits exactly).
std::string checksum_hex(uint64_t x) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(x));
    return buffer;
}

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
         << ", \"overflow_high\": " << num(acc.overflow_high)
         << ", \"checksum\": \"" << checksum_hex(acc.checksum()) << "\"},\n"
         << "  \"timing\": {\"total_s\": " << num(timing.total_s)
         << ", \"compute_s\": " << num(timing.compute_s)
         << ", \"reduce_s\": " << num(timing.reduce_s) << "}\n"
         << "}\n";
    return static_cast<bool>(file);
}

}  // namespace mc

namespace mc {

bool finish_run(const std::string& program, const Config& cfg, const BinLayout& layout,
                const Accumulator& acc, const ParallelInfo& parallel, const Timing& timing,
                std::string& error) {
    const auto fan = compute_fan_chart(acc, layout, cfg.initial);
    const auto final_stats = compute_final_stats(acc, layout);
    const auto risk = compute_risk(acc, layout, cfg.initial, cfg.confidence);

    if (!ensure_directory(cfg.out, error)) return false;
    const std::string fan_path = cfg.out + "/fan_chart.csv";
    if (!write_fan_chart_csv(fan_path, fan)) {
        error = "cannot write " + fan_path;
        return false;
    }
    const std::string summary_path = cfg.out + "/summary.json";
    if (!write_summary_json(summary_path, cfg, parallel, final_stats, risk, acc, timing)) {
        error = "cannot write " + summary_path;
        return false;
    }

    std::printf("%s: %llu trials x %d months, %d process(es) x %d thread(s)\n", program.c_str(),
                static_cast<unsigned long long>(cfg.trials), cfg.months, parallel.mpi_processes,
                parallel.omp_threads);
    std::printf("  time: total %.3f s (compute %.3f s, reduce %.3f s)\n", timing.total_s,
                timing.compute_s, timing.reduce_s);
    std::printf("  final value: mean %.2f  median %.2f  P5 %.2f  P95 %.2f\n", final_stats.mean,
                final_stats.median, final_stats.p05, final_stats.p95);
    std::printf("  VaR(%.0f%%): %.2f   P(loss): %.4f   histogram overflow: low %llu, high %llu\n",
                cfg.confidence * 100.0, risk.var, risk.prob_loss,
                static_cast<unsigned long long>(acc.overflow_low),
                static_cast<unsigned long long>(acc.overflow_high));
    std::printf("  checksum %s   wrote %s and %s\n", checksum_hex(acc.checksum()).c_str(),
                fan_path.c_str(), summary_path.c_str());
    return true;
}

}  // namespace mc
