#include "io.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace mc {

namespace {

// Formats a number with up to 10 significant digits ("10000", "0.07", "12345.6789").
// JSON has no NaN or infinity, so those become null. Input validation should make
// that impossible, but a broken file would break every Python script.
std::string num(double x) {
    if (!std::isfinite(x)) return "null";
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

// Relative difference |a - b| / |b| (b is the reference value). 0 if both are 0.
double rel_diff(double a, double b) {
    if (a == b) return 0.0;
    return std::fabs(a - b) / std::fabs(b);
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

bool write_final_hist_csv(const std::string& path, const Accumulator& acc,
                          const BinLayout& layout) {
    std::ofstream file(path);
    if (!file) return false;
    const int T = acc.months;
    const double lo = layout.log_lo[T - 1];
    const double width = layout.log_width[T - 1];
    const uint64_t* counts = acc.month_hist(T);
    file << "bin_lower,bin_upper,count\n";
    for (int b = 0; b < layout.bins; ++b) {
        file << num(std::exp(lo + b * width)) << ',' << num(std::exp(lo + (b + 1) * width)) << ','
             << counts[b] << '\n';
    }
    return static_cast<bool>(file);
}

bool write_summary_json(const std::string& path, const Config& cfg,
                        const ParallelInfo& parallel, const FinalStats& f,
                        const RiskStats& risk, const Accumulator& acc, const Timing& timing,
                        const ExactStats* exact) {
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
         << ", \"threshold\": " << num(risk.threshold)
         << ", \"prob_loss\": " << num(risk.prob_loss)
         << ", \"prob_total_loss\": " << num(risk.prob_total_loss)
         << ", \"confidence\": " << num(risk.confidence) << "},\n"
         << "  \"histogram\": {\"bins\": " << cfg.bins
         << ", \"overflow_low\": " << num(acc.overflow_low)
         << ", \"overflow_high\": " << num(acc.overflow_high)
         << ", \"checksum\": \"" << checksum_hex(acc.checksum()) << "\"},\n";
    if (exact != nullptr) {
        // Only with --exact: true values from all final values, and how far the
        // histogram answers are from them (relative, 0.001 = 0.1 %).
        file << "  \"exact\": {\"median\": " << num(exact->median)
             << ", \"p05\": " << num(exact->p05)
             << ", \"p95\": " << num(exact->p95)
             << ", \"var\": " << num(exact->var)
             << ", \"rel_diff_median\": " << num(rel_diff(f.median, exact->median))
             << ", \"rel_diff_p05\": " << num(rel_diff(f.p05, exact->p05))
             << ", \"rel_diff_p95\": " << num(rel_diff(f.p95, exact->p95))
             << ", \"rel_diff_var\": " << num(rel_diff(risk.var, exact->var)) << "},\n";
    }
    file << "  \"timing\": {\"total_s\": " << num(timing.total_s)
         << ", \"compute_s\": " << num(timing.compute_s)
         << ", \"reduce_s\": " << num(timing.reduce_s) << "}\n"
         << "}\n";
    return static_cast<bool>(file);
}

bool finish_run(const std::string& program, const Config& cfg, const BinLayout& layout,
                const Accumulator& acc, const ParallelInfo& parallel, const Timing& timing,
                std::string& error) {
    const auto fan = compute_fan_chart(acc, layout, cfg.initial);
    const auto final_stats = compute_final_stats(acc, layout);
    const auto risk = compute_risk(acc, layout, cfg.initial, cfg.confidence);
    ExactStats exact;
    if (cfg.exact) exact = compute_exact_stats(acc.final_values, cfg.initial, cfg.confidence);

    if (!ensure_directory(cfg.out, error)) return false;
    const std::string fan_path = cfg.out + "/fan_chart.csv";
    const std::string hist_path = cfg.out + "/final_hist.csv";
    const std::string summary_path = cfg.out + "/summary.json";
    if (!write_fan_chart_csv(fan_path, fan)) {
        error = "cannot write " + fan_path;
        return false;
    }
    if (!write_final_hist_csv(hist_path, acc, layout)) {
        error = "cannot write " + hist_path;
        return false;
    }
    if (!write_summary_json(summary_path, cfg, parallel, final_stats, risk, acc, timing,
                            cfg.exact ? &exact : nullptr)) {
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
    if (cfg.exact) {
        std::printf("  exact check:  median %.2f (%.4f%%)  P5 %.2f (%.4f%%)  P95 %.2f (%.4f%%)"
                    "  VaR %.2f (%.4f%%)\n",
                    exact.median, 100 * rel_diff(final_stats.median, exact.median), exact.p05,
                    100 * rel_diff(final_stats.p05, exact.p05), exact.p95,
                    100 * rel_diff(final_stats.p95, exact.p95), exact.var,
                    100 * rel_diff(risk.var, exact.var));
    }
    std::printf("  checksum %s   wrote %s/{fan_chart.csv, final_hist.csv, summary.json}\n",
                checksum_hex(acc.checksum()).c_str(), cfg.out.c_str());
    return true;
}

}  // namespace mc
