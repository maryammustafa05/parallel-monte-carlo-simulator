#include "analytics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mc {

double histogram_percentile(const uint64_t* counts, int bins, double log_lo,
                            double log_width, double q) {
    uint64_t total = 0;
    for (int b = 0; b < bins; ++b) total += counts[b];
    if (total == 0) return std::numeric_limits<double>::quiet_NaN();

    // Walk the bins until the running count reaches q * total, then place the
    // answer proportionally inside that bin.
    const double target = q * static_cast<double>(total);
    double cumulative = 0.0;
    for (int b = 0; b < bins; ++b) {
        if (counts[b] == 0) continue;
        const double count = static_cast<double>(counts[b]);
        if (cumulative + count >= target) {
            const double fraction = std::clamp((target - cumulative) / count, 0.0, 1.0);
            return std::exp(log_lo + (b + fraction) * log_width);
        }
        cumulative += count;
    }
    return std::exp(log_lo + bins * log_width);  // only reached through rounding at q = 1
}

double value_at_risk(double initial, double quantile_value) {
    return std::max(0.0, initial - quantile_value);
}

// Percentile q of month t (1-based) from the accumulator.
static double month_percentile(const Accumulator& acc, const BinLayout& layout, int t,
                               double q) {
    return histogram_percentile(acc.month_hist(t), layout.bins, layout.log_lo[t - 1],
                                layout.log_width[t - 1], q);
}

std::vector<FanRow> compute_fan_chart(const Accumulator& acc, const BinLayout& layout,
                                      double initial) {
    std::vector<FanRow> rows;
    rows.reserve(static_cast<size_t>(acc.months) + 1);

    FanRow start;
    start.month = 0;
    start.p05 = start.p25 = start.median = start.p75 = start.p95 = start.mean = initial;
    rows.push_back(start);

    const double n = static_cast<double>(acc.trials);
    for (int t = 1; t <= acc.months; ++t) {
        FanRow row;
        row.month = t;
        row.p05 = month_percentile(acc, layout, t, 0.05);
        row.p25 = month_percentile(acc, layout, t, 0.25);
        row.median = month_percentile(acc, layout, t, 0.50);
        row.p75 = month_percentile(acc, layout, t, 0.75);
        row.p95 = month_percentile(acc, layout, t, 0.95);
        row.mean = acc.month_sum[t - 1] / n;
        rows.push_back(row);
    }
    return rows;
}

FinalStats compute_final_stats(const Accumulator& acc, const BinLayout& layout) {
    FinalStats s;
    const double n = static_cast<double>(acc.trials);
    s.mean = acc.final_sum / n;
    // Var = E[V^2] - E[V]^2. Rounding can make it a hair below 0 when sigma = 0.
    const double variance = acc.final_sum_sq / n - s.mean * s.mean;
    s.std = std::sqrt(std::max(0.0, variance));
    s.min = acc.final_min;
    s.max = acc.final_max;
    const int T = acc.months;
    s.median = month_percentile(acc, layout, T, 0.50);
    s.p05 = month_percentile(acc, layout, T, 0.05);
    s.p95 = month_percentile(acc, layout, T, 0.95);
    return s;
}

RiskStats compute_risk(const Accumulator& acc, const BinLayout& layout, double initial,
                       double confidence) {
    RiskStats r;
    r.confidence = confidence;
    // At 95 % confidence we look at the 5th percentile of the final value.
    const double worst_case = month_percentile(acc, layout, acc.months, 1.0 - confidence);
    r.var = value_at_risk(initial, worst_case);
    r.prob_loss = static_cast<double>(acc.loss_count) / static_cast<double>(acc.trials);
    return r;
}

}  // namespace mc
