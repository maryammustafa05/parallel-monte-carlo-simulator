// Tests for analytics.cpp on small hand-made histograms whose answers we can
// work out on paper.

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/analytics.hpp"
#include "test_util.hpp"

static void test_uniform_histogram() {
    // 10 bins in log space starting at 0, width 0.1, 100 counts each (1000 total).
    std::vector<uint64_t> counts(10, 100);
    const double lo = 0.0, w = 0.1;
    // Median: 500 counts = end of bin 4 -> log value 0.5.
    CHECK_NEAR(mc::histogram_percentile(counts.data(), 10, lo, w, 0.50), std::exp(0.5), 1e-12);
    // P5: 50 counts = halfway through bin 0 -> 0.05.
    CHECK_NEAR(mc::histogram_percentile(counts.data(), 10, lo, w, 0.05), std::exp(0.05), 1e-12);
    // P95: 950 counts = halfway through bin 9 -> 0.95.
    CHECK_NEAR(mc::histogram_percentile(counts.data(), 10, lo, w, 0.95), std::exp(0.95), 1e-12);
}

static void test_single_bin_and_edges() {
    // All 40 counts in bin 3 of 8, log range starts at ln(1000), width 0.01.
    std::vector<uint64_t> counts(8, 0);
    counts[3] = 40;
    const double lo = std::log(1000.0), w = 0.01;
    CHECK_NEAR(mc::histogram_percentile(counts.data(), 8, lo, w, 0.25),
               std::exp(lo + 3.25 * w), 1e-9);
    // q = 0 -> lower edge of first non-empty bin; q = 1 -> upper edge of last one.
    CHECK_NEAR(mc::histogram_percentile(counts.data(), 8, lo, w, 0.0), std::exp(lo + 3 * w), 1e-9);
    CHECK_NEAR(mc::histogram_percentile(counts.data(), 8, lo, w, 1.0), std::exp(lo + 4 * w), 1e-9);

    std::vector<uint64_t> empty(8, 0);
    CHECK(std::isnan(mc::histogram_percentile(empty.data(), 8, lo, w, 0.5)));
}

static void test_var() {
    CHECK_NEAR(mc::value_at_risk(10000.0, 8000.0), 2000.0, 1e-12);
    CHECK_NEAR(mc::value_at_risk(10000.0, 12000.0), 0.0, 1e-12);  // no loss -> VaR 0
}

static void test_stats_from_accumulator() {
    // One month, 4 bins covering log values [ln 5000, ln 20000).
    mc::BinLayout layout;
    layout.months = 1;
    layout.bins = 4;
    layout.log_lo = {std::log(5000.0)};
    layout.log_width = {(std::log(20000.0) - std::log(5000.0)) / 4.0};
    layout.inv_width = {1.0 / layout.log_width[0]};

    // Final values 6000, 9000, 12000, 15000 with V0 = 10000.
    mc::Accumulator acc(1, 4);
    const double values[] = {6000.0, 9000.0, 12000.0, 15000.0};
    for (double v : values) {
        acc.month_sum[0] += v;
        const double x = (std::log(v) - layout.log_lo[0]) * layout.inv_width[0];
        acc.month_hist(1)[static_cast<int>(x)] += 1;
        acc.add_final(v, 10000.0);
    }

    const mc::FinalStats s = mc::compute_final_stats(acc, layout);
    CHECK_NEAR(s.mean, 10500.0, 1e-9);
    // Population std of {6000, 9000, 12000, 15000} = sqrt(11250000) = 3354.10...
    CHECK_NEAR(s.std, std::sqrt(11250000.0), 1e-6);
    CHECK(s.min == 6000.0 && s.max == 15000.0);

    const mc::RiskStats r = mc::compute_risk(acc, layout, 10000.0, 0.95);
    CHECK_NEAR(r.prob_loss, 0.5, 1e-12);  // 6000 and 9000 are losses
    CHECK_NEAR(r.var, mc::value_at_risk(10000.0, s.p05), 1e-9);
    CHECK(r.var > 0.0);

    const auto fan = mc::compute_fan_chart(acc, layout, 10000.0);
    CHECK(fan.size() == 2);
    CHECK(fan[0].month == 0 && fan[0].median == 10000.0 && fan[0].p05 == 10000.0);
    CHECK_NEAR(fan[1].mean, 10500.0, 1e-9);
    CHECK_NEAR(fan[1].median, s.median, 1e-9);
    CHECK(fan[1].p05 <= fan[1].p25 && fan[1].p25 <= fan[1].median &&
          fan[1].median <= fan[1].p75 && fan[1].p75 <= fan[1].p95);
}

int main() {
    test_uniform_histogram();
    test_single_bin_and_edges();
    test_var();
    test_stats_from_accumulator();
    return test::summary("test_analytics");
}
