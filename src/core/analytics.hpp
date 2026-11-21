#pragma once
// Turns an Accumulator into the numbers people care about:
// percentiles per month (the fan chart), final-value statistics and VaR.

#include <cstdint>
#include <vector>

#include "accumulator.hpp"
#include "simulation.hpp"

namespace mc {

// One row of fan_chart.csv (values in currency units).
struct FanRow {
    int month = 0;
    double p05 = 0, p25 = 0, median = 0, p75 = 0, p95 = 0, mean = 0;
};

// Statistics of the final value V_T.
struct FinalStats {
    double mean = 0, std = 0, min = 0, max = 0, median = 0, p05 = 0, p95 = 0;
};

// Risk numbers for the whole horizon.
struct RiskStats {
    double var = 0;              // Value at Risk at `confidence`, in currency units
    double threshold = 0;        // the (1 - confidence) quantile of V_T; VaR = max(0, V0 - threshold)
    double prob_loss = 0;        // fraction of trials with V_T < V0
    double prob_total_loss = 0;  // fraction of trials wiped out (V_T = 0)
    double confidence = 0;
};

// --exact mode: the same numbers computed from every final value (no histogram).
struct ExactStats {
    double median = 0, p05 = 0, p95 = 0;
    double var = 0;         // VaR at the run's confidence level
};

// Value at quantile q (0..1) of a log-spaced histogram plus `zeros` values that
// are exactly 0 (wiped-out trials, kept outside the histogram). Interpolates
// linearly inside the bin (in log space). Returns NaN if there are no values.
double histogram_percentile(const uint64_t* counts, int bins, double log_lo,
                            double log_width, double q, uint64_t zeros = 0);

// VaR = max(0, V0 - value_at_quantile): the loss not exceeded with the given confidence.
double value_at_risk(double initial, double quantile_value);

// Fan chart rows for months 0..T (month 0 is V0 everywhere).
std::vector<FanRow> compute_fan_chart(const Accumulator& acc, const BinLayout& layout,
                                      double initial);

// Mean, standard deviation (population), min, max and percentiles of V_T.
FinalStats compute_final_stats(const Accumulator& acc, const BinLayout& layout);

// VaR at `confidence` and probability of ending below V0.
RiskStats compute_risk(const Accumulator& acc, const BinLayout& layout, double initial,
                       double confidence);

// Quantile q of already SORTED values, interpolating linearly between the two
// nearest values at position q*(n-1) (the same rule as NumPy's default).
double sorted_percentile(const std::vector<double>& sorted, double q);

// Exact median, P5, P95 and VaR from all final values (takes a copy and sorts it).
ExactStats compute_exact_stats(std::vector<double> values, double initial, double confidence);

}  // namespace mc
