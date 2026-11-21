#pragma once
// Accumulator: everything we remember about a batch of trials (design decision D1).
//
// We never store whole paths. Instead, for every month we keep a histogram
// (how many trials ended up in each value range) plus a few running sums.
// Two accumulators can be merged just by adding their numbers, which is
// exactly what OpenMP threads and MPI processes will do in later phases.

#include <cstdint>
#include <vector>

namespace mc {

struct Accumulator {
    int months = 0;  // horizon T
    int bins = 0;    // histogram bins per month

    // Histogram counts, one row of `bins` per month. Row (t-1) is month t.
    // Month 0 is not stored: every trial starts at exactly V0.
    std::vector<uint64_t> hist;

    // Sum of V_t over all trials, for each month t = 1..T (index t-1). Used for the mean.
    std::vector<double> month_sum;

    uint64_t trials = 0;          // number of trials added
    double final_sum = 0.0;       // sum of V_T
    double final_sum_sq = 0.0;    // sum of V_T^2 (for the standard deviation)
    double final_min = 0.0;       // smallest V_T seen (valid only if trials > 0)
    double final_max = 0.0;       // largest V_T seen (valid only if trials > 0)
    uint64_t loss_count = 0;      // trials with V_T < V0 (for prob_loss)
    uint64_t overflow_low = 0;    // values below a month's histogram range (all months)
    uint64_t overflow_high = 0;   // values above a month's histogram range (all months)

    // Creates an empty accumulator for `months` months and `bins` bins per month.
    Accumulator(int months, int bins);

    // Pointer to the histogram row of month t (1-based, 1..months).
    uint64_t* month_hist(int t) { return hist.data() + static_cast<size_t>(t - 1) * bins; }
    const uint64_t* month_hist(int t) const {
        return hist.data() + static_cast<size_t>(t - 1) * bins;
    }

    // Records one finished trial's final value V_T.
    void add_final(double value, double initial);

    // Adds all counts and sums of `other` into this accumulator.
    // Both must have the same months and bins.
    void merge(const Accumulator& other);

    // A 64-bit fingerprint (FNV-1a hash) of all integer counts: histograms, trials,
    // loss_count and overflow counters. Two runs have the same checksum only if all
    // those counts are identical, so it proves serial == parallel in one number.
    uint64_t checksum() const;
};

}  // namespace mc
