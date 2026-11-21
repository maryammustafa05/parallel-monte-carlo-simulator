#pragma once
// The Monte Carlo model itself: histogram bin layout and simulating one chunk.

#include <cstdint>
#include <vector>

#include "accumulator.hpp"
#include "config.hpp"

namespace mc {

// Trials per chunk (design decision D2). Each chunk has its own RNG seed,
// so results do not depend on how chunks are shared between workers.
constexpr uint64_t CHUNK_SIZE = 4096;

// Where the histogram bins of each month lie (design decision D1).
// Bins are evenly spaced in LOG space: bin b of month t covers
//   [ exp(log_lo + b*width), exp(log_lo + (b+1)*width) )
struct BinLayout {
    int months = 0;
    int bins = 0;
    std::vector<double> log_lo;     // per month (index t-1): log of the lowest bin edge
    std::vector<double> log_width;  // per month: width of one bin in log space
    std::vector<double> inv_width;  // per month: 1 / log_width (multiplying is faster than dividing)
};

// Monthly expected return: mu / 12.
double monthly_mu(const Config& cfg);

// Monthly volatility: sigma / sqrt(12).
double monthly_sigma(const Config& cfg);

// Builds the bin layout for cfg: each month is centred on the expected ln(V_t)
// and spans +-6 standard deviations of ln(V_t).
BinLayout make_bin_layout(const Config& cfg);

// Number of chunks needed for cfg.trials trials: ceil(trials / CHUNK_SIZE).
uint64_t num_chunks(uint64_t trials);

// Contiguous block of chunks that worker `rank` of `workers` should run
// (design decision D4). Each worker gets total/workers chunks, and the first
// total%workers workers get one extra, so the split works for any numbers.
// Sets first = index of its first chunk, count = how many (may be 0).
void chunk_block(uint64_t total_chunks, int workers, int rank, uint64_t& first,
                 uint64_t& count);

// Histogram bin of value v in month t; counts out-of-range values in acc's overflow
// counters. Returns -1 for v <= 0 (a wiped-out trial), which is counted in
// acc.month_zero instead of the histogram.
int bin_index(const BinLayout& layout, int t, double v, Accumulator& acc);

// Simulates every trial of chunk `chunk` and adds the results to acc.
void simulate_chunk(const Config& cfg, const BinLayout& layout, uint64_t chunk,
                    Accumulator& acc);

}  // namespace mc
