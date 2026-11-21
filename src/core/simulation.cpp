#include "simulation.hpp"

#include <algorithm>
#include <cmath>

#include "rng.hpp"

namespace mc {

double monthly_mu(const Config& cfg) { return cfg.mu / 12.0; }

double monthly_sigma(const Config& cfg) { return cfg.sigma / std::sqrt(12.0); }

BinLayout make_bin_layout(const Config& cfg) {
    BinLayout layout;
    layout.months = cfg.months;
    layout.bins = cfg.bins;
    layout.log_lo.resize(cfg.months);
    layout.log_width.resize(cfg.months);
    layout.inv_width.resize(cfg.months);

    // ln(V_t) is a sum of t independent terms ln(1 + r). Approximate one term's
    // mean and standard deviation (first-order Taylor expansion around 1 + mu_m).
    const double mu_m = monthly_mu(cfg);
    const double sigma_m = monthly_sigma(cfg);
    const double growth = 1.0 + mu_m;
    const double drift = std::log(growth) - 0.5 * (sigma_m * sigma_m) / (growth * growth);
    const double step_sd = sigma_m / growth;

    // With sigma = 0 every trial has the same value and the range would have
    // zero width. A tiny minimum half-width (0.1 % in log space) avoids that.
    const double MIN_HALF_WIDTH = 1e-3;
    const double NUM_SD = 6.0;

    for (int t = 1; t <= cfg.months; ++t) {
        const double centre = std::log(cfg.initial) + t * drift;
        const double half = std::max(NUM_SD * step_sd * std::sqrt(static_cast<double>(t)),
                                     MIN_HALF_WIDTH);
        const double width = (2.0 * half) / cfg.bins;
        layout.log_lo[t - 1] = centre - half;
        layout.log_width[t - 1] = width;
        layout.inv_width[t - 1] = 1.0 / width;
    }
    return layout;
}

uint64_t num_chunks(uint64_t trials) { return (trials + CHUNK_SIZE - 1) / CHUNK_SIZE; }

void chunk_block(uint64_t total_chunks, int workers, int rank, uint64_t& first,
                 uint64_t& count) {
    const uint64_t p = static_cast<uint64_t>(workers);
    const uint64_t r = static_cast<uint64_t>(rank);
    const uint64_t base = total_chunks / p;
    const uint64_t extra = total_chunks % p;  // this many ranks get one more chunk
    count = base + (r < extra ? 1 : 0);
    // Ranks before r each took `base` chunks, and min(r, extra) of them took one more.
    first = r * base + std::min(r, extra);
}

int bin_index(const BinLayout& layout, int t, double v, Accumulator& acc) {
    // A value of 0 (a total loss after a -100 % month) has no logarithm.
    if (v <= 0.0) {
        ++acc.month_zero[t - 1];
        return -1;
    }
    const double x = (std::log(v) - layout.log_lo[t - 1]) * layout.inv_width[t - 1];
    // Written as !(x >= 0) rather than x < 0 so that a NaN also lands here
    // instead of becoming an out-of-range array index.
    if (!(x >= 0.0)) {
        ++acc.overflow_low;
        return 0;
    }
    if (x >= layout.bins) {
        ++acc.overflow_high;
        return layout.bins - 1;
    }
    return static_cast<int>(x);
}

void simulate_chunk(const Config& cfg, const BinLayout& layout, uint64_t chunk,
                    Accumulator& acc) {
    const uint64_t first = chunk * CHUNK_SIZE;
    const uint64_t last = std::min(cfg.trials, first + CHUNK_SIZE);  // last chunk may be short

    // The chunk's random numbers depend only on (seed, chunk): see rng.hpp.
    NormalGenerator normal(chunk_seed(cfg.seed, chunk));
    const double mu_m = monthly_mu(cfg);
    const double sigma_m = monthly_sigma(cfg);

    for (uint64_t trial = first; trial < last; ++trial) {
        double value = cfg.initial;
        for (int t = 1; t <= cfg.months; ++t) {
            double r = mu_m + sigma_m * normal.next();
            if (r < -1.0) r = -1.0;  // you cannot lose more than 100 % of your money
            value *= 1.0 + r;

            acc.month_sum[t - 1] += value;
            const int bin = bin_index(layout, t, value, acc);
            if (bin >= 0) acc.month_hist(t)[bin] += 1;
        }
        acc.add_final(value, cfg.initial);
    }
}

}  // namespace mc
