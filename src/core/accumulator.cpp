#include "accumulator.hpp"

#include <algorithm>
#include <stdexcept>

namespace mc {

Accumulator::Accumulator(int months_, int bins_)
    : months(months_),
      bins(bins_),
      hist(static_cast<size_t>(months_) * static_cast<size_t>(bins_), 0),
      month_sum(static_cast<size_t>(months_), 0.0),
      month_zero(static_cast<size_t>(months_), 0) {}

void Accumulator::add_final(double value, double initial) {
    // The first trial sets min/max; after that we compare.
    if (trials == 0) {
        final_min = value;
        final_max = value;
    } else {
        final_min = std::min(final_min, value);
        final_max = std::max(final_max, value);
    }
    ++trials;
    final_sum += value;
    final_sum_sq += value * value;
    if (value < initial) ++loss_count;
    if (keep_final_values) final_values.push_back(value);
}

void Accumulator::merge(const Accumulator& other) {
    if (other.months != months || other.bins != bins) {
        throw std::invalid_argument("Accumulator::merge: shapes differ");
    }
    if (other.trials == 0) return;  // nothing to add (and its min/max are meaningless)

    for (size_t i = 0; i < hist.size(); ++i) hist[i] += other.hist[i];
    for (size_t i = 0; i < month_sum.size(); ++i) month_sum[i] += other.month_sum[i];
    for (size_t i = 0; i < month_zero.size(); ++i) month_zero[i] += other.month_zero[i];

    if (trials == 0) {
        final_min = other.final_min;
        final_max = other.final_max;
    } else {
        final_min = std::min(final_min, other.final_min);
        final_max = std::max(final_max, other.final_max);
    }
    trials += other.trials;
    final_sum += other.final_sum;
    final_sum_sq += other.final_sum_sq;
    loss_count += other.loss_count;
    overflow_low += other.overflow_low;
    overflow_high += other.overflow_high;
    final_values.insert(final_values.end(), other.final_values.begin(),
                        other.final_values.end());
}

uint64_t Accumulator::checksum() const {
    // FNV-1a: for each byte, XOR it into the hash and multiply by a fixed prime.
    uint64_t hash = 0xCBF29CE484222325ULL;
    auto mix = [&hash](uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (8 * byte)) & 0xFFULL;
            hash *= 0x100000001B3ULL;
        }
    };
    for (uint64_t count : hist) mix(count);
    mix(trials);
    mix(loss_count);
    mix(overflow_low);
    mix(overflow_high);
    // Zero counts are mixed in only where they are non-zero, so runs without any
    // wiped-out trials keep exactly the same checksum as before this was added.
    for (size_t i = 0; i < month_zero.size(); ++i) {
        if (month_zero[i] != 0) {
            mix(i);
            mix(month_zero[i]);
        }
    }
    return hash;
}

}  // namespace mc
