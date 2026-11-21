// Tests for accumulator.cpp: merging adds everything correctly, and splitting
// chunks between "workers" then merging gives exactly the single-worker result.
// That second test is a preview of design decision D2 (needed in Phases 2-3).

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/accumulator.hpp"
#include "core/simulation.hpp"
#include "test_util.hpp"

static void test_manual_merge() {
    mc::Accumulator a(2, 4), b(2, 4);
    CHECK(a.hist.size() == 8);
    CHECK(a.month_sum.size() == 2);

    a.month_hist(1)[0] = 3;
    a.month_hist(2)[3] = 1;
    a.month_sum[0] = 10.0;
    a.add_final(5.0, 10.0);   // a loss
    a.add_final(20.0, 10.0);  // a gain
    a.overflow_low = 1;

    b.month_hist(1)[0] = 2;
    b.month_hist(2)[1] = 7;
    b.month_sum[0] = 2.5;
    b.add_final(1.0, 10.0);   // a loss, new minimum
    b.overflow_high = 4;

    a.merge(b);
    CHECK(a.month_hist(1)[0] == 5);
    CHECK(a.month_hist(2)[1] == 7);
    CHECK(a.month_hist(2)[3] == 1);
    CHECK_NEAR(a.month_sum[0], 12.5, 1e-12);
    CHECK(a.trials == 3);
    CHECK_NEAR(a.final_sum, 26.0, 1e-12);
    CHECK_NEAR(a.final_sum_sq, 25.0 + 400.0 + 1.0, 1e-12);
    CHECK(a.final_min == 1.0);
    CHECK(a.final_max == 20.0);
    CHECK(a.loss_count == 2);
    CHECK(a.overflow_low == 1);
    CHECK(a.overflow_high == 4);

    // Merging an empty accumulator changes nothing (its min/max must be ignored).
    mc::Accumulator empty(2, 4);
    a.merge(empty);
    CHECK(a.trials == 3);
    CHECK(a.final_min == 1.0);

    // Merging INTO an empty accumulator copies min/max.
    mc::Accumulator fresh(2, 4);
    fresh.merge(a);
    CHECK(fresh.final_min == 1.0 && fresh.final_max == 20.0);
}

static void test_split_equals_whole() {
    mc::Config cfg;
    cfg.months = 12;
    cfg.bins = 256;
    cfg.trials = 10 * mc::CHUNK_SIZE + 123;  // last chunk is partial on purpose
    const mc::BinLayout layout = mc::make_bin_layout(cfg);
    const uint64_t chunks = mc::num_chunks(cfg.trials);
    CHECK(chunks == 11);

    // One worker does every chunk.
    mc::Accumulator whole(cfg.months, cfg.bins);
    for (uint64_t c = 0; c < chunks; ++c) mc::simulate_chunk(cfg, layout, c, whole);

    // Two "workers" take even and odd chunks, then merge (in the other order).
    mc::Accumulator even(cfg.months, cfg.bins), odd(cfg.months, cfg.bins);
    for (uint64_t c = 0; c < chunks; ++c) {
        mc::simulate_chunk(cfg, layout, c, (c % 2 == 0) ? even : odd);
    }
    odd.merge(even);

    CHECK(whole.trials == cfg.trials);
    CHECK(odd.trials == cfg.trials);
    CHECK(odd.hist == whole.hist);  // integer counts: bit-identical
    CHECK(odd.loss_count == whole.loss_count);
    CHECK(odd.overflow_low == whole.overflow_low);
    CHECK(odd.overflow_high == whole.overflow_high);
    CHECK(odd.final_min == whole.final_min);
    CHECK(odd.final_max == whole.final_max);
    // Floating sums: same numbers added in a different order -> tiny rounding difference.
    CHECK_NEAR(odd.final_sum, whole.final_sum, 1e-12 * whole.final_sum);
    CHECK_NEAR(odd.final_sum_sq, whole.final_sum_sq, 1e-12 * whole.final_sum_sq);
    CHECK_NEAR(odd.month_sum[5], whole.month_sum[5], 1e-12 * whole.month_sum[5]);
}

// chunk_block must give every chunk to exactly one rank, in order, with sizes
// differing by at most 1 — for even splits, uneven splits and more ranks than chunks.
static void test_chunk_block() {
    const uint64_t totals[] = {0, 1, 2, 13, 100, 245};
    const int worker_counts[] = {1, 2, 3, 4, 7, 16};
    for (uint64_t total : totals) {
        for (int workers : worker_counts) {
            uint64_t expected_first = 0, min_count = total, max_count = 0;
            bool contiguous = true;
            for (int r = 0; r < workers; ++r) {
                uint64_t first = 0, count = 0;
                mc::chunk_block(total, workers, r, first, count);
                if (first != expected_first) contiguous = false;
                expected_first += count;
                min_count = std::min(min_count, count);
                max_count = std::max(max_count, count);
            }
            CHECK(contiguous);
            CHECK(expected_first == total);       // every chunk assigned exactly once
            CHECK(max_count - min_count <= 1);    // balanced
        }
    }
    // The case in the consistency test: 13 chunks on 3 ranks -> 5, 4, 4.
    uint64_t first = 0, count = 0;
    mc::chunk_block(13, 3, 0, first, count);
    CHECK(first == 0 && count == 5);
    mc::chunk_block(13, 3, 2, first, count);
    CHECK(first == 9 && count == 4);
}

int main() {
    test_manual_merge();
    test_split_equals_whole();
    test_chunk_block();
    return test::summary("test_accumulator");
}
