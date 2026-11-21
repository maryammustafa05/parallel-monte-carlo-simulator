// Tests for rng.hpp: determinism, uniform range, and Normal(0,1) shape.
// Statistical checks use a tolerance of 4 standard errors (fails by chance
// with probability ~0.006 %, and the seed is fixed so the result is repeatable).

#include <cmath>
#include <cstdint>

#include "core/rng.hpp"
#include "test_util.hpp"

static void test_determinism() {
    mc::Xoshiro256ss a(123), b(123), c(124);
    bool same = true, differs = false;
    for (int i = 0; i < 1000; ++i) {
        const uint64_t x = a.next();
        if (x != b.next()) same = false;
        if (x != c.next()) differs = true;
    }
    CHECK(same);     // same seed -> same sequence
    CHECK(differs);  // different seed -> different sequence

    mc::NormalGenerator n1(7), n2(7);
    bool normals_same = true;
    for (int i = 0; i < 1001; ++i) {  // odd count also exercises the "spare" value
        if (n1.next() != n2.next()) normals_same = false;
    }
    CHECK(normals_same);
}

static void test_chunk_seeds() {
    CHECK(mc::chunk_seed(42, 5) == mc::chunk_seed(42, 5));
    CHECK(mc::chunk_seed(42, 0) != mc::chunk_seed(42, 1));
    CHECK(mc::chunk_seed(42, 3) != mc::chunk_seed(43, 3));
    CHECK(mc::chunk_seed(0, 0) != 0);  // no degenerate all-zero seed
}

static void test_uniform() {
    mc::Xoshiro256ss rng(2024);
    const int N = 1000000;
    double sum = 0.0;
    bool in_range = true;
    for (int i = 0; i < N; ++i) {
        const double u = rng.uniform();
        const double v = rng.uniform_open_zero();
        if (!(u >= 0.0 && u < 1.0)) in_range = false;
        if (!(v > 0.0 && v <= 1.0)) in_range = false;
        sum += u;
    }
    CHECK(in_range);
    const double se = std::sqrt(1.0 / 12.0 / N);  // std of a uniform is 1/sqrt(12)
    CHECK_NEAR(sum / N, 0.5, 4 * se);
}

static void test_normal() {
    mc::NormalGenerator normal(99);
    const int N = 1000000;
    double sum = 0.0, sum_sq = 0.0;
    int inside_196 = 0;
    for (int i = 0; i < N; ++i) {
        const double z = normal.next();
        sum += z;
        sum_sq += z * z;
        if (std::fabs(z) < 1.959963984540054) ++inside_196;
    }
    const double mean = sum / N;
    const double var = sum_sq / N - mean * mean;
    const double frac = static_cast<double>(inside_196) / N;
    std::printf("  normal: mean %.5f  variance %.5f  P(|Z|<1.96) %.5f  (N = %d)\n", mean, var,
                frac, N);
    CHECK_NEAR(mean, 0.0, 4.0 / std::sqrt(N));                    // SE of mean = 1/sqrt(N)
    CHECK_NEAR(var, 1.0, 4.0 * std::sqrt(2.0 / N));               // SE of variance = sqrt(2/N)
    CHECK_NEAR(frac, 0.95, 4.0 * std::sqrt(0.95 * 0.05 / N));     // binomial SE
}

int main() {
    test_determinism();
    test_chunk_seeds();
    test_uniform();
    test_normal();
    return test::summary("test_rng");
}
