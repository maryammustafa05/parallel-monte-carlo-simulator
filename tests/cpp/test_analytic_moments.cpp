// The key correctness test of the engine (CLAUDE.md section 4.4).
//
// Monthly growth factors (1 + r_t) are independent with mean a = 1 + mu_m and
// variance s^2 = sigma_m^2, so (ignoring the -100 % clamp, negligible here):
//   E[V_T]   = V0   * a^T
//   E[V_T^2] = V0^2 * (a^2 + s^2)^T
//   E[V_T^4] = V0^4 * (a^4 + 6 a^2 s^2 + 3 s^4)^T     (used for the error bar of E[V_T^2])
// The simulated values must match within 4 standard errors.

#include <cmath>
#include <cstdio>

#include "core/accumulator.hpp"
#include "core/simulation.hpp"
#include "test_util.hpp"

static void check_moments(const mc::Config& cfg) {
    const mc::BinLayout layout = mc::make_bin_layout(cfg);
    mc::Accumulator acc(cfg.months, cfg.bins);
    const uint64_t chunks = mc::num_chunks(cfg.trials);
    for (uint64_t c = 0; c < chunks; ++c) mc::simulate_chunk(cfg, layout, c, acc);

    const double n = static_cast<double>(acc.trials);
    const double a = 1.0 + mc::monthly_mu(cfg);
    const double s2 = mc::monthly_sigma(cfg) * mc::monthly_sigma(cfg);
    const double T = cfg.months;
    const double V0 = cfg.initial;

    const double e1 = V0 * std::pow(a, T);
    const double e2 = V0 * V0 * std::pow(a * a + s2, T);
    const double e4 = std::pow(V0, 4) * std::pow(a * a * a * a + 6 * a * a * s2 + 3 * s2 * s2, T);

    const double sim_e1 = acc.final_sum / n;
    const double sim_e2 = acc.final_sum_sq / n;
    const double se1 = std::sqrt((e2 - e1 * e1) / n);  // standard error of the mean
    const double se2 = std::sqrt((e4 - e2 * e2) / n);  // standard error of mean(V^2)

    std::printf("  mu=%.2f sigma=%.2f T=%d N=%llu\n", cfg.mu, cfg.sigma, cfg.months,
                static_cast<unsigned long long>(cfg.trials));
    std::printf("    E[V_T]   formula %.4f  simulated %.4f  diff = %+.2f SE\n", e1, sim_e1,
                (sim_e1 - e1) / se1);
    std::printf("    E[V_T^2] formula %.6g  simulated %.6g  diff = %+.2f SE\n", e2, sim_e2,
                (sim_e2 - e2) / se2);

    CHECK(acc.trials == cfg.trials);
    CHECK_NEAR(sim_e1, e1, 4 * se1);
    CHECK_NEAR(sim_e2, e2, 4 * se2);

    // The monthly mean (used in fan_chart.csv) must follow the same formula, e.g. at T/2.
    const int half = cfg.months / 2;
    const double e1_half = V0 * std::pow(a, half);
    const double e2_half = V0 * V0 * std::pow(a * a + s2, half);
    const double se_half = std::sqrt((e2_half - e1_half * e1_half) / n);
    CHECK_NEAR(acc.month_sum[half - 1] / n, e1_half, 4 * se_half);
}

int main() {
    mc::Config base;  // defaults: V0 10000, mu 7 %, sigma 15 %, 120 months
    base.trials = 400000;
    check_moments(base);

    mc::Config risky;  // a more volatile, shorter case
    risky.mu = 0.12;
    risky.sigma = 0.30;
    risky.months = 36;
    risky.trials = 200000;
    risky.seed = 7;
    check_moments(risky);

    return test::summary("test_analytic_moments");
}
