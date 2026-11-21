// Tests for config.cpp: command-line parsing and input validation.
// Several rejected inputs below crashed the program or wrote inf/nan into the
// results before Phase 7; the comments say which.

#include <string>
#include <vector>

#include "core/config.hpp"
#include "test_util.hpp"

// Parses `args` as if typed after the program name. Returns the result; fills cfg/error.
static mc::ParseResult parse(std::vector<std::string> args, mc::Config& cfg, std::string& error) {
    args.insert(args.begin(), "mc_test");
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(a.data());
    return mc::parse_args(static_cast<int>(argv.size()), argv.data(), cfg, error);
}

static bool accepted(const std::vector<std::string>& args) {
    mc::Config cfg;
    std::string error;
    return parse(args, cfg, error) == mc::ParseResult::Ok;
}

static void test_defaults_and_values() {
    mc::Config cfg;
    std::string error;
    CHECK(parse({}, cfg, error) == mc::ParseResult::Ok);
    CHECK(cfg.initial == 10000.0 && cfg.mu == 0.07 && cfg.sigma == 0.15);
    CHECK(cfg.months == 120 && cfg.trials == 1000000 && cfg.seed == 42);
    CHECK(cfg.confidence == 0.95 && cfg.threads == 1 && cfg.bins == 2048 && !cfg.exact);

    mc::Config c2;
    CHECK(parse({"--initial", "2500.5", "--mu", "-0.2", "--sigma", "0", "--months", "1",
                 "--trials", "1e6", "--confidence", "0.99", "--seed", "18446744073709551615",
                 "--threads", "4", "--bins", "256", "--out", "x/y", "--exact"},
                c2, error) == mc::ParseResult::Ok);
    CHECK(c2.initial == 2500.5 && c2.mu == -0.2 && c2.sigma == 0.0 && c2.months == 1);
    CHECK(c2.trials == 1000000);                    // "1e6" accepted
    CHECK(c2.seed == 18446744073709551615ULL);      // full 64-bit seed range
    CHECK(c2.confidence == 0.99 && c2.threads == 4 && c2.bins == 256);
    CHECK(c2.out == "x/y" && c2.exact);

    mc::Config c3;
    CHECK(parse({"--help"}, c3, error) == mc::ParseResult::Help);
}

static void test_malformed_input() {
    CHECK(!accepted({"--trials"}));                  // missing value
    CHECK(!accepted({"--trials", "abc"}));
    CHECK(!accepted({"--trials", "12abc"}));
    CHECK(!accepted({"--trials", "-5"}));            // would wrap to a huge number if not checked
    CHECK(!accepted({"--trials", "1.5"}));           // not a whole number
    CHECK(!accepted({"--seed", "18446744073709551616"}));  // 2^64: one too many
    CHECK(!accepted({"--mu", "nan"}));
    CHECK(!accepted({"--sigma", "inf"}));
    CHECK(!accepted({"--months", "3.5"}));
    CHECK(!accepted({"--out", ""}));
    CHECK(!accepted({"--unknown", "1"}));

    mc::Config cfg;
    std::string error;
    parse({"--trials", "abc"}, cfg, error);
    CHECK(error.find("--trials") != std::string::npos);  // message names the flag
}

static void test_ranges() {
    // Edge values that must be accepted.
    CHECK(accepted({"--trials", "1"}));
    CHECK(accepted({"--months", "1"}));
    CHECK(accepted({"--months", "1200"}));
    CHECK(accepted({"--sigma", "0"}));
    CHECK(accepted({"--mu", "-0.99"}));
    CHECK(accepted({"--bins", "2"}));
    CHECK(accepted({"--exact", "--trials", "10000000"}));

    // Out of range: each of these used to crash, hang or write inf/nan.
    CHECK(!accepted({"--initial", "0"}));
    CHECK(!accepted({"--initial", "-100"}));
    CHECK(!accepted({"--initial", "1e305"}));       // overflowed to inf
    CHECK(!accepted({"--mu", "-1"}));
    CHECK(!accepted({"--mu", "-12"}));              // log(0) -> crash (segfault)
    CHECK(!accepted({"--mu", "1000"}));             // overflowed to inf
    CHECK(!accepted({"--sigma", "-0.1"}));
    CHECK(!accepted({"--sigma", "11"}));
    CHECK(!accepted({"--months", "0"}));
    CHECK(!accepted({"--months", "1201"}));
    CHECK(!accepted({"--months", "100000000"}));    // out of memory -> crash
    CHECK(!accepted({"--trials", "0"}));
    CHECK(!accepted({"--confidence", "0"}));
    CHECK(!accepted({"--confidence", "1"}));
    CHECK(!accepted({"--threads", "0"}));
    CHECK(!accepted({"--threads", "257"}));
    CHECK(!accepted({"--bins", "1"}));
    CHECK(!accepted({"--bins", "65537"}));
    CHECK(!accepted({"--exact", "--trials", "10000001"}));

    // Allowed one by one, but together they could exceed a double (~1e308).
    CHECK(accepted({"--sigma", "10", "--months", "120"}));
    CHECK(!accepted({"--sigma", "10", "--months", "1200"}));
}

int main() {
    test_defaults_and_values();
    test_malformed_input();
    test_ranges();
    return test::summary("test_config");
}
