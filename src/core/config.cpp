#include "config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>

namespace mc {

namespace {

// Parses a whole string as a finite double. Rejects "12abc", "", "nan", "inf".
bool parse_double(const std::string& text, double& value) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (errno != 0 || *end != '\0' || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
}

// Parses a whole string as a signed integer (e.g. "120", "-5").
bool parse_int(const std::string& text, long long& value) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long long parsed = std::strtoll(text.c_str(), &end, 10);
    if (errno != 0 || *end != '\0') return false;
    value = parsed;
    return true;
}

// Parses a non-negative count. Accepts plain integers up to the full 64-bit range
// ("1000000", "18446744073709551615") and, for convenience, whole numbers in
// scientific notation ("1e6").
bool parse_count(const std::string& text, uint64_t& value) {
    if (text.empty() || text[0] == '-' || text[0] == '+') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long as_int = std::strtoull(text.c_str(), &end, 10);
    if (*end == '\0') {
        if (errno != 0) return false;  // larger than 2^64 - 1
        value = static_cast<uint64_t>(as_int);
        return true;
    }
    double as_double = 0.0;
    if (parse_double(text, as_double) && as_double >= 0.0 && as_double <= 1e18 &&
        std::floor(as_double) == as_double) {
        value = static_cast<uint64_t>(as_double);
        return true;
    }
    return false;
}

// Parses an int flag value and checks it fits in an int.
bool parse_small_int(const std::string& text, int& value) {
    long long parsed = 0;
    if (!parse_int(text, parsed) || parsed < -1000000000LL || parsed > 1000000000LL) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

}  // namespace

ParseResult parse_args(int argc, char** argv, Config& cfg, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];

        if (flag == "--help" || flag == "-h") return ParseResult::Help;
        if (flag == "--exact") {
            cfg.exact = true;
            continue;
        }

        // Every other flag needs a value after it.
        if (i + 1 >= argc) {
            error = "missing value after " + flag;
            return ParseResult::Error;
        }
        const std::string value = argv[++i];
        bool ok = true;

        if (flag == "--initial") {
            ok = parse_double(value, cfg.initial);
        } else if (flag == "--mu") {
            ok = parse_double(value, cfg.mu);
        } else if (flag == "--sigma") {
            ok = parse_double(value, cfg.sigma);
        } else if (flag == "--months") {
            ok = parse_small_int(value, cfg.months);
        } else if (flag == "--trials") {
            ok = parse_count(value, cfg.trials);
        } else if (flag == "--confidence") {
            ok = parse_double(value, cfg.confidence);
        } else if (flag == "--seed") {
            ok = parse_count(value, cfg.seed);
        } else if (flag == "--threads") {
            ok = parse_small_int(value, cfg.threads);
        } else if (flag == "--bins") {
            ok = parse_small_int(value, cfg.bins);
        } else if (flag == "--out") {
            cfg.out = value;
            ok = !value.empty();
        } else {
            error = "unknown flag " + flag;
            return ParseResult::Error;
        }

        if (!ok) {
            error = "invalid value '" + value + "' for " + flag;
            return ParseResult::Error;
        }
    }

    return validate(cfg, error) ? ParseResult::Ok : ParseResult::Error;
}

// ln of the largest value one trial could plausibly reach: every month a
// 6-standard-deviation good month. Used to refuse inputs that would overflow
// double precision (about 1.8e308, ln = 709) and write "inf" into the results.
static double worst_case_log_value(const Config& cfg) {
    const double best_month = 1.0 + cfg.mu / 12.0 + 6.0 * cfg.sigma / std::sqrt(12.0);
    return std::log(cfg.initial) + cfg.months * std::log(best_month);
}

bool validate(const Config& cfg, std::string& error) {
    // Upper limits keep the program away from crashes (memory) and from numbers
    // too large for a double; they are far beyond any realistic investment.
    if (!(cfg.initial > 0.0 && cfg.initial <= MAX_INITIAL)) {
        error = "--initial must be > 0 and <= 1e12";
    } else if (!(cfg.mu > -1.0 && cfg.mu <= MAX_MU)) {
        // mu = -1 means losing 100 % per year on average; below that the model breaks.
        error = "--mu must be > -1 (-100 % per year) and <= 10 (1000 % per year)";
    } else if (!(cfg.sigma >= 0.0 && cfg.sigma <= MAX_SIGMA)) {
        error = "--sigma must be >= 0 and <= 10 (1000 % per year)";
    } else if (cfg.months < 1 || cfg.months > MAX_MONTHS) {
        error = "--months must be between 1 and 1200 (100 years)";
    } else if (cfg.trials < 1) {
        error = "--trials must be >= 1";
    } else if (!(cfg.confidence > 0.0 && cfg.confidence < 1.0)) {
        error = "--confidence must be strictly between 0 and 1";
    } else if (cfg.threads < 1 || cfg.threads > MAX_THREADS) {
        error = "--threads must be between 1 and 256";
    } else if (cfg.bins < 2 || cfg.bins > MAX_BINS) {
        error = "--bins must be between 2 and 65536";
    } else if (worst_case_log_value(cfg) > 700.0) {
        error = "--initial, --mu, --sigma and --months together could produce values too "
                "large for a double (> 1e304); use a smaller horizon or volatility";
    } else if (cfg.exact && cfg.trials > MAX_EXACT_TRIALS) {
        error = "--exact keeps every final value in memory; use --trials <= " +
                std::to_string(MAX_EXACT_TRIALS);
    } else {
        return true;
    }
    return false;
}

std::string usage(const std::string& program_name) {
    return "Usage: " + program_name + " [options]\n"
           "\n"
           "  --initial X      initial investment V0            (default 10000)\n"
           "  --mu X           expected annual return, 0.07=7%  (default 0.07)\n"
           "  --sigma X        annual volatility, 0.15=15%      (default 0.15)\n"
           "  --months N       horizon in months                (default 120)\n"
           "  --trials N       number of simulated futures      (default 1000000)\n"
           "  --confidence X   VaR confidence level, 0<X<1      (default 0.95)\n"
           "  --seed N         global random seed               (default 42)\n"
           "  --threads N      OpenMP threads per process       (default 1, hybrid only)\n"
           "  --bins N         histogram bins per month         (default 2048)\n"
           "  --out DIR        output directory                 (default results/run)\n"
           "  --exact          also compute exact percentiles from all final values\n"
           "                   (validation only, needs --trials <= 10000000)\n"
           "  --help           show this text\n";
}

}  // namespace mc
