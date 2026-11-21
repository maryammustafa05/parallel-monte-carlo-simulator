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

// Parses a non-negative count. Accepts plain integers ("1000000") and, for
// convenience, whole numbers in scientific notation ("1e6").
bool parse_count(const std::string& text, uint64_t& value) {
    if (text.empty() || text[0] == '-') return false;
    long long as_int = 0;
    if (parse_int(text, as_int)) {
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

bool validate(const Config& cfg, std::string& error) {
    if (!(cfg.initial > 0.0)) {
        error = "--initial must be > 0";
    } else if (cfg.months < 1) {
        error = "--months must be >= 1";
    } else if (cfg.trials < 1) {
        error = "--trials must be >= 1";
    } else if (!(cfg.sigma >= 0.0)) {
        error = "--sigma must be >= 0";
    } else if (!(cfg.confidence > 0.0 && cfg.confidence < 1.0)) {
        error = "--confidence must be strictly between 0 and 1";
    } else if (cfg.threads < 1) {
        error = "--threads must be >= 1";
    } else if (cfg.bins < 2) {
        error = "--bins must be >= 2";
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
           "  --exact          keep all final values for validation (Phase 4)\n"
           "  --help           show this text\n";
}

}  // namespace mc
