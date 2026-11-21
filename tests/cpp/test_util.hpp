#pragma once
// A tiny test helper so we don't need GoogleTest.
// Usage: CHECK(cond); CHECK_NEAR(a, b, tol); ... return test::summary("name");

#include <cmath>
#include <cstdio>

namespace test {

// Number of failed checks in this test program.
inline int& failures() {
    static int count = 0;
    return count;
}

// Number of checks run in this test program.
inline int& checks() {
    static int count = 0;
    return count;
}

// Prints a one-line result and returns the process exit code (0 = all passed).
inline int summary(const char* name) {
    std::printf("[%s] %d checks, %d failed -> %s\n", name, checks(), failures(),
                failures() == 0 ? "PASS" : "FAIL");
    return failures() == 0 ? 0 : 1;
}

}  // namespace test

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++test::checks();                                                        \
        if (!(cond)) {                                                           \
            ++test::failures();                                                  \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                    \
    do {                                                                         \
        ++test::checks();                                                        \
        const double a_ = (a), b_ = (b), tol_ = (tol);                           \
        if (!(std::fabs(a_ - b_) <= tol_)) {                                     \
            ++test::failures();                                                  \
            std::printf("  FAIL %s:%d: |%s - %s| = |%.12g - %.12g| > %.3g\n",    \
                        __FILE__, __LINE__, #a, #b, a_, b_, tol_);               \
        }                                                                        \
    } while (0)
