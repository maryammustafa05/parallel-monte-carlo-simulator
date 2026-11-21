#pragma once
// Random number generation for the simulator (design decision D3).
//
// Why not std::mt19937 + std::normal_distribution?
//   The C++ standard does not fix the algorithm behind std::normal_distribution,
//   so different compilers can give different numbers for the same seed. We want
//   results that are reproducible everywhere, so we write our own small RNG:
//     splitmix64   -> turns one 64-bit seed into well-mixed starting state
//     xoshiro256** -> fast, high-quality uniform 64-bit random numbers
//     Box-Muller   -> turns two uniforms into two Normal(0,1) values

#include <cmath>
#include <cstdint>

namespace mc {

// Advances a splitmix64 state and returns the next mixed 64-bit value.
inline uint64_t splitmix64_next(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed for chunk `chunk` of the run (design decision D2).
// Every chunk gets its own seed computed only from (global_seed, chunk), so it
// produces the same numbers no matter which process or thread runs it.
inline uint64_t chunk_seed(uint64_t global_seed, uint64_t chunk) {
    uint64_t chunk_state = chunk;
    const uint64_t chunk_hash = splitmix64_next(chunk_state);
    uint64_t mixed = global_seed ^ chunk_hash;
    return splitmix64_next(mixed);
}

// xoshiro256** uniform random generator (Blackman & Vigna, 2018).
class Xoshiro256ss {
public:
    // Fills the 256-bit state from one 64-bit seed using splitmix64.
    explicit Xoshiro256ss(uint64_t seed) {
        uint64_t sm = seed;
        for (uint64_t& word : s_) {
            word = splitmix64_next(sm);
        }
    }

    // Returns the next uniformly distributed 64-bit integer.
    uint64_t next() {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Returns a uniform double in [0, 1) using the top 53 bits (a double's precision).
    double uniform() {
        return static_cast<double>(next() >> 11) * INV_2_POW_53;
    }

    // Returns a uniform double in (0, 1]; never 0, so log() of it is always finite.
    double uniform_open_zero() {
        return static_cast<double>((next() >> 11) + 1) * INV_2_POW_53;
    }

private:
    static constexpr double INV_2_POW_53 = 1.0 / 9007199254740992.0;  // 1 / 2^53

    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t s_[4];
};

// Standard normal generator using the Box-Muller transform.
// Each transform makes TWO independent normals; we keep the second one
// ("spare") for the next call instead of throwing it away (twice as fast).
class NormalGenerator {
public:
    // Creates a generator whose whole sequence is fixed by `seed`.
    explicit NormalGenerator(uint64_t seed) : rng_(seed) {}

    // Returns one Normal(0, 1) sample.
    double next() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        const double u1 = rng_.uniform_open_zero();  // (0,1]: avoids log(0)
        const double u2 = rng_.uniform();
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = TWO_PI * u2;
        spare_ = radius * std::sin(angle);
        has_spare_ = true;
        return radius * std::cos(angle);
    }

private:
    static constexpr double TWO_PI = 6.283185307179586476925286766559;

    Xoshiro256ss rng_;
    bool has_spare_ = false;
    double spare_ = 0.0;
};

}  // namespace mc
