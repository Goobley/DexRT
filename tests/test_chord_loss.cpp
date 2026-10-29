#include "catch_amalgamated.hpp"
#include "RayMarching.hpp"
#include <cmath>

using Catch::Matchers::WithinRel;

namespace {
/// Mean of exp(-2 tau x) over x in [0, 1] (the full chord, in units of the
/// face to centre tau), by composite Simpson in f64
f64 numerical_chord_mean_transmission(f64 tau) {
    constexpr int n = 200000;
    const f64 h = 1.0 / n;
    f64 sum = 1.0 + std::exp(-2.0 * tau);
    for (int i = 1; i < n; ++i) {
        sum += ((i % 2) ? 4.0 : 2.0) * std::exp(-2.0 * tau * i * h);
    }
    return sum * h / 3.0;
}
}

TEST_CASE("Chord mean transmission", "[rad_loss]") {
    SECTION("Matches the numerical chord average") {
        for (f64 log_tau = -8.0; log_tau <= 2.0; log_tau += 0.5) {
            const f64 tau = std::pow(10.0, log_tau);
            const f64 expected = numerical_chord_mean_transmission(tau);
            REQUIRE_THAT(f64(chord_mean_transmission(fp_t(tau))), WithinRel(expected, 1e-6));
        }
    }
    SECTION("Limits") {
        REQUIRE(chord_mean_transmission(FP(0.0)) == FP(1.0));
        REQUIRE_THAT(f64(chord_mean_transmission(FP(1e-30))), WithinRel(1.0, 1e-7));
        // thick: 1 / (2 tau)
        for (f64 tau : {1e2, 1e4, 1e6}) {
            REQUIRE_THAT(f64(chord_mean_transmission(fp_t(tau))), WithinRel(0.5 / tau, 1e-6));
        }
    }
    SECTION("Chord loss is the change in intensity across the cell") {
        // chi tr_chord (S - I_f) * 2h == I_exit - I_f for constant material
        const f64 chi = 3.0, S = 2.0, I_f = 0.5;
        for (f64 h : {1e-6, 0.01, 0.4, 5.0}) {
            const f64 tau = chi * h;
            const f64 loss = chi * chord_mean_transmission(fp_t(tau)) * (S - I_f);
            const f64 I_exit = S + (I_f - S) * std::exp(-2.0 * tau);
            REQUIRE_THAT(loss * 2.0 * h, WithinRel(I_exit - I_f, 1e-6));
        }
    }
}
