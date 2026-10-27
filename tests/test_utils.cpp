#include "catch_amalgamated.hpp"
#include "Utils.hpp"
#include "CompensatedSum.hpp"
#include "Rad16BitStorage.hpp"
#include <random>
#include <limits>

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Interp", "[interp]") {
    constexpr int n = 10;
    namespace Const = ConstantsFP;
    constexpr fp_t two_pi = FP(2.0) * Const::pi;
    {
        yakl::Array<fp_t, 1, yakl::memHost> x("x", n);
        yakl::Array<fp_t, 1, yakl::memHost> y("y", n);

        for (int i = 0; i < n; ++i) {
            x(i) = fp_t(i) * two_pi / fp_t(n - 1);
            y(i) = std::sin(x(i));
        }

        fp_t sample = interp(FP(5.5) * two_pi / fp_t(n-1), x, y);
        REQUIRE_THAT(sample, WithinRel(FP(0.5) * (y(5) + y(6)), FP(1e-6)));
        sample = interp(FP(100.0), x, y);
        REQUIRE(sample == y(y.extent(0) - 1));
        sample = interp(FP(-1000.0), x, y);
        REQUIRE(sample == y(0));

        constexpr fp_t samples[n] = {
            FP(-400.0), FP(0.0), FP(1.12), FP(2.0), FP(5.7),
            FP(6.0), FP(4.678), two_pi, FP(11.1), FP(1e10)
        };
        auto xd = x.createDeviceCopy();
        auto yd = y.createDeviceCopy();
        Fp1d test_samples("test samples", n);
        dex_parallel_for(
            FlatLoop<1>(1),
            YAKL_LAMBDA (int x) {
                for (int i = 0; i < n; ++i) {
                    test_samples(i) = interp(samples[i], xd, yd);
                }
            }
        );
        auto test_samples_host = test_samples.createHostCopy();

        const fp_t sin2pi = std::sin(two_pi);
        const fp_t expected[n] = {
            FP(0.0),  FP(0.0),  FP(8.49464167e-01),  FP(8.82086087e-01),
            FP(-5.36953542e-01), FP(-2.60735913e-01), FP(-9.49261115e-01),
            sin2pi, sin2pi, sin2pi
        };
        for (int i = 0; i < n; ++i) {
            REQUIRE_THAT(test_samples_host(i), WithinRel(expected[i], FP(1e-6)));
        }
    }
}

TEMPLATE_TEST_CASE("Compensated sum", "[compensated_sum]", float, double) {
    using Sum = CompensatedSumT<TestType, true>;
    constexpr TestType eps = std::numeric_limits<TestType>::epsilon();
    // A dominant term plus small ones each lost to rounding in a
    // plain sum.
    const TestType small = TestType(0.25) * eps;
    constexpr int num_small = 6;
    Sum sum;
    sum.add(TestType(1.0));
    for (int i = 0; i < num_small; ++i) {
        sum.add(small);
    }
    const TestType rest = sum.minus(TestType(1.0));
    REQUIRE_THAT(rest, WithinRel(TestType(num_small) * small, TestType(4.0) * eps));
    // minus leaves the sum unchanged; subtract changes it.
    REQUIRE(sum.minus(TestType(1.0)) == rest);
    Sum diff = sum;
    diff.subtract(TestType(1.0));
    REQUIRE(diff.value() == rest);
    REQUIRE(sum.plus(-TestType(1.0)) == rest);

    // minus(b) is the original fused (sum + error) - b, bitwise.
    const TestType vals[] = {TestType(1.0), TestType(3.0e-3), TestType(-7.5e-5), TestType(2.0e2), TestType(1.0) / TestType(3.0)};
    Sum acc;
    for (TestType v : vals) {
        acc.add(v);
        for (TestType b : vals) {
            const TestType s = acc.sum - b;
            const TestType bb = s - acc.sum;
            const TestType err = (acc.sum - (s - bb)) - (b + bb);
            REQUIRE(acc.minus(b) == s + (err + acc.error));
        }
    }
}

TEST_CASE("Uncompensated sum", "[compensated_sum]") {
    CompensatedSumT<double, false> sum;
    const double vals[] = {1.0, 3.0e-3, -7.5e-5, 2.0e2, 1.0 / 3.0};
    double plain = 0.0;
    for (double v : vals) {
        sum.add(v);
        plain += v;
        REQUIRE(sum.value() == plain);
        REQUIRE(sum.minus(v) == plain - v);
        REQUIRE(sum.plus(v) == plain + v);
    }
    sum.subtract(1.0);
    REQUIRE(sum.value() == plain - 1.0);
}

TEST_CASE("16-bit radiation storage", "[rad16]") {
    // Every value round trips exactly.
    for (u32 h = 0; h < 0x10000; ++h) {
        REQUIRE(float_to_rad16(rad16_to_float(u16(h))) == u16(h));
    }
    REQUIRE(rad16_to_float(float_to_rad16(1.0f)) == 1.0f);
    REQUIRE(rad16_to_float(0x0001) == std::ldexp(1.0f, -49));
    REQUIRE(rad16_to_float(0x0400) == std::ldexp(1.0f, -39));
    // Ties to even, saturation, negative/NaN to 0.
    REQUIRE(float_to_rad16(1.0f + std::ldexp(1.0f, -11)) == float_to_rad16(1.0f));
    REQUIRE(float_to_rad16(1.0f + 3.0f * std::ldexp(1.0f, -11)) == float_to_rad16(1.0f) + 2);
    REQUIRE(float_to_rad16(1e30f) == RAD16_MAX_BITS);
    REQUIRE(float_to_rad16(-1.0f) == 0);
    REQUIRE(float_to_rad16(std::nanf("")) == 0);
    // Nearest: neither neighbouring value is closer (ties: even).
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> e(-52, 23);
    std::uniform_real_distribution<f32> m(1.0f, 2.0f);
    for (int i = 0; i < 200000; ++i) {
        const f32 f = std::ldexp(m(rng), e(rng));
        const u16 h = float_to_rad16(f);
        const f64 d = std::abs(f64(rad16_to_float(h)) - f);
        for (int s : {-1, 1}) {
            const i32 n = i32(h) + s;
            if (n < 0 || n > RAD16_MAX_BITS) {
                continue;
            }
            const f64 dn = std::abs(f64(rad16_to_float(u16(n))) - f);
            REQUIRE((d < dn || (d == dn && (h & 1) == 0)));
        }
        if (f >= std::ldexp(1.0f, -39) && f < 1.6e7f) {
            REQUIRE(d <= std::ldexp(f64(f), -11));
        }
    }
}
