#include "catch_amalgamated.hpp"
#include "Utils.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Interp", "[interp]") {
    constexpr int n = 10;
    constexpr fp_t two_pi = FP(2.0) * FP(M_PI);
    yakl::init();
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
        parallel_for(
            SimpleBounds<1>(1),
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
    yakl::finalize();
}