#include "catch_amalgamated.hpp"
#include "Types.hpp"
#include <magma_v2.h>

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Magma solve", "[magma]") {
    yakl::init();
    magma_init();
    {
        // NOTE(cmo): Magma is column major!
        Fp2dHost A_cpu("A", 3, 3);
        A_cpu(0, 0) = FP(100.0);
        A_cpu(1, 0) = FP(3.0);
        A_cpu(2, 0) = FP(-10.0);
        A_cpu(0, 1) = FP(-1000.0);
        A_cpu(1, 1) = FP(0.5);
        A_cpu(2, 1) = FP(5000.0);
        A_cpu(0, 2) = FP(-1e-4);
        A_cpu(1, 2) = FP(5e5);
        A_cpu(2, 2) = FP(1.0);
        Fp1dHost b_cpu("b", 3);
        b_cpu(0) = FP(37614.0);
        b_cpu(1) = FP(-37811.0);
        b_cpu(2) = FP(6188999991.9996);

        auto A = A_cpu.createDeviceCopy();
        auto b = b_cpu.createDeviceCopy();
        yakl::Array<i32, 1, yakl::memHost> ipiv("ipiv", 3);
        i32 info;

        // TODO(cmo): Switch to dgesv in double precision
        magma_sgesv_gpu(
            A.extent(0), 
            1, 
            A.get_data(), 
            A.extent(0), 
            ipiv.get_data(), 
            b.get_data(), 
            b.extent(0),
            &info
        );

        REQUIRE(info == 0);

        auto result = b.createHostCopy();

        CAPTURE(result(0), result(1), result(2));
        REQUIRE_THAT(result(0), WithinRel(FP(4.0), FP(1e-4)));
        REQUIRE_THAT(result(1), WithinRel(FP(12378.0), FP(1e-4)));
        REQUIRE_THAT(result(2), WithinRel(FP(-8.0), FP(1e-4)));
    }
    magma_finalize();
    yakl::finalize();
}

// TODO(cmo): Batched test