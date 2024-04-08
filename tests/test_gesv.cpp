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

TEST_CASE("Magma batched solve", "[magma]") {
    yakl::init();
    magma_init();
    {
        magma_queue_t mag_queue = nullptr;
        magma_queue_create(0, &mag_queue);

        // NOTE(cmo): Magma is column major!
        Fp3dHost A_cpu("A", 2, 3, 3);
        A_cpu(0, 0, 0) = FP(100.0);
        A_cpu(0, 1, 0) = FP(3.0);
        A_cpu(0, 2, 0) = FP(-10.0);
        A_cpu(0, 0, 1) = FP(-1000.0);
        A_cpu(0, 1, 1) = FP(0.5);
        A_cpu(0, 2, 1) = FP(5000.0);
        A_cpu(0, 0, 2) = FP(-1e-4);
        A_cpu(0, 1, 2) = FP(5e5);
        A_cpu(0, 2, 2) = FP(1.0);
        A_cpu(1, 0, 0) = FP(1.0);
        A_cpu(1, 1, 0) = FP(0.0);
        A_cpu(1, 2, 0) = FP(0.0);
        A_cpu(1, 0, 1) = FP(0.0);
        A_cpu(1, 1, 1) = FP(1.0);
        A_cpu(1, 2, 1) = FP(0.0);
        A_cpu(1, 0, 2) = FP(0.0);
        A_cpu(1, 1, 2) = FP(0.0);
        A_cpu(1, 2, 2) = FP(1.0);
        Fp2dHost b_cpu("b", 2, 3);
        b_cpu(0, 0) = FP(37614.0);
        b_cpu(0, 1) = FP(-37811.0);
        b_cpu(0, 2) = FP(6188999991.9996);
        b_cpu(1, 0) = FP(1.0);
        b_cpu(1, 1) = FP(320.0);
        b_cpu(1, 2) = FP(-48.2);

        auto A = A_cpu.createDeviceCopy();
        auto b = b_cpu.createDeviceCopy();
        yakl::Array<fp_t*, 1, yakl::memHost> As_host("As_host", 2);
        As_host(0) = A.get_data();
        As_host(1) = A.get_data() + A.extent(1) * A.extent(2);
        auto As = As_host.createDeviceCopy();
        yakl::Array<fp_t*, 1, yakl::memHost> bs_host("bs_host", 2);
        bs_host(0) = b.get_data();
        bs_host(1) = b.get_data() + b.extent(1);
        auto bs = bs_host.createDeviceCopy();
        yakl::Array<i32, 2, yakl::memDevice> ipiv("ipiv", 2, 3);
        yakl::Array<i32*, 1, yakl::memHost> ipivs_host("ipivs_host", 2);
        ipivs_host(0) = ipiv.get_data();
        ipivs_host(1) = ipiv.get_data() + ipiv.extent(1);
        auto ipivs = ipivs_host.createDeviceCopy();
        yakl::Array<i32, 1, yakl::memDevice> infos("infos", 2);
        i32 batch_count = 2;

        // TODO(cmo): Switch to dgesv in double precision
        // NOTE(cmo): A "d" prefix in the magma docs means device array.
        magma_sgesv_batched_small(
            A.extent(1),
            1,
            As.get_data(),
            A.extent(1),
            ipivs.get_data(),
            bs.get_data(),
            b.extent(1),
            infos.get_data(),
            batch_count,
            mag_queue
        );

        magma_queue_sync(mag_queue);

        auto infos_host = infos.createHostCopy();
        REQUIRE(infos_host(0) == 0);
        REQUIRE(infos_host(1) == 0);

        auto result = b.createHostCopy();

        CAPTURE(result(0, 0), result(0, 1), result(0, 2));
        REQUIRE_THAT(result(0, 0), WithinRel(FP(4.0), FP(1e-4)));
        REQUIRE_THAT(result(0, 1), WithinRel(FP(12378.0), FP(1e-4)));
        REQUIRE_THAT(result(0, 2), WithinRel(FP(-8.0), FP(1e-4)));
        REQUIRE_THAT(result(1, 0), WithinRel(FP(1.0), FP(1e-4)));
        REQUIRE_THAT(result(1, 1), WithinRel(FP(320.0), FP(1e-4)));
        REQUIRE_THAT(result(1, 2), WithinRel(FP(-48.2), FP(1e-4)));

        magma_queue_destroy(mag_queue);
    }
    magma_finalize();
    yakl::finalize();
}