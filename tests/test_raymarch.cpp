#include "catch_amalgamated.hpp"
#include "RayMarching.hpp"
#include "Types.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE( "Clipping rays to box", "[clip_rays]" ) {
    Box box{};
    box.dims[0](0) = FP(0.0);
    box.dims[0](1) = FP(63.0);
    box.dims[1](0) = FP(0.0);
    box.dims[1](1) = FP(31.0);

    // Start inside, end outside
    vec2 start;
    vec2 end;
    start(0) = 32;
    start(1) = 16;
    end(0) = 80;
    end(1) = 24;
    auto clipped = clip_ray_to_box({start, end}, box).value();
    REQUIRE( clipped.start(0) == FP(32.0) );
    REQUIRE( clipped.start(1) == FP(16.0) );
    REQUIRE( clipped.end(0) == FP(63.0) );
    REQUIRE_THAT( clipped.end(1), WithinRel(FP(21.0) + FP(1.0) / FP(6.0), FP(1e-4)) );

    // Start outside, end in
    start(0) = FP(-12.0);
    start(1) = FP(-10.0);
    end(0) = FP(20.0);
    end(1) = FP(24.0);
    clipped = clip_ray_to_box({start, end}, box).value();
    REQUIRE( clipped.start(0) == FP(0.0) );
    REQUIRE_THAT( clipped.start(1), WithinRel(FP(2.75), FP(1e-4)) );
    REQUIRE( clipped.end(0) == FP(20.0) );
    REQUIRE( clipped.end(1) == FP(24.0) );

    // Previous case reversed
    start(0) = FP(20.0);
    start(1) = FP(24.0);
    end(0) = FP(-12.0);
    end(1) = FP(-10.0);
    clipped = clip_ray_to_box({start, end}, box).value();
    REQUIRE( clipped.start(0) == FP(20.0) );
    REQUIRE( clipped.start(1) == FP(24.0) );
    REQUIRE( clipped.end(0) == FP(0.0) );
    REQUIRE_THAT( clipped.end(1), WithinRel(FP(2.75), FP(1e-4)) );

    // Entirely inside
    start(0) = FP(20.0);
    start(1) = FP(24.0);
    end(0) = FP(12.0);
    end(1) = FP(10.0);
    clipped = clip_ray_to_box({start, end}, box).value();
    REQUIRE( clipped.start(0) == FP(20.0) );
    REQUIRE( clipped.start(1) == FP(24.0) );
    REQUIRE( clipped.end(0) == FP(12.0) );
    REQUIRE( clipped.end(1) == FP(10.0) );

    // Vertical, outside to in
    start(0) = FP(1.0);
    start(1) = FP(-20.0);
    end(0) = FP(1.0);
    end(1) = FP(10.0);
    clipped = clip_ray_to_box({start, end}, box).value();
    REQUIRE( clipped.start(0) == FP(1.0) );
    REQUIRE( clipped.start(1) == FP(0.0) );
    REQUIRE( clipped.end(0) == FP(1.0) );
    REQUIRE( clipped.end(1) == FP(10.0) );

    // No hit
    start(0) = FP(1.0);
    start(1) = FP(-20.0);
    end(0) = FP(-1.5);
    end(1) = FP(10.0);
    auto maybe = clip_ray_to_box({start, end}, box);
    REQUIRE( !maybe.has_value() );

    // No hit
    start(0) = FP(-2.0);
    start(1) = FP(30.0);
    end(0) = FP(2.5);
    end(1) = FP(35.0);
    maybe = clip_ray_to_box({start, end}, box);
    REQUIRE( !maybe.has_value() );

    // Axis-aligned along edge
    start(0) = FP(63.0);
    start(1) = FP(-2.5);
    end(0) = FP(63.0);
    end(1) = FP(35.0);
    clipped = clip_ray_to_box({start, end}, box).value();
    REQUIRE( clipped.start(0) == FP(63.0) );
    REQUIRE( clipped.start(1) >= FP(0.0) );
    REQUIRE_THAT( clipped.start(1), WithinAbs(FP(0.0), FP(1e-6)) );
    REQUIRE( clipped.end(0) == FP(63.0) );
    REQUIRE( clipped.end(1) <= FP(31.0) );
    REQUIRE_THAT( clipped.end(1), WithinAbs(FP(31.0), FP(1e-6)) );

    // Same test but with bigger numbers (precision)
    box.dims[0](0) = FP(0.0);
    box.dims[0](1) = FP(16384.0);
    box.dims[1](0) = FP(0.0);
    box.dims[1](1) = FP(16384.0);
    start(0) = FP(16384.0);
    start(1) = FP(-2.5);
    end(0) = FP(16384.0);
    end(1) = FP(24e3);
    clipped = clip_ray_to_box({start, end}, box).value();
    REQUIRE( clipped.start(0) == FP(16384.0) );
    REQUIRE( clipped.start(1) >= FP(0.0) );
    REQUIRE_THAT( clipped.start(1), WithinAbs(FP(0.0), FP(1e-6)) );
    REQUIRE( clipped.end(0) == FP(16384.0) );
    REQUIRE( clipped.end(1) <= FP(16384.0) );
    REQUIRE_THAT( clipped.end(1), WithinAbs(FP(16384.0), FP(1e-6)) );

    start(0) = FP(16384.0) + FP(1e-3);
    start(1) = FP(-2.5);
    end(0) = FP(16384.0) + FP(1e-3);
    end(1) = FP(24e3);
    maybe = clip_ray_to_box({start, end}, box);
    REQUIRE( !maybe.has_value() );

    // Test case that failed due to precision issues
    box.dims[0](0) = FP(0.0);
    box.dims[0](1) = FP(15.0);
    box.dims[1](0) = FP(0.0);
    box.dims[1](1) = FP(15.0);
    start(0) = FP(2.8);
    start(1) = FP(18.4);
    end(0) = FP(21.0);
    end(1) = FP(23423423.0);
    maybe = clip_ray_to_box({start, end}, box);
    REQUIRE( !maybe.has_value() );
}

TEST_CASE( "2D Grid Raymarch", "[raymarch]") {
    Catch::StringMaker<float>::precision = 15;
    ivec2 domain_size;
    domain_size(0) = 16;
    domain_size(1) = 16;
    vec2 start, end;
    start(0) = FP(2.2);
    start(1) = FP(3.5);
    end(0) = FP(3.1);
    end(1) = FP(10.23);

    // NOTE(cmo): Test cases from python impl verified by hand
    auto marcher = RayMarch2d_new(start, end, domain_size).value();
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(0.504451), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 2 );
    REQUIRE( marcher.curr_coord(1) == 3 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.00890218), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 2 );
    REQUIRE( marcher.curr_coord(1) == 4 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.00890218), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 2 );
    REQUIRE( marcher.curr_coord(1) == 5 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.00890218), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 2 );
    REQUIRE( marcher.curr_coord(1) == 6 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.00890218), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 2 );
    REQUIRE( marcher.curr_coord(1) == 7 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.00890218), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 2 );
    REQUIRE( marcher.curr_coord(1) == 8 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(0.48651557), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 2 );
    REQUIRE( marcher.curr_coord(1) == 9 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(0.5223866), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 3 );
    REQUIRE( marcher.curr_coord(1) == 9 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(0.23204706), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 3 );
    REQUIRE( marcher.curr_coord(1) == 10 );
    REQUIRE( !next_intersection(&marcher) );

    start(0) = FP(1.91421);
    start(1) = FP(3.085815);
    end(0) = FP(0.5);
    end(1) = FP(4.5);
    marcher = RayMarch2d_new(start, end, domain_size).value();
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.2928661), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 1 );
    REQUIRE( marcher.curr_coord(1) == 3 );
    next_intersection(&marcher);
    REQUIRE( marcher.dt < 2e-5 );
    REQUIRE( marcher.curr_coord(0) == 1 );
    REQUIRE( marcher.curr_coord(1) == 4 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(0.70709967), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 0 );
    REQUIRE( marcher.curr_coord(1) == 4 );
    REQUIRE( !next_intersection(&marcher) );

    start(0) = FP(2.1);
    start(1) = FP(3.1);
    end(0) = FP(4.8);
    end(1) = FP(5.8);
    marcher = RayMarch2d_new(start, end, domain_size).value();
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.27279236), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 2 );
    REQUIRE( marcher.curr_coord(1) == 3 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(0.0), FP(1e-6)) );
    REQUIRE( marcher.curr_coord(0) == 3 );
    REQUIRE( marcher.curr_coord(1) == 3 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.4142135865), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 3 );
    REQUIRE( marcher.curr_coord(1) == 4 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(0.0), FP(1e-6)) );
    REQUIRE( marcher.curr_coord(0) == 4 );
    REQUIRE( marcher.curr_coord(1) == 4 );
    next_intersection(&marcher);
    REQUIRE_THAT( marcher.dt, WithinAbs(FP(1.13137114), FP(1e-4)) );
    REQUIRE( marcher.curr_coord(0) == 4 );
    REQUIRE( marcher.curr_coord(1) == 5 );
    REQUIRE( !next_intersection(&marcher) );

    start(0) = FP(2.8);
    start(1) = FP(18.4);
    end(0) = FP(21.0);
    end(1) = FP(2342.0);
    auto maybe = RayMarch2d_new(start, end, domain_size);
    REQUIRE( !maybe );
}

#ifdef FLATLAND
TEST_CASE( "Raymarch Against mipmaps", "[raymarch_mips]") {
    yakl::init();
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    constexpr int grid_size = 1024;

    {
    Fp3d eta("eta", grid_size, grid_size, NUM_WAVELENGTHS);
    Fp3d chi("chi", grid_size, grid_size, NUM_WAVELENGTHS);
    fp_t chi_wall = FP(16.0);
    fp_t eta_wall = FP(8.0);
    parallel_for(
        SimpleBounds<2>(grid_size, grid_size),
        YAKL_LAMBDA (int x, int y) {
            fp_t val = FP(0.0);
            fp_t chi_val = FP(1e-10);
            if (x > 512) {
                val = eta_wall;
                chi_val = chi_wall;
            }
            for (int w = 0; w < NUM_WAVELENGTHS; ++w) {
                eta(x, y, w) = val;
                chi(x, y, w) = chi_val;
            }
        }
    );
    yakl::fence();
    fp_t expected_soln = eta_wall / chi_wall;

    yakl::ScalarLiveOut<fp_t> intensity_full_res(FP(0.0));
    CascadeRTState rt_state;
    rt_state.chi = chi;
    rt_state.eta = eta;
    rt_state.mipmap_factor = 0;
    parallel_for(
        SimpleBounds<1>(1),
        YAKL_LAMBDA (int x) {
            yakl::SArray<fp_t, 1, 1> az_rays(FP(1.0));
            vec2 ray_start;
            ray_start(0) = FP(0.0);
            ray_start(1) = FP(512.0);
            vec2 ray_end;
            ray_end(0) = FP(1024.0);
            ray_end(1) = FP(512.0);

            auto result = raymarch_2d(rt_state, ray_start, ray_end, az_rays);
            intensity_full_res = result(0, 0);
        }
    );

    fp_t full_res_host = intensity_full_res.hostRead();
    CAPTURE(full_res_host);
    REQUIRE_THAT(full_res_host, WithinRel(expected_soln, FP(1e-6)));


    const int mip_factor = 4;
    Fp3d eta_mip("eta_mip", grid_size / (1 << mip_factor), grid_size / (1 << mip_factor), NUM_WAVELENGTHS);
    Fp3d chi_mip("chi_mip", grid_size / (1 << mip_factor), grid_size / (1 << mip_factor), NUM_WAVELENGTHS);
    auto mip_dims = eta_mip.get_dimensions();

    parallel_for(
        SimpleBounds<2>(mip_dims(0), mip_dims(1)),
        mipmap_arr(eta, eta_mip, mip_factor)
    );
    parallel_for(
        SimpleBounds<2>(mip_dims(0), mip_dims(1)),
        mipmap_arr(chi, chi_mip, mip_factor)
    );
    yakl::fence();

    yakl::ScalarLiveOut<fp_t> intensity_mipped(FP(0.0));
    rt_state.chi = chi_mip;
    rt_state.eta = eta_mip;
    rt_state.mipmap_factor = mip_factor;
    parallel_for(
        SimpleBounds<1>(1),
        YAKL_LAMBDA (int x) {
            yakl::SArray<fp_t, 1, 1> az_rays(FP(1.0));
            vec2 ray_start;
            ray_start(0) = FP(0.0);
            ray_start(1) = FP(32.0);
            vec2 ray_end;
            ray_end(0) = FP(64.0);
            ray_end(1) = FP(32.0);

            auto result = raymarch_2d(rt_state, ray_start, ray_end, az_rays);
            intensity_mipped = result(0, 0);
        }
    );

    fp_t mipped = intensity_mipped.hostRead();
    CAPTURE(mipped);
    REQUIRE_THAT(mipped, WithinRel(expected_soln, FP(1e-6)));

    }

    yakl::finalize();
}
#endif