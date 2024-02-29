#if !defined(DEXRT_TYPES_HPP)
#define DEXRT_TYPES_HPP
#include "Config.hpp"

constexpr auto memDevice = yakl::memDevice;
// constexpr auto memDevice = yakl::memHost;
typedef yakl::Array<fp_t, 1, memDevice> Fp1d;
typedef yakl::Array<fp_t, 2, memDevice> Fp2d;
typedef yakl::Array<fp_t, 3, memDevice> Fp3d;
typedef yakl::Array<fp_t, 4, memDevice> Fp4d;

typedef yakl::Array<fp_t const, 1, memDevice> FpConst1d;
typedef yakl::Array<fp_t const, 2, memDevice> FpConst2d;
typedef yakl::Array<fp_t const, 3, memDevice> FpConst3d;
typedef yakl::Array<fp_t const, 4, memDevice> FpConst4d;

typedef yakl::Array<fp_t, 1, yakl::memHost> Fp1dHost;
typedef yakl::Array<fp_t, 2, yakl::memHost> Fp2dHost;
typedef yakl::Array<fp_t, 3, yakl::memHost> Fp3dHost;
typedef yakl::Array<fp_t, 4, yakl::memHost> Fp4dHost;

typedef yakl::SArray<fp_t, 1, 2> vec2;
typedef yakl::SArray<int32_t, 1, 2> ivec2;

#include <vector>
struct State {
    std::vector<Fp4d> cascades;
    Fp3d emission;
    Fp3d absorption;
};

struct RayMarchState {
    // Start pos
    vec2 p0;
    // end pos
    vec2 p1;

    // Current cell idx
    ivec2 p;
    // Integer step dir
    ivec2 step;
    // Cell to stop in
    ivec2 end;

    // t to next hit per axis
    vec2 tm;
    // t increment per step per axis
    vec2 td;

    // axis increment
    vec2 d;
    // hit location
    vec2 hit;
    // value of t at prev intersection
    fp_t prev_t = FP(0.0);
    // length of step
    fp_t ds = FP(0.0);
};

struct RayStartEnd {
    vec2 start;
    vec2 end;
};

struct Box {
    vec2 dims[NUM_DIM];
};

#else
#endif
