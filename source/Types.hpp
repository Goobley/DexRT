#if !defined(DEXRT_TYPES_HPP)
#define DEXRT_TYPES_HPP
#include "Config.hpp"

constexpr auto memDevice = yakl::memDevice;
// constexpr auto memDevice = yakl::memHost;
typedef yakl::Array<fp_t, 1, memDevice> Fp1d;
typedef yakl::Array<fp_t, 2, memDevice> Fp2d;
typedef yakl::Array<fp_t, 3, memDevice> Fp3d;
typedef yakl::Array<fp_t, 4, memDevice> Fp4d;
typedef yakl::Array<fp_t, 5, memDevice> Fp5d;

typedef yakl::Array<fp_t const, 1, memDevice> FpConst1d;
typedef yakl::Array<fp_t const, 2, memDevice> FpConst2d;
typedef yakl::Array<fp_t const, 3, memDevice> FpConst3d;
typedef yakl::Array<fp_t const, 4, memDevice> FpConst4d;
typedef yakl::Array<fp_t const, 5, memDevice> FpConst5d;

typedef yakl::Array<fp_t, 1, yakl::memHost> Fp1dHost;
typedef yakl::Array<fp_t, 2, yakl::memHost> Fp2dHost;
typedef yakl::Array<fp_t, 3, yakl::memHost> Fp3dHost;
typedef yakl::Array<fp_t, 4, yakl::memHost> Fp4dHost;
typedef yakl::Array<fp_t, 5, yakl::memHost> Fp5dHost;

typedef yakl::SArray<fp_t, 1, 2> vec2;
typedef yakl::SArray<int32_t, 1, 2> ivec2;

#include <vector>
struct State {
    std::vector<Fp5d> cascades;
    Fp3d emission;
    Fp3d absorption;
};

struct RayMarchState {
    // Start pos
    vec2 p0;
    // end pos
    vec2 p1;

    // Current cell coordinate
    ivec2 curr_coord;
    // Next cell coordinate
    ivec2 next_coord;
    // Integer step dir
    ivec2 step;

    // t to next hit per axis
    vec2 next_hit;
    // t increment per step per axis
    vec2 delta;
    // t to stop at
    fp_t max_t;

    // axis increment
    vec2 direction;
    // value of t at current intersection (far side of curr_coord, just before entering next_coord)
    fp_t t = FP(0.0);
    // length of step
    fp_t dt = FP(0.0);
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
