#if !defined(DEXRT_TYPES_HPP)
#define DEXRT_TYPES_HPP
#include "Config.hpp"

typedef yakl::Array<fp_t, 1, yakl::memDevice> Fp1d;
typedef yakl::Array<fp_t, 2, yakl::memDevice> Fp2d;
typedef yakl::Array<fp_t, 3, yakl::memDevice> Fp3d;
typedef yakl::Array<fp_t, 4, yakl::memDevice> Fp4d;

typedef yakl::Array<fp_t const, 1, yakl::memDevice> FpConst1d;
typedef yakl::Array<fp_t const, 2, yakl::memDevice> FpConst2d;
typedef yakl::Array<fp_t const, 3, yakl::memDevice> FpConst3d;
typedef yakl::Array<fp_t const, 4, yakl::memDevice> FpConst4d;

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
};

#else
#endif
