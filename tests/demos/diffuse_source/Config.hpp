#if !defined(DEXRT_CONFIG_HPP)
#define DEXRT_CONFIG_HPP
#include "YAKL.h"

#define DEXRT_SINGLE_PREC
#ifdef DEXRT_SINGLE_PREC
typedef float fp_t;
#define FP(X) (X##f)
#else
typedef double fp_t;
#define FP(X) (X)
#endif
typedef float f32;
typedef double f64;
typedef int32_t i32;

constexpr int CANVAS_X = 1024;
constexpr int CANVAS_Y = 1024;
constexpr fp_t PROBE0_LENGTH = FP(2.0);
constexpr int PROBE0_NUM_RAYS = 4;
constexpr fp_t PROBE0_SPACING = FP(1.0);
constexpr int PROBES_IN_CASCADE_0 = CANVAS_X / PROBE0_SPACING;

constexpr int CASCADE_BRANCHING_FACTOR = 2;
constexpr int MAX_LEVEL = 4;

constexpr bool LAST_CASCADE_TO_INFTY = true;
constexpr fp_t LAST_CASCADE_MAX_DIST = FP(1e8);

constexpr bool BRANCH_RAYS = false;
constexpr bool BILINEAR_FIX = false;

constexpr bool USE_MIPMAPS = false;
constexpr int MIPMAP_FACTORS[MAX_LEVEL+1] = {0, 0, 1, 1, 1};
constexpr bool USE_ATMOSPHERE = false;

constexpr int NUM_WAVELENGTHS = 3;
constexpr int NUM_COMPONENTS = 2 * NUM_WAVELENGTHS;
constexpr int NUM_DIM = 2;

constexpr bool PINGPONG_BUFFERS = false;

#define FLATLAND
#ifdef FLATLAND
constexpr int NUM_AZ = 1;
constexpr int NUM_GAUSS_LOBATTO = yakl::max(NUM_AZ - 1, 1);
constexpr fp_t TRACE_AZ_RAYS[NUM_GAUSS_LOBATTO] = {FP(0.000000)};
constexpr fp_t TRACE_INCL_RAYS[NUM_GAUSS_LOBATTO] = {FP(1.000000)};
constexpr fp_t TRACE_AZ_WEIGHTS[NUM_GAUSS_LOBATTO] = {FP(1.0)};
constexpr fp_t AZ_RAYS[NUM_AZ] = {FP(0.000000)};
constexpr fp_t INCL_RAYS[NUM_AZ] = {FP(1.000000)};
constexpr fp_t AZ_WEIGHTS[NUM_AZ] = {FP(1.0)};
#else
constexpr int NUM_AZ = 4;
constexpr int NUM_GAUSS_LOBATTO = yakl::max(NUM_AZ - 1, 1);
constexpr fp_t TRACE_AZ_RAYS[NUM_GAUSS_LOBATTO] = {FP(1.000000), FP(0.723607), FP(0.276393)};
constexpr fp_t TRACE_INCL_RAYS[NUM_GAUSS_LOBATTO] = {FP(0.000000), FP(0.690212), FP(0.961045)};
constexpr fp_t TRACE_AZ_WEIGHTS[NUM_GAUSS_LOBATTO] = {FP(0.083333), FP(0.416667), FP(0.416667)};
constexpr fp_t AZ_RAYS[NUM_AZ] = {FP(1.000000), FP(0.723607), FP(0.276393), FP(0.000000)};
constexpr fp_t INCL_RAYS[NUM_AZ] = {FP(0.000000), FP(0.690212), FP(0.961045), FP(1.000000)};
constexpr fp_t AZ_WEIGHTS[NUM_AZ] = {FP(0.083333), FP(0.416667), FP(0.416667), FP(0.083333)};
#endif

template <int NumAz=NUM_AZ>
yakl::SArray<fp_t, 1, NumAz> get_az_rays() {
    yakl::SArray<fp_t, 1, NumAz> az_rays;
    for (int r = 0; r < NumAz; ++r) {
        az_rays(r) = INCL_RAYS[r];
    }
    return az_rays;
}

template <int NumAz=NUM_AZ>
yakl::SArray<fp_t, 1, NumAz> get_az_weights() {
    yakl::SArray<fp_t, 1, NumAz> az_weights;
    for (int r = 0; r < NumAz; ++r) {
        az_weights(r) = AZ_WEIGHTS[r];
    }
    return az_weights;
}


#else
#endif