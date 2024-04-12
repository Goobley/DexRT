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
typedef int64_t i64;

constexpr fp_t PROBE0_LENGTH = FP(2.0);
constexpr int PROBE0_NUM_RAYS = 4;
constexpr fp_t PROBE0_SPACING = FP(1.0);

constexpr int CASCADE_BRANCHING_FACTOR = 2;
constexpr int MAX_LEVEL = 4;

constexpr bool LAST_CASCADE_TO_INFTY = true;
constexpr fp_t LAST_CASCADE_MAX_DIST = FP(1e8);

constexpr bool BRANCH_RAYS = false;
constexpr bool BILINEAR_FIX = false;

constexpr bool USE_MIPMAPS = false;
constexpr int MIPMAP_FACTORS[MAX_LEVEL+1] = {0, 0, 1, 1, 1};

constexpr bool PINGPONG_BUFFERS = true;

// #define FLATLAND
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
constexpr int NUM_AZ = 5;
// constexpr int NUM_GAUSS_LOBATTO = yakl::max(NUM_AZ - 1, 1);
// constexpr fp_t TRACE_AZ_RAYS[NUM_GAUSS_LOBATTO] = {FP(1.000000), FP(0.723607), FP(0.276393)};
// constexpr fp_t TRACE_INCL_RAYS[NUM_GAUSS_LOBATTO] = {FP(0.000000), FP(0.690212), FP(0.961045)};
// constexpr fp_t TRACE_AZ_WEIGHTS[NUM_GAUSS_LOBATTO] = {FP(0.083333), FP(0.416667), FP(0.416667)};
// constexpr fp_t AZ_RAYS[NUM_AZ] = {FP(1.000000), FP(0.723607), FP(0.276393), FP(0.000000)};
// constexpr fp_t INCL_RAYS[NUM_AZ] = {FP(0.000000), FP(0.690212), FP(0.961045), FP(1.000000)};
// constexpr fp_t AZ_WEIGHTS[NUM_AZ] = {FP(0.083333), FP(0.416667), FP(0.416667), FP(0.083333)};
constexpr fp_t INCL_RAYS[NUM_AZ] = {FP(0.04691008), FP(0.23076534), FP(0.5), FP(0.76923466), FP(0.95308992)};
constexpr fp_t AZ_WEIGHTS[NUM_AZ] = {FP(0.11846344), FP(0.23931434), FP(0.28444444), FP(0.23931434), FP(0.11846344)};
#endif

/// Whether to load an atmosphere or use the LIGHT_MODEL to determine eta/chi.
constexpr bool USE_ATMOSPHERE = true;
constexpr int MODEL_X = 1024;
constexpr int MODEL_Y = 1024;
#define LIGHT_MODEL model_F_emission
#define ABSORPTION_MODEL model_F_absorption

/// Number of wavelengths in an RGB batch
constexpr int NUM_WAVELENGTHS = USE_ATMOSPHERE ? 1 : 3;
constexpr int NUM_COMPONENTS = 2 * NUM_WAVELENGTHS;
constexpr int NUM_DIM = 2;

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