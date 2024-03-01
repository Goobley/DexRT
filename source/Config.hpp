#if !defined(DEXRT_CONFIG_HPP)
#define DEXRT_CONFIG_HPP
#include "YAKL.h"

// #define HAVE_MPI
#define DEXRT_SINGLE_PREC
#ifdef DEXRT_SINGLE_PREC
typedef float fp_t;
#define FP(X) (X##f)
#else
typedef double fp_t;
#define FP(X) (X)
#endif

constexpr int CANVAS_X = 1024;
constexpr int CANVAS_Y = 1024;
constexpr fp_t PROBE0_LENGTH = FP(2.0);
constexpr int PROBE0_NUM_RAYS = 4;
constexpr fp_t PROBE0_SPACING = FP(1.0);
constexpr int PROBES_IN_CASCADE_0 = CANVAS_X / PROBE0_SPACING;
constexpr int CASCADE_BRANCHING_FACTOR = 2;
constexpr bool BRANCH_RAYS = false;
constexpr bool BILINEAR_FIX = false;
constexpr int MAX_LEVEL = 7;
constexpr int NUM_WAVELENGTHS = 3;
constexpr int NUM_COMPONENTS = 2 * NUM_WAVELENGTHS;
constexpr int NUM_DIM = 2;

// The last one is just in/out of page, so resolve at the end
constexpr int NUM_AZ = 4;
constexpr int NUM_GAUSS_LOBATTO = yakl::max(NUM_AZ - 1, 1);
constexpr fp_t TRACE_AZ_RAYS[NUM_GAUSS_LOBATTO] = {FP(1.000000), FP(0.723607), FP(0.276393)};
constexpr fp_t TRACE_AZ_WEIGHTS[NUM_GAUSS_LOBATTO] = {FP(0.083333), FP(0.416667), FP(0.416667)};
constexpr fp_t AZ_RAYS[NUM_AZ] = {FP(1.000000), FP(0.723607), FP(0.276393), FP(0.000000)};
constexpr fp_t AZ_WEIGHTS[NUM_AZ] = {FP(0.083333), FP(0.416667), FP(0.416667), FP(0.083333)};

// #define OLD_RAYMARCH 1
// #define TRACE_OPAQUE_LIGHTS 1
#define LIGHT_MODEL model_D_emission
#define ABSORPTION_MODEL model_D_absorption

#else
#endif