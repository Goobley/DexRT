#if !defined(DEXRT_CONFIG_HPP)
#define DEXRT_CONFIG_HPP

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
constexpr int NUM_COMPONENTS = 4;

// #define OLD_RAYMARCH 1

#else
#endif