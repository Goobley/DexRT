#if !defined(DEXRT_CONFIG_HPP)
#define DEXRT_CONFIG_HPP
#include "YAKL.h"
#include "AlwaysFalse.hpp"
#include <cassert>

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
typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef uint16_t u16;
typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;

#ifdef YAKL_DEBUG
#define DEXRT_DEBUG
#endif
typedef f64 StatEqPrecision;

constexpr int DEXRT_WARP_SIZE = 32;

constexpr fp_t PROBE0_LENGTH = FP(1.5);
constexpr int PROBE0_NUM_RAYS = 4;
constexpr fp_t PROBE0_SPACING = FP(1.0);

constexpr int CASCADE_BRANCHING_FACTOR = 2;
constexpr int MAX_CASCADE = 5;

constexpr bool LAST_CASCADE_TO_INFTY = true;
constexpr fp_t LAST_CASCADE_MAX_DIST = FP(1e4);

constexpr bool PREAVERAGE = false;
constexpr bool DIR_BY_DIR = true;
static_assert(! (PREAVERAGE && DIR_BY_DIR), "Cannot enable both DIR_BY_DIR treatment and PREAVERAGING");
static_assert(!PREAVERAGE, "Don't use preaveraging for DexRT unless you know what you're doing! (Then disable me)");

enum class LineCoeffCalc {
    Classic,
    VelocityInterp,
    CoreAndVoigt
};
constexpr const char* LineCoeffCalcNames[3] = {
    "Classic",
    "VelocityInterp",
    "CoreAndVoigt"
};
constexpr LineCoeffCalc LINE_SCHEME = LineCoeffCalc::VelocityInterp;

constexpr int INTERPOLATE_DIRECTIONAL_BINS = 21;
// NOTE(cmo): Code will warn if insufficient bins to provide less than this
constexpr fp_t INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH = FP(2.0);
constexpr int CORE_AND_VOIGT_MAX_LINES = 4;
enum class BaseMipContents {
    Continua,
    LinesAtRest,
    VelocityDependent
};
constexpr BaseMipContents BASE_MIP_CONTAINS =
    (LINE_SCHEME == LineCoeffCalc::VelocityInterp) ? BaseMipContents::LinesAtRest
        : (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) ? BaseMipContents::Continua
        : BaseMipContents::VelocityDependent;

enum class RcConfiguration {
    Vanilla,
    ParallaxFix,
    ParallaxFixInner,
    BilinearFix,
};
constexpr const char* RcConfigurationNames[4] = {
    "Vanilla",
    "ParallaxFix",
    "ParallaxFixInner",
    "BilinearFix"
};
constexpr RcConfiguration RC_CONFIG = RcConfiguration::Vanilla;
constexpr int PARALLAX_MERGE_ABOVE_CASCADE = 1;
constexpr int INNER_PARALLAX_MERGE_ABOVE_CASCADE = -1;
static_assert(
    !(
        (DIR_BY_DIR)
        && (
            RC_CONFIG == RcConfiguration::ParallaxFix
            || RC_CONFIG == RcConfiguration::ParallaxFixInner
        )
    ),
    "Parallax (reprojection) methods cannot be used with DIR_BY_DIR"
);
constexpr bool STORE_TAU_CASCADES = false;
static_assert(! (!STORE_TAU_CASCADES && RC_CONFIG == RcConfiguration::ParallaxFixInner), "Need to store tau cascades for the inner parallax fix");

constexpr int BLOCK_SIZE = 16;
constexpr bool HYPERBLOCK2x2 = true;
constexpr int ENTRY_SIZE = 3;

constexpr bool USE_MIPMAPS = true;
// constexpr int MIPMAP_FACTORS[MAX_CASCADE+1] = {0, 0, 0, 0, 0, 0};
// constexpr int MIP_LEVEL[MAX_CASCADE+1] = {0, 0, 0, 0, 0, 0};
constexpr int MIPMAP_FACTORS[MAX_CASCADE+1] = {0, 0, 1, 1, 1, 0};
constexpr int MIP_LEVEL[MAX_CASCADE+1] = {0, 0, 1, 2, 3, 3};

constexpr bool PINGPONG_BUFFERS = true;

constexpr fp_t ANGLE_INVARIANT_THERMAL_VEL_FRAC = FP(0.5);
// constexpr bool USE_BC = true;
constexpr bool PWBC_USE_VECTOR_FORM = true;
constexpr bool PWBC_CONSIDER_HORIZONTAL_OFFSET = true;

// #define FLATLAND
#ifdef FLATLAND
constexpr int NUM_INCL = 1;
constexpr int NUM_GAUSS_LOBATTO = yakl::max(NUM_INCL - 1, 1);
constexpr fp_t INCL_RAYS[NUM_INCL] = {FP(0.000000)};
constexpr fp_t INCL_WEIGHTS[NUM_INCL] = {FP(1.0)};
#else
constexpr int NUM_INCL = 8;
constexpr fp_t INCL_RAYS_4[4] = {FP(0.0), FP(0.21234053823915322), FP(0.5905331355592653), FP(0.9114120404872961)};
constexpr fp_t INCL_WEIGHTS_4[4] = {FP(0.0625), FP(0.32884431998005864), FP(0.38819346884317174), FP(0.22046221117676768)};
constexpr fp_t INCL_RAYS_8[8] = {FP(0.0), FP(0.0562625605369218), FP(0.1802406917368919), FP(0.3526247171131696), FP(0.5471536263305554), FP(0.7342101772154105), FP(0.8853209468390957), FP(0.9775206135612882)};
constexpr fp_t INCL_WEIGHTS_8[8] = {FP(0.015625), FP(0.09267907740149031), FP(0.15206531032339413), FP(0.1882587726945594), FP(0.19578608372624642), FP(0.173507397817251), FP(0.1248239506649343), FP(0.05725440737209736)};
constexpr fp_t INCL_RAYS_16[16] = {FP(0.0), FP(0.01426945473682606), FP(0.047299590094167676), FP(0.09771329932062206), FP(0.16356903939438944), FP(0.24233526096865737), FP(0.3309848049700401), FP(0.42611083909331415), FP(0.5240576915367652), FP(0.6210613113530221), FP(0.7133939137424725), FP(0.7975072449498961), FP(0.8701689744464087), FP(0.9285870468848408), FP(0.9705177013520576), FP(0.9943593110274891)};
constexpr fp_t INCL_WEIGHTS_16[16] = {FP(0.00390625), FP(0.02385111347384263), FP(0.041992640722481094), FP(0.05851017655192987), FP(0.07277777261010238), FP(0.08424819892496091), FP(0.09248089074433263), FP(0.09715954485578383), FP(0.09810439411951585), FP(0.0952791471276767), FP(0.08879239637637339), FP(0.07889346090209981), FP(0.06596284996653853), FP(0.0504978398108993), FP(0.0330947543050648), FP(0.014448569508336421)};

static_assert(NUM_INCL == 4 || NUM_INCL == 8 || NUM_INCL == 16);
constexpr fp_t const* INCL_RAYS = (NUM_INCL == 4) ?
    INCL_RAYS_4
    : (NUM_INCL == 8) ?
        INCL_RAYS_8
        : (NUM_INCL == 16) ?
            INCL_RAYS_16
            : 0;
constexpr fp_t const* INCL_WEIGHTS = (NUM_INCL == 4) ?
    INCL_WEIGHTS_4
    : (NUM_INCL == 8) ?
        INCL_WEIGHTS_8
        : (NUM_INCL == 16) ?
            INCL_WEIGHTS_16
            : 0;

#endif

constexpr int MODEL_X = 1024;
constexpr int MODEL_Y = 1024;

constexpr int NUM_DIM = 2;

constexpr int WAVE_BATCH = DEXRT_WARP_SIZE / NUM_INCL;

template <int NumIncl=NUM_INCL>
yakl::SArray<fp_t, 1, NumIncl> get_incl_rays() {
    yakl::SArray<fp_t, 1, NumIncl> incl_rays;
    for (int r = 0; r < NumIncl; ++r) {
        incl_rays(r) = INCL_RAYS[r];
    }
    return incl_rays;
}

template <int NumIncl=NUM_INCL>
yakl::SArray<fp_t, 1, NumIncl> get_incl_weights() {
    yakl::SArray<fp_t, 1, NumIncl> incl_weights;
    for (int r = 0; r < NumIncl; ++r) {
        incl_weights(r) = INCL_WEIGHTS[r];
    }
    return incl_weights;
}


#if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP)
#define DEXRT_USE_MAGMA
#include <magma_v2.h>
#endif

#else
#endif
