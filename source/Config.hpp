#if !defined(DEXRT_CONFIG_HPP)
#define DEXRT_CONFIG_HPP
#include "YAKL.h"
#include "AlwaysFalse.hpp"
#include <cassert>
#include <bit>

#include "UserConfig2d.hpp"
#include "UserConfig3d.hpp"

#ifdef KOKKOS_ENABLE_DEBUG
#define DEXRT_DEBUG
#endif

#define YAKL_INLINE KOKKOS_INLINE_FUNCTION
#define YAKL_LAMBDA KOKKOS_LAMBDA
#define YAKL_CLASS_LAMBDA KOKKOS_CLASS_LAMBDA
#define YAKL_EXECUTE_ON_HOST_ONLY KOKKOS_IF_ON_HOST

namespace yakl {
    // NOTE(cmo): Stubs for functions I can't be bothered to rewrite right now.
    inline void fence() {
        Kokkos::fence();
    }

    template <typename str>
    YAKL_INLINE void yakl_throw(str in) {
        Kokkos::abort(in);
    }
}

// NOTE(cmo): A function to define in main so functions that don't get state
// passed to them can query dimensionality (either 2 or 3)
extern int get_dexrt_dimensionality();

// NOTE(cmo): The spacing between probes on cascade 0 -- this isn't actually configurable
constexpr fp_t PROBE0_SPACING = FP(1.0);

typedef yakl::Array<GammaFp, 3, yakl::memDevice> GammaMat;

constexpr bool VARY_BRANCHING_FACTOR = (BRANCHING_FACTOR_SWITCH > 0);
static_assert(!(PREAVERAGE && VARY_BRANCHING_FACTOR), "Can't use preaveraging and variable branching factor because I was too lazy to implement it");
static_assert(!(PREAVERAGE && RAYMARCH_TYPE == RaymarchType::LineSweep), "Currently can't use preaveraging and linesweeping");

// Compile time errors for RC misconfig
static_assert(! (PREAVERAGE && DIR_BY_DIR), "Cannot enable both DIR_BY_DIR treatment and PREAVERAGING");
static_assert(!PREAVERAGE, "Don't use preaveraging for DexRT unless you know what you're doing! (Then disable me)");
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
static_assert(! (!STORE_TAU_CASCADES && RC_CONFIG == RcConfiguration::ParallaxFixInner), "Need to store tau cascades for the inner parallax fix");

// NOTE(cmo): The contents of mip0 is determined by LINE_SCHEME, and determines
// how we handle computing variance for mip-mapping.
enum class BaseMipContents {
    Continua,
    LinesAtRest,
    VelocityDependent
};
constexpr BaseMipContents BASE_MIP_CONTAINS =
    (LINE_SCHEME == LineCoeffCalc::VelocityInterp) ? BaseMipContents::LinesAtRest
        : (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) ? BaseMipContents::Continua
        : BaseMipContents::VelocityDependent;
constexpr BaseMipContents BASE_MIP_CONTAINS_3D =
    (LINE_SCHEME_3D == LineCoeffCalc::VelocityInterp) ? BaseMipContents::LinesAtRest
        : (LINE_SCHEME_3D == LineCoeffCalc::CoreAndVoigt) ? BaseMipContents::Continua
        : BaseMipContents::VelocityDependent;


static_assert(std::has_single_bit(u32(BLOCK_SIZE)), "BLOCK_SIZE must be a power of two (>= 2)");

#ifdef FLATLAND
constexpr int NUM_GAUSS_LOBATTO = std::max(NUM_INCL - 1, 1);
constexpr fp_t INCL_RAYS[NUM_INCL] = {FP(0.000000)};
constexpr fp_t INCL_WEIGHTS[NUM_INCL] = {FP(1.0)};
#else
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

// NOTE(cmo): These are vestigial, but still needed for now.
constexpr int NUM_DIM = 2;
constexpr bool PWBC_USE_VECTOR_FORM = true;
constexpr bool PWBC_CONSIDER_HORIZONTAL_OFFSET = true;

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

// NOTE(cmo): The kokkos-kernels approach is faster than magma. If you _really_ want MAGMA, you should define it in your build script
// #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#ifdef DEXRT_USE_MAGMA
    #include <magma_v2.h>
    #include <magmablas.h>
#endif

#else
#endif
