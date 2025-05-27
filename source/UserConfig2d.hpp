#if !defined(DEXRT_USER_CONFIG_HPP)
#define DEXRT_USER_CONFIG_HPP
#include "YAKL.h"
#include "stdint.h"

/*== Precision ================================================================*/

// So far everything works in single precision, other than where overridden (i.e.
// StatEq), but you can adjust here, especially if you're lucky enough to have a
// GPU with good f64 throughput.
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

// Set the precision used in statistical equilibrium and the the Newton-Raphson
// charge conservation. Don't change from double precision unless you have a
// good idea what you're doing!
typedef f64 StatEqPrecision;

// Set the precision used in the accumulation of the Gamma matrix. Single
// precision is usually fine.
typedef fp_t GammaFp;

/*== 2D Config ===============================================================*/
// To consider a flatland setup (no inclination rays), uncomment the #define
// below. This can be useful for comparing against other radiance cascade works.
// For science, do not use flatland mode, and NUM_INCL must be set to 4, 8, or
// 16, defining the number of inclination rays used in a Gauss-Radau quadrature
// over 1 hemisphere (of y).

// #define FLATLAND
#ifdef FLATLAND
constexpr int NUM_INCL = 1;
#else
constexpr int NUM_INCL = 4;
#endif

/*== Memory/Warp size ========================================================*/

// Number of threads to set up to treat contiguously. These all take the same
// steps through the grid, at different wavelengths and inclinations as
// necessary. 32 works well in most places, but can be lowered to reduce memory.
constexpr int DEXRT_WARP_SIZE = 32;

// Whether to store the cascades containing tau. These aren't directly necessary
// for RC methods other than ParallaxFixInner, and can halve the memory
// consumption of the cascades -- which aren't currently stored sparsely.
constexpr bool STORE_TAU_CASCADES = false;

// Whether to store all cascades, e.g. for saving or debugging, or simply
// "ping-pong" between two cascades, merging as we go between cascade i+1 and i.
// Can be a significant memory saving
constexpr bool PINGPONG_BUFFERS = true;

// The wavelength batch is typically set based on the warp size and number of
// inclinations, but can be overridden here. Override not well tested.
constexpr int WAVE_BATCH = std::max(DEXRT_WARP_SIZE / NUM_INCL, 1);

/*== BlockMap ================================================================*/

// The size chunks to consider in the BlockMap. 8 or 16 are typically good
// choices.
constexpr int BLOCK_SIZE = 16;
// Whether to apply a simple 2x2 Morton pattern to the compressed tile
// mip-levels stored in the MultiResBlockMap. Has a tiny impact on cache
// locality.
constexpr bool HYPERBLOCK2x2 = true;
// The size of the entry (in bits) to compress each tile's mip level into for
// traversal in the MultiResBlockMap. If not using mips, this can be 1.
// The code will raise an error if this is too small.
constexpr int ENTRY_SIZE = 3;

/*== Ray-Marching & Cascades =================================================*/

// The raymarch length on cascade 0
constexpr fp_t PROBE0_LENGTH = FP(1.5);
// The number of rays to trace on cascade 0
constexpr int PROBE0_NUM_RAYS = 4;

// The angular branching and interval length factor to use for each cascade
// (i.e. num_rays_i = PROBE0_NUM_RAYS * 2^(CASCADE_BRANCHING_FACTOR * i))
constexpr int CASCADE_BRANCHING_FACTOR = 2;
// Set this factor if a different factor is to be used for the higher cascades.
constexpr int UPPER_BRANCHING_FACTOR = 1;
// The cascade at which to transition to UPPER_BRANCHING_FACTOR (set to 0 to
// disable).
constexpr int BRANCHING_FACTOR_SWITCH = 3;

// Whether to run the rays of the last cascade off the grid regardless of where
// they start, to correctly sample the boundary conditions.
constexpr bool LAST_CASCADE_TO_INFTY = true;
// A sensible length (in voxels) that represents the edge of the grid from
// anywhere (it's not actually infinity!). May need to be increased if running
// huge models.
constexpr fp_t LAST_CASCADE_MAX_DIST = FP(2e4);

// Whether to use preaveraging (i.e. each texel of the cascade stores the
// contributions for cascade i-1 already averaged). Don't do this -- it's not
// compatible with the full non-LTE solution!
constexpr bool PREAVERAGE = false;
// Whether to treat the subset of each cascade associated with each ray of C0
// separately. This is a good default with good memory savings, but is
// incompatible with the parallax fixes.
constexpr bool DIR_BY_DIR = true;

constexpr const char* RcConfigurationNames[4] = {
    "Vanilla",
    "ParallaxFix",
    "ParallaxFixInner",
    "BilinearFix"
};
enum class RcConfiguration {
    Vanilla,
    ParallaxFix,
    ParallaxFixInner,
    BilinearFix,
};
// The cascade calculation mode to use:
// - Vanilla: basic method described in the paper, can produce ringing artefacts
// around small bright optically thick sources.
// - ParallaxFix: a cheap approximation with many of the benefits of the
// bilinear fix, but incompatible with DIR_BY_DIR
// - ParallaxFixInner: a different formulation of ParallaxFix, brings its own
// subtle set of artefacts... both parallax fixes need more proper
// investigation.
// - BilinearFix: The full bilinear fix, close to perfect accuracy, but 4x more
// rays.
constexpr RcConfiguration RC_CONFIG = RcConfiguration::Vanilla;

// Not using parallax fixes on the lowest cascades can reduce the effect of their specific artefacts.
// When using RC_CONFIG == ParallaxFix, apply the modified merge above this cascade level (exclusive)
constexpr int PARALLAX_MERGE_ABOVE_CASCADE = 1;
// When using RC_CONFIG == rParallaxFixInner, apply the modified merge above this cascade level (exclusive)
constexpr int INNER_PARALLAX_MERGE_ABOVE_CASCADE = -1;

constexpr const char* RaymarchTypeNames[2] = {
    "Raymarch",
    "LineSweep"
};
enum class RaymarchType {
    Raymarch,
    LineSweep
};
// Whether to raymarch or linesweep
constexpr RaymarchType RAYMARCH_TYPE = RaymarchType::Raymarch;
// Level to start sweeping on (classical march at levels lower than this)
constexpr int LINE_SWEEP_START_CASCADE = 2;

/*== Line emissivity/opacity =================================================*/

constexpr const char* LineCoeffCalcNames[3] = {
    "Classic",
    "VelocityInterp",
    "CoreAndVoigt"
};
enum class LineCoeffCalc {
    Classic,
    VelocityInterp,
    CoreAndVoigt
};
// The scheme to use for handling evaluating the directional emissivity/opacity
// of lines:
// - Classic: evaluate everything from scratch, everywhere. The approach taken
// in the original paper. Does not allow for mipmapping.
// - VelocityInterp: Create a grid of `INTERPOLATE_DIRECTIONAL_BINS` (see below)
// bins, and use this as a basis that is linearly interpolated. Fast,
// memory-hungry if good accuracy is needed in dynamic models. Allows for mipmapping.
// - CoreAndVoigt: A good tradeoff that allows for mipmapping. The emissivity
// and opacity for up to `CORE_AND_VOIGT_MAX_LINES` are considered without the
// effects of the line profile, and these are then modulated by the line profile
// for each direction.
constexpr LineCoeffCalc LINE_SCHEME = LineCoeffCalc::CoreAndVoigt;

// The fraction of the thermal velocity in a cell below which to consider it
// static (minimising line profile calculations). Only considered if LINE_SCHEME
// == Classic.
constexpr fp_t ANGLE_INVARIANT_THERMAL_VEL_FRAC = FP(0.5);

// The number of bins to use if LINE_SCHEME == VelocityInterp. Ignored otherwise.
constexpr int INTERPOLATE_DIRECTIONAL_BINS = 21;
// If LINE_SCHEME == VelocityInterp code will warn if insufficient bins to
// provide less than this spacing (can be used to have an idea of accuracy).
constexpr fp_t INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH = FP(2.0);

// The maximum number of lines to consider simultaneously if LINE_SCHEME ==
// CoreAndVoigt. Code will throw an error if this is too small, but there are
// performance benefits for keeping it smaller.
constexpr int CORE_AND_VOIGT_MAX_LINES = 4;

// Whether to conserve pressure in the charge conservation Newton-Raphson, or in
// the separate secondary iteration.
constexpr bool CONSERVE_PRESSURE_NR = true;

#else
#endif