#if !defined(DEXRT_USER_CONFIG_3D_HPP)
#define DEXRT_USER_CONFIG_3D_HPP

#include "UserConfig2d.hpp"


/*== BlockMap ================================================================*/

// The size chunks to consider in the BlockMap. 8 or 16 are typically good
// choices.
constexpr int BLOCK_SIZE_3D = 8;
// The size of the entry (in bits) to compress each tile's mip level into for
// traversal in the MultiResBlockMap. If not using mips, this can be 1.
// The code will raise an error if this is too small.
constexpr int ENTRY_SIZE_3D = 3;

/*== Ray-Marching & Cascades =================================================*/

// The raymarch length on cascade 0
constexpr fp_t C0_LENGTH_3D = FP(4.0);
constexpr int C0_AZ_RAYS_3D = 4;
constexpr int C0_POLAR_RAYS_3D = 4;

// rays = C0 * 2 ** (BRANCHING_EXP * n)
constexpr int AZ_BRANCHING_EXP_3D = 1;
constexpr int POLAR_BRANCHING_EXP_3D = 1;

// rays = C0 * BRANCHING_FACTOR ** n
constexpr int AZ_BRANCHING_FACTOR_3D = 3;
constexpr int POLAR_BRANCHING_FACTOR_3D = 3;

// Whether to use BRANCHING_FACTOR over BRANCHING_EXP. Requires more
// calculation, but allows for e.g. 3.
constexpr bool USE_BRANCHING_FACTOR_3D = true;

// Spatial scale exp: t_i = C0_LENGTH_3D * 2 ** (n * SPATIAL_SCALE_EXP_3D)
constexpr int SPATIAL_SCALE_EXP_3D = 1;
// Spatial length scaling factor: t_i = C0_LENGTH_3D * SPATIAL_SCALE_FACTOR_3D ** n
constexpr int SPATIAL_SCALE_FACTOR_3D = 3;
// Whether to use SCALE_FACTOR over SCALE_EXP. Equivalent to USE_BRANCHING_FACTOR_3D
constexpr bool USE_SCALE_FACTOR_3D = true;

// Whether to run the rays of the last cascade off the grid regardless of where
// they start, to correctly sample the boundary conditions.
constexpr bool LAST_CASCADE_TO_INFTY_3D = true;
// A sensible length (in voxels) that represents the edge of the grid from
// anywhere (it's not actually infinity!). May need to be increased if running
// huge models.
constexpr fp_t LAST_CASCADE_MAX_DIST_3D = FP(2e4);

// Whether to treat the subset of each cascade associated with each ray of C0
// separately. This is a good default with good memory savings, but is
// incompatible with the parallax fixes.
constexpr bool DIR_BY_DIR_3D = true;


constexpr LineCoeffCalc LINE_SCHEME_3D = LineCoeffCalc::CoreAndVoigt;

#else
#endif