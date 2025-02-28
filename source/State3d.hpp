#if !defined(DEXRT_STATE_3D_HPP)
#define DEXRT_STATE_3D_HPP

#include "Config.hpp"
#include "DexrtConfig.hpp"
#include "RcUtilsModes3d.hpp"
#include "BlockMap.hpp"

struct GivenEmisOpac3d {
    fp_t voxel_scale;
    Fp4d emis;
    Fp4d opac;
};

struct State3d {
    DexrtConfig config;
    CascadeStorage3d c0_size;
    MultiResBlockMap<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3> mr_block_map;
    GivenEmisOpac3d given_state;
    Fp2d J; // [la, ks]
};

#else
#endif