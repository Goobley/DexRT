#if !defined(DEXRT_MIPMAPS_3D_HPP)
#define DEXRT_MIPMAPS_3D_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State3d.hpp"
#include "BlockMap.hpp"
// #include "DirectionalEmisOpacInterp.hpp"
#include "CoreAndVoigtEmisOpac.hpp"

struct ClassicEmisOpacData3d {
    Fp1d dynamic_opac; // [ks] -- whether this cell needs its lines to be computed separately.

    void init(i64 buffer_len);
};

struct MultiResMipChain3d {
    i32 max_mip_factor;
    Fp1d emis; // [ks]
    Fp1d opac; // [ks]
    // DirectionalEmisOpacInterp dir_data;
    CoreAndVoigtData3d cav_data;
    ClassicEmisOpacData3d classic_data;
    Fp1d vx; // [ks], present if not LineCoeffCalc::Classic
    Fp1d vy; // [ks], present if not LineCoeffCalc::Classic
    Fp1d vz; // [ks], present if not LineCoeffCalc::Classic

    /// buffer_len is expected to be the one from mr_block_map, i.e. to hold all the mips.
    void init(const State3d& state, i64 buffer_len);
    void fill_mip0_atomic(const State3d& state, const Fp2d& lte_scratch, int la) const;
    // NOTE(cmo): subset bits are only needed by Directional, so ignore them for now.
    // void fill_subset_mip0_atomic(const State& state, const CascadeCalcSubset& subset, const Fp2d& n_star) const;

    /// compute the mips from mip0 (stored in the start of the arrays), for
    /// direction independent terms. Also update state.mr_block_map as
    /// necessary.
    void compute_mips(const State3d& state, int la) const;
    /// compute the mips from mip0 being stored in the start of these arrays.
    /// Currently these aren't allowed to modify state.mr_block_map.
    // void compute_subset_mips(const State& state, const CascadeCalcSubset& subset) const;
};

#else
#endif