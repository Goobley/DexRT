#if !defined(DEXRT_MIPMAPS_HPP)
#define DEXRT_MIPMAPS_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "BlockMap.hpp"
#include "DirectionalEmisOpacInterp.hpp"
#include "CoreAndVoigtEmisOpac.hpp"

struct ClassicEmisOpacData {
    Fp2d dynamic_opac; // [ks, wave_batch] -- whether this cell needs its lines to be computed separately.

    void init(i64 buffer_len, i32 wave_batch);
};

struct MultiResMipChain {
    i32 max_mip_factor;
    Fp2d emis; // [ks, wave_batch]
    Fp2d opac; // [ks, wave_batch]
    DirectionalEmisOpacInterp dir_data;
    CoreAndVoigtData cav_data;
    ClassicEmisOpacData classic_data;
    Fp1d vx; // [ks], present if not LineCoeffCalc::Classic
    Fp1d vy; // [ks], present if not LineCoeffCalc::Classic
    Fp1d vz; // [ks], present if not LineCoeffCalc::Classic

    /// buffer_len is expected to be the one from mr_block_map, i.e. to hold all the mips.
    void init(const State& state, i64 buffer_len, i32 wave_batch);
    void fill_mip0_atomic(const State& state, const Fp2d& lte_scratch, int la_start, int la_end) const;
    void fill_subset_mip0_atomic(const State& state, const CascadeCalcSubset& subset, const Fp2d& n_star) const;

    /// compute the mips from mip0 (stored in the start of the arrays), for
    /// direction independent terms. Also update state.mr_block_map as
    /// necessary.
    void compute_mips(const State& state, int la_start, int la_end) const;
    /// compute the mips from mip0 being stored in the start of these arrays.
    /// Currently these aren't allowed to modify state.mr_block_map.
    void compute_subset_mips(const State& state, const CascadeCalcSubset& subset) const;
};

#else
#endif