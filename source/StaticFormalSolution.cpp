#include "StaticFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "GammaMatrix.hpp"
#include "Atmosphere.hpp"
#include "RcUtilsModes.hpp"
#include "MergeToJ.hpp"

void static_formal_sol_given_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate, int la_start, int la_end) {
    assert(state.config.mode == DexrtMode::GivenFs);
    JasUnpack(state, mr_block_map);
    JasUnpack(casc_state, mip_chain);
    const auto& block_map = mr_block_map.block_map;

    if (la_end == -1) {
        la_end = la_start + 1;
    }
    if ((la_end - la_start) > WAVE_BATCH) {
        assert(false && "Wavelength batch too big.");
    }
    int wave_batch = la_end - la_start;

    auto& eta_store = state.given_state.emis;
    auto& chi_store = state.given_state.opac;
    auto bounds = block_map.loop_bounds();
    parallel_for(
        "Copy eta, chi",
        MDRange<3>({0, 0, 0}, {bounds.m_upper[0], bounds.m_upper[1], wave_batch}),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
            IndexGen<BLOCK_SIZE> idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            mip_chain.emis(ks, wave) = eta_store(coord.z, coord.x, la_start + wave);
            mip_chain.opac(ks, wave) = chi_store(coord.z, coord.x, la_start + wave);
        }
    );
    yakl::fence();
    mip_chain.compute_mips(state, la_start, la_end);

    constexpr int RcModeBc = RC_flags_pack(RcFlags{
        .dynamic = false,
        .preaverage = PREAVERAGE,
        .sample_bc = true,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR
    });
    constexpr int RcModeNoBc = RC_flags_pack(RcFlags{
        .dynamic = false,
        .preaverage = PREAVERAGE,
        .sample_bc = false,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR
    });
    constexpr int RcStorage = RC_flags_storage();

    // NOTE(cmo): Compute RC FS
    constexpr int num_subsets = subset_tasks_per_cascade<RcStorage>();
    for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
    // for (int subset_idx = 0; subset_idx < 1; ++subset_idx) {
        CascadeCalcSubset subset{
            .la_start=la_start,
            .la_end=la_end,
            .subset_idx=subset_idx
        };
        mip_chain.compute_subset_mips(state, subset);

        cascade_i_25d<RcModeBc>(
            state,
            casc_state,
            casc_state.num_cascades,
            subset,
            mip_chain
        );
        yakl::fence();
        for (int casc_idx = casc_state.num_cascades - 1; casc_idx >= 0; --casc_idx) {
            cascade_i_25d<RcModeNoBc>(
                state,
                casc_state,
                casc_idx,
                subset,
                mip_chain
            );
            yakl::fence();
        }
        merge_c0_to_J(
            casc_state,
            state.mr_block_map,
            state.J,
            state.incl_quad,
            la_start,
            la_end
        );
        yakl::fence();
    }
}