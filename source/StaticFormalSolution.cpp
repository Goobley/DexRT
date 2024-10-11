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
    const auto& block_map = mr_block_map.block_map;

    if (la_end == -1) {
        la_end = la_start + 1;
    }
    if ((la_end - la_start) > WAVE_BATCH) {
        assert(false && "Wavelength batch too big.");
    }
    int wave_batch = la_end - la_start;

    MultiResMipChain mip_chain;
    mip_chain.init(state, mr_block_map.buffer_len(), wave_batch);
    auto& eta_store = state.given_state.emis;
    auto& chi_store = state.given_state.opac;
    auto bounds = block_map.loop_bounds();
    parallel_for(
        "Copy eta, chi",
        SimpleBounds<3>(bounds.dim(0), bounds.dim(1), wave_batch),
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
        CascadeCalcSubset subset{
            .la_start=la_start,
            .la_end=la_end,
            .subset_idx=subset_idx
        };
        mip_chain.compute_subset_mips(state, subset, la_start, la_end);

        std::vector<Fp3d> eta_mips;
        std::vector<Fp3d> chi_mips;
        yakl::Array<i32, 2, yakl::memDevice> max_mip_level("max mip level reshape", state.J.extent(1) / BLOCK_SIZE, state.J.extent(2) / BLOCK_SIZE);
        // std::vector<Fp4d> eta_a_mips;
        // std::vector<Fp4d> chi_a_mips;
        for (int mip_level=0; mip_level <= state.mr_block_map.max_mip_level; ++mip_level) {
            const int vox_size = (1 << mip_level);
            Fp3d emis_entry("eta", state.J.extent(1) / vox_size, state.J.extent(2) / vox_size, wave_batch);
            Fp3d opac_entry("chi", state.J.extent(1) / vox_size, state.J.extent(2) / vox_size, wave_batch);

            auto bounds = block_map.loop_bounds(vox_size);
            parallel_for(
                SimpleBounds<3>(
                    bounds.dim(0),
                    bounds.dim(1),
                    wave_batch
                ),
                YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
                    MRIdxGen idx_gen(mr_block_map);
                    i64 ks = idx_gen.loop_idx(mip_level, tile_idx, block_idx);
                    Coord2 coord = idx_gen.loop_coord(mip_level, tile_idx, block_idx);

                    emis_entry(coord.z / vox_size, coord.x / vox_size, wave) = mip_chain.emis(ks, wave);
                    opac_entry(coord.z / vox_size, coord.x / vox_size, wave) = mip_chain.opac(ks, wave);
                }
            );
            parallel_for(
                SimpleBounds<1>(
                    bounds.dim(0)
                ),
                YAKL_LAMBDA (i64 tile_idx) {
                    MRIdxGen idx_gen(mr_block_map);
                    Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);

                    max_mip_level(tile_coord.z, tile_coord.x) = mr_block_map.lookup.get(tile_coord.x, tile_coord.z);
                }
            );
            yakl::fence();
            eta_mips.push_back(emis_entry);
            chi_mips.push_back(opac_entry);
        }
        yakl::SimpleNetCDF nc;
        // NOTE(cmo): Mips are being generated, but max_mip_level, i.e. mr_block_map not being filled
        nc.create("mip_data.nc", yakl::NETCDF_MODE_REPLACE);
        for (int mip_level = 0; mip_level < eta_mips.size(); ++mip_level) {
            std::string zn = fmt::format("z{}", mip_level);
            std::string xn = fmt::format("x{}", mip_level);
            nc.write(eta_mips[mip_level], fmt::format("emis_{}", mip_level), {zn, xn, "wave"});
            nc.write(chi_mips[mip_level], fmt::format("opac_{}", mip_level), {zn, xn, "wave"});
        }
        nc.write(max_mip_level, "max_mip_level", {"z_tiles", "x_tiles"});
        nc.close();

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
            casc_state.i_cascades[0],
            state.c0_size,
            state.J,
            state.incl_quad,
            la_start,
            la_end
        );
        yakl::fence();
    }
}