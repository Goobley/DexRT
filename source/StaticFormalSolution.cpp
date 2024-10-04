#include "StaticFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "GammaMatrix.hpp"
#include "Atmosphere.hpp"
#include "RcUtilsModes.hpp"
#include "MergeToJ.hpp"

void static_compute_gamma(
    const State& state,
    const CascadeState& casc_state,
    const Fp3d& lte_scratch,
    const CascadeCalcSubset& subset
) {
    JasUnpack(subset, la_start, la_end, subset_idx);
    using namespace ConstantsFP;
    const auto flat_atmos = flatten<const fp_t>(state.atmos);
    const auto& adata = state.adata;
    const auto& phi = state.phi;
    const auto& pops = state.pops;
    const auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    const auto flat_lte_pops = lte_scratch.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    const bool sparse_calc = state.config.sparse_calculation;

    constexpr int RcMode = RC_flags_storage();

    for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
        const auto& Gamma = state.Gamma[ia];
        const auto flat_Gamma = Gamma.reshape<3>(Dims(
            Gamma.extent(0),
            Gamma.extent(1),
            Gamma.extent(2) * Gamma.extent(3)
        ));
        const auto& alo = state.alo;
        const auto& I = casc_state.i_cascades[0];
        const auto& incl_quad = state.incl_quad;
        int wave_batch = la_end - la_start;
        const auto& wphi = state.wphi.reshape<2>(Dims(state.wphi.extent(0), state.wphi.extent(1) * state.wphi.extent(2)));

        CascadeStorage dims = state.c0_size;
        i64 num_active_space = flat_atmos.temperature.extent(0);
        yakl::Array<i32, 2, yakl::memDevice> probe_spatial_lookup;
        if (sparse_calc) {
            probe_spatial_lookup = casc_state.probes_to_compute[0];
            num_active_space = probe_spatial_lookup.extent(0);
        }

        const auto& wavelength = adata.wavelength;
        parallel_for(
            "compute Gamma",
            SimpleBounds<4>(
                num_active_space,
                dims.num_flat_dirs,
                wave_batch,
                dims.num_incl
            ),
            YAKL_LAMBDA (i64 k_active, /*i64 k, */ int phi_idx, int wave, int theta_idx) {
                // k_active may or may not be k. For sparse calculation, it's
                // the index into the active probe array, otherwise, it is k.
                i64 k = k_active;
                if (sparse_calc) {
                    int u = probe_spatial_lookup(k_active, 0);
                    int v = probe_spatial_lookup(k_active, 1);
                    k = v * dims.num_probes(0) + u;
                }

                // NOTE(cmo): As in the loop over probes we iterate as [v, u] (u
                // fast-running), but index as [u, v], i.e. dims.num_probes(0) =
                // dim(u). Typical definition of k = u * Nv + v, but here we do
                // loop index k = v * Nu + u where Nu = dims.num_probes(0). This
                // preserves our iteration ordering
                ivec2 probe_coord;
                probe_coord(0) = k % dims.num_probes(0);
                probe_coord(1) = k / dims.num_probes(0);
                const ProbeStorageIndex probe_idx{
                    .coord=probe_coord,
                    .dir=phi_idx,
                    .incl=theta_idx,
                    .wave=wave
                };
                const fp_t intensity = probe_fetch<RcMode>(I, dims, probe_idx);
                const fp_t alo_entry = probe_fetch<RcMode>(alo, dims, probe_idx);

                const int la = la_start + wave;
                fp_t lambda = wavelength(la);
                const fp_t hnu_4pi = hc_kJ_nm / (four_pi * lambda);
                fp_t wl_weight = FP(1.0) / hnu_4pi;
                if (la == 0) {
                    wl_weight *= FP(0.5) * (wavelength(1) - wavelength(0));
                } else if (la == wavelength.extent(0) - 1) {
                    wl_weight *= FP(0.5) * (
                        wavelength(wavelength.extent(0) - 1) - wavelength(wavelength.extent(0) - 2)
                    );
                } else {
                    wl_weight *= FP(0.5) * (wavelength(la + 1) - wavelength(la - 1));
                }
                const fp_t wl_ray_weight = wl_weight / fp_t(c0_dirs_to_average<RcMode>());

                AtmosPointParams local_atmos;
                local_atmos.temperature = flat_atmos.temperature(k);
                local_atmos.ne = flat_atmos.ne(k);
                local_atmos.vturb = flat_atmos.vturb(k);
                local_atmos.nhtot = flat_atmos.nh_tot(k);
                local_atmos.nh0 = flat_atmos.nh0(k);
                local_atmos.vel = FP(0.0);

                const int kr_base = adata.line_start(ia);
                for (int kr_atom = 0; kr_atom < adata.num_line(ia); ++kr_atom) {
                    const int kr = kr_base + kr_atom;
                    const auto& l = adata.lines(kr);
                    if (!l.is_active(la)) {
                        continue;
                    }

                    const UV uv = compute_uv_line(
                        EmisOpacState<>{
                            .adata = adata,
                            .profile = phi,
                            .la = la,
                            .n = flat_pops,
                            .n_star_scratch = flat_lte_pops,
                            .k = k,
                            .atmos = local_atmos
                        },
                        kr
                    );

                    const int offset = adata.level_start(ia);
                    const fp_t eta = flat_pops(offset + l.j, k) * uv.Uji;
                    const fp_t chi = flat_pops(offset + l.i, k) * uv.Vij - flat_pops(offset + l.j, k) * uv.Vji;


                    add_to_gamma<true>(GammaAccumState{
                        .eta = eta,
                        .chi = chi,
                        .uv = uv,
                        .I = intensity,
                        .alo = alo_entry,
                        .wlamu = wl_ray_weight * incl_quad.wmuy(theta_idx) * wphi(kr, k),
                        .Gamma = flat_Gamma,
                        .i = l.i,
                        .j = l.j,
                        .k = k
                    });
                }
                const int kr_base_c = adata.cont_start(ia);
                for (int kr_atom = 0; kr_atom < adata.num_cont(ia); ++kr_atom) {
                    const int kr = kr_base_c + kr_atom;
                    const auto& cont = adata.continua(kr);
                    if (!cont.is_active(la)) {
                        continue;
                    }

                    const UV uv = compute_uv_cont(
                        EmisOpacState<>{
                            .adata = adata,
                            .profile = phi,
                            .la = la,
                            .n = flat_pops,
                            .n_star_scratch = flat_lte_pops,
                            .k = k,
                            .atmos = local_atmos
                        },
                        kr
                    );

                    const int offset = adata.level_start(ia);
                    const fp_t eta = flat_pops(offset + cont.j, k) * uv.Uji;
                    const fp_t chi = flat_pops(offset + cont.i, k) * uv.Vij - flat_pops(offset + cont.j, k) * uv.Vji;

                    add_to_gamma<true>(GammaAccumState{
                        .eta = eta,
                        .chi = chi,
                        .uv = uv,
                        .I = intensity,
                        .alo = alo_entry,
                        .wlamu = wl_ray_weight * incl_quad.wmuy(theta_idx),
                        .Gamma = flat_Gamma,
                        .i = cont.i,
                        .j = cont.j,
                        .k = k
                    });
                }
            }
        );
    }
    yakl::fence();
}

// void static_formal_sol_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate, int la_start, int la_end) {
//     assert(state.config.mode == DexrtMode::Lte || state.config.mode == DexrtMode::NonLte);
//     auto& atmos = state.atmos;
//     auto& phi = state.phi;
//     auto& pops = state.pops;
//     auto& adata = state.adata;
//     auto& eta = casc_state.eta;
//     auto& chi = casc_state.chi;

//     // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
//     // it, for now, trust the pool allocator
//     auto pops_dims = pops.get_dimensions();
//     Fp3d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1), pops_dims(2));
//     const bool sparse_calc = state.config.sparse_calculation;

//     if (la_end == -1) {
//         la_end = la_start + 1;
//     }
//     if ((la_end - la_start) > WAVE_BATCH) {
//         assert(false && "Wavelength batch too big.");
//     }
//     int wave_batch = la_end - la_start;

//     auto flat_temperature = atmos.temperature.collapse();
//     auto flat_ne = atmos.ne.collapse();
//     auto flat_vturb = atmos.vturb.collapse();
//     auto flat_nhtot = atmos.nh_tot.collapse();
//     auto flat_nh0 = atmos.nh0.collapse();
//     auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
//     auto flat_n_star = lte_scratch.reshape<2>(Dims(lte_scratch.extent(0), lte_scratch.extent(1) * lte_scratch.extent(2)));
//     auto flat_eta = eta.reshape<2>(Dims(eta.extent(0) * eta.extent(1), eta.extent(2)));
//     auto flat_chi = chi.reshape<2>(Dims(chi.extent(0) * chi.extent(1), chi.extent(2)));
//     auto flat_active = state.active.reshape<1>(Dims(state.active.extent(0), state.active.extent(1)));
//     // NOTE(cmo): Compute emis/opac
//     parallel_for(
//         "Compute eta, chi",
//         SimpleBounds<2>(flat_temperature.extent(0), wave_batch),
//         YAKL_LAMBDA (int64_t k, int wave) {
//             if (sparse_calc && !flat_active(k)) {
//                 return;
//             }
//             AtmosPointParams local_atmos;
//             local_atmos.temperature = flat_temperature(k);
//             local_atmos.ne = flat_ne(k);
//             local_atmos.vturb = flat_vturb(k);
//             local_atmos.nhtot = flat_nhtot(k);
//             local_atmos.nh0 = flat_nh0(k);
//             const int la = la_start + wave;
//             auto active_set = slice_active_set(adata, la);
//             auto active_cont = slice_active_cont_set(adata, la);
//             auto result = emis_opac(
//                 EmisOpacState<fp_t>{
//                     .adata = adata,
//                     .profile = phi,
//                     .la = la_start + wave,
//                     .n = flat_pops,
//                     .n_star_scratch = flat_n_star,
//                     .k = k,
//                     .atmos = local_atmos,
//                     .active_set = active_set,
//                     .active_set_cont = active_cont
//                 }
//             );
//             flat_eta(k, wave) = result.eta;
//             flat_chi(k, wave) = result.chi;
//         }
//     );
//     if (state.alo.initialized()) {
//         state.alo = FP(0.0);
//     }
//     yakl::fence();
//     constexpr int RcModeBc = RC_flags_pack(RcFlags{
//         .dynamic = false,
//         .preaverage = PREAVERAGE,
//         .sample_bc = true,
//         .compute_alo = false,
//         .dir_by_dir = DIR_BY_DIR,
//         .dynamic_interp = false
//     });
//     constexpr int RcModeNoBc = RC_flags_pack(RcFlags{
//         .dynamic = false,
//         .preaverage = PREAVERAGE,
//         .sample_bc = false,
//         .compute_alo = false,
//         .dir_by_dir = DIR_BY_DIR,
//         .dynamic_interp = false
//     });
//     constexpr int RcModeAlo = RC_flags_pack(RcFlags{
//         .dynamic = false,
//         .preaverage = PREAVERAGE,
//         .sample_bc = false,
//         .compute_alo = true,
//         .dir_by_dir = DIR_BY_DIR,
//         .dynamic_interp = false
//     });
//     constexpr int RcStorage = RC_flags_storage();

//     // NOTE(cmo): Compute RC FS
//     constexpr int num_subets = subset_tasks_per_cascade<RcStorage>();
//     for (int subset_idx = 0; subset_idx < num_subets; ++subset_idx) {
//         CascadeCalcSubset subset{
//             .la_start=la_start,
//             .la_end=la_end,
//             .subset_idx=subset_idx
//         };
//         cascade_i_25d<RcModeBc>(
//             state,
//             casc_state,
//             casc_state.num_cascades,
//             subset
//         );
//         yakl::fence();
//         for (int casc_idx = casc_state.num_cascades - 1; casc_idx >= 1; --casc_idx) {
//             cascade_i_25d<RcModeNoBc>(
//                 state,
//                 casc_state,
//                 casc_idx,
//                 subset
//             );
//             yakl::fence();
//         }
//         if (state.alo.initialized() && !lambda_iterate) {
//             cascade_i_25d<RcModeAlo>(
//                 state,
//                 casc_state,
//                 0,
//                 subset
//             );
//         } else {
//             cascade_i_25d<RcModeNoBc>(
//                 state,
//                 casc_state,
//                 0,
//                 subset
//             );
//         }
//         yakl::fence();
//         if (state.alo.initialized()) {
//             // NOTE(cmo): Add terms to Gamma
//             if (lambda_iterate) {
//                 state.alo = FP(0.0);
//                 yakl::fence();
//             }
//             static_compute_gamma(
//                 state,
//                 casc_state,
//                 lte_scratch,
//                 subset
//             );
//         }
//         merge_c0_to_J(
//             casc_state.i_cascades[0],
//             state.c0_size,
//             state.J,
//             state.incl_quad,
//             la_start,
//             la_end
//         );
//         yakl::fence();
//     }
// }

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
        mip_chain.compute_mips(state, subset);

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