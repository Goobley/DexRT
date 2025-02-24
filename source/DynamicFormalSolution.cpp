#include "DynamicFormalSolution.hpp"
#include "StaticFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "GammaMatrix.hpp"
#include "Atmosphere.hpp"
#include "RcUtilsModes.hpp"
#include "MergeToJ.hpp"
#include "DirectionalEmisOpacInterp.hpp"
#include "Mipmaps.hpp"

void dynamic_compute_gamma_atomic(
    const State& state,
    const CascadeState& casc_state,
    const Fp2d& lte_scratch,
    const CascadeCalcSubset& subset
) {
    JasUnpack(subset, la_start, la_end, subset_idx);
    JasUnpack(state, phi, pops, adata, wphi, mr_block_map);
    using namespace ConstantsFP;
    const auto flat_atmos = flatten<const fp_t>(state.atmos);

    constexpr int RcMode = RC_flags_storage_2d();
    if constexpr (RcMode & RC_PREAVERAGE) {
        throw std::runtime_error("Dynamic Non-LTE calculation of Gamma incompatible with PREAVERAGE. Try DIR_BY_DIR instead.");
    }

    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<RcMode>(dims, 0);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    const int num_cascades = casc_state.num_cascades;
    const auto spatial_bounds = mr_block_map.block_map.loop_bounds();

    for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
        const auto& Gamma = state.Gamma[ia];
        const auto& alo = casc_state.alo;
        const auto& I = casc_state.i_cascades[0];
        const auto& incl_quad = state.incl_quad;
        int wave_batch = la_end - la_start;
        wave_batch = std::min(wave_batch, ray_subset.wave_batch);

        const auto& wavelength = adata.wavelength;
        dex_parallel_for(
            "compute Gamma",
            FlatLoop<5>(
                spatial_bounds.dim(0),
                spatial_bounds.dim(1),
                ray_subset.num_flat_dirs,
                wave_batch,
                ray_subset.num_incl
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, int phi_idx, int wave, int theta_idx) {
                IdxGen idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 cell_coord = idx_gen.loop_coord(tile_idx, block_idx);
                ivec2 probe_coord;
                probe_coord(0) = cell_coord.x;
                probe_coord(1) = cell_coord.z;

                phi_idx += ray_subset.start_flat_dirs;
                wave += ray_subset.start_wave_batch;
                theta_idx += ray_subset.start_incl;

                const ProbeIndex probe_idx{
                    .coord=probe_coord,
                    .dir=phi_idx,
                    .incl=theta_idx,
                    .wave=wave
                };
                RayProps ray = ray_props(ray_set, num_cascades, 0, probe_idx);
                const fp_t intensity = probe_fetch<RcMode>(I, ray_set, probe_idx);
                const fp_t alo_entry = probe_fetch<RcMode>(alo, ray_set, probe_idx);

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

                vec3 mu = inverted_mu(ray, incl_quad.muy(probe_idx.incl));

                AtmosPointParams local_atmos;
                local_atmos.temperature = flat_atmos.temperature(ks);
                local_atmos.ne = flat_atmos.ne(ks);
                local_atmos.vturb = flat_atmos.vturb(ks);
                local_atmos.nhtot = flat_atmos.nh_tot(ks);
                local_atmos.nh0 = flat_atmos.nh0(ks);
                local_atmos.vel = (
                        flat_atmos.vx(ks) * mu(0)
                        + flat_atmos.vy(ks) * mu(1)
                        + flat_atmos.vz(ks) * mu(2)
                );

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
                            .n = pops,
                            .n_star_scratch = lte_scratch,
                            .k = ks,
                            .atmos = local_atmos
                        },
                        kr
                    );

                    const int offset = adata.level_start(ia);
                    const fp_t eta = pops(offset + l.j, ks) * uv.Uji;
                    const fp_t chi = pops(offset + l.i, ks) * uv.Vij - pops(offset + l.j, ks) * uv.Vji + FP(1e-20);

                    add_to_gamma<true>(GammaAccumState{
                        .eta = eta,
                        .chi = chi,
                        .uv = uv,
                        .I = intensity,
                        .alo = alo_entry,
                        .wlamu = wl_ray_weight * incl_quad.wmuy(theta_idx) * wphi(kr, ks),
                        .Gamma = Gamma,
                        .i = l.i,
                        .j = l.j,
                        .k = ks
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
                            .n = pops,
                            .n_star_scratch = lte_scratch,
                            .k = ks,
                            .atmos = local_atmos
                        },
                        kr
                    );

                    const int offset = adata.level_start(ia);
                    const fp_t eta = pops(offset + cont.j, ks) * uv.Uji;
                    const fp_t chi = pops(offset + cont.i, ks) * uv.Vij - pops(offset + cont.j, ks) * uv.Vji + FP(1e-20);

                    add_to_gamma<true>(GammaAccumState{
                        .eta = eta,
                        .chi = chi,
                        .uv = uv,
                        .I = intensity,
                        .alo = alo_entry,
                        .wlamu = wl_ray_weight * incl_quad.wmuy(theta_idx),
                        .Gamma = Gamma,
                        .i = cont.i,
                        .j = cont.j,
                        .k = ks
                    });
                }
            }
        );
    }
    yakl::fence();
}

void dynamic_compute_gamma_nonatomic(
    const State& state,
    const CascadeState& casc_state,
    const Fp2d& lte_scratch,
    const CascadeCalcSubset& subset
) {
    JasUnpack(subset, la_start, la_end, subset_idx);
    JasUnpack(state, phi, pops, adata, wphi, mr_block_map);
    using namespace ConstantsFP;
    const auto flat_atmos = flatten<const fp_t>(state.atmos);

    constexpr int RcMode = RC_flags_storage_2d();
    if constexpr (RcMode & RC_PREAVERAGE) {
        throw std::runtime_error("Dynamic Non-LTE calculation of Gamma incompatible with PREAVERAGE. Try DIR_BY_DIR instead.");
    }

    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<RcMode>(dims, 0);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    const int num_cascades = casc_state.num_cascades;
    const auto spatial_bounds = mr_block_map.block_map.loop_bounds();

    int wave_batch = la_end - la_start;
    wave_batch = std::min(wave_batch, ray_subset.wave_batch);
    Fp1dHost wl_ray_weights_h("wl_ray_weights", wave_batch);
    constexpr bool include_hc_4pi = false;
    constexpr fp_t hc_4pi = hc_kJ_nm / four_pi;
    for (int wave = 0; wave < wave_batch; ++wave) {
        const int la = la_start + wave;
        const auto& wavelength_h= state.adata_host.wavelength;
        fp_t lambda = wavelength_h(la);
        fp_t hnu_4pi = FP(1.0);
        if (include_hc_4pi) {
            hnu_4pi *= hc_4pi;
        }
        fp_t wl_weight = lambda / hnu_4pi;
        if (la == 0) {
            wl_weight *= FP(0.5) * (wavelength_h(1) - wavelength_h(0));
        } else if (la == wavelength_h.extent(0) - 1) {
            wl_weight *= FP(0.5) * (
                wavelength_h(wavelength_h.extent(0) - 1) - wavelength_h(wavelength_h.extent(0) - 2)
            );
        } else {
            wl_weight *= FP(0.5) * (wavelength_h(la + 1) - wavelength_h(la - 1));
        }
        const fp_t wl_ray_weight = wl_weight / fp_t(c0_dirs_to_average<RcMode>());
        wl_ray_weights_h(wave) = wl_ray_weight;
    }
    Fp1d wl_ray_weights(wl_ray_weights_h.createDeviceCopy());

    for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
        const auto& Gamma = state.Gamma[ia];
        const auto& alo = casc_state.alo;
        const auto& I = casc_state.i_cascades[0];
        const auto& incl_quad = state.incl_quad;

        dex_parallel_for(
            "compute Gamma",
            FlatLoop<2>(
                spatial_bounds.dim(0),
                spatial_bounds.dim(1)
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 cell_coord = idx_gen.loop_coord(tile_idx, block_idx);
                ivec2 probe_coord;
                probe_coord(0) = cell_coord.x;
                probe_coord(1) = cell_coord.z;

                AtmosPointParams local_atmos;
                local_atmos.temperature = flat_atmos.temperature(ks);
                local_atmos.ne = flat_atmos.ne(ks);
                local_atmos.vturb = flat_atmos.vturb(ks);
                local_atmos.nhtot = flat_atmos.nh_tot(ks);
                local_atmos.nh0 = flat_atmos.nh0(ks);

                for (
                    int wave = ray_subset.start_wave_batch;
                    wave < ray_subset.start_wave_batch + wave_batch;
                    ++wave
                ) {
                    const int la = la_start + wave;
                    const auto active_set = slice_active_set(adata, la);
                    const auto active_cont_set = slice_active_cont_set(adata, la);

                    for (
                        int phi_idx = ray_subset.start_flat_dirs;
                        phi_idx < ray_subset.start_flat_dirs + ray_subset.num_flat_dirs;
                        phi_idx += 1
                    ) {
                        for (
                            int theta_idx = ray_subset.start_incl;
                            theta_idx < ray_subset.start_incl + ray_subset.num_incl;
                            theta_idx += 1
                        ) {
                            const ProbeIndex probe_idx{
                                .coord=probe_coord,
                                .dir=phi_idx,
                                .incl=theta_idx,
                                .wave=wave
                            };
                            RayProps ray = ray_props(ray_set, num_cascades, 0, probe_idx);
                            const fp_t intensity = probe_fetch<RcMode>(I, ray_set, probe_idx);
                            const fp_t alo_entry = probe_fetch<RcMode>(alo, ray_set, probe_idx);

                            vec3 mu = inverted_mu(ray, incl_quad.muy(probe_idx.incl));
                            local_atmos.vel = (
                                    flat_atmos.vx(ks) * mu(0)
                                    + flat_atmos.vy(ks) * mu(1)
                                    + flat_atmos.vz(ks) * mu(2)
                            );

                            const int offset = adata.level_start(ia);
                            const int kr_base = adata.line_start(ia);
                            const int max_line = kr_base + adata.num_line(ia);
                            for (int kra = 0; kra < active_set.extent(0); ++kra) {
                                const int kr = active_set(kra);
                                if (!(kr >= kr_base && kr < max_line)) {
                                    continue;
                                }
                                const auto& l = adata.lines(kr);
                                const UV uv = compute_uv_line(
                                    EmisOpacState<>{
                                        .adata = adata,
                                        .profile = phi,
                                        .la = la,
                                        .n = pops,
                                        .n_star_scratch = lte_scratch,
                                        .k = ks,
                                        .atmos = local_atmos
                                    },
                                    kr,
                                    UvOptions{
                                        .include_hc_4pi = false
                                    }
                                );

                                fp_t eta = pops(offset + l.j, ks) * uv.Uji;
                                fp_t chi = pops(offset + l.i, ks) * uv.Vij - pops(offset + l.j, ks) * uv.Vji;
                                if (!include_hc_4pi) {
                                    eta *= hc_4pi;
                                    chi *= hc_4pi;
                                }
                                chi += FP(1e-20);
                                const fp_t wlamu = wl_ray_weights(wave) * incl_quad.wmuy(theta_idx) * wphi(kr, ks);

                                add_to_gamma<false>(GammaAccumState{
                                    .eta = eta,
                                    .chi = chi,
                                    .uv = uv,
                                    .I = intensity,
                                    .alo = alo_entry,
                                    .wlamu = wlamu,
                                    .Gamma = Gamma,
                                    .i = l.i,
                                    .j = l.j,
                                    .k = ks
                                });
                            }
                            const int kr_base_c = adata.cont_start(ia);
                            const int max_cont = kr_base_c + adata.num_cont(ia);
                            for (int kra = 0; kra < active_cont_set.extent(0); ++kra) {
                                const int kr = active_cont_set(kra);
                                if (!(kr >= kr_base_c && kr < max_cont)) {
                                    continue;
                                }
                                const auto& cont = adata.continua(kr);

                                const UV uv = compute_uv_cont(
                                    EmisOpacState<>{
                                        .adata = adata,
                                        .profile = phi,
                                        .la = la,
                                        .n = pops,
                                        .n_star_scratch = lte_scratch,
                                        .k = ks,
                                        .atmos = local_atmos
                                    },
                                    kr,
                                    UvOptions{
                                        .include_hc_4pi = false
                                    }
                                );

                                fp_t eta = pops(offset + cont.j, ks) * uv.Uji;
                                fp_t chi = pops(offset + cont.i, ks) * uv.Vij - pops(offset + cont.j, ks) * uv.Vji;
                                if (!include_hc_4pi) {
                                    eta *= hc_4pi;
                                    chi *= hc_4pi;
                                }
                                chi += FP(1e-20);

                                add_to_gamma<false>(GammaAccumState{
                                    .eta = eta,
                                    .chi = chi,
                                    .uv = uv,
                                    .I = intensity,
                                    .alo = alo_entry,
                                    .wlamu = wl_ray_weights(wave) * incl_quad.wmuy(theta_idx),
                                    .Gamma = Gamma,
                                    .i = cont.i,
                                    .j = cont.j,
                                    .k = ks
                                });
                            }
                        }
                    }
                }
            }
        );
        yakl::fence();
    }
}

void dynamic_compute_gamma(
    const State& state,
    const CascadeState& casc_state,
    const Fp2d& lte_scratch,
    const CascadeCalcSubset& subset
) {
    constexpr bool atomic = false;
    if constexpr (atomic) {
        dynamic_compute_gamma_atomic(state, casc_state, lte_scratch, subset);
    } else {
        dynamic_compute_gamma_nonatomic(state, casc_state, lte_scratch, subset);
    }
}

void dynamic_formal_sol_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate, int la_start, int la_end) {
    // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
    // it, for now, trust the pool allocator
    auto pops_dims = state.pops.get_dimensions();
    Fp2d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1));

    JasUnpack(casc_state, mip_chain);

    if (la_end == -1) {
        la_end = la_start + 1;
    }
    if ((la_end - la_start) > WAVE_BATCH) {
        assert(false && "Wavelength batch too big.");
    }
    // NOTE(cmo): lte_scratch filled here
    mip_chain.fill_mip0_atomic(state, lte_scratch, la_start, la_end);
    mip_chain.compute_mips(state, la_start, la_end);

    constexpr int RcModeBc = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = PREAVERAGE,
        .sample_bc = true,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR
    });
    constexpr int RcModeNoBc = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = PREAVERAGE,
        .sample_bc = false,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR
    });
    constexpr int RcModeAlo = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = PREAVERAGE,
        .sample_bc = false,
        .compute_alo = true,
        .dir_by_dir = DIR_BY_DIR
    });
    constexpr int RcStorage = RC_flags_storage_2d();
    // NOTE(cmo): Compute RC FS
    constexpr int num_subsets = subset_tasks_per_cascade<RcStorage>();
    for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
        if (casc_state.alo.initialized()) {
            casc_state.alo = FP(0.0);
        }
        CascadeCalcSubset subset{
            .la_start=la_start,
            .la_end=la_end,
            .subset_idx=subset_idx
        };
        mip_chain.fill_subset_mip0_atomic(state, subset, lte_scratch);
        mip_chain.compute_subset_mips(state, subset);
        cascade_i_25d<RcModeBc>(
            state,
            casc_state,
            casc_state.num_cascades,
            subset,
            mip_chain
        );
        for (int casc_idx = casc_state.num_cascades - 1; casc_idx >= 1; --casc_idx) {
            cascade_i_25d<RcModeNoBc>(
                state,
                casc_state,
                casc_idx,
                subset,
                mip_chain
            );
        }
        if (casc_state.alo.initialized() && !lambda_iterate) {
            cascade_i_25d<RcModeAlo>(
                state,
                casc_state,
                0,
                subset,
                mip_chain
            );
        } else {
            cascade_i_25d<RcModeNoBc>(
                state,
                casc_state,
                0,
                subset,
                mip_chain
            );
        }
        if (casc_state.alo.initialized()) {
            // NOTE(cmo): Add terms to Gamma
            dynamic_compute_gamma(
                state,
                casc_state,
                lte_scratch,
                subset
            );
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
