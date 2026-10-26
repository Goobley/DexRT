#include "DynamicFormalSolution.hpp"
#include "StaticFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "GammaMatrix.hpp"
#include "GammaRayAccumulate.hpp"
#include "Atmosphere.hpp"
#include "RcUtilsModes.hpp"
#include "MergeToJ.hpp"
#include "DirectionalEmisOpacInterp.hpp"
#include "Mipmaps.hpp"

void dynamic_compute_gamma(
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
    yakl::SArray<fp_t, 1, WAVE_BATCH> wl_ray_weights(FP(0.0));
    for (int wave = 0; wave < wave_batch; ++wave) {
        wl_ray_weights(wave) = gamma_wavelength_weight(state.adata_host, la_start + wave) / fp_t(c0_dirs_to_average<RcMode>());
    }

    const GammaAccumArrayState store = gamma_accum_array_state(state.Gamma, &state.rate_diag);
    const auto& psi_star = casc_state.psi_star;
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
                const GammaCell cell{
                    .adata = adata,
                    .phi = phi,
                    .pops = pops,
                    .lte_scratch = lte_scratch,
                    .wphi = wphi,
                    .store = store,
                    .active_set = active_set,
                    .active_cont_set = active_cont_set,
                    .ks = ks,
                    .la = la,
                    .wl_ray_weight = wl_ray_weights(wave)
                };
                GammaRayAngularIntegral pre_int;

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
                        const GammaRay gamma_ray{
                            .I = probe_fetch<RcMode>(I, ray_set, probe_idx),
                            .psi_star = probe_fetch<RcMode>(psi_star, ray_set, probe_idx),
                            .weight = incl_quad.wmuy(theta_idx)
                        };

                        vec3 mu = inverted_mu(ray, incl_quad.muy(probe_idx.incl));
                        local_atmos.vel = (
                                flat_atmos.vx(ks) * mu(0)
                                + flat_atmos.vy(ks) * mu(1)
                                + flat_atmos.vz(ks) * mu(2)
                        );
                        accumulate_gamma_ray_lines(cell, local_atmos, gamma_ray);
                        pre_int.add(gamma_ray);
                    }
                }
                accumulate_gamma_continua(cell, local_atmos, pre_int);
            }
        }
    );
    yakl::fence();
}

template <int RcMode = RC_flags_storage_2d() | RC_DYNAMIC>
void compute_rad_loss(
    const State& state,
    const CascadeState& casc_state,
    const CascadeCalcSubset& subset,
    const MultiResMipChain& mip_chain
) {
    JasUnpack(subset, la_start, la_end, subset_idx);
    JasUnpack(state, phi, pops, adata, mr_block_map, rad_loss);
    using namespace ConstantsFP;
    const auto sparse_atmos = state.atmos;

    bool integrate_rad_loss = state.config.rad_loss == RadLossType::Integrated;
    int rl_la_start = la_start;
    if (state.config.store_J_on_cpu || integrate_rad_loss) {
        rl_la_start = 0;
    }

    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<RcMode>(dims, 0);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    const int num_cascades = casc_state.num_cascades;
    const auto spatial_bounds = mr_block_map.block_map.loop_bounds();

    int wave_batch = la_end - la_start;
    wave_batch = std::min(wave_batch, ray_subset.wave_batch);
    yakl::SArray<fp_t, 1, WAVE_BATCH> wl_ray_weights(FP(0.0));
    for (int wave = 0; wave < wave_batch; ++wave) {
        const int la = la_start + wave;
        const fp_t wl_weight = state.adata_host.wavelength_bin(la);
        const fp_t wl_ray_weight = four_pi * wl_weight / fp_t(c0_dirs_to_average<RcMode>());
        wl_ray_weights(wave) = wl_ray_weight;
    }

    const auto& I = casc_state.i_cascades[0];
    const auto& incl_quad = state.incl_quad;
    typedef typename RcDynamicState<RcMode>::type DynamicState;
    const bool include_flux_divergence = state.config.rad_loss_use_flux_divergence;
    constexpr fp_t flux_div_tau_0 = FP(0.1);

    dex_parallel_for(
        "compute radiative losses",
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


            for (
                int wave = ray_subset.start_wave_batch;
                wave < ray_subset.start_wave_batch + wave_batch;
                ++wave
            ) {
                fp_t loss_entry = FP(0.0);
                const int la = la_start + wave;
                const fp_t lambda = adata.wavelength(la);
                DynamicState dyn_state = get_dyn_state<DynamicState>(
                    la,
                    sparse_atmos,
                    adata,
                    phi,
                    pops,
                    mip_chain
                );

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

                        vec3 mu = inverted_mu(ray, incl_quad.muy(probe_idx.incl));
                        auto emis_opac = sample_emis_opac(SampleEmisOpacArgs<DynamicState>{
                            .ks=ks,
                            .wave=wave,
                            .la=la,
                            .lambda=lambda,
                            .mu=mu,
                            .mip_chain=mip_chain,
                            .dyn_state=dyn_state
                        });

                        const fp_t wlamu = wl_ray_weights(wave) * incl_quad.wmuy(theta_idx);
                        const fp_t intensity = probe_fetch<RcMode>(I, ray_set, probe_idx);

                        const fp_t S = emis_opac.eta / (emis_opac.chi + FP(1e-20));
                        // const fp_t direct_contrib = wlamu * emis_opac.chi * (S - intensity);
                        const fp_t direct_contrib = wlamu * (emis_opac.eta - emis_opac.chi * intensity);

                        JasUse(include_flux_divergence, flux_div_tau_0);
                        if (include_flux_divergence) {
                            ProbeIndex idx(probe_idx);
                            idx.coord(0) = std::max(probe_idx.coord(0) - 1, mr_block_map.block_map.bbox.min(0));
                            const fp_t im = probe_fetch<RcMode>(I, ray_set, idx);
                            idx.coord(0) = std::min(probe_idx.coord(0) + 1, mr_block_map.block_map.bbox.max(0) - 1);
                            const fp_t ip = probe_fetch<RcMode>(I, ray_set, idx);

                            const fp_t dFdx = (ip - im) * mu(0) / (FP(2.0) * sparse_atmos.voxel_scale);

                            idx.coord(0) = probe_idx.coord(0);
                            idx.coord(1) = std::max(probe_idx.coord(1) - 1, mr_block_map.block_map.bbox.min(1));
                            const fp_t imz = probe_fetch<RcMode>(I, ray_set, idx);
                            idx.coord(1) = std::min(probe_idx.coord(1) + 1, mr_block_map.block_map.bbox.max(1) - 1);
                            const fp_t ipz = probe_fetch<RcMode>(I, ray_set, idx);

                            const fp_t dFdz = (ipz - imz) * mu(2) / (FP(2.0) * sparse_atmos.voxel_scale);
                            const fp_t divF = dFdx + dFdz;
                            const fp_t divF_contrib = wlamu * divF;
                            // Do the exp weighting
                            const fp_t expw = std::exp(-emis_opac.chi * sparse_atmos.voxel_scale / flux_div_tau_0);
                            loss_entry += expw * direct_contrib + (FP(1.0) - expw) * divF_contrib;
                        } else {
                            // loss_entry += wlamu * (emis_opac.eta - emis_opac.chi * intensity);
                            // loss_entry += wlamu * emis_opac.chi * (S - intensity);
                            loss_entry += direct_contrib;
                        }
                    }
                }
                int rad_loss_plane = rl_la_start;
                if (!integrate_rad_loss) {
                    rad_loss_plane += wave;
                }
                rad_loss(rad_loss_plane, ks) += loss_entry;
            }
        }
    );
    yakl::fence();
}

void dynamic_formal_sol_rc(
    const State& state,
    const CascadeState& casc_state,
    DynamicFormalSolRcOptions opts
) {
    // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
    // it, for now, trust the pool allocator
    auto pops_dims = state.pops.get_dimensions();
    Fp2d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1));

    JasUnpack(casc_state, mip_chain);
    JasUnpack(opts, lambda_iterate, la_start, la_end);

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
        if (casc_state.psi_star.initialized()) {
            casc_state.psi_star = FP(0.0);
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
        if (casc_state.psi_star.initialized() && !lambda_iterate) {
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
        if (opts.compute_rad_loss && state.rad_loss.initialized()) {
            compute_rad_loss<RcModeNoBc>(
                state,
                casc_state,
                subset,
                mip_chain
            );
        }
        if (opts.compute_gamma && casc_state.psi_star.initialized()) {
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
