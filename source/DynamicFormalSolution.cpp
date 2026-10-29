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

template <int MaxLines>
static void dynamic_compute_gamma_impl(
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

    const GammaAccumArrayState store = gamma_accum_array_state(state.precon_R, &state.rate_diag);
    const auto& I_ext = casc_state.I_ext;
    const auto& one_minus_alo = casc_state.one_minus_alo;
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
                GammaCell cell{
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
                cell.statics = gamma_static_sums(cell, local_atmos);
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
                            .I_ext = probe_fetch<RcMode>(I_ext, ray_set, probe_idx),
                            .one_minus_alo = probe_fetch<RcMode>(one_minus_alo, ray_set, probe_idx),
                            .weight = incl_quad.wmuy(theta_idx)
                        };

                        vec3 mu = inverted_mu(ray, incl_quad.muy(probe_idx.incl));
                        local_atmos.vel = (
                                flat_atmos.vx(ks) * mu(0)
                                + flat_atmos.vy(ks) * mu(1)
                                + flat_atmos.vz(ks) * mu(2)
                        );
                        const GammaRayDerived derived = accumulate_gamma_ray_lines<MaxLines>(cell, local_atmos, gamma_ray);
                        pre_int.add(gamma_ray, derived);
                    }
                }
                accumulate_gamma_continua(cell, local_atmos, pre_int);
            }
        }
    );
    yakl::fence();
}

void dynamic_compute_gamma(
    const State& state,
    const CascadeState& casc_state,
    const Fp2d& lte_scratch,
    const CascadeCalcSubset& subset
) {
    const int max_lines = gamma_max_active_lines(state.adata_host, subset.la_start, subset.la_end);
    if (max_lines <= 1) {
        dynamic_compute_gamma_impl<1>(state, casc_state, lte_scratch, subset);
    } else if (max_lines <= CORE_AND_VOIGT_MAX_LINES) {
        dynamic_compute_gamma_impl<CORE_AND_VOIGT_MAX_LINES>(state, casc_state, lte_scratch, subset);
    } else {
        dynamic_compute_gamma_impl<0>(state, casc_state, lte_scratch, subset);
    }
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
    // NOTE(cmo): Losses in this form are only supported in Vanilla  (checked in the config)
    constexpr bool march_loss = RC_CONFIG == RcConfiguration::Vanilla;
    constexpr int RcModeAloLoss = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = PREAVERAGE,
        .sample_bc = false,
        .compute_alo = true,
        .dir_by_dir = DIR_BY_DIR,
        .compute_loss = march_loss
    });
    constexpr int RcModeNoBcLoss = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = PREAVERAGE,
        .sample_bc = false,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR,
        .compute_loss = march_loss
    });
    const bool rad_loss_in_march = opts.compute_rad_loss && state.rad_loss.initialized();
    constexpr int RcStorage = RC_flags_storage_2d();
    // NOTE(cmo): Compute RC FS
    constexpr int num_subsets = subset_tasks_per_cascade<RcStorage>();
    for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
        if (casc_state.I_ext.initialized()) {
            casc_state.I_ext = FP(0.0);
            casc_state.one_minus_alo = FP(0.0);
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
        const bool c0_alo = casc_state.I_ext.initialized() && !lambda_iterate;
        if (c0_alo && rad_loss_in_march) {
            cascade_i_25d<RcModeAloLoss>(state, casc_state, 0, subset, mip_chain);
        } else if (c0_alo) {
            cascade_i_25d<RcModeAlo>(state, casc_state, 0, subset, mip_chain);
        } else if (rad_loss_in_march) {
            cascade_i_25d<RcModeNoBcLoss>(state, casc_state, 0, subset, mip_chain);
        } else {
            cascade_i_25d<RcModeNoBc>(state, casc_state, 0, subset, mip_chain);
        }
        if (opts.compute_gamma && casc_state.I_ext.initialized()) {
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
