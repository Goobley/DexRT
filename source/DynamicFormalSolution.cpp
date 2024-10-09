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

void dynamic_compute_gamma(
    const State& state,
    const CascadeState& casc_state,
    const Fp3d& lte_scratch,
    const CascadeCalcSubset& subset
) {
    JasUnpack(subset, la_start, la_end, subset_idx);
    using namespace ConstantsFP;
    const auto flat_atmos = flatten<const fp_t>(state.atmos);
    const auto& phi = state.phi;
    const auto& pops = state.pops;
    const auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    const auto flat_lte_pops = lte_scratch.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    const auto& adata = state.adata;
    const auto& wphi = state.wphi.reshape<2>(Dims(state.wphi.extent(0), state.wphi.extent(1) * state.wphi.extent(2)));
    const bool sparse_calc = state.config.sparse_calculation;

    constexpr int RcMode = RC_flags_storage();
    if constexpr (RcMode & RC_PREAVERAGE) {
        throw std::runtime_error("Dynamic Non-LTE calculation of Gamma incompatible with PREAVERAGE. Try DIR_BY_DIR instead.");
    }

    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    const int num_cascades = casc_state.num_cascades;

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
        wave_batch = std::min(wave_batch, ray_subset.wave_batch);

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
                ray_subset.num_flat_dirs,
                wave_batch,
                ray_subset.num_incl
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

                vec3 mu;
                const fp_t cos_theta = incl_quad.muy(probe_idx.incl);
                const fp_t sin_theta = std::sqrt(FP(1.0) - square(cos_theta));
                mu(0) = ray.dir(0) * sin_theta;
                mu(1) = cos_theta;
                mu(2) = ray.dir(1) * sin_theta;

                AtmosPointParams local_atmos;
                local_atmos.temperature = flat_atmos.temperature(k);
                local_atmos.ne = flat_atmos.ne(k);
                local_atmos.vturb = flat_atmos.vturb(k);
                local_atmos.nhtot = flat_atmos.nh_tot(k);
                local_atmos.nh0 = flat_atmos.nh0(k);
                local_atmos.vel = (
                        flat_atmos.vx(k) * mu(0)
                        + flat_atmos.vy(k) * mu(1)
                        + flat_atmos.vz(k) * mu(2)
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
                            .n = flat_pops,
                            .n_star_scratch = flat_lte_pops,
                            .k = k,
                            .atmos = local_atmos
                        },
                        kr
                    );

                    const int offset = adata.level_start(ia);
                    const fp_t eta = flat_pops(offset + l.j, k) * uv.Uji;
                    const fp_t chi = flat_pops(offset + l.i, k) * uv.Vij - flat_pops(offset + l.j, k) * uv.Vji + FP(1e-20);

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
                    const fp_t chi = flat_pops(offset + cont.i, k) * uv.Vij - flat_pops(offset + cont.j, k) * uv.Vji + FP(1e-20);

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

void dynamic_formal_sol_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate, int la_start, int la_end) {
    auto& atmos = state.atmos;
    auto& phi = state.phi;
    auto& pops = state.pops;
    auto& adata = state.adata;
    auto& dynamic_opac = state.dynamic_opac;

    const bool sparse_calc = state.config.sparse_calculation;
    // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
    // it, for now, trust the pool allocator
    auto pops_dims = pops.get_dimensions();
    Fp3d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1), pops_dims(2));

    if (la_end == -1) {
        la_end = la_start + 1;
    }
    if ((la_end - la_start) > WAVE_BATCH) {
        assert(false && "Wavelength batch too big.");
    }
    int wave_batch = la_end - la_start;
    MultiResMipChain mip_chain;
    mip_chain.init(state, state.mr_block_map.buffer_len(), wave_batch);

    const auto flatmos = flatten<const fp_t>(atmos);
    auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    auto flat_n_star = lte_scratch.reshape<2>(Dims(lte_scratch.extent(0), lte_scratch.extent(1) * lte_scratch.extent(2)));
    auto flat_dynamic_opac = dynamic_opac.reshape<2>(
        Dims(
            dynamic_opac.extent(0) * dynamic_opac.extent(1),
            dynamic_opac.extent(2)
        )
    );
    auto flat_active = state.active.reshape<1>(Dims(state.active.extent(0) * state.active.extent(1)));
    const auto& block_map = state.mr_block_map.block_map;
    auto bounds = block_map.loop_bounds();
    parallel_for(
        "Compute eta, chi",
        SimpleBounds<3>(bounds.dim(0), bounds.dim(1), wave_batch),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx, int wave) {
            IndexGen<BLOCK_SIZE> idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            i64 k = idx_gen.full_flat_idx(coord.x, coord.z);

            AtmosPointParams local_atmos;
            local_atmos.temperature = flatmos.temperature(k);
            local_atmos.ne = flatmos.ne(k);
            local_atmos.vturb = flatmos.vturb(k);
            local_atmos.nhtot = flatmos.nh_tot(k);
            local_atmos.nh0 = flatmos.nh0(k);
            const fp_t v_norm = std::sqrt(
                    square(flatmos.vx(k))
                    + square(flatmos.vy(k))
                    + square(flatmos.vz(k))
            );
            const int la = la_start + wave;
            int governing_atom = adata.governing_trans(la).atom;

            bool static_only = v_norm >= (ANGLE_INVARIANT_THERMAL_VEL_FRAC * thermal_vel(
                adata.mass(governing_atom),
                local_atmos.temperature
            ));
            auto active_set = slice_active_set(adata, la);
            const bool no_lines = (active_set.extent(0) == 0);
            flat_dynamic_opac(k, wave) = static_only || no_lines;
            EmisOpacMode mode = static_only ? EmisOpacMode::StaticOnly : EmisOpacMode::All;
            if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
                mode = EmisOpacMode::StaticOnly;
            }

            auto result = emis_opac(
                EmisOpacState<fp_t>{
                    .adata = adata,
                    .profile = phi,
                    .la = la,
                    .n = flat_pops,
                    .n_star_scratch = flat_n_star,
                    .k = k,
                    .atmos = local_atmos,
                    .active_set = active_set,
                    .active_set_cont = slice_active_cont_set(adata, la),
                    .mode = mode
                }
            );
            mip_chain.emis(ks, wave) = result.eta;
            mip_chain.opac(ks, wave) = result.chi;
        }
    );
    parallel_for(
        "Copy vels",
        SimpleBounds<2>(bounds),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            IndexGen<BLOCK_SIZE> idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            i64 k = idx_gen.full_flat_idx(coord.x, coord.z);

            mip_chain.vx(ks) = flatmos.vx(k);
            mip_chain.vy(ks) = flatmos.vy(k);
            mip_chain.vz(ks) = flatmos.vz(k);
        }
    );
    yakl::fence();
    if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
        mip_chain.cav_data.fill(state, la_start, la_end);
    }
    mip_chain.compute_mips(state);

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
    constexpr int RcStorage = RC_flags_storage();
    // NOTE(cmo): Compute RC FS
    constexpr int num_subsets = subset_tasks_per_cascade<RcStorage>();
    for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
        if (state.alo.initialized()) {
            state.alo = FP(0.0);
        }
        CascadeCalcSubset subset{
            .la_start=la_start,
            .la_end=la_end,
            .subset_idx=subset_idx
        };
        if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
            FlatVelocity vels{
                .vx = mip_chain.vx,
                .vy = mip_chain.vy,
                .vz = mip_chain.vz,
            };
            mip_chain.dir_data.fill<RcModeNoBc>(state, casc_state, subset, vels, lte_scratch);
        }
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
        if (state.alo.initialized() && !lambda_iterate) {
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
        if (state.alo.initialized()) {
            // NOTE(cmo): Add terms to Gamma
            dynamic_compute_gamma(
                state,
                casc_state,
                lte_scratch,
                subset
            );
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