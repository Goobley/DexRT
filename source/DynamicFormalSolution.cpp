#include "DynamicFormalSolution.hpp"
#include "StaticFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "GammaMatrix.hpp"
#include "Atmosphere.hpp"
#include "RcUtilsModes.hpp"

void dynamic_compute_gamma(
    const State& state,
    const CascadeState& casc_state,
    int la_start,
    int la_end,
    const Fp3d& lte_scratch
) {
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

    for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
        const auto& Gamma = state.Gamma[ia];
        const auto flat_Gamma = Gamma.reshape<3>(Dims(
            Gamma.extent(0),
            Gamma.extent(1),
            Gamma.extent(2) * Gamma.extent(3)
        ));
        const auto& alo = state.alo.reshape<4>(Dims(
            state.alo.extent(0) * state.alo.extent(1),
            state.alo.extent(2),
            state.alo.extent(3),
            state.alo.extent(4)
        ));
        const auto& I = casc_state.i_cascades[0];
        const auto& incl_quad = state.incl_quad;
        int wave_batch = la_end - la_start;

        CascadeStorage dims = state.c0_size;
        CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
        const int num_cascades = casc_state.num_cascades;

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
                const ProbeIndex probe_idx{
                    .coord=probe_coord,
                    .dir=phi_idx,
                    .incl=theta_idx,
                    .wave=wave
                };
                RayProps ray = ray_props(ray_set, num_cascades, 0, probe_idx);
                const fp_t intensity = probe_fetch<RcMode>(I, dims, probe_idx);

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
                const fp_t wl_ray_weight = wl_weight / fp_t(dims.num_flat_dirs);

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
                        .alo = alo(k, phi_idx, wave, theta_idx),
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
                        .alo = alo(k, phi_idx, wave, theta_idx),
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
    auto& eta = casc_state.eta;
    auto& chi = casc_state.chi;

    // if (!atmos.moving) {
    //     return static_formal_sol_rc(state, casc_state, la_start, la_end);
    // }

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

    const auto flatmos = flatten<const fp_t>(atmos);
    auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    auto flat_n_star = lte_scratch.reshape<2>(Dims(lte_scratch.extent(0), lte_scratch.extent(1) * lte_scratch.extent(2)));
    auto flat_eta = eta.reshape<2>(Dims(eta.extent(0) * eta.extent(1), eta.extent(2)));
    auto flat_chi = chi.reshape<2>(Dims(chi.extent(0) * chi.extent(1), chi.extent(2)));
    auto flat_dynamic_opac = dynamic_opac.reshape<2>(Dims(chi.extent(0) * chi.extent(1), chi.extent(2)));
    auto flat_active = state.active.reshape<1>(Dims(state.active.extent(0) * state.active.extent(1)));
    // NOTE(cmo): Compute emis/opac
    parallel_for(
        "Compute eta, chi",
        SimpleBounds<2>(flatmos.temperature.extent(0), wave_batch),
        YAKL_LAMBDA (int64_t k, int wave) {
            if (sparse_calc && !flat_active(k)) {
                return;
            }
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

            bool static_only = v_norm > (ANGLE_INVARIANT_THERMAL_VEL_FRAC * thermal_vel(
                adata.mass(governing_atom),
                local_atmos.temperature
            ));
            flat_dynamic_opac(k, wave) = static_only;
            EmisOpacMode mode = static_only ? EmisOpacMode::StaticOnly : EmisOpacMode::All;

            auto result = emis_opac(
                EmisOpacState<fp_t>{
                    .adata = adata,
                    .profile = phi,
                    .la = la,
                    .n = flat_pops,
                    .n_star_scratch = flat_n_star,
                    .k = k,
                    .atmos = local_atmos,
                    .active_set = slice_active_set(adata, la),
                    .active_set_cont = slice_active_cont_set(adata, la),
                    .mode = mode
                }
            );
            flat_eta(k, wave) = result.eta;
            flat_chi(k, wave) = result.chi;
        }
    );
    if (state.alo.initialized()) {
        state.alo = FP(0.0);
    }
    yakl::fence();
    // NOTE(cmo): Compute RC FS
    constexpr int RcModeBc = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = PREAVERAGE,
        .sample_bc = true,
        .compute_alo = false
    });
    constexpr int RcModeNoBc = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = PREAVERAGE,
        .sample_bc = false,
        .compute_alo = false
    });
    constexpr int RcModeAlo = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = PREAVERAGE,
        .sample_bc = false,
        .compute_alo = true
    });
    cascade_i_25d<RcModeBc>(
        state,
        casc_state,
        casc_state.num_cascades,
        la_start,
        la_end
    );
    yakl::fence();
    for (int casc_idx = casc_state.num_cascades - 1; casc_idx >= 1; --casc_idx) {
        cascade_i_25d<RcModeNoBc>(
            state,
            casc_state,
            casc_idx,
            la_start,
            la_end
        );
        yakl::fence();
    }
    if (state.alo.initialized()) {
        cascade_i_25d<RcModeAlo>(
            state,
            casc_state,
            0,
            la_start,
            la_end
        );
    } else {
        cascade_i_25d<RcModeNoBc>(
            state,
            casc_state,
            0,
            la_start,
            la_end
        );
    }
    yakl::fence();
    if (state.alo.initialized()) {
        // NOTE(cmo): Add terms to Gamma
        if (lambda_iterate) {
            state.alo = FP(0.0);
            yakl::fence();
        } else {
            dynamic_compute_gamma(
                state,
                casc_state,
                la_start,
                la_end,
                lte_scratch
            );
        }
    }
    // NOTE(cmo): J is not computed in this function, but done in main for now
}