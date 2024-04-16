#include "StaticFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "GammaMatrix.hpp"
#include "Atmosphere.hpp"

void static_compute_gamma(State* state, int la, const Fp3d& lte_scratch) {
    using namespace ConstantsFP;
    const auto flat_atmos = flatten<const fp_t>(state->atmos);
    const auto& atom = state->atom;
    const auto& phi = state->phi;
    const auto& pops = state->pops;
    const auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    const auto flat_lte_pops = lte_scratch.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    const auto& Gamma = state->Gamma;
    const auto flat_Gamma = Gamma.reshape<3>(Dims(
        Gamma.extent(0),
        Gamma.extent(1),
        Gamma.extent(2) * Gamma.extent(3)
    ));
    const auto& alo = state->alo.reshape<2>(Dims(
        state->alo.extent(0) * state->alo.extent(1),
        state->alo.extent(2)
    ));
    const auto casc_dims = state->cascades[0].get_dimensions();
    const auto& I = state->cascades[0].reshape<4>(Dims(
        casc_dims(0) * casc_dims(1),
        casc_dims(2),
        casc_dims(3),
        casc_dims(4)
    ));
    const auto& nh_lte = state->nh_lte;
    auto az_rays = get_az_rays();
    auto az_weights = get_az_weights();
    auto I_dims = I.get_dimensions();

    const auto& wavelength = state->wavelength_h;
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
    const fp_t wl_ray_weight = wl_weight / fp_t(I_dims(1));

    parallel_for(
        "compute Gamma",
        SimpleBounds<1>(flat_atmos.temperature.extent(0)),
        YAKL_LAMBDA (i64 k) {
            AtmosPointParams local_atmos;
            local_atmos.temperature = flat_atmos.temperature(k);
            local_atmos.ne = flat_atmos.ne(k);
            local_atmos.vturb = flat_atmos.vturb(k);
            local_atmos.nhtot = flat_atmos.nh_tot(k);
            local_atmos.nh0 = nh_lte(local_atmos.temperature, local_atmos.ne, local_atmos.nhtot);
            local_atmos.vel = FP(0.0);

            for (int kr = 0; kr < atom.lines.extent(0); ++kr) {
                const auto& l = atom.lines(kr);
                if (!l.is_active(la)) {
                    continue;
                }

                const UV uv = compute_uv_line(
                    EmisOpacState<>{
                        .atom = atom,
                        .profile = phi,
                        .la = la,
                        .n = flat_pops,
                        .n_star_scratch = flat_lte_pops,
                        .k = k,
                        .atmos = local_atmos
                    },
                    kr
                );

                const fp_t eta = flat_pops(l.j, k) * uv.Uji;
                const fp_t chi = flat_pops(l.i, k) * uv.Vij - flat_pops(l.j, k) * uv.Vji;

                for (int ray_idx = 0; ray_idx < I_dims(1); ++ray_idx) {
                    for (int batch_la = 0; batch_la < I_dims(2) / 2; ++batch_la) {
                        for (int r = 0; r < I_dims(3); ++r) {
                            add_to_gamma<false>(GammaAccumState{
                                .eta = eta,
                                .chi = chi,
                                .uv = uv,
                                .I = I(k, ray_idx, 2 * batch_la, r),
                                .alo = alo(k, r),
                                .wlamu = wl_ray_weight * az_weights(r),
                                .Gamma = flat_Gamma,
                                .i = l.i,
                                .j = l.j,
                                .k = k
                            });
                        }
                    }
                }
            }
            for (int kr = 0; kr < atom.continua.extent(0); ++kr) {
                const auto& cont = atom.continua(kr);
                if (!cont.is_active(la)) {
                    continue;
                }

                const UV uv = compute_uv_cont(
                    EmisOpacState<>{
                        .atom = atom,
                        .profile = phi,
                        .la = la,
                        .n = flat_pops,
                        .n_star_scratch = flat_lte_pops,
                        .k = k,
                        .atmos = local_atmos
                    },
                    kr
                );

                const fp_t eta = flat_pops(cont.j, k) * uv.Uji;
                const fp_t chi = flat_pops(cont.i, k) * uv.Vij - flat_pops(cont.j, k) * uv.Vji;

                for (int ray_idx = 0; ray_idx < I_dims(1); ++ray_idx) {
                    for (int batch_la = 0; batch_la < I_dims(2) / 2; ++batch_la) {
                        for (int r = 0; r < I_dims(3); ++r) {
                            add_to_gamma<false>(GammaAccumState{
                                .eta = eta,
                                .chi = chi,
                                .uv = uv,
                                .I = I(k, ray_idx, 2 * batch_la, r),
                                .alo = alo(k, r),
                                .wlamu = wl_ray_weight * az_weights(r),
                                .Gamma = flat_Gamma,
                                .i = cont.i,
                                .j = cont.j,
                                .k = k
                            });
                        }
                    }
                }
            }
        }
    );
    yakl::fence();
}

void static_formal_sol_rc(State* state, int la) {
    auto& march_state = state->raymarch_state;

    auto& atmos = state->atmos;
    auto& phi = state->phi;
    auto& pops = state->pops;
    auto& atom = state->atom;
    auto& eta = march_state.emission;
    auto& chi = march_state.absorption;
    const auto& nh_lte = state->nh_lte;

    // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
    // it, for now, trust the pool allocator
    auto pops_dims = pops.get_dimensions();
    Fp3d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1), pops_dims(2));

    auto flat_temperature = atmos.temperature.collapse();
    auto flat_ne = atmos.ne.collapse();
    auto flat_vturb = atmos.vturb.collapse();
    auto flat_nhtot = atmos.nh_tot.collapse();
    auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    auto flat_n_star = lte_scratch.reshape<2>(Dims(lte_scratch.extent(0), lte_scratch.extent(1) * lte_scratch.extent(2)));
    auto flat_eta = eta.reshape<2>(Dims(eta.extent(0) * eta.extent(1), eta.extent(2)));
    auto flat_chi = chi.reshape<2>(Dims(chi.extent(0) * chi.extent(1), chi.extent(2)));
    // NOTE(cmo): Compute emis/opac
    parallel_for(
        "Compute eta, chi",
        SimpleBounds<1>(flat_temperature.extent(0)),
        YAKL_LAMBDA (int64_t k) {
            AtmosPointParams local_atmos;
            local_atmos.temperature = flat_temperature(k);
            local_atmos.ne = flat_ne(k);
            local_atmos.vturb = flat_vturb(k);
            local_atmos.nhtot = flat_nhtot(k);
            local_atmos.nh0 = nh_lte(local_atmos.temperature, local_atmos.ne, local_atmos.nhtot);
            auto result = emis_opac(
                EmisOpacState<fp_t>{
                    .atom = atom,
                    .profile = phi,
                    .la = la,
                    .n = flat_pops,
                    .n_star_scratch = flat_n_star,
                    .k = k,
                    .atmos = local_atmos
                }
            );
            flat_eta(k, 0) = result.eta;
            flat_chi(k, 0) = result.chi;
        }
    );
    if (state->alo.initialized()) {
        state->alo = FP(0.0);
    }
    yakl::fence();
    // NOTE(cmo): Regenerate mipmaps
    if constexpr (USE_MIPMAPS) {
        int current_mip_factor = 0;
        for (int i = 0; i < march_state.emission_mipmaps.size(); ++i) {
            if (march_state.cumulative_mipmap_factor(i) == current_mip_factor) {
                continue;
            }
            current_mip_factor = march_state.cumulative_mipmap_factor(i);
            auto new_eta = march_state.emission_mipmaps[i];
            auto new_chi = march_state.absorption_mipmaps[i];
            auto dims = new_eta.get_dimensions();
            parallel_for(
                SimpleBounds<2>(dims(0), dims(1)),
                YAKL_LAMBDA (int x, int y) {
                    mipmap_arr(eta, new_eta, current_mip_factor, x, y);
                }
            );
            parallel_for(
                SimpleBounds<2>(dims(0), dims(1)),
                YAKL_LAMBDA (int x, int y) {
                    mipmap_arr(chi, new_chi, current_mip_factor, x, y);
                }
            );
        }
        yakl::fence();
    }
    // NOTE(cmo): Compute RC FS
    for (int i = MAX_LEVEL; i >= 0; --i) {
        const bool compute_alo = ((i == 0) && state->alo.initialized());
        if constexpr (BILINEAR_FIX) {
            compute_cascade_i_bilinear_fix_2d(state, i, compute_alo);
        } else {
            compute_cascade_i_2d(state, i, compute_alo);
        }
        yakl::fence();
    }
    // NOTE(cmo): J is not computed in this function, but done in main for now

    if (state->alo.initialized()) {
        static_compute_gamma(state, la, lte_scratch);
    }
}