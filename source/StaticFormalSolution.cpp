#include "StaticFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"

void static_compute_gamma(State* state, int la, const Fp3d& lte_scratch) {
    using namespace ConstantsFP;
    auto& atmos = state->atmos;
    auto atmos_dims = atmos.temperature.get_dimensions();
    const auto& atom = state->atom;
    const auto& phi = state->phi;
    const auto& pops = state->pops;
    const auto& Gamma = state->Gamma;
    const auto& alo = state->alo;
    const auto& I = state->cascades[0];
    const auto& wavelength = atom.wavelength;
    const auto& nh_lte = state->nh_lte;
    auto az_rays = get_az_rays();
    auto az_weights = get_az_weights();
    auto I_dims = I.get_dimensions();


    parallel_for(
        "compute Gamma",
        SimpleBounds<2>(atmos_dims(0), atmos_dims(1)),
        YAKL_LAMBDA (int x, int y) {
            AtmosPointParams local_atmos;
            local_atmos.temperature = atmos.temperature(x, y);
            local_atmos.ne = atmos.ne(x, y);
            local_atmos.vturb = atmos.vturb(x, y);
            local_atmos.nhtot = atmos.nh_tot(x, y);
            local_atmos.nh0 = nh_lte(local_atmos.temperature, local_atmos.ne, local_atmos.nhtot);

            fp_t lambda = atom.wavelength(la);
            for (int kr = 0; kr < atom.lines.extent(0); ++kr) {
                const auto& l = atom.lines(kr);
                if (!l.is_active(la)) {
                    continue;
                }

                LineParams params;
                params.dop_width = doppler_width(
                    l.lambda0,
                    atom.mass,
                    local_atmos.temperature,
                    local_atmos.vturb
                );
                params.gamma = gamma_from_broadening(
                    l,
                    atom.broadening,
                    local_atmos.temperature,
                    local_atmos.ne,
                    local_atmos.nh0
                );
                // TODO(cmo): Come back to this!
                params.vel = FP(0.0);

                const UV uv = compute_uv(
                    l,
                    phi,
                    params,
                    lambda
                );

                const fp_t eta = pops(l.j, x, y) * uv.Uji;
                const fp_t chi = pops(l.i, x, y) * uv.Vij - pops(l.j, x, y) * uv.Vji;
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

                fp_t G_ij = FP(0.0);
                fp_t G_ji = FP(0.0);
                for (int ray_idx = 0; ray_idx < I_dims(2); ++ray_idx) {
                    for (int batch_la = 0; batch_la < I_dims(3) / 2; ++batch_la) {
                        for (int r = 0; r < I_dims(4); ++r) {
                            const fp_t psi_star = alo(x, y, r) / chi;
                            const fp_t I_eff = I(x, y, ray_idx, 2 * batch_la, r) - psi_star * eta;
                            const fp_t wlamu = wl_weight * FP(1.0) / fp_t(I_dims(2)) * az_weights(r);
                            fp_t integrand = (FP(1.0) - chi * psi_star) * uv.Uji + uv.Vji * I_eff;
                            G_ij += integrand * wlamu;
                            // if (x == 0 && y == 255 && G_ij < FP(0.0)) {
                            //     printf("neg: int %e, wla %e, alo %e, Ieff %e\n", integrand, wlamu, alo(x, y), I_eff);
                            // }

                            integrand = uv.Vij * I_eff;
                            G_ji += integrand * wlamu;
                            // if (x == 0 && y == 255 && G_ji < FP(0.0)) {
                            //     printf("neg: int %e, wla %e, alo %e, Ieff %e\n", integrand, wlamu, alo(x, y), I_eff);
                            // }
                        }
                    }
                }
                Gamma(l.i, l.j, x, y) += G_ij;
                Gamma(l.j, l.i, x, y) += G_ji;
            }
            for (int kr = 0; kr < atom.continua.extent(0); ++kr) {
                const auto& cont = atom.continua(kr);
                if (!cont.is_active(la)) {
                    continue;
                }

                ContParams params;
                params.la = la;
                params.thermal_ratio = lte_scratch(cont.i, x, y) / lte_scratch(cont.j, x, y) * std::exp(-hc_k_B_nm / (lambda * local_atmos.temperature));
                params.sigma_grid = get_sigma(atom, cont);

                const UV uv = compute_uv(
                    cont,
                    params,
                    lambda
                );
                const fp_t eta = pops(cont.j, x, y) * uv.Uji;
                const fp_t chi = pops(cont.i, x, y) * uv.Vij - pops(cont.j, x, y) * uv.Vji;
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

                fp_t G_ij = FP(0.0);
                fp_t G_ji = FP(0.0);
                for (int ray_idx = 0; ray_idx < I_dims(2); ++ray_idx) {
                    for (int batch_la = 0; batch_la < I_dims(3) / 2; ++batch_la) {
                        for (int r = 0; r < I_dims(4); ++r) {
                            const fp_t psi_star = alo(x, y, r) / chi;
                            const fp_t I_eff = I(x, y, ray_idx, 2 * batch_la, r) - psi_star * eta;
                            const fp_t wlamu = wl_weight * FP(1.0) / fp_t(I_dims(2)) * az_weights(r);
                            fp_t integrand = (FP(1.0) - chi * psi_star) * uv.Uji + uv.Vji * I_eff;
                            G_ij += integrand * wlamu;

                            integrand = uv.Vij * I_eff;
                            G_ji += integrand * wlamu;
                        }
                    }
                }
                Gamma(cont.i, cont.j, x, y) += G_ij;
                Gamma(cont.j, cont.i, x, y) += G_ji;
            }
        }
    );
    yakl::fence();
    parallel_for(
        "Gamma fixup",
        SimpleBounds<2>(atmos_dims(0), atmos_dims(1)),
        YAKL_LAMBDA (int x, int y) {
            for (int i = 0; i < Gamma.extent(1); ++i) {
                fp_t diag = FP(0.0);
                Gamma(i, i, x, y) = FP(0.0);
                for (int j = 0; j < Gamma.extent(0); ++j) {
                    diag += Gamma(j, i, x, y);
                }
                Gamma(i, i, x, y) = -diag;
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