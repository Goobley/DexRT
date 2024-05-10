#include "StaticFormalSolution.hpp"
#include "RadianceCascades2.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "GammaMatrix.hpp"
#include "Atmosphere.hpp"
#include "RcUtilsModes.hpp"

void static_compute_gamma(
    const State& state,
    const CascadeState& casc_state,
    int la_start,
    int la_end,
    const Fp3d& lte_scratch
) {
    // using namespace ConstantsFP;
    // const auto flat_atmos = flatten<const fp_t>(state.atmos);
    // const auto& atom = state.atom;
    // const auto& phi = state.phi;
    // const auto& pops = state.pops;
    // const auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    // const auto flat_lte_pops = lte_scratch.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    // const auto& Gamma = state.Gamma;
    // const auto flat_Gamma = Gamma.reshape<3>(Dims(
    //     Gamma.extent(0),
    //     Gamma.extent(1),
    //     Gamma.extent(2) * Gamma.extent(3)
    // ));
    // const auto& alo = state.alo.reshape<4>(Dims(
    //     state.alo.extent(0) * state.alo.extent(1),
    //     state.alo.extent(2),
    //     state.alo.extent(3),
    //     state.alo.extent(4)
    // ));
    // const auto& I = casc_state.i_cascades[0];
    // const auto& nh_lte = state.nh_lte;
    // const auto& incl_quad = state.incl_quad;
    // int wave_batch = la_end - la_start;

    // CascadeStorage dims = state.c0_size;

    // const auto& wavelength = state.atom.wavelength;
    // parallel_for(
    //     "compute Gamma",
    //     SimpleBounds<4>(
    //         flat_atmos.temperature.extent(0),
    //         dims.num_flat_dirs,
    //         wave_batch,
    //         dims.num_incl
    //     ),
    //     YAKL_LAMBDA (i64 k, int phi_idx, int wave, int theta_idx) {
    //         ivec2 probe_coord;
    //         probe_coord(0) = k % dims.num_probes(1);
    //         probe_coord(1) = k / dims.num_probes(1);
    //         const ProbeIndex probe_idx{
    //             .coord=probe_coord,
    //             .dir=phi_idx,
    //             .incl=theta_idx,
    //             .wave=wave
    //         };
    //         const fp_t intensity = probe_fetch(I, dims, probe_idx);

    //         const int la = la_start + wave;
    //         fp_t lambda = wavelength(la);
    //         const fp_t hnu_4pi = hc_kJ_nm / (four_pi * lambda);
    //         fp_t wl_weight = FP(1.0) / hnu_4pi;
    //         if (la == 0) {
    //             wl_weight *= FP(0.5) * (wavelength(1) - wavelength(0));
    //         } else if (la == wavelength.extent(0) - 1) {
    //             wl_weight *= FP(0.5) * (
    //                 wavelength(wavelength.extent(0) - 1) - wavelength(wavelength.extent(0) - 2)
    //             );
    //         } else {
    //             wl_weight *= FP(0.5) * (wavelength(la + 1) - wavelength(la - 1));
    //         }
    //         const fp_t wl_ray_weight = wl_weight / fp_t(dims.num_flat_dirs);

    //         AtmosPointParams local_atmos;
    //         local_atmos.temperature = flat_atmos.temperature(k);
    //         local_atmos.ne = flat_atmos.ne(k);
    //         local_atmos.vturb = flat_atmos.vturb(k);
    //         local_atmos.nhtot = flat_atmos.nh_tot(k);
    //         local_atmos.nh0 = nh_lte(local_atmos.temperature, local_atmos.ne, local_atmos.nhtot);
    //         // TODO(cmo): Update for dynamic case.
    //         local_atmos.vel = FP(0.0);

    //         for (int kr = 0; kr < atom.lines.extent(0); ++kr) {
    //             const auto& l = atom.lines(kr);
    //             if (!l.is_active(la)) {
    //                 continue;
    //             }

    //             const UV uv = compute_uv_line(
    //                 EmisOpacState<>{
    //                     .atom = atom,
    //                     .profile = phi,
    //                     .la = la,
    //                     .n = flat_pops,
    //                     .n_star_scratch = flat_lte_pops,
    //                     .k = k,
    //                     .atmos = local_atmos
    //                 },
    //                 kr
    //             );

    //             const fp_t eta = flat_pops(l.j, k) * uv.Uji;
    //             const fp_t chi = flat_pops(l.i, k) * uv.Vij - flat_pops(l.j, k) * uv.Vji;


    //             add_to_gamma<true>(GammaAccumState{
    //                 .eta = eta,
    //                 .chi = chi,
    //                 .uv = uv,
    //                 .I = intensity,
    //                 .alo = alo(k, phi_idx, wave, theta_idx),
    //                 .wlamu = wl_ray_weight * incl_quad.wmuy(theta_idx),
    //                 .Gamma = flat_Gamma,
    //                 .i = l.i,
    //                 .j = l.j,
    //                 .k = k
    //             });
    //         }
    //         for (int kr = 0; kr < atom.continua.extent(0); ++kr) {
    //             const auto& cont = atom.continua(kr);
    //             if (!cont.is_active(la)) {
    //                 continue;
    //             }

    //             const UV uv = compute_uv_cont(
    //                 EmisOpacState<>{
    //                     .atom = atom,
    //                     .profile = phi,
    //                     .la = la,
    //                     .n = flat_pops,
    //                     .n_star_scratch = flat_lte_pops,
    //                     .k = k,
    //                     .atmos = local_atmos
    //                 },
    //                 kr
    //             );

    //             const fp_t eta = flat_pops(cont.j, k) * uv.Uji;
    //             const fp_t chi = flat_pops(cont.i, k) * uv.Vij - flat_pops(cont.j, k) * uv.Vji;

    //             add_to_gamma<true>(GammaAccumState{
    //                 .eta = eta,
    //                 .chi = chi,
    //                 .uv = uv,
    //                 .I = intensity,
    //                 .alo = alo(k, phi_idx, wave, theta_idx),
    //                 .wlamu = wl_ray_weight * incl_quad.wmuy(theta_idx),
    //                 .Gamma = flat_Gamma,
    //                 .i = cont.i,
    //                 .j = cont.j,
    //                 .k = k
    //             });
    //         }
    //     }
    // );
    // yakl::fence();
}

void static_formal_sol_rc(const State& state, const CascadeState& casc_state, int la_start, int la_end) {
    auto& atmos = state.atmos;
    auto& phi = state.phi;
    auto& pops = state.pops;
    auto& adata = state.adata;
    auto& eta = casc_state.eta;
    auto& chi = casc_state.chi;
    const auto& nh_lte = state.nh_lte;

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
        SimpleBounds<2>(flat_temperature.extent(0), wave_batch),
        YAKL_LAMBDA (int64_t k, int wave) {
            AtmosPointParams local_atmos;
            local_atmos.temperature = flat_temperature(k);
            local_atmos.ne = flat_ne(k);
            local_atmos.vturb = flat_vturb(k);
            local_atmos.nhtot = flat_nhtot(k);
            local_atmos.nh0 = nh_lte(local_atmos.temperature, local_atmos.ne, local_atmos.nhtot);
            static_assert(false, "See below");
            auto result = emis_opac(
                EmisOpacState<fp_t>{
                    .adata = adata,
                    .profile = phi,
                    .la = la_start + wave,
                    .n = flat_pops,
                    .n_star_scratch = flat_n_star,
                    .k = k,
                    .atmos = local_atmos,
                    // THIS IS SLICING host data on device somehow???? Pointers must be copied wrong when we make device version
                    .active_set = slice_active_set(adata, la_start + wave),
                    .active_set_cont = slice_active_cont_set(adata, la_start + wave)
                }
            );
            flat_eta(k, wave) = result.eta;
            flat_chi(k, wave) = result.chi;
        }
    );
    state.alo = FP(0.0);
    yakl::fence();
    // NOTE(cmo): Compute RC FS
    constexpr int RcModeBc = RC_flags_pack(RcFlags{
        .dynamic = false,
        .preaverage = PREAVERAGE,
        .sample_bc = USE_BC,
        .compute_alo = false
    });
    constexpr int RcModeNoBc = RC_flags_pack(RcFlags{
        .dynamic = false,
        .preaverage = PREAVERAGE,
        .sample_bc = false,
        .compute_alo = false
    });
    constexpr int RcModeAlo = RC_flags_pack(RcFlags{
        .dynamic = false,
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
    cascade_i_25d<RcModeAlo>(
        state,
        casc_state,
        0,
        la_start,
        la_end
    );
    yakl::fence();
    if (state.alo.initialized()) {
        // NOTE(cmo): Add terms to Gamma
        static_compute_gamma(
            state,
            casc_state,
            la_start,
            la_end,
            lte_scratch
        );
    }
    // NOTE(cmo): J is not computed in this function, but done in main for now
}