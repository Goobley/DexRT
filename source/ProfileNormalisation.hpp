#if !defined(DEXRT_PROFILE_NORMALISATION_HPP)
#define DEXRT_PROFILE_NORMALISATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "Atmosphere.hpp"
#include "Voigt.hpp"
#include "EmisOpac.hpp"

inline
void compute_profile_normalisation(const State& state, const CascadeState& casc_state) {
    const auto flatmos = flatten(state.atmos);
    Fp2d wphi = state.wphi.reshape<2>(Dims(state.wphi.extent(0), state.wphi.extent(1) * state.wphi.extent(2)));
    const auto& adata = state.adata;
    const auto& wavelength = adata.wavelength;
    const auto& incl_quad = state.incl_quad;
    const auto& profile = state.phi;
    const int num_cascades = casc_state.num_cascades;

    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<PREAVERAGE>(state.c0_size, 0);
    // NOTE(cmo): This is pretty thrown-together
    parallel_for(
        "Compute wphi",
        SimpleBounds<2>(wphi.extent(0), wphi.extent(1)),
        YAKL_LAMBDA (int kr, i64 k) {
            using namespace ConstantsFP;
            wphi(kr, k) = FP(0.0);

            const auto& line = adata.lines(kr);
            ivec2 probe_coord;
            probe_coord(0) = k % dims.num_probes(1);
            probe_coord(1) = k / dims.num_probes(1);
            AtmosPointParams local_atmos;
            local_atmos.temperature = flatmos.temperature(k);
            local_atmos.ne = flatmos.ne(k);
            local_atmos.vturb = flatmos.vturb(k);
            local_atmos.nhtot = flatmos.nh_tot(k);
            local_atmos.nh0 = flatmos.nh0(k);

            const fp_t dop_width = doppler_width(line.lambda0, adata.mass(line.atom), local_atmos.temperature, local_atmos.vturb);
            const fp_t gamma = gamma_from_broadening(line, adata.broadening, local_atmos.temperature, local_atmos.ne, local_atmos.nh0);

            for (int phi_idx = 0; phi_idx < dims.num_flat_dirs; ++phi_idx) {
                for (int theta_idx = 0; theta_idx < dims.num_incl; ++theta_idx) {
                    const ProbeIndex probe_idx{
                        .coord=probe_coord,
                        .dir=phi_idx,
                        .incl=theta_idx,
                        .wave=0
                    };
                    RayProps ray = ray_props(ray_set, num_cascades, 0, probe_idx);
                    vec3 mu;
                    const fp_t cos_theta = incl_quad.muy(probe_idx.incl);
                    const fp_t sin_theta = std::sqrt(FP(1.0) - square(cos_theta));
                    mu(0) = ray.dir(0) * sin_theta;
                    mu(1) = cos_theta;
                    mu(2) = ray.dir(1) * sin_theta;

                    local_atmos.vel = (
                            flatmos.vx(k) * mu(0)
                            + flatmos.vy(k) * mu(1)
                            + flatmos.vz(k) * mu(2)
                    );

                    const fp_t ray_weight = FP(1.0) / fp_t(dims.num_flat_dirs) * incl_quad.wmuy(theta_idx);

                    for (int la = 0; la < wavelength.extent(0); ++la) {
                        if (!line.is_active(la)) {
                            continue;
                        }
                        const fp_t lambda = wavelength(la);
                        fp_t wl_weight = FP(1.0);
                        if (la == 0) {
                            wl_weight *= FP(0.5) * (wavelength(1) - wavelength(0));
                        } else if (la == wavelength.extent(0) - 1) {
                            wl_weight *= FP(0.5) * (
                                wavelength(wavelength.extent(0) - 1) - wavelength(wavelength.extent(0) - 2)
                            );
                        } else {
                            wl_weight *= FP(0.5) * (wavelength(la + 1) - wavelength(la - 1));
                        }
                        const fp_t a = damping_from_gamma(gamma, lambda, dop_width);
                        const fp_t v = ((lambda - line.lambda0) + (local_atmos.vel * line.lambda0) / c) / dop_width;
                        // [nm-1]
                        const fp_t p = profile(a, v) / (sqrt_pi * dop_width);
                        wphi(kr, k) += p * wl_weight * ray_weight;
                    }
                }
            }
            // NOTE(cmo): Store the multiplicative normalisation factor
            wphi(kr, k) = FP(1.0) / wphi(kr, k);
        }
    );
    yakl::fence();
    // auto wphi_host = wphi.createHostCopy();
    // yakl::fence();
    // fmt::print("Normalisation factors: ");
    // for (int kr = 0; kr < wphi_host.extent(0); ++kr) {
    //     fmt::print("{:e}, ", wphi_host(kr, 0));
    // }
    // fmt::println("\n-----");
}

#else
#endif