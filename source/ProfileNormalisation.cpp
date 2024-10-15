#include "ProfileNormalisation.hpp"

void compute_profile_normalisation(const State& state, const CascadeState& casc_state) {
    const auto flatmos = flatten(state.atmos);
    Fp2d wphi = state.wphi;
    const auto& adata = state.adata;
    const auto& wavelength = adata.wavelength;
    const auto& incl_quad = state.incl_quad;
    const auto& profile = state.phi;
    const int num_cascades = casc_state.num_cascades;

    constexpr int RcMode = RC_flags_storage();
    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<RcMode>(dims, 0);
    DeviceProbesToCompute probes_to_compute = casc_state.probes_to_compute.bind(0);
    // NOTE(cmo): This is pretty thrown-together
    parallel_for(
        "Compute wphi",
        SimpleBounds<2>(wphi.extent(0), probes_to_compute.num_active_probes()),
        YAKL_LAMBDA (int kr, i64 ks) {
            using namespace ConstantsFP;
            fp_t entry = FP(0.0);

            const auto& line = adata.lines(kr);
            ivec2 probe_coord = probes_to_compute(ks);

            AtmosPointParams local_atmos;
            local_atmos.temperature = flatmos.temperature(ks);
            local_atmos.ne = flatmos.ne(ks);
            local_atmos.vturb = flatmos.vturb(ks);
            local_atmos.nhtot = flatmos.nh_tot(ks);
            local_atmos.nh0 = flatmos.nh0(ks);

            const fp_t dop_width = doppler_width(line.lambda0, adata.mass(line.atom), local_atmos.temperature, local_atmos.vturb);
            const fp_t gamma = gamma_from_broadening(line, adata.broadening, local_atmos.temperature, local_atmos.ne, local_atmos.nh0);

            for (int phi_idx = 0; phi_idx < ray_set.num_flat_dirs; ++phi_idx) {
                for (int theta_idx = 0; theta_idx < ray_set.num_incl; ++theta_idx) {
                    const ProbeIndex probe_idx{
                        .coord=probe_coord,
                        .dir=phi_idx,
                        .incl=theta_idx,
                        .wave=0
                    };
                    RayProps ray = ray_props(ray_set, num_cascades, 0, probe_idx);
                    vec3 mu = inverted_mu(ray, probe_idx.incl);

                    local_atmos.vel = (
                            flatmos.vx(ks) * mu(0)
                            + flatmos.vy(ks) * mu(1)
                            + flatmos.vz(ks) * mu(2)
                    );

                    const fp_t ray_weight = FP(1.0) / fp_t(ray_set.num_flat_dirs) * incl_quad.wmuy(theta_idx);

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
                        entry += p * wl_weight * ray_weight;
                    }
                }
            }
            // NOTE(cmo): Store the multiplicative normalisation factor
            if (entry != FP(0.0)) {
                wphi(kr, ks) = FP(1.0) / entry;
            } else {
                wphi(kr, ks) = FP(1.0);
            }
        }
    );
    yakl::fence();
    const i64 min_loc = yakl::intrinsics::minloc(wphi.collapse());
    const i64 min_k = min_loc % wphi.extent(1);
    auto wphi_host = wphi.createHostCopy();
    yakl::fence();
    fmt::print("Normalisation factors: ");
    for (int kr = 0; kr < wphi_host.extent(0); ++kr) {
        fmt::print("{:e}, ", wphi_host(kr, min_k));
    }
    fmt::println("\n-----");
}