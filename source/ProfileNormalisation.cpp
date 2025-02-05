#include "ProfileNormalisation.hpp"

void compute_profile_normalisation(const State& state, const CascadeState& casc_state) {
    const auto flatmos = flatten(state.atmos);
    JasUnpack(state, adata, wphi, mr_block_map, incl_quad);
    const auto& profile = state.phi;
    const auto& wavelength = adata.wavelength;
    const int num_cascades = casc_state.num_cascades;
    auto bounds = mr_block_map.block_map.loop_bounds();

    constexpr int RcMode = RC_flags_storage();
    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<RcMode>(dims, 0);
    // NOTE(cmo): This is pretty thrown-together
    dex_parallel_for(
        "Compute wphi",
        FlatLoop<3>(wphi.extent(0), bounds.dim(0), bounds.dim(1)),
        YAKL_LAMBDA (int kr, i64 tile_idx, i32 block_idx) {
            using namespace ConstantsFP;
            fp_t entry = FP(0.0);

            IdxGen idx_gen(mr_block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 cell_coord = idx_gen.loop_coord(tile_idx, block_idx);
            ivec2 probe_coord;
            probe_coord(0) = cell_coord.x;
            probe_coord(1) = cell_coord.z;

            const auto& line = adata.lines(kr);

            AtmosPointParams local_atmos;
            local_atmos.temperature = flatmos.temperature(ks);
            local_atmos.ne = flatmos.ne(ks);
            local_atmos.vturb = flatmos.vturb(ks);
            local_atmos.nhtot = flatmos.nh_tot(ks);
            local_atmos.nh0 = flatmos.nh0(ks);

            const fp_t dop_width = doppler_width(line.lambda0, adata.mass(line.atom), local_atmos.temperature, local_atmos.vturb);
            const fp_t gamma = gamma_from_broadening(line, adata.broadening, local_atmos.temperature, local_atmos.ne, local_atmos.nh0);

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

                for (int phi_idx = 0; phi_idx < ray_set.num_flat_dirs; ++phi_idx) {
                    for (int theta_idx = 0; theta_idx < ray_set.num_incl; ++theta_idx) {
                        const ProbeIndex probe_idx{
                            .coord=probe_coord,
                            .dir=phi_idx,
                            .incl=theta_idx,
                            .wave=0
                        };
                        RayProps ray = ray_props(ray_set, num_cascades, 0, probe_idx);
                        vec3 mu = inverted_mu(ray, incl_quad.muy(theta_idx));

                        local_atmos.vel = (
                                flatmos.vx(ks) * mu(0)
                                + flatmos.vy(ks) * mu(1)
                                + flatmos.vz(ks) * mu(2)
                        );

                        const fp_t ray_weight = FP(1.0) / fp_t(ray_set.num_flat_dirs) * incl_quad.wmuy(theta_idx);

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
    const auto& wphi_flat = wphi.collapse();
    const i64 min_loc = yakl::intrinsics::minloc(wphi_flat);
    const i64 min_k = min_loc % wphi.extent(1);
    auto wphi_host = wphi.createHostCopy();
    yakl::fence();
    std::string output("  Lowest normalisation factors (wphi): ");
    for (int kr = 0; kr < wphi_host.extent(0); ++kr) {
        output += fmt::format("{:e}", wphi_host(kr, min_k));
        if (kr != wphi_host.extent(0) - 1) {
            output += ", ";
        }
    }
    output += "\n";
    state.println("{}", output);
}