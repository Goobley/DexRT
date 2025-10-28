#include "ProfileNormalisation.hpp"
#include "CascadeState.hpp"
#include "CascadeState3d.hpp"
#include "RcUtilsModes3d.hpp"

template <int NumDim>
KOKKOS_INLINE_FUNCTION ivec<NumDim> coord_to_ivec(Coord<NumDim> c);

template <>
KOKKOS_INLINE_FUNCTION ivec<2> coord_to_ivec<2>(Coord<2> c) {
    ivec2 probe_coord;
    probe_coord(0) = c.x;
    probe_coord(1) = c.z;
    return probe_coord;
}

template <>
KOKKOS_INLINE_FUNCTION ivec<3> coord_to_ivec<3>(Coord<3> c) {
    ivec3 probe_coord;
    probe_coord(0) = c.x;
    probe_coord(1) = c.y;
    probe_coord(2) = c.z;
    return probe_coord;
}

template <int NumDim, class State, class CascadeState>
void compute_profile_normalisation(const State& state, const CascadeState& casc_state, bool print_worst_wphi) {
    const auto flatmos = flatten(state.atmos);
    JasUnpack(state, adata, wphi, mr_block_map);
    const auto& profile = state.phi;
    const auto& wavelength = adata.wavelength;
    const int num_cascades = casc_state.num_cascades;
    auto bounds = mr_block_map.block_map.loop_bounds();

    auto dims = state.c0_size;
    InclQuadrature incl_quad{};
    CascadeRays ray_set{};
    CascadeRays3d ray_set_3d{};
    if constexpr (NumDim == 2) {
        ray_set = cascade_compute_size<RC_flags_storage_2d()>(dims, 0);
        incl_quad = state.incl_quad;
    } else {
        ray_set_3d = cascade_compute_size<RC_flags_storage_3d()>(dims, 0);
    }

    using IdxGen_t = std::conditional_t<NumDim == 2, IdxGen, IdxGen3d>;

    dex_parallel_for(
        "Compute wphi",
        FlatLoop<3>(wphi.extent(0), bounds.dim(0), bounds.dim(1)),
        YAKL_LAMBDA (int kr, i64 tile_idx, i32 block_idx) {
            using namespace ConstantsFP;
            fp_t entry = FP(0.0);

            IdxGen_t idx_gen(mr_block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord<NumDim> cell_coord = idx_gen.loop_coord(tile_idx, block_idx);
            ivec<NumDim> probe_coord = coord_to_ivec(cell_coord);

            const auto& line = adata.lines(kr);

            AtmosPointParams local_atmos;
            local_atmos.temperature = flatmos.temperature(ks);
            local_atmos.ne = flatmos.ne(ks);
            local_atmos.vturb = flatmos.vturb(ks);
            local_atmos.nhtot = flatmos.nh_tot(ks);
            local_atmos.nh0 = flatmos.nh0(ks);

            const fp_t dop_width = doppler_width(line.lambda0, adata.mass(line.atom), local_atmos.temperature, local_atmos.vturb);
            const fp_t gamma = gamma_from_broadening(line, adata.broadening, local_atmos.temperature, local_atmos.ne, local_atmos.nh0);

            JasUse(ray_set, incl_quad, ray_set_3d, num_cascades);
            for (int la = line.blue_idx; la < line.red_idx; ++la) {
                const fp_t lambda = wavelength(la);
                fp_t wl_weight = adata.wavelength_bin(la);

                // NOTE(cmo): Evaluate the profile with the current value in
                // local_atmos, and add it to entry with the provided ray_weight
                auto add_profile_term = [&](const fp_t ray_weight) {
                    const fp_t a = damping_from_gamma(gamma, lambda, dop_width);
                    const fp_t v = ((lambda - line.lambda0) + (local_atmos.vel * line.lambda0) / c) / dop_width;
                    // [nm-1]
                    const fp_t p = profile(a, v) / (sqrt_pi * dop_width);
                    entry += p * wl_weight * ray_weight;
                };

                if constexpr (NumDim == 2) {
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

                            add_profile_term(ray_weight);
                        }
                    }
                } else if constexpr (NumDim == 3) {
                    const fp_t ray_weight = FP(1.0) / fp_t(ray_set_3d.num_az_rays * ray_set_3d.num_polar_rays);
                    for (int phi_idx = 0; phi_idx < ray_set_3d.num_az_rays; ++phi_idx) {
                        for (int theta_idx = 0; theta_idx < ray_set_3d.num_polar_rays; ++theta_idx) {
                            const ProbeIndex3d probe_idx {
                                .coord = probe_coord,
                                .polar = theta_idx,
                                .az = phi_idx
                            };

                            RaySegment<3> ray = probe_ray(ray_set_3d, num_cascades, 0, probe_idx);
                            local_atmos.vel = (
                                flatmos.vx(ks) * ray.d(0)
                                + flatmos.vy(ks) * ray.d(1)
                                + flatmos.vz(ks) * ray.d(2)
                            );

                            add_profile_term(ray_weight);
                        }
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

    if (print_worst_wphi) {
        const auto loop = FlatLoop<2>(wphi.extent(0), wphi.extent(1));
        typedef Kokkos::MaxLoc<fp_t, Kokkos::pair<i32, i32>> Reducer;
        typedef Reducer::value_type ReducerType;
        Fp1d wphi_plane("wphi max err plane", wphi.extent(0));

        ReducerType max_err_loc;

        dex_parallel_reduce(
            "Compute wphi err",
            loop,
            KOKKOS_LAMBDA (const int kr, const int k, ReducerType& rvar) {
                fp_t err = std::abs(FP(1.0) - wphi(kr, k));
                if (err > rvar.val) {
                    rvar.val = err;
                    rvar.loc = Kokkos::make_pair(kr, k);
                }
            },
            Reducer(max_err_loc)
        );
        i32 max_err_k = max_err_loc.loc.second;
        dex_parallel_for(
            "Copy wphi plane",
            FlatLoop<1>(wphi.extent(0)),
            KOKKOS_LAMBDA (const int kr) {
                wphi_plane(kr) = wphi(kr, max_err_k);
            }
        );
        Fp1d max_err_temp("temp_view", &flatmos.temperature(max_err_k), 1);
        Fp1d max_err_nh("temp_view", &flatmos.nh_tot(max_err_k), 1);
        Fp1d max_err_ne("temp_view", &flatmos.ne(max_err_k), 1);
        yakl::fence();
        auto wphi_plane_host = wphi_plane.createHostCopy();
        auto max_err_temp_host = max_err_temp.createHostCopy();
        auto max_err_nh_host = max_err_nh.createHostCopy();
        auto max_err_ne_host = max_err_ne.createHostCopy();
        std::string output = fmt::format("  Highest error normalisation factors (wphi) @ ks = {} (T={:.2e}, nh={:.2e}, ne={:.2e}): ", max_err_k, max_err_temp_host(0), max_err_nh_host(0), max_err_ne_host(0));
        for (int kr = 0; kr < wphi_plane_host.extent(0); ++kr) {
            output += fmt::format("({:.3f} nm: {:e})", state.adata_host.lines(kr).lambda0, wphi_plane_host(kr));
            if (kr != wphi_plane_host.extent(0) - 1) {
                output += ", ";
            }
        }
        state.println("{}", output);
    }
}

/// 2D
void compute_profile_normalisation(const State& state, const CascadeState& casc_state, bool print_worst_wphi) {
    compute_profile_normalisation<2>(state, casc_state, print_worst_wphi);
}

/// 3D
void compute_profile_normalisation(const State3d& state, const CascadeState3d& casc_state, bool print_worst_wphi) {
    compute_profile_normalisation<3>(state, casc_state, print_worst_wphi);
}