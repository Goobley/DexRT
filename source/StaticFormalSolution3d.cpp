#include "StaticFormalSolution3d.hpp"
#include "CascadeState.hpp"
#include "Mipmaps3d.hpp"
#include "RayMarching.hpp" // only for merge_intervals
#include "RayMarching3d.hpp"
#include "RadianceCascades3d.hpp"

// struct Raymarch3dArgs {
//     const ProbeIndex3d& this_probe;
//     const DeviceCascadeState3d& casc_state;
//     const MultiResBlockMap<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>& mr_block_map;
//     const RaySegment<3>& ray;
//     const fp_t distance_scale;
//     const MultiResMipChain3d& mip_chain;
//     const i32 max_mip_to_sample;
// };

// NOTE(cmo): None of this file is really used anymore. It was for initial
// testing of 3d, or doing arbitrary (non-atomic line) scenes.

static void merge_c0_to_J_3d(const State3d& state, const CascadeState3d& casc_state, int la, fp_t ray_weight=FP(-1.0)) {
    constexpr int RcMode = RC_flags_storage_3d();
    const CascadeStorage3d& c0_size(state.c0_size);
    CascadeRays3d ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
    if (ray_weight < FP(0.0)) {
        ray_weight = FP(1.0) / fp_t(ray_set.num_az_rays * ray_set.num_polar_rays);
    }

    const auto& c0 = casc_state.i_cascades[0];
    const auto& J = state.J;
    // TODO(cmo): This is storing in flattened [z, y, x] order, not ks order.
    FlatLoop<3> probe_loop(ray_set.num_probes(2), ray_set.num_probes(1), ray_set.num_probes(0));

    dex_parallel_for(
        FlatLoop<3>(probe_loop.num_iter, c0_size.num_polar_rays, c0_size.num_az_rays),
        KOKKOS_LAMBDA (i64 flat_probe_idx, int theta_idx, int phi_idx) {
            auto rev_probe_coord = probe_loop.unpack(flat_probe_idx);
            ivec3 probe_coord;
            probe_coord(0) = rev_probe_coord[2];
            probe_coord(1) = rev_probe_coord[1];
            probe_coord(2) = rev_probe_coord[0];

            ProbeStorageIndex3d this_probe {
                .coord = probe_coord,
                .polar = theta_idx,
                .az = phi_idx
            };
            const fp_t sample = probe_fetch<RcMode>(c0, c0_size, this_probe);
            // JasUse(J, ray_weight);
            // if constexpr (DIR_BY_DIR_3D) {
            //     J(la, flat_probe_idx) += ray_weight * sample;
            // } else {
            // }
            Kokkos::atomic_add(&J(la, flat_probe_idx), ray_weight * sample);
        }
    );
    Kokkos::fence();
}

void static_formal_sol_long_char_3d(const State3d& state, const CascadeState3d& casc_state) {
    assert(state.config.mode == DexrtMode::GivenFs);
    JasUnpack(state, mr_block_map, given_state, periodic);
    JasUnpack(casc_state, mip_chain);
    const auto& block_map = mr_block_map.block_map;
    const i32 num_wavelengths = state.J.extent(0);

    const fp_t distance_scale = state.given_state.voxel_scale;
    state.J = FP(0.0);
    Kokkos::fence();
    auto& eta_store = given_state.emis;
    auto& chi_store = given_state.opac;
    for (int la = 0; la < num_wavelengths; ++la) {
        dex_parallel_for(
            "Copy eta, chi",
            block_map.loop_bounds(),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen3d idx_gen(block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord3 coord = idx_gen.loop_coord(tile_idx, block_idx);
                mip_chain.emis(ks) = eta_store(la, coord.z, coord.y, coord.x);
                mip_chain.opac(ks) = chi_store(la, coord.z, coord.y, coord.x);
            }
        );
        Kokkos::fence();

        constexpr int RcMode = RC_flags_storage_3d();

        FpConst1d lc_qx = FpConst1dHost("lc_qx", (const fp_t*)LC_QUAD_X, NUM_LC_QUAD).createDeviceCopy();
        FpConst1d lc_qy = FpConst1dHost("lc_qy", (const fp_t*)LC_QUAD_Y, NUM_LC_QUAD).createDeviceCopy();
        FpConst1d lc_qz = FpConst1dHost("lc_qy", (const fp_t*)LC_QUAD_Z, NUM_LC_QUAD).createDeviceCopy();

        constexpr int num_subsets = NUM_LC_QUAD;
        for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
            fmt::println("Subset {} of {}...", subset_idx, num_subsets);

            CascadeStorage3d dims = cascade_size(state.c0_size, 0);
            Fp1d i_cascade_i = casc_state.i_cascades[0];
            Fp1d tau_cascade_i = casc_state.tau_cascades[0];
            FpConst1d i_cascade_ip, tau_cascade_ip;

            DeviceCascadeState3d dev_casc_state {
                .num_cascades = casc_state.num_cascades,
                .n = 0,
                .casc_dims = dims,
                .cascade_I = i_cascade_i,
                .cascade_tau = tau_cascade_i,
                .upper_I = i_cascade_ip,
                .upper_tau = tau_cascade_ip
            };

            std::string name("long char");
            yakl::timer_start(name);

            FlatLoop<3> probe_loop(dims.num_probes(2), dims.num_probes(1), dims.num_probes(0));

            dex_parallel_for(
                "RC Loop 3D",
                FlatLoop<1>(probe_loop.num_iter),
                KOKKOS_LAMBDA (i64 flat_probe_idx) {
                    auto rev_probe_coord = probe_loop.unpack(flat_probe_idx);
                    ivec3 probe_coord;
                    probe_coord(0) = rev_probe_coord[2];
                    probe_coord(1) = rev_probe_coord[1];
                    probe_coord(2) = rev_probe_coord[0];

                    ProbeIndex3d probe_idx {
                        .coord=probe_coord,
                        .polar = 0,
                        .az = 0
                    };

                    vec3 d;
                    d(0) = lc_qx(subset_idx);
                    d(1) = lc_qy(subset_idx);
                    d(2) = lc_qz(subset_idx);
                    vec3 o = probe_pos(probe_coord, 0);
                    RaySegment<3> ray(o, d, -LAST_CASCADE_MAX_DIST_3D, FP(0.0));

                    // compute_ri
                    Raymarch3dArgs<ZeroBc, DexEmpty> args {
                        .this_probe = probe_idx,
                        .casc_state = dev_casc_state,
                        .mr_block_map = mr_block_map,
                        .periodic = periodic,
                        .ray = ray,
                        .distance_scale = distance_scale,
                        .mip_chain = mip_chain,
                        .max_mip_to_sample = 0
                    };
                    RadianceInterval ri = multi_level_dda_raymarch_3d(
                        args
                    );
                    i64 lin_idx = probe_linear_index<RcMode>(dims, probe_idx);
                    dev_casc_state.cascade_I(lin_idx) = ri.I;
                    if constexpr (STORE_TAU_CASCADES) {
                        dev_casc_state.cascade_tau(lin_idx) = ri.tau;
                    }

                }
            );
            Kokkos::fence();

            yakl::timer_stop(name);
            merge_c0_to_J_3d(
                state,
                casc_state,
                la,
                LC_WEIGHT[subset_idx]
            );
            Kokkos::fence();
        }
    }
}

void static_formal_sol_rc_given_3d(const State3d& state, const CascadeState3d& casc_state) {
    assert(state.config.mode == DexrtMode::GivenFs);
    JasUnpack(state, mr_block_map, given_state, periodic);
    JasUnpack(casc_state, mip_chain);
    const auto& block_map = mr_block_map.block_map;
    const i32 num_wavelengths = state.J.extent(0);

    if constexpr (FORCE_LC_QUADRATURE) {
        static_formal_sol_long_char_3d(state, casc_state);
        return;
    }

    const fp_t distance_scale = state.given_state.voxel_scale;
    state.J = FP(0.0);
    Kokkos::fence();
    auto& eta_store = given_state.emis;
    auto& chi_store = given_state.opac;
    for (int la = 0; la < num_wavelengths; ++la) {
        dex_parallel_for(
            "Copy eta, chi",
            block_map.loop_bounds(),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen3d idx_gen(block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord3 coord = idx_gen.loop_coord(tile_idx, block_idx);
                mip_chain.emis(ks) = eta_store(la, coord.z, coord.y, coord.x);
                mip_chain.opac(ks) = chi_store(la, coord.z, coord.y, coord.x);
            }
        );
        Kokkos::fence();
        mip_chain.compute_mips(state, la);

        bool any_periodic = false;
        for (int i = 0; i < get_dexrt_dimensionality(); ++i) {
            any_periodic |= periodic(i);
        }

        constexpr int RcMode = RC_flags_storage_3d();

        constexpr int num_subsets = subset_tasks_per_cascade_3d<RcMode>();
        for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
            fmt::println("Subset {} of {}...", subset_idx, num_subsets);
            for (int casc_idx = casc_state.num_cascades; casc_idx >= 0; --casc_idx) {
                CascadeStorage3d dims = cascade_size(state.c0_size, casc_idx);
                CascadeStorage3d upper_dims = cascade_size(state.c0_size, casc_idx+1);
                CascadeRays3d ray_set = cascade_compute_size<RcMode>(state.c0_size, casc_idx);
                CascadeRaysSubset3d ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);

                CascadeIdxs lookup = cascade_indices(casc_state, casc_idx);
                Fp1d i_cascade_i = casc_state.i_cascades[lookup.i];
                Fp1d tau_cascade_i = casc_state.tau_cascades[lookup.i];
                FpConst1d i_cascade_ip, tau_cascade_ip;
                if (lookup.ip != -1) {
                    i_cascade_ip = casc_state.i_cascades[lookup.ip];
                    tau_cascade_ip = casc_state.tau_cascades[lookup.ip];
                }

                const int max_mip_to_sample = std::min(
                    state.config.mip_config.mip_levels[casc_idx],
                    mip_chain.max_mip_factor
                );

                DeviceCascadeState3d dev_casc_state {
                    .num_cascades = casc_state.num_cascades,
                    .n = casc_idx,
                    .casc_dims = dims,
                    .upper_dims = upper_dims,
                    .cascade_I = i_cascade_i,
                    .cascade_tau = tau_cascade_i,
                    .upper_I = i_cascade_ip,
                    .upper_tau = tau_cascade_ip
                };

                std::string name = fmt::format("Cascade {}", casc_idx);
                yakl::timer_start(name);

                FlatLoop<3> probe_loop(ray_set.num_probes(2), ray_set.num_probes(1), ray_set.num_probes(0));

                dex_parallel_for(
                    "RC Loop 3D",
                    FlatLoop<3>(probe_loop.num_iter, ray_subset.num_polar_rays, ray_subset.num_az_rays),
                    KOKKOS_LAMBDA (i64 flat_probe_idx, int theta_idx, int phi_idx) {
                        auto rev_probe_coord = probe_loop.unpack(flat_probe_idx);
                        ivec3 probe_coord;
                        probe_coord(0) = rev_probe_coord[2];
                        probe_coord(1) = rev_probe_coord[1];
                        probe_coord(2) = rev_probe_coord[0];

                        theta_idx += ray_subset.start_polar_rays;
                        phi_idx += ray_subset.start_az_rays;
                        ProbeIndex3d probe_idx {
                            .coord=probe_coord,
                            .polar = theta_idx,
                            .az = phi_idx
                        };

                        RaySegment<3> ray = probe_ray(ray_set, dev_casc_state.num_cascades, casc_idx, probe_idx);

                        // compute_ri
                        constexpr bool trilinear_fix = false;
                        RadianceInterval ri;
                        Raymarch3dArgs<ZeroBc, DexEmpty> args {
                            .this_probe = probe_idx,
                            .casc_state = dev_casc_state,
                            .mr_block_map = mr_block_map,
                            .periodic = periodic,
                            .ray = ray,
                            .distance_scale = distance_scale,
                            .mip_chain = mip_chain,
                            .max_mip_to_sample = max_mip_to_sample
                        };
                        JasUse(any_periodic);
                        if constexpr (trilinear_fix) {
                            if (any_periodic) {
                                ri = march_and_merge_trilinear_interval_3d<RcMode | RC_PERIODIC>(
                                    args
                                );
                            } else {
                                ri = march_and_merge_trilinear_interval_3d<RcMode>(
                                    args
                                );
                            }
                        } else {
                            if (any_periodic) {
                                ri = march_and_merge_average_interval_3d<RcMode | RC_PERIODIC>(
                                    args
                                );
                            } else {
                                ri = march_and_merge_average_interval_3d<RcMode>(
                                    args
                                );
                            }
                        }
                        i64 lin_idx = probe_linear_index<RcMode>(dims, probe_idx);
                        dev_casc_state.cascade_I(lin_idx) = ri.I;
                        if constexpr (STORE_TAU_CASCADES) {
                            dev_casc_state.cascade_tau(lin_idx) = ri.tau;
                        }

                        // TODO(cmo): ALO
                    }
                );
                Kokkos::fence();

                yakl::timer_stop(name);
            }
            merge_c0_to_J_3d(
                state,
                casc_state,
                la
            );
            Kokkos::fence();

        }

    }
}