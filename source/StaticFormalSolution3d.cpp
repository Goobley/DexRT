#include "StaticFormalSolution3d.hpp"
#include "CascadeState.hpp"
#include "Mipmaps3d.hpp"
#include "RayMarching.hpp" // only for merge_intervals

struct Raymarch3dArgs {
    const ProbeIndex3d& this_probe;
    const DeviceCascadeState3d& casc_state;
    const MultiResBlockMap<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>& mr_block_map;
    const RaySegment<3>& ray;
    const fp_t distance_scale;
    const MultiResMipChain3d& mip_chain;
    const i32 max_mip_to_sample;
};

YAKL_INLINE RadianceInterval<DexEmpty> multi_level_dda_raymarch_3d(
    const Raymarch3dArgs& args
) {
    JasUnpack(args, mr_block_map, ray, distance_scale, mip_chain);
    RadianceInterval<DexEmpty> result;

    MRIdxGen3d idx_gen(mr_block_map);
    auto s = MultiLevelDDA<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>(idx_gen);
    const bool marcher = s.init(ray, args.max_mip_to_sample, nullptr);
    if (!marcher) {
        return result;
    }
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> eta_k(
        mip_chain.emis.data(),
        mip_chain.emis.extent(0)
    );
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> chi_k(
        mip_chain.opac.data(),
        mip_chain.opac.extent(0)
    );

    fp_t eta_s = FP(0.0), chi_s = FP(1e-20);
    do {
        if (s.can_sample()) {
            i64 ks = idx_gen.idx(
                s.current_mip_level,
                Coord3{.x = s.curr_coord(0), .y = s.curr_coord(1), .z = s.curr_coord(2)}
            );
            eta_s = eta_k(ks);
            chi_s = chi_k(ks) + FP(1e-15);

            fp_t tau = chi_s * s.dt * distance_scale;
            fp_t source_fn = eta_s / chi_s;
            fp_t edt = std::exp(-tau);
            fp_t one_m_edt = -std::expm1(-tau);
            result.tau += tau;
            result.I = result.I * edt + source_fn * one_m_edt;
        }

    } while (s.step_through_grid());

    return result;
}

template <int RcMode=0>
YAKL_INLINE RadianceInterval<DexEmpty> march_and_merge_average_interval_3d(const Raymarch3dArgs& args) {
    JasUnpack(args, this_probe, casc_state);
    RadianceInterval<DexEmpty> ri = multi_level_dda_raymarch_3d(args);

    RadianceInterval<DexEmpty> interp;
    if (casc_state.upper_I.initialized()) {
        JasUnpack(casc_state, upper_I, upper_tau, casc_dims, upper_dims);

        TexelsPerRay3d upper_tex = upper_texels_per_ray_3d<RcMode>(casc_state.n);
        int upper_polar_start_idx = this_probe.polar * upper_dims.num_polar_rays / casc_dims.num_polar_rays;
        int upper_az_start_idx = this_probe.az * upper_dims.num_az_rays / casc_dims.num_az_rays;
        const fp_t ray_weight = FP(1.0) / fp_t(upper_tex.az * upper_tex.polar);

        TrilinearCorner base = trilinear_corner(this_probe.coord);
        vec<8> weights = trilinear_weights(base);
        for (int tri_idx = 0; tri_idx < 8; ++tri_idx) {
            ivec3 upper_coord = trilinear_coord(base, upper_dims.num_probes, tri_idx);
            for (
                int upper_polar_idx = upper_polar_start_idx;
                upper_polar_idx < upper_polar_start_idx + upper_tex.polar;
                ++upper_polar_idx
            ) {
                for (
                    int upper_az_idx = upper_az_start_idx;
                    upper_az_idx < upper_az_start_idx + upper_tex.az;
                    ++upper_az_idx
                ) {
                    ProbeIndex3d upper_probe {
                        .coord = upper_coord,
                        .polar = upper_polar_idx,
                        .az = upper_az_idx
                    };
                    i64 lin_idx = probe_linear_index<RcMode>(upper_dims, upper_probe);
                    interp.I += ray_weight * weights(tri_idx) * upper_I(lin_idx);
                    if constexpr (STORE_TAU_CASCADES) {
                        interp.tau += ray_weight * weights(tri_idx) * upper_tau(lin_idx);
                    }
                }
            }
        }
    }
    return merge_intervals(ri, interp);
}

template <int RcMode=0>
YAKL_INLINE RadianceInterval<DexEmpty> march_and_merge_trilinear_interval_3d(const Raymarch3dArgs& args) {
    JasUnpack(args, this_probe, casc_state);

    RadianceInterval<DexEmpty> interp;
    if (casc_state.upper_I.initialized()) {
        JasUnpack(casc_state, upper_I, upper_tau, casc_dims, upper_dims);
        CascadeRays3d casc_rays = cascade_storage_to_rays<RcMode>(casc_dims);
        CascadeRays3d upper_casc_rays = cascade_storage_to_rays<RcMode>(upper_dims);

        TexelsPerRay3d upper_tex = upper_texels_per_ray_3d<RcMode>(casc_state.n);
        int upper_polar_start_idx = this_probe.polar * upper_dims.num_polar_rays / casc_dims.num_polar_rays;
        int upper_az_start_idx = this_probe.az * upper_dims.num_az_rays / casc_dims.num_az_rays;
        const fp_t ray_weight = FP(1.0) / fp_t(upper_tex.az * upper_tex.polar);

        TrilinearCorner base = trilinear_corner(this_probe.coord);
        vec<8> weights = trilinear_weights(base);
        for (int tri_idx = 0; tri_idx < 8; ++tri_idx) {
            ivec3 upper_coord = trilinear_coord(base, upper_dims.num_probes, tri_idx);
            RaySegment<3> tri_ray = trilinear_probe_ray(
                casc_rays,
                upper_casc_rays,
                casc_state.num_cascades,
                casc_state.n,
                this_probe,
                upper_coord
            );
            RadianceInterval<DexEmpty> ri = multi_level_dda_raymarch_3d(
                Raymarch3dArgs {
                    .this_probe = args.this_probe,
                    .casc_state = args.casc_state,
                    .mr_block_map = args.mr_block_map,
                    .ray = tri_ray,
                    .distance_scale = args.distance_scale,
                    .mip_chain = args.mip_chain,
                    .max_mip_to_sample = args.max_mip_to_sample
                }
            );

            RadianceInterval<DexEmpty> upper_interp{};
            for (
                int upper_polar_idx = upper_polar_start_idx;
                upper_polar_idx < upper_polar_start_idx + upper_tex.polar;
                ++upper_polar_idx
            ) {
                for (
                    int upper_az_idx = upper_az_start_idx;
                    upper_az_idx < upper_az_start_idx + upper_tex.az;
                    ++upper_az_idx
                ) {
                    ProbeIndex3d upper_probe {
                        .coord = upper_coord,
                        .polar = upper_polar_idx,
                        .az = upper_az_idx
                    };
                    i64 lin_idx = probe_linear_index<RcMode>(upper_dims, upper_probe);
                    upper_interp.I += ray_weight * upper_I(lin_idx);
                    if constexpr (STORE_TAU_CASCADES) {
                        upper_interp.tau += ray_weight * upper_tau(lin_idx);
                    }
                }
            }
            RadianceInterval<DexEmpty> merged = merge_intervals(ri, upper_interp);
            interp.I += weights(tri_idx) * merged.I;
            if constexpr (STORE_TAU_CASCADES) {
                interp.tau += weights(tri_idx) * merged.tau;
            }
        }
    } else {
        interp = multi_level_dda_raymarch_3d(args);
    }
    return interp;
}

void merge_c0_to_J_3d(const State3d& state, const CascadeState3d& casc_state, int la, fp_t ray_weight=FP(-1.0)) {
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
    JasUnpack(state, mr_block_map, given_state);
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
                    Raymarch3dArgs args {
                        .this_probe = probe_idx,
                        .casc_state = dev_casc_state,
                        .mr_block_map = mr_block_map,
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
    JasUnpack(state, mr_block_map, given_state);
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

        // TODO(cmo): Update this
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
                        Raymarch3dArgs args {
                            .this_probe = probe_idx,
                            .casc_state = dev_casc_state,
                            .mr_block_map = mr_block_map,
                            .ray = ray,
                            .distance_scale = distance_scale,
                            .mip_chain = mip_chain,
                            .max_mip_to_sample = max_mip_to_sample
                        };
                        if constexpr (trilinear_fix) {
                            ri = march_and_merge_trilinear_interval_3d<RcMode>(
                                args
                            );
                        } else {
                            ri = march_and_merge_average_interval_3d<RcMode>(
                                args
                            );
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