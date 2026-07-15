#include "DynamicFormalSolution3d.hpp"
#include "CascadeState.hpp"
#include "Mipmaps3d.hpp"
#include "RayMarching.hpp" // only for merge_intervals
#include "RayMarching3d.hpp"
#include "RadianceCascades3d.hpp"
#include "BoundaryDispatch.hpp"
#include "GammaMatrix.hpp"

template <typename BcType=void>
KOKKOS_INLINE_FUNCTION
BcType get_bc(const DeviceBoundaries& bounds) {

}

template <>
KOKKOS_INLINE_FUNCTION
ZeroBc get_bc(const DeviceBoundaries& bounds) {
    return bounds.zero_bc;
}

template <>
KOKKOS_INLINE_FUNCTION
PwBc<> get_bc(const DeviceBoundaries& bounds) {
    return bounds.pw_bc;
}

template <>
KOKKOS_INLINE_FUNCTION
PlaneBc<> get_bc(const DeviceBoundaries& bounds) {
    return bounds.plane_bc;
}


void dynamic_compute_gamma(
    const State3d& state,
    const CascadeState3d& casc_state,
    const Fp2d& lte_scratch,
    int la,
    int subset_idx
) {
    JasUnpack(state, phi, pops, adata, wphi, mr_block_map);
    const auto flatmos = flatten<const fp_t>(state.atmos);
    constexpr int RcMode = RC_flags_storage_3d();
    const auto& wavelength = adata.wavelength;

    CascadeStorage3d dims = state.c0_size;
    CascadeRays3d ray_set = cascade_compute_size<RcMode>(dims, 0);
    CascadeRaysSubset3d ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    const int num_cascades = casc_state.num_cascades;
    const auto spatial_bounds = mr_block_map.block_map.loop_bounds();
    for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
        const auto& Gamma = state.Gamma[ia];
        const auto& psi_star = casc_state.psi_star;
        const auto& I = casc_state.i_cascades[0];
        dex_parallel_for(
            "Compute Gamma",
            FlatLoop<4>(
                spatial_bounds.dim(0),
                spatial_bounds.dim(1),
                ray_subset.num_polar_rays,
                ray_subset.num_az_rays
            ),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx, int theta_idx, int phi_idx) {
                IdxGen3d idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord3 cell_coord = idx_gen.loop_coord(tile_idx, block_idx);
                ivec3 probe_coord;
                probe_coord(0) = cell_coord.x;
                probe_coord(1) = cell_coord.y;
                probe_coord(2) = cell_coord.z;

                theta_idx += ray_subset.start_polar_rays;
                phi_idx += ray_subset.start_az_rays;
                const ProbeIndex3d probe_idx{
                    .coord=probe_coord,
                    .polar = theta_idx,
                    .az = phi_idx
                };
                RaySegment<3> ray = probe_ray(ray_set, num_cascades, 0, probe_idx);
                const fp_t intensity = probe_fetch<RcMode>(I, ray_set, probe_idx);
                const fp_t psi_star_entry = probe_fetch<RcMode>(psi_star, ray_set, probe_idx);

                const fp_t lambda = wavelength(la);
                using namespace ConstantsFP;

                // NOTE(cmo): We can drop 4pi/hc from the wavelength/angle
                // integral weight if we divide U and V by hc/4pi. For lines,
                // these terms normally cancel: for continua it's an extra
                // operation. In both cases, to construct eta and chi, we need
                // to multiply those terms by the populations (as usual), but
                // _also_ hc/4pi .
                constexpr bool include_4pi_hc = false;
                constexpr fp_t hc_4pi = hc_kJ_nm / four_pi;
                fp_t hc_4pi_eff = FP(1.0);
                if (include_4pi_hc) {
                    hc_4pi_eff = hc_kJ_nm / four_pi;
                }
                fp_t wl_weight = lambda / hc_4pi_eff * adata.wavelength_bin(la);
                const fp_t wl_ray_weight = wl_weight / fp_t(ray_set.num_az_rays * ray_set.num_polar_rays);
                AtmosPointParams local_atmos;
                local_atmos.temperature = flatmos.temperature(ks);
                local_atmos.ne = flatmos.ne(ks);
                local_atmos.vturb = flatmos.vturb(ks);
                local_atmos.nhtot = flatmos.nh_tot(ks);
                local_atmos.nh0 = flatmos.nh0(ks);
                local_atmos.vel = (
                        flatmos.vx(ks) * ray.d(0)
                        + flatmos.vy(ks) * ray.d(1)
                        + flatmos.vz(ks) * ray.d(2)
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
                            .n = pops,
                            .n_star_scratch = lte_scratch,
                            .k = ks,
                            .atmos = local_atmos
                        },
                        kr,
                        UvOptions {
                            .divide_by_hc_4pi = !include_4pi_hc
                        }
                    );

                    const int offset = adata.level_start(ia);
                    fp_t eta = pops(offset + l.j, ks) * uv.Uji;
                    fp_t chi = pops(offset + l.i, ks) * uv.Vij - pops(offset + l.j, ks) * uv.Vji;
                    if (!include_4pi_hc) {
                        // NOTE(cmo): If we have dropped the hc/4pi
                        // from U and V, these still need the
                        // dimensioning
                        eta *= hc_4pi;
                        chi *= hc_4pi;
                    }
                    chi += FP(1e-20);

                    add_to_gamma<true>(GammaAccumState{
                        .eta = eta,
                        .chi = chi,
                        .uv = uv,
                        .I = intensity,
                        .psi_star = psi_star_entry,
                        .wlamu = wl_ray_weight * wphi(kr, ks),
                        .Gamma = Gamma,
                        .i = l.i,
                        .j = l.j,
                        .k = ks
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
                            .n = pops,
                            .n_star_scratch = lte_scratch,
                            .k = ks,
                            .atmos = local_atmos
                        },
                        kr,
                        UvOptions {
                            .divide_by_hc_4pi = !include_4pi_hc
                        }
                    );

                    const int offset = adata.level_start(ia);
                    fp_t eta = pops(offset + cont.j, ks) * uv.Uji;
                    fp_t chi = pops(offset + cont.i, ks) * uv.Vij - pops(offset + cont.j, ks) * uv.Vji;
                    if (!include_4pi_hc) {
                        // NOTE(cmo): If we have dropped the hc/4pi
                        // from U and V, these still need the
                        // dimensioning
                        eta *= hc_4pi;
                        chi *= hc_4pi;
                    }
                    chi += FP(1e-20);

                    add_to_gamma<true>(GammaAccumState{
                        .eta = eta,
                        .chi = chi,
                        .uv = uv,
                        .I = intensity,
                        .psi_star = psi_star_entry,
                        .wlamu = wl_ray_weight,
                        .Gamma = Gamma,
                        .i = cont.i,
                        .j = cont.j,
                        .k = ks
                    });
                }
            }
        );
    }
    Kokkos::fence();
}


static void merge_c0_to_J_3d(const State3d& state, const CascadeState3d& casc_state, int la, fp_t ray_weight=FP(-1.0)) {
    constexpr int RcMode = RC_flags_storage_3d();
    const bool sparse = casc_state.probes_to_compute.sparse;
    const CascadeStorage3d c0_dims = casc_state.probes_to_compute.c0_size;
    CascadeRays3d ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
    if (ray_weight < FP(0.0)) {
        ray_weight = FP(1.0) / fp_t(ray_set.num_az_rays * ray_set.num_polar_rays);
    }

    const auto& c0 = casc_state.i_cascades[0];
    const auto& J = state.J;

    JasUnpack(state, mr_block_map);
    const FlatLoop<2> spatial_bounds = mr_block_map.block_map.loop_bounds();

    const bool J_slice = (J.extent(0) == 1);
    DeviceProbesToCompute<3> probes_to_compute = casc_state.probes_to_compute.bind(0);

    dex_parallel_for(
        "Final cascade to J",
        FlatLoop<3>(probes_to_compute.num_active_probes(), c0_dims.num_polar_rays, c0_dims.num_az_rays),
        KOKKOS_LAMBDA (i64 k, int theta_idx, int phi_idx) {
            ivec3 probe_coord = probes_to_compute(k);
            ProbeStorageIndex3d this_probe {
                .coord = probe_coord,
                .polar = theta_idx,
                .az = phi_idx
            };

            i64 ks;
            if (sparse) {
                IdxGen3d idx_gen(mr_block_map);
                ks = idx_gen.idx(Coord3{.x = probe_coord(0), .y = probe_coord(1), .z = probe_coord(2)});
            } else {
                ks = i64(probe_coord(2) * c0_dims.num_probes(1) + probe_coord(1)) * c0_dims.num_probes(0) + probe_coord(0);
            }

            int inner_la = J_slice ? 0 : la;
            const fp_t sample = probe_fetch<RcMode>(c0, c0_dims, this_probe);
            Kokkos::atomic_add(&J(inner_la, ks), ray_weight * sample);
        }
    );
    Kokkos::fence();
}

template <int RcMode>
void compute_cascade_i_3d(const State3d& state, const CascadeState3d& casc_state, int la, int subset_idx, int casc_idx) {
    JasUnpack(state, atmos, phi, adata, pops, mr_block_map, periodic);
    const auto& mip_chain = casc_state.mip_chain;
    constexpr bool compute_alo = RcMode & RC_COMPUTE_ALO;
    using Alo = std::conditional_t<compute_alo, fp_t, DexEmpty>;
    typedef typename RcDynamicState3d<RcMode>::type DynamicState;

    const fp_t distance_scale = atmos.voxel_scale;
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
    if constexpr (compute_alo) {
        dev_casc_state.psi_star = casc_state.psi_star;
    }
    DeviceBoundaries boundaries_h{
        .boundary = state.boundary,
        .zero_bc = state.zero_bc,
        .pw_bc = state.pw_bc,
        .plane_bc = state.plane_bc
    };
    auto offset = get_offsets(atmos);

    std::string name = fmt::format("Cascade {}", casc_idx);
    yakl::timer_start(name);

    i64 spatial_bounds = casc_state.probes_to_compute.num_active_probes(casc_idx);
    DeviceProbesToCompute<3> probe_coord_lookup = casc_state.probes_to_compute.bind(casc_idx);

    dex_parallel_for(
        "RC Loop 3D",
        FlatLoop<3>(spatial_bounds, ray_subset.num_polar_rays, ray_subset.num_az_rays),
        KOKKOS_LAMBDA (i64 flat_probe_idx, int theta_idx, int phi_idx) {
            ivec3 probe_coord = probe_coord_lookup(flat_probe_idx);

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
            RadianceInterval<Alo> ri;

            const auto& boundaries = boundaries_h;
            bool print_debug = false;
            if (casc_idx == 1 && probe_idx.coord(0) == 2 && probe_idx.coord(1) == 3 && probe_idx.coord(2) == 27 && probe_idx.polar==0 && probe_idx.az==3) {
                print_debug = true;

            }

            auto dispatch = [&]<typename BcType, typename DynamicState>(const BcType& bc, const DynamicState& ds) {
                Raymarch3dArgs<BcType, DynamicState> args {
                    .this_probe = probe_idx,
                    .casc_state = dev_casc_state,
                    .mr_block_map = mr_block_map,
                    .periodic = periodic,
                    .ray = ray,
                    .distance_scale = distance_scale,
                    .mip_chain = mip_chain,
                    .max_mip_to_sample = max_mip_to_sample,
                    .bc = get_bc<BcType>(boundaries),
                    .dyn_state = ds,
                    .la = la,
                    .offset = offset,
                    .print_debug = print_debug
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
            };
            DynamicState dyn_state = get_dyn_state_3d<DynamicState>(
                la,
                atmos,
                adata,
                phi,
                pops,
                mip_chain
            );
            if constexpr (RcMode & RC_SAMPLE_BC) {
                switch (boundaries.boundary) {
                    case BoundaryType::Zero: {
                        dispatch(ZeroBc{}, dyn_state);
                    } break;
                    case BoundaryType::Promweaver:{
                        dispatch(PwBc<>{}, dyn_state);
                    } break;
                    case BoundaryType::Plane:{
                        dispatch(PlaneBc<>{}, dyn_state);
                    } break;
                    default: {
                        Kokkos::abort("Unknown BC type");
                    }
                }
            } else {
                dispatch(ZeroBc{}, dyn_state);
            }
            i64 lin_idx = probe_linear_index<RcMode>(dims, probe_idx);
            dev_casc_state.cascade_I(lin_idx) = ri.I;
            if constexpr (STORE_TAU_CASCADES) {
                dev_casc_state.cascade_tau(lin_idx) = ri.tau;
            }
            constexpr bool dev_compute_alo = bool(RcMode & RC_COMPUTE_ALO);
            if constexpr (dev_compute_alo) {
                dev_casc_state.psi_star(lin_idx) = ri.psi_star;
            }
        }
    );
    Kokkos::fence();

    yakl::timer_stop(name);
}

// NOTE(cmo): Needed to pass an integer into a lambda as a template param
template <int N>
using Constant = std::integral_constant<int, N>;

void dynamic_formal_sol_rc_3d_subset(
    const State3d& state,
    const CascadeState3d& casc_state,
    bool lambda_iterate,
    int la,
    int subset_idx
) {
    constexpr int RcModeBc = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = false,
        .sample_bc = true,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR_3D
    });
    constexpr int RcModeNoBc = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = false,
        .sample_bc = false,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR_3D
    });
    constexpr int RcModeAlo = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = false,
        .sample_bc = false,
        .compute_alo = true,
        .dir_by_dir = DIR_BY_DIR_3D
    });
    bool any_periodic = false;
    for (int i = 0; i < get_dexrt_dimensionality(); ++i) {
        any_periodic |= state.periodic(i);
    }

    auto dispatch_cascade_i = [&]<int RcMode>(
        Constant<RcMode>,
        const State3d& state,
        const CascadeState3d& casc_state,
        int la,
        int subset_idx,
        int casc_idx
    ) {
        if (any_periodic) {
            return compute_cascade_i_3d<RcMode | RC_PERIODIC>(
                state,
                casc_state,
                la,
                subset_idx,
                casc_idx
            );
        } else {
            return compute_cascade_i_3d<RcMode>(
                state,
                casc_state,
                la,
                subset_idx,
                casc_idx
            );
        }
    };

    if (casc_state.num_cascades > 0) {
        dispatch_cascade_i(
            Constant<RcModeBc>{},
            state,
            casc_state,
            la,
            subset_idx,
            casc_state.num_cascades
        );
    }
    for (int casc_idx = casc_state.num_cascades - 1; casc_idx >= 1; --casc_idx) {
        dispatch_cascade_i(
            Constant<RcModeNoBc>{},
            state,
            casc_state,
            la,
            subset_idx,
            casc_idx
        );
    }
    if (casc_state.psi_star.initialized() && !lambda_iterate) {
        dispatch_cascade_i(
            Constant<RcModeAlo>{},
            state,
            casc_state,
            la,
            subset_idx,
            0
        );
    } else {
        dispatch_cascade_i(
            Constant<RcModeNoBc>{},
            state,
            casc_state,
            la,
            subset_idx,
            0
        );
    }
}

void dynamic_formal_sol_rc_3d(const State3d& state, const CascadeState3d& casc_state, bool lambda_iterate, int la) {
    JasUnpack(casc_state, mip_chain);

    // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
    // it, for now, trust the pool allocator
    auto pops_dims = state.pops.get_dimensions();
    Fp2d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1));

    mip_chain.fill_mip0_atomic(state, lte_scratch, la);
    mip_chain.compute_mips(state, la);

    constexpr int RcStorage = RC_flags_storage_3d();

    constexpr int num_subsets = subset_tasks_per_cascade_3d<RcStorage>();
    for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
        if (casc_state.psi_star.initialized()) {
            casc_state.psi_star = FP(0.0);
        }
        dynamic_formal_sol_rc_3d_subset(state, casc_state, lambda_iterate, la, subset_idx);
        if (casc_state.psi_star.initialized()) {
            dynamic_compute_gamma(
                state,
                casc_state,
                lte_scratch,
                la,
                subset_idx
            );
        }
        merge_c0_to_J_3d(
            state,
            casc_state,
            la
        );
        Kokkos::fence();
    }
}