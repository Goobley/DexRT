#if !defined(DEXRT_RAY_MARCHING_3D_HPP)
#define DEXRT_RAY_MARCHING_3D_HPP
#include "State3d.hpp"
#include "CascadeState3d.hpp"
#include "CascadeState.hpp"
#include "Mipmaps3d.hpp"

template <typename DynamicState=void>
KOKKOS_INLINE_FUNCTION
DynamicState get_dyn_state_3d(
    int la,
    const SparseAtmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const MultiResMipChain3d& mip_chain
) {
    return DynamicState{};
}

struct Raymarch3dDynamicState {
    const yakl::Array<const u16, 1, yakl::memDevice> active_set;
    const SparseAtmosphere& atmos;
    const AtomicData<fp_t>& adata;
    const VoigtProfile<fp_t, false>& profile;
    const Fp1d& nh0;
    const Fp2d& n;
};

template <>
KOKKOS_INLINE_FUNCTION
Raymarch3dDynamicState get_dyn_state_3d(
    int la,
    const SparseAtmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const MultiResMipChain3d& mip_chain
) {
    return Raymarch3dDynamicState{
        .active_set = slice_active_set(adata, la),
        .atmos = atmos,
        .adata = adata,
        .profile = profile,
        .nh0 = atmos.nh0,
        .n = flat_pops
    };
}

struct Raymarch3dDynamicCavState {
    const yakl::SArray<i32, 1, CORE_AND_VOIGT_MAX_LINES_3D> active_set;
    const VoigtProfile<fp_t, false>& profile;
    const AtomicData<fp_t>& adata;
};

template <>
KOKKOS_INLINE_FUNCTION
Raymarch3dDynamicCavState get_dyn_state_3d(
    int la,
    const SparseAtmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const MultiResMipChain3d& mip_chain
) {
    auto basic_a_set = slice_active_set(adata, la);
    yakl::SArray<i32, 1, CORE_AND_VOIGT_MAX_LINES_3D> local_active_set; // In krl indices
    const auto& krl_mapping = mip_chain.cav_data.active_set_mapping;
    int l_idx = 0;
    for (int a = 0; a < basic_a_set.extent(0); ++a) {
        i32 kr = basic_a_set(a);
        for (int krl = 0; krl < CORE_AND_VOIGT_MAX_LINES_3D; ++krl) {
            if (krl_mapping(krl) == kr) {
                local_active_set(l_idx++) = krl;
            }
        }
    }
    if (l_idx < CORE_AND_VOIGT_MAX_LINES_3D) {
        local_active_set(l_idx) = -1;
    }
    return Raymarch3dDynamicCavState{
        .active_set = local_active_set,
        .profile = profile,
        .adata = adata
    };
}


template <int RcMode=0>
struct RcDynamicState3d {
    typedef typename std::conditional_t<
        RcMode & RC_DYNAMIC && (LINE_SCHEME_3D == LineCoeffCalc::CoreAndVoigt),
        Raymarch3dDynamicCavState,
        std::conditional_t<
            RcMode & RC_DYNAMIC,
            Raymarch3dDynamicState,
            DexEmpty
        >
    > type;
};

template <typename Bc, typename DynamicState>
struct Raymarch3dArgs {
    const ProbeIndex3d& this_probe;
    const DeviceCascadeState3d& casc_state;
    const MultiResBlockMap<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>& mr_block_map;
    const yakl::SArray<bool, 1, 3> periodic;
    const RaySegment<3>& ray;
    const fp_t distance_scale;
    const MultiResMipChain3d& mip_chain;
    const i32 max_mip_to_sample;
    Bc bc;
    DynamicState dyn_state;
    int la;
    vec3 offset;
    bool print_debug = false;
};

template <class DynamicState = DexEmpty>
struct SampleEmisOpac3dArgs {
    i64 ks;
    i32 la;
    fp_t lambda;
    vec3 mu;
    const MultiResMipChain3d& mip_chain;
    DynamicState dyn_state;
};

template <typename DynamicState, std::enable_if_t<std::is_same_v<DynamicState, DexEmpty>, int> = 0>
KOKKOS_INLINE_FUNCTION EmisOpac sample_emis_opac(const SampleEmisOpac3dArgs<DynamicState>& args) {
    JasUnpack(args, ks, mip_chain);
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> eta_k(
        mip_chain.emis.data(),
        mip_chain.emis.extent(0)
    );
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> chi_k(
        mip_chain.opac.data(),
        mip_chain.opac.extent(0)
    );

    fp_t eta_s = eta_k(ks);
    fp_t chi_s = chi_k(ks);
    return EmisOpac {
        .eta = eta_s,
        .chi = chi_s
    };
}

template <typename DynamicState, std::enable_if_t<std::is_same_v<DynamicState, Raymarch3dDynamicState>, int> = 0>
KOKKOS_INLINE_FUNCTION EmisOpac sample_emis_opac(const SampleEmisOpac3dArgs<DynamicState>& args) {
    JasUnpack(args, ks, la, mu, mip_chain, dyn_state);
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> eta_k(
        mip_chain.emis.data(),
        mip_chain.emis.extent(0)
    );
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> chi_k(
        mip_chain.opac.data(),
        mip_chain.opac.extent(0)
    );
    fp_t eta_s = eta_k(ks);
    fp_t chi_s = chi_k(ks) + FP(1e-15);
    const SparseAtmosphere& atmos = dyn_state.atmos;
    if (
        mip_chain.classic_data.dynamic_opac(ks)
        && dyn_state.active_set.extent(0) > 0
    ) {
        const fp_t vel = (
            atmos.vx(ks) * mu(0)
            + atmos.vy(ks) * mu(1)
            + atmos.vz(ks) * mu(2)
        );
        AtmosPointParams local_atmos{
            .temperature = atmos.temperature.get_data()[ks],
            .ne = atmos.ne.get_data()[ks],
            .vturb = atmos.vturb.get_data()[ks],
            .nhtot = atmos.nh_tot.get_data()[ks],
            .vel = vel,
            .nh0 = dyn_state.nh0.get_data()[ks]
        };
        auto lines = emis_opac(
            EmisOpacState<fp_t>{
                .adata = dyn_state.adata,
                .profile = dyn_state.profile,
                .la = la,
                .n = dyn_state.n,
                .k = ks,
                .atmos = local_atmos,
                .active_set = dyn_state.active_set,
                .mode = EmisOpacMode::DynamicOnly
            }
        );
        eta_s += lines.eta;
        chi_s += lines.chi;
    }
    return EmisOpac {
        .eta = eta_s,
        .chi = chi_s
    };
}

template <typename DynamicState, std::enable_if_t<std::is_same_v<DynamicState, Raymarch3dDynamicCavState>, int> = 0>
KOKKOS_INLINE_FUNCTION EmisOpac sample_emis_opac(const SampleEmisOpac3dArgs<DynamicState>& args) {
    JasUnpack(args, ks, lambda, mu, mip_chain, dyn_state);
    JasUnpack(dyn_state, active_set, profile, adata);
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> eta_k(
        mip_chain.emis.data(),
        mip_chain.emis.extent(0)
    );
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> chi_k(
        mip_chain.opac.data(),
        mip_chain.opac.extent(0)
    );
    fp_t eta_s = eta_k(ks);
    fp_t chi_s = chi_k(ks) + FP(1e-15);

    const fp_t vel = (
        mip_chain.vx(ks) * mu(0)
        + mip_chain.vy(ks) * mu(1)
        + mip_chain.vz(ks) * mu(2)
    );
    CavEmisOpacState emis_opac_state {
        .ks = ks,
        .krl = 0,
        .wave = 0,
        .lambda = lambda,
        .vel = vel,
        .phi = profile
    };

    #pragma unroll
    for (int kri = 0; kri < CORE_AND_VOIGT_MAX_LINES_3D; ++kri) {
        i32 krl = active_set(kri);
        if (krl < 0) {
            break;
        }
        emis_opac_state.krl = krl;
        i32 kr = mip_chain.cav_data.active_set_mapping(krl);
        EmisOpac eta_chi = mip_chain.cav_data.emis_opac(
            emis_opac_state
        );
        eta_s += eta_chi.eta;
        chi_s += eta_chi.chi;
    }
    return EmisOpac {
        .eta = eta_s,
        .chi = chi_s
    };
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
YAKL_INLINE RadianceInterval<Alo> multi_level_dda_raymarch_3d_periodic(
    const Raymarch3dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, mr_block_map, distance_scale, mip_chain, la, bc, offset, dyn_state, periodic);
    JasUnpack(args, print_debug);
    // NOTE(cmo): Need to be able to modify this in the periodic case
    auto ray = args.ray;
    ray.update_origin(ray.t0);
    constexpr bool dynamic = (RcMode & RC_DYNAMIC);
    RadianceInterval<Alo> result;

    MRIdxGen3d idx_gen(mr_block_map);
    auto s = MultiLevelDDA<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>(idx_gen);
    int start_clipped_axis;
    bool marcher = s.init(ray, args.max_mip_to_sample, &start_clipped_axis);
    constexpr bool always_sample_bc = (RcMode & RC_SAMPLE_BC) && LAST_CASCADE_TO_INFTY && !(RcMode & RC_LINE_SWEEP);
    const bool ray_starts_outside = (RcMode & RC_SAMPLE_BC) && (!marcher || start_clipped_axis != -1);
    if (always_sample_bc || ray_starts_outside) {
        // NOTE(cmo): Check the ray is going up along z.
        if ((ray.d(2) > FP(0.0)) && la != -1) {
            vec3 pos;
            pos(0) = ray.o(0) * distance_scale + offset(0);
            pos(1) = ray.o(1) * distance_scale + offset(1);
            pos(2) = ray.o(2) * distance_scale + offset(2);

            fp_t I_sample = sample_boundary(bc, la, pos, ray.d);
            result.I = I_sample;
        }
    }

    const auto& aabb = args.mr_block_map.block_map.bbox;
    constexpr int num_dim = 3;
    bool has_ext_contrib = segment_has_external_periodic_contribution<num_dim>(
        aabb,
        periodic,
        ray
    );
    if (print_debug) {
        auto start = ray(ray.t0);
        auto end = ray(ray.t1);
        printf("<%f, %f, %f> -> <%f, %f, %f> (%d), ext: %s \n", start(0), start(1), start(2), end(0), end(1), end(2), start_clipped_axis, has_ext_contrib ? "yes" : "no");
    }
    if (!marcher && has_ext_contrib) {
        // NOTE(cmo): Wrap the segment end point into the domain and compute the
        // adjusted start point
        // In all that follows, the end point refers to the head of the ray
        // segment, i.e. the upper bound of the integral stored into the probe
        // (the near point). This is due to the back-to-front integration.
        auto end_point = ray(ray.t1);
        if (print_debug) {
            printf("Remapping from <%f, %f, %f>\n", end_point(0), end_point(1), end_point(2));
            printf("aabb [%d, %d, %d], [%d, %d, %d]\n", aabb.min(0), aabb.min(1), aabb.min(2), aabb.max(0), aabb.max(1), aabb.max(2));
        }
        for (int i = 0; i < num_dim; ++i) {
            if (periodic(i)) {
                end_point(i) = (
                    end_point(i) < FP(0.0) ? aabb.max(i) : aabb.min(i)
                ) + std::fmod(end_point(i), aabb.max(i) - aabb.min(i));
                // NOTE(cmo): In case we're right on the boundary (technically
                // there is a contribution, but the ray length for this wrap has
                // collapsed, force a wrap now).
                if (std::abs(end_point(i) - aabb.min(i)) < FP(1e-3) || std::abs(end_point(i) - aabb.max(i)) < FP(1e-3)) {
                    end_point(i) += std::copysign(FP(1.0), ray.d(i)) * (aabb.max(i) - aabb.min(i));
                }
            }
        }
        if (print_debug) {
            auto new_start = end_point - ray.d * (ray.t1 - ray.t0);
            printf("After wrap <%f, %f, %f> <- <%f, %f, %f> [%f, %f]\n", end_point(0), end_point(1), end_point(2), new_start(0), new_start(1), new_start(2), ray.t0, ray.t1);
        }
        ray = RaySegment<num_dim>(
            end_point - ray.d * (ray.t1 - ray.t0),
            ray.d,
            ray.t0,
            ray.t1
        );
        marcher = s.init(ray, args.max_mip_to_sample, &start_clipped_axis);
    if (print_debug) {
        auto start = ray(ray.t0);
        auto end = ray(ray.t1);
        printf("remapped <%f, %f, %f> -> <%f, %f, %f> (%d), valid: %s \n", start(0), start(1), start(2), end(0), end(1), end(2), start_clipped_axis, marcher ? "yes" : "no");
        start = s.ray(s.ray.t0);
        end = s.ray(s.ray.t1);
        printf("in stepper: <%f, %f, %f> -> <%f, %f, %f>\n", start(0), start(1), start(2), end(0), end(1), end(2));
        printf("periodic: %d, %d, %d\n\n\n", int(periodic(0)), int(periodic(1)), int(periodic(2)));
    }
    }

    // NOTE(cmo): If, after all that, we failed to initialise a valid trace,
    // then it's incredibly likely that there isn't one.
    if (!marcher) {
        return result;
    }

    RadianceInterval<Alo> trace_result{
        .I = FP(0.0),
        .tau = FP(0.0)
    };

    int num_wraps = 0;
    fp_t t_remaining = ray.t1 - ray.t0;
    constexpr fp_t min_seg_length = FP(0.1);
    auto check_ray_done = [&](fp_t t_traversed) {
        // NOTE(cmo): The check order is important here to avoid reading before the start of periodic's data.
        // If the ray wasn't clipped, then we should have arrived at its termination on this traversal.
        return (
            start_clipped_axis == -1 ||
            !periodic(start_clipped_axis) ||
            (t_remaining - t_traversed) < FP(0.1)
        );
    };

    fp_t lambda;
    if constexpr (dynamic && std::is_same_v<DynamicState, Raymarch3dDynamicCavState>) {
        lambda = dyn_state.adata.wavelength(la);
    }
    fp_t eta_s = FP(0.0), chi_s = FP(1e-20), one_m_edt = FP(0.0);
    while (marcher && num_wraps < MAX_PERIODIC_WRAPS && trace_result.tau < PERIODIC_TAU_CUT) {
        RadianceInterval<Alo> current_interval{
            .I = FP(0.0),
            .tau = FP(0.0)
        };
        do {
            one_m_edt = FP(0.0);
            if (s.can_sample()) {
                i64 ks = idx_gen.idx(
                    s.current_mip_level,
                    Coord3{.x = s.curr_coord(0), .y = s.curr_coord(1), .z = s.curr_coord(2)}
                );

                EmisOpac emis_opac = sample_emis_opac(SampleEmisOpac3dArgs<DynamicState> {
                    .ks = ks,
                    .la = la,
                    .lambda = lambda,
                    .mu = ray.d,
                    .mip_chain = mip_chain,
                    .dyn_state = dyn_state
                });
                eta_s = emis_opac.eta;
                chi_s = emis_opac.chi;

                if constexpr (EXTRA_SAFE_SOURCE_FN) {
                    chi_s += (std::abs(chi_s) < FP(1e-15)) * FP(1e-15);
                }
                fp_t tau = chi_s * s.dt * distance_scale;
                fp_t source_fn = eta_s / chi_s;
                fp_t edt = std::exp(-tau);
                one_m_edt = -std::expm1(-tau);
                current_interval.tau += tau;
                current_interval.I = current_interval.I * edt + source_fn * one_m_edt;
            }

        } while (s.step_through_grid());
        // NOTE(cmo): These are back-to-front traces, but the intervals connect
        // front-to-back. This is messy, but it's how it fits in the current
        // framework -- performance doesn't seem as bad as anticipated.
        trace_result.I += std::exp(-trace_result.tau) * current_interval.I;
        trace_result.tau += current_interval.tau;

        // NOTE(cmo): The ALO should get computed at the end of the first trace (since that has to end at the probe in question)
        if constexpr ((RcMode & RC_COMPUTE_ALO) && !std::is_same_v<Alo, DexEmpty>) {
            if (num_wraps == 0) {
                result.psi_star = std::max(one_m_edt / chi_s, FP(0.0));
            }
        }

        // NOTE(cmo): If our escape axis isn't periodic or we're done, then leave
        fp_t t_traversed = s.ray.t1 - s.ray.t0;
        bool escape = false;
        if (check_ray_done(t_traversed)) {
            escape = true;
        } else {
            // NOTE(cmo): Compute new ray segment. If the ray collapses to ~0
            // length because it hits a tiny corner of the grid then we cycle
            // immediately onto the segment (having checked that it also exists
            // through a periodic boundary and not a fixed one)
            do {
                num_wraps += 1;
                t_remaining -= s.ray.t1 - s.ray.t0;
                // Wrap over the periodic axis
                auto new_end = s.ray(s.ray.t0);
                new_end(start_clipped_axis) += s.step(start_clipped_axis) * (aabb.max(start_clipped_axis) - aabb.min(start_clipped_axis));
                ray = RaySegment<num_dim>(
                    new_end - ray.d * t_remaining,
                    ray.d,
                    FP(0.0),
                    t_remaining
                );
                marcher = s.init(ray, args.max_mip_to_sample, &start_clipped_axis);
                // Check if the ray will escape after this wrap, and if it's
                // also short enough to not bother with
                if (
                    (start_clipped_axis == -1 || !periodic(start_clipped_axis))
                    && (s.ray.t1 - s.ray.t0 < min_seg_length)
                ) {
                    escape = true;
                }
            } while(!escape && (s.ray.t1 - s.ray.t0) < FP(1e-2) && num_wraps < MAX_PERIODIC_WRAPS);
        }
        if (escape) {
            break;
        }
    }

    // NOTE(cmo): Merge boundary into trace result
    result.I = result.I * std::exp(-trace_result.tau) + trace_result.I;
    result.tau = trace_result.tau;

    return result;
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
YAKL_INLINE RadianceInterval<Alo> multi_level_dda_raymarch_3d(
    const Raymarch3dArgs<Bc, DynamicState>& args
) {
    if constexpr (RcMode & RC_PERIODIC) {
        return multi_level_dda_raymarch_3d_periodic<RcMode>(args);
    }

    JasUnpack(args, mr_block_map, ray, distance_scale, mip_chain, la, bc, offset, dyn_state);
    constexpr bool dynamic = (RcMode & RC_DYNAMIC);
    RadianceInterval<Alo> result;

    MRIdxGen3d idx_gen(mr_block_map);
    auto s = MultiLevelDDA<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>(idx_gen);
    int start_clipped_axis;
    const bool marcher = s.init(ray, args.max_mip_to_sample, &start_clipped_axis);
    constexpr bool always_sample_bc = (RcMode & RC_SAMPLE_BC) && LAST_CASCADE_TO_INFTY && !(RcMode & RC_LINE_SWEEP);
    const bool ray_starts_outside = (RcMode & RC_SAMPLE_BC) && (!marcher || start_clipped_axis != -1);
    if (always_sample_bc || ray_starts_outside) {
        // NOTE(cmo): Check the ray is going up along z.
        if ((ray.d(2) > FP(0.0)) && la != -1) {
            vec3 pos;
            pos(0) = ray.o(0) * distance_scale + offset(0);
            pos(1) = ray.o(1) * distance_scale + offset(1);
            pos(2) = ray.o(2) * distance_scale + offset(2);

            fp_t I_sample = sample_boundary(bc, la, pos, ray.d);
            result.I = I_sample;
        }
    }
    if (!marcher) {
        return result;
    }
    fp_t lambda;
    if constexpr (dynamic && std::is_same_v<DynamicState, Raymarch3dDynamicCavState>) {
        lambda = dyn_state.adata.wavelength(la);
    }

    fp_t eta_s = FP(0.0), chi_s = FP(1e-20), one_m_edt = FP(0.0);
    do {
        one_m_edt = FP(0.0);
        if (s.can_sample()) {
            i64 ks = idx_gen.idx(
                s.current_mip_level,
                Coord3{.x = s.curr_coord(0), .y = s.curr_coord(1), .z = s.curr_coord(2)}
            );

            EmisOpac emis_opac = sample_emis_opac(SampleEmisOpac3dArgs<DynamicState> {
                .ks = ks,
                .la = la,
                .lambda = lambda,
                .mu = ray.d,
                .mip_chain = mip_chain,
                .dyn_state = dyn_state
            });
            eta_s = emis_opac.eta;
            chi_s = emis_opac.chi;

            if constexpr (EXTRA_SAFE_SOURCE_FN) {
                chi_s += (std::abs(chi_s) < FP(1e-15)) * FP(1e-15);
            }
            fp_t tau = chi_s * s.dt * distance_scale;
            fp_t source_fn = eta_s / chi_s;
            fp_t edt = std::exp(-tau);
            one_m_edt = -std::expm1(-tau);
            result.tau += tau;
            result.I = result.I * edt + source_fn * one_m_edt;
        }

    } while (s.step_through_grid());

    if constexpr ((RcMode & RC_COMPUTE_ALO) && !std::is_same_v<Alo, DexEmpty>) {
        result.psi_star = std::max(one_m_edt / chi_s, FP(0.0));
    }
    return result;
}

#else
#endif