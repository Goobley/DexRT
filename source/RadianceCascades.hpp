#if !defined(DEXRT_RADIANCE_CASCADES_2_HPP)
#define DEXRT_RADIANCE_CASCADES_2_HPP

#include "RcUtilsModes.hpp"
#include "BoundaryDispatch.hpp"
#include "RayMarching.hpp"
#include "Atmosphere.hpp"

template <typename DynamicState>
struct RaymarchParams {
    fp_t distance_scale;
    fp_t incl;
    fp_t incl_weight;
    int la;
    vec3 offset;
    const yakl::Array<bool, 2, yakl::memDevice>& active;
    const DynamicState& dyn_state;
};

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>>
YAKL_INLINE RadianceInterval<Alo> march_and_merge_average_interval(
    const CascadeStateAndBc<Bc>& casc_state,
    const CascadeStorage& dims,
    const ProbeIndex& this_probe,
    RayProps ray,
    const RaymarchParams<DynamicState>& params
) {
    ray = invert_direction(ray);
    RadianceInterval<Alo> ri = dda_raymarch_2d<RcMode, Bc>(
        Raymarch2dArgs<Bc, DynamicState>{
            .casc_state_bc = casc_state,
            .ray = ray,
            .distance_scale = params.distance_scale,
            .incl = params.incl,
            .incl_weight = params.incl_weight,
            .wave = this_probe.wave,
            .la = params.la,
            .offset = params.offset,
            .active = params.active,
            .dyn_state = params.dyn_state
        }
    );
    constexpr bool preaverage = RcMode & RC_PREAVERAGE;

    int num_rays_per_ray = 1 << CASCADE_BRANCHING_FACTOR;
    if constexpr (preaverage) {
        num_rays_per_ray = 1;
    }

    const int upper_ray_start_idx = this_probe.dir * num_rays_per_ray;
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval<Alo> interp{};
    if (casc_state.state.upper_I.initialized()) {
        BilinearCorner base = bilinear_corner(this_probe.coord);
        vec4 weights = bilinear_weights(base);
        JasUnpack(casc_state.state, upper_I, upper_tau);
        // NOTE(cmo): The cascade_size function works with any cascade and a relative level offset for n.
        CascadeStorage upper_dims = cascade_size(dims, 1);
        for (int bilin = 0; bilin < 4; ++bilin) {
            ivec2 bilin_offset = bilinear_offset(base, upper_dims.num_probes, bilin);
            for (
                int upper_ray_idx = upper_ray_start_idx;
                upper_ray_idx < upper_ray_start_idx + num_rays_per_ray;
                ++upper_ray_idx
            ) {
                ProbeIndex upper_probe{
                    .coord = base.corner + bilin_offset,
                    .dir = upper_ray_idx,
                    .incl = this_probe.incl,
                    .wave = this_probe.wave
                };
                interp.I += ray_weight * weights(bilin) * probe_fetch(upper_I, upper_dims, upper_probe);
                interp.tau += ray_weight * weights(bilin) * probe_fetch(upper_tau, upper_dims, upper_probe);
            }
        }
    }
    return merge_intervals(ri, interp);
}

template <typename DynamicState>
YAKL_INLINE
DynamicState get_dyn_state(
    int la,
    const RayProps& ray,
    const fp_t incl,
    const Atmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const yakl::Array<bool, 3, yakl::memDevice>& dynamic_opac
) {
    return DynamicState{};
}

template <>
YAKL_INLINE
Raymarch2dDynamicState get_dyn_state(
    int la,
    const RayProps& ray,
    const fp_t incl,
    const Atmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const yakl::Array<bool, 3, yakl::memDevice>& dynamic_opac
) {
    const fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
    vec3 mu;
    mu(0) = ray.dir(0) * sin_theta;
    mu(1) = incl;
    mu(2) = ray.dir(1) * sin_theta;
    return Raymarch2dDynamicState{
        .mu = mu,
        .active_set = slice_active_set(adata, la),
        .dynamic_opac = dynamic_opac,
        .atmos = atmos,
        .adata = adata,
        .profile = profile,
        .nh0 = atmos.nh0,
        .n = flat_pops
    };
}

template <int RcMode=0>
void cascade_i_25d(
    const State& state,
    const CascadeState& casc_state,
    int cascade_idx,
    int la_start = -1,
    int la_end = -1
) {
    JasUnpack(state, atmos, incl_quad, adata, pops, dynamic_opac, active);
    const auto& profile = state.phi;
    constexpr bool compute_alo = RcMode & RC_COMPUTE_ALO;
    using AloType = std::conditional_t<compute_alo, fp_t, DexEmpty>;
    constexpr bool dynamic = RcMode & RC_DYNAMIC;
    using DynamicState = std::conditional_t<dynamic, Raymarch2dDynamicState, DexEmpty>;
    const bool sparse_calc = state.config.sparse_calculation;

    CascadeIdxs lookup = cascade_indices(casc_state, cascade_idx);
    Fp1d i_cascade_i = casc_state.i_cascades[lookup.i];
    Fp1d tau_cascade_i = casc_state.tau_cascades[lookup.i];
    FpConst1d i_cascade_ip, tau_cascade_ip;
    if (lookup.ip != -1) {
        i_cascade_ip = casc_state.i_cascades[lookup.ip];
        tau_cascade_ip = casc_state.tau_cascades[lookup.ip];
    }
    DeviceCascadeState dev_casc_state {
        .num_cascades = casc_state.num_cascades,
        .n = cascade_idx,
        .cascade_I = i_cascade_i,
        .cascade_tau = tau_cascade_i,
        .upper_I = i_cascade_ip,
        .upper_tau = tau_cascade_ip,
        .eta = casc_state.eta,
        .chi = casc_state.chi,
    };
    if constexpr (compute_alo) {
        dev_casc_state.alo = state.alo;
    }

    CascadeStorage dims = cascade_size(state.c0_size, cascade_idx);
    CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, cascade_idx);
    int wave_batch = la_end - la_start;

    DeviceBoundaries boundaries_h{
        .boundary = state.boundary,
        .zero_bc = state.zero_bc,
        .pw_bc = state.pw_bc
    };
    auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));

    auto offset = get_offsets(atmos);

    i64 spatial_bounds = dims.num_probes(1) * dims.num_probes(0);
    yakl::Array<i32, 2, yakl::memDevice> probe_space_lookup;
    if (sparse_calc) {
        probe_space_lookup = casc_state.probes_to_compute[cascade_idx];
        spatial_bounds = probe_space_lookup.extent(0);
    }

    parallel_for(
        "RC Loop",
        SimpleBounds<4>(
            spatial_bounds,
            // dims.num_probes(1),
            // dims.num_probes(0),
            dims.num_flat_dirs,
            wave_batch,
            dims.num_incl
        ),
        YAKL_LAMBDA (i64 k, /* int v, int u, */ int phi_idx, int wave, int theta_idx) {
            constexpr bool dev_compute_alo = RcMode & RC_COMPUTE_ALO;
            int u, v;
            if (sparse_calc) {
                u = probe_space_lookup(k, 0);
                v = probe_space_lookup(k, 1);
            } else {
                // NOTE(cmo): As in the loop over probes we iterate as [v, u] (u
                // fast-running), but index as [u, v], i.e. dims.num_probes(0) =
                // dim(u). Typical definition of k = u * Nv + v, but here we do
                // loop index k = v * Nu + u where Nu = dims.num_probes(0). This
                // preserves our iteration ordering
                u = k % dims.num_probes(0);
                v = k / dims.num_probes(0);
            }
            ivec2 probe_coord;
            probe_coord(0) = u;
            probe_coord(1) = v;
            int la = la_start + wave;

            RadianceInterval<AloType> average_ri{};
            static_assert(false, "This isn't correct in the case of DIR_BY_DIR");
            const int num_rays_per_texel = ray_set.num_flat_dirs / dims.num_flat_dirs;
            const fp_t sample_weight = FP(1.0) / fp_t(num_rays_per_texel);
            for (int i = 0; i < num_rays_per_texel; ++i) {
                ProbeIndex probe_idx{
                    .coord=probe_coord,
                    // NOTE(cmo): Handles pre-averaging case
                    .dir=phi_idx * num_rays_per_texel + i,
                    .incl=theta_idx,
                    .wave=wave
                };
                RayProps ray = ray_props(ray_set, dev_casc_state.num_cascades, cascade_idx, probe_idx);
                DynamicState dyn_state = get_dyn_state<DynamicState>(
                    la,
                    ray,
                    incl_quad.muy(theta_idx),
                    atmos,
                    adata,
                    profile,
                    flat_pops,
                    dynamic_opac
                );
                RaymarchParams<DynamicState> params {
                    .distance_scale = atmos.voxel_scale,
                    .incl = incl_quad.muy(theta_idx),
                    .incl_weight = incl_quad.wmuy(theta_idx),
                    .la = la,
                    .offset = offset,
                    .active = active,
                    .dyn_state = dyn_state
                };

                RadianceInterval<AloType> ri;
                auto& casc_dims = dims;
#if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
                const auto boundaries = boundaries_h;
#else
                const auto& boundaries = boundaries_h;
#endif
                if constexpr (RcMode & RC_SAMPLE_BC) {
                    switch (boundaries.boundary) {
                        case BoundaryType::Zero: {
                            auto casc_and_bc = get_bc<ZeroBc>(dev_casc_state, boundaries);
                            ri = march_and_merge_average_interval<RcMode>(
                                casc_and_bc,
                                casc_dims,
                                probe_idx,
                                ray,
                                params
                            );
                        } break;
                        case BoundaryType::Promweaver: {
                            auto casc_and_bc = get_bc<PwBc<>>(dev_casc_state, boundaries);
                            ri = march_and_merge_average_interval<RcMode>(
                                casc_and_bc,
                                casc_dims,
                                probe_idx,
                                ray,
                                params
                            );
                        } break;
                        default: {
                            assert(false && "Unknown BC type");
                        }
                    }
                } else {
                    auto casc_and_bc = get_bc<ZeroBc>(dev_casc_state, boundaries);
                    ri = march_and_merge_average_interval<RcMode>(
                        casc_and_bc,
                        casc_dims,
                        probe_idx,
                        ray,
                        params
                    );
                }
                average_ri.I += sample_weight * ri.I;
                average_ri.tau += sample_weight * ri.tau;
                if constexpr (dev_compute_alo) {
                    average_ri.alo += sample_weight * ri.alo;
                }
            }

            ProbeIndex probe_storage_idx{
                .coord=probe_coord,
                .dir=phi_idx,
                .incl=theta_idx,
                .wave=wave
            };
            i64 lin_idx = probe_linear_index<RcMode>(dims, probe_storage_idx);
            dev_casc_state.cascade_I(lin_idx) = average_ri.I;
            dev_casc_state.cascade_tau(lin_idx) = average_ri.tau;
            if constexpr (dev_compute_alo) {
                dev_casc_state.alo(v, u, phi_idx, wave, theta_idx) = average_ri.alo;
            }
        }
    );
}

#else
#endif