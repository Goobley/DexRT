#if !defined(DEXRT_RADIANCE_CASCADES_2_HPP)
#define DEXRT_RADIANCE_CASCADES_2_HPP

#include "RcUtilsModes.hpp"
#include "BoundaryDispatch.hpp"
#include "RayMarching2.hpp"
#include "Atmosphere.hpp"

template <typename DynamicState>
struct RaymarchParams {
    fp_t distance_scale;
    fp_t incl;
    fp_t incl_weight;
    int la;
    vec3 offset;
    DynamicState dyn_state;
};

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<RcMode & RC_COMPUTE_ALO, fp_t, DexEmpty>>
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
    const CompAtom<fp_t>& atom,
    const VoigtProfile<fp_t>& profile,
    const Fp3d& n,
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
    const CompAtom<fp_t>& atom,
    const VoigtProfile<fp_t>& profile,
    const Fp3d& n,
    const yakl::Array<bool, 3, yakl::memDevice>& dynamic_opac
) {
    const fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
    vec3 mu;
    mu(0) = ray.dir(0) * sin_theta;
    mu(1) = incl;
    mu(2) = ray.dir(1) * sin_theta;
    return Raymarch2dDynamicState{
        .mu = mu,
        .active_set = slice_active_set(atom, la),
        .dynamic_opac = dynamic_opac,
        .atmos = atmos,
        .atom = atom,
        .profile = profile,
        .nh0 = atmos.nh0,
        .n = n.reshape<2>(Dims(n.extent(0), n.extent(1) * n.extent(2))),
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
    JasUnpack(state, atmos, incl_quad, atom, pops, dynamic_opac);
    const auto& profile = state.phi;
    constexpr bool compute_alo = RcMode & RC_COMPUTE_ALO;
    using AloType = std::conditional_t<compute_alo, fp_t, DexEmpty>;
    constexpr bool dynamic = RcMode & RC_DYNAMIC;
    using DynamicState = std::conditional_t<dynamic, Raymarch2dDynamicState, DexEmpty>;

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
    constexpr bool preaverage = RcMode & RC_PREAVERAGE;

    CascadeStorage dims = cascade_size(state.c0_size, cascade_idx);
    CascadeRays ray_set = cascade_compute_size<preaverage>(state.c0_size, cascade_idx);
    int wave_batch = la_end - la_start;

    auto offset = get_offsets(atmos);

    parallel_for(
        "RC Loop",
        SimpleBounds<5>(
            dims.num_probes(1),
            dims.num_probes(0),
            dims.num_flat_dirs,
            wave_batch,
            dims.num_incl
        ),
        YAKL_LAMBDA (int v, int u, int phi_idx, int wave, int theta_idx) {
            constexpr bool dev_compute_alo = RcMode & RC_COMPUTE_ALO;
            ivec2 probe_coord;
            probe_coord(0) = u;
            probe_coord(1) = v;
            int la = la_start + wave;

            RadianceInterval<AloType> average_ri{};
            const int num_rays_per_texel = ray_set.num_flat_dirs / dims.num_flat_dirs;
            const fp_t sample_weight = FP(1.0) / fp_t(num_rays_per_texel);
            for (int i = 0; i < num_rays_per_texel; ++i) {
                ProbeIndex probe_idx{
                    .coord=probe_coord,
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
                    atom,
                    profile,
                    pops,
                    dynamic_opac
                );
                RaymarchParams<DynamicState> params {
                    .distance_scale = atmos.voxel_scale,
                    .incl = incl_quad.muy(theta_idx),
                    .incl_weight = incl_quad.wmuy(theta_idx),
                    .la = la,
                    .offset = offset,
                    .dyn_state = dyn_state
                };

                RadianceInterval<AloType> ri;
                BoundaryType boundary = state.boundary;
                auto& casc_dims = dims;
                if constexpr (RcMode && RC_SAMPLE_BC) {
                    switch (boundary) {
                        case BoundaryType::Zero: {
                            auto casc_and_bc = get_bc<ZeroBc>(dev_casc_state, state);
                            ri = march_and_merge_average_interval<RcMode>(
                                casc_and_bc,
                                casc_dims,
                                probe_idx,
                                ray,
                                params
                            );
                        } break;
                        case BoundaryType::Promweaver: {
                            auto casc_and_bc = get_bc<PwBc<>>(dev_casc_state, state);
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
                    auto casc_and_bc = get_bc<ZeroBc>(dev_casc_state, state);
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
            i64 lin_idx = probe_linear_index(dims, probe_storage_idx);
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