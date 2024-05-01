#if !defined(DEXRT_RADIANCE_CASCADES_2_HPP)
#define DEXRT_RADIANCE_CASCADES_2_HPP

#include "RcUtilsModes.hpp"
#include "BoundaryDispatch.hpp"
#include "RayMarching2.hpp"
#include "Atmosphere.hpp"

struct RaymarchParams {
    fp_t distance_scale;
    fp_t incl;
    fp_t incl_weight;
    int la;
    vec3 offset;
};

template <int RcMode=0, typename Bc>
YAKL_INLINE RadianceInterval march_and_merge_average_interval(
    const CascadeStateAndBc<Bc>& casc_state,
    const CascadeStorage& dims,
    const ProbeIndex& this_probe,
    RayProps ray,
    const RaymarchParams& params
) {
    ray = invert_direction(ray);
    RadianceInterval ri = dda_raymarch_2d<RcMode, Bc>(
        Raymarch2dArgs<Bc>{
            .casc_state_bc = casc_state,
            .ray = ray,
            .distance_scale = params.distance_scale,
            .incl = params.incl,
            .incl_weight = params.incl_weight,
            .wave = this_probe.wave,
            .la = params.la,
            .offset = params.offset,
        }
    );
    constexpr bool preaverage = RcMode & RC_PREAVERAGE;

    int num_rays_per_ray = 1 << CASCADE_BRANCHING_FACTOR;
    if constexpr (preaverage) {
        num_rays_per_ray = 1;
    }

    const int upper_ray_start_idx = this_probe.dir * num_rays_per_ray;
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval interp{};
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

template <int RcMode=0>
void cascade_i_25d(
    const State& state,
    const CascadeState& casc_state,
    int cascade_idx,
    int la_start = -1,
    int la_end = -1
) {
    JasUnpack(state, atmos, incl_quad);

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
            ivec2 probe_coord;
            probe_coord(0) = u;
            probe_coord(1) = v;

            RadianceInterval average_ri{};
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
                RaymarchParams params {
                    .distance_scale = atmos.voxel_scale,
                    .incl = incl_quad.muy(theta_idx),
                    .incl_weight = incl_quad.wmuy(theta_idx),
                    .la = la_start + wave,
                    .offset = offset
                };

                RadianceInterval ri;
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
        }
    );
}

#else
#endif