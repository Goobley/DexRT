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
    vec3 offset;
};

template <int RcMode=0, typename Bc>
YAKL_INLINE RadianceInterval march_and_merge_average_interval(
    const CascadeStateAndBc<Bc>& casc_state,
    const CascadeDims& dims,
    const ProbeIndex& this_probe,
    const RayProps& ray,
    const RaymarchParams& params
) {
    RadianceInterval ri = dda_raymarch_2d<RcMode, Bc>(
        Raymarch2dArgs<Bc>{
            .casc_state_bc = casc_state,
            .ray = ray,
            .distance_scale = params.distance_scale,
            .incl = params.incl,
            .incl_weight = params.incl_weight,
            .wave = this_probe.wave,
            .offset = params.offset,
        }
    );

    const int num_rays_per_ray = 1 << CASCADE_BRANCHING_FACTOR;
    const int upper_ray_start_idx = this_probe.dir * num_rays_per_ray;
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval interp{};
    if (casc_state.state.upper_I.initialized()) {
        BilinearCorner base = bilinear_corner(this_probe.coord);
        vec4 weights = bilinear_weights(base);
        // NOTE(cmo): The cascade_size function works with any cascade and a relative level offset for n.
        JasUnpack(casc_state.state, upper_I, upper_tau);
        CascadeDims upper_dims = cascade_size(dims, 1);
        for (int bilin = 0; bilin < 4; ++bilin) {
            ivec2 bilin_offset = bilinear_offset(base, dims, bilin);
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
                ri.I += ray_weight * weights(bilin) * probe_fetch(upper_I, upper_dims, upper_probe);
                ri.tau += ray_weight * weights(bilin) * probe_fetch(upper_tau, upper_dims, upper_probe);
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

    CascadeDims dims = cascade_size(state.c0_size, cascade_idx);
    dims.wave_batch = la_end - la_start;

    auto offset = get_offsets(atmos);

    parallel_for(
        "RC Loop",
        SimpleBounds<5>(
            dims.num_probes(1),
            dims.num_probes(0),
            dims.num_flat_dirs,
            dims.wave_batch,
            dims.num_incl
        ),
        YAKL_LAMBDA (int v, int u, int phi_idx, int wave, int theta_idx) {
            ivec2 probe_coord;
            probe_coord(0) = u;
            probe_coord(1) = v;

            ProbeIndex probe_idx{
                .coord=probe_coord,
                .dir=phi_idx,
                .incl=theta_idx,
                .wave=wave
            };

            RayProps ray = ray_props(dims, dev_casc_state.num_cascades, cascade_idx, probe_idx);
            RaymarchParams params {
                .distance_scale = atmos.voxel_scale,
                .incl = incl_quad.muy(theta_idx),
                .incl_weight = incl_quad.wmuy(theta_idx),
                .offset = offset
            };

            RadianceInterval ri;
            BoundaryType boundary = state.boundary;
            if constexpr (RcMode && RC_SAMPLE_BC) {
                switch (boundary) {
                    case BoundaryType::Zero: {
                        auto casc_and_bc = get_bc<ZeroBc>(dev_casc_state, state);
                        ri = march_and_merge_average_interval<RcMode>(
                            casc_and_bc,
                            dims,
                            probe_idx,
                            ray,
                            params
                        );
                    } break;
                    case BoundaryType::Promweaver: {
                        auto casc_and_bc = get_bc<PwBc<>>(dev_casc_state, state);
                        ri = march_and_merge_average_interval<RcMode>(
                            casc_and_bc,
                            dims,
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
                    dims,
                    probe_idx,
                    ray,
                    params
                );
            }

            i64 lin_idx = probe_linear_index(dims, probe_idx);

            i_cascade_i(lin_idx) = ri.I;
            tau_cascade_i(lin_idx) = ri.tau;
        }
    );
}

#else
#endif