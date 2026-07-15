#if !defined(DEXRT_RADIANCE_CASCADES_3D_HPP)
#define DEXRT_RADIANCE_CASCADES_3D_HPP
#include "RayMarching3d.hpp"

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
KOKKOS_INLINE_FUNCTION RadianceInterval<Alo> march_and_merge_average_interval_3d(
    const Raymarch3dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, this_probe, casc_state, periodic);
    RadianceInterval<Alo> ri = multi_level_dda_raymarch_3d<RcMode>(args);

    RadianceInterval<Alo> interp;
    if (casc_state.upper_I.initialized()) {
        JasUnpack(casc_state, upper_I, upper_tau, casc_dims, upper_dims);

        TexelsPerRay3d upper_tex = upper_texels_per_ray_3d<RcMode>(casc_state.n);
        int upper_polar_start_idx = this_probe.polar * upper_dims.num_polar_rays / casc_dims.num_polar_rays;
        int upper_az_start_idx = this_probe.az * upper_dims.num_az_rays / casc_dims.num_az_rays;
        const fp_t ray_weight = FP(1.0) / fp_t(upper_tex.az * upper_tex.polar);

        TrilinearCorner base = trilinear_corner(this_probe.coord, upper_dims.num_probes, periodic);
        vec<8> weights = trilinear_weights(base);
        for (int tri_idx = 0; tri_idx < 8; ++tri_idx) {
            ivec3 upper_coord = trilinear_coord(base, upper_dims.num_probes, periodic, tri_idx);
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

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
YAKL_INLINE RadianceInterval<Alo> march_and_merge_trilinear_interval_3d(
    const Raymarch3dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, this_probe, casc_state, periodic);

    RadianceInterval<Alo> interp;
    if (casc_state.upper_I.initialized()) {
        JasUnpack(casc_state, upper_I, upper_tau, casc_dims, upper_dims);
        CascadeRays3d casc_rays = cascade_storage_to_rays<RcMode>(casc_dims);
        CascadeRays3d upper_casc_rays = cascade_storage_to_rays<RcMode>(upper_dims);

        TexelsPerRay3d upper_tex = upper_texels_per_ray_3d<RcMode>(casc_state.n);
        int upper_polar_start_idx = this_probe.polar * upper_dims.num_polar_rays / casc_dims.num_polar_rays;
        int upper_az_start_idx = this_probe.az * upper_dims.num_az_rays / casc_dims.num_az_rays;
        const fp_t ray_weight = FP(1.0) / fp_t(upper_tex.az * upper_tex.polar);

        TrilinearCorner base = trilinear_corner(this_probe.coord, upper_dims.num_probes, periodic);
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
            RadianceInterval<Alo> ri = multi_level_dda_raymarch_3d<RcMode>(
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

            RadianceInterval<Alo> upper_interp{};
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
            RadianceInterval<Alo> merged = merge_intervals(ri, upper_interp);
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

#else
#endif
