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
    const CascadeRays& rays,
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

    constexpr int num_rays_per_ray = upper_texels_per_ray<RcMode>();
    // const int upper_ray_start_idx = this_probe.dir * num_rays_per_ray;
    const int upper_ray_start_idx = this_probe.dir * (1 << CASCADE_BRANCHING_FACTOR);
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval<Alo> interp{};
    if (casc_state.state.upper_I.initialized()) {
        BilinearCorner base = bilinear_corner(this_probe.coord);
        vec4 weights = bilinear_weights(base);
        JasUnpack(casc_state.state, upper_I, upper_tau);
        // NOTE(cmo): The cascade_compute_size function works with any cascade and a relative level offset for n.
        CascadeRays upper_rays = cascade_compute_size(rays, 1);
        for (int bilin = 0; bilin < 4; ++bilin) {
            ivec2 bilin_offset = bilinear_offset(base, upper_rays.num_probes, bilin);
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
                interp.I += ray_weight * weights(bilin) * probe_fetch<RcMode>(upper_I, upper_rays, upper_probe);
                // interp.tau += ray_weight * weights(bilin) * std::exp(-probe_fetch<RcMode>(upper_tau, upper_rays, upper_probe));
                interp.tau += ray_weight * weights(bilin) * probe_fetch<RcMode>(upper_tau, upper_rays, upper_probe);
            }
        }
        // interp.tau = -std::log(interp.tau);
    }
    return merge_intervals(ri, interp);
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>>
YAKL_INLINE RadianceInterval<Alo> march_and_merge_bilinear_fix(
    const CascadeStateAndBc<Bc>& casc_state,
    const CascadeRays& rays,
    const ProbeIndex& this_probe,
    RayProps ray,
    const RaymarchParams<DynamicState>& params
) {
    ray = invert_direction(ray);

    constexpr int num_rays_per_ray = upper_texels_per_ray<RcMode>();
    // const int upper_ray_start_idx = this_probe.dir * num_rays_per_ray;
    const int upper_ray_start_idx = this_probe.dir * (1 << CASCADE_BRANCHING_FACTOR);
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval<Alo> interp{};
    if (casc_state.state.upper_I.initialized()) {
        BilinearCorner base = bilinear_corner(this_probe.coord);
        vec4 weights = bilinear_weights(base);
        JasUnpack(casc_state.state, upper_I, upper_tau);
        // NOTE(cmo): The cascade_compute_size function works with any cascade and a relative level offset for n.
        CascadeRays upper_rays = cascade_compute_size(rays, 1);
        for (int bilin = 0; bilin < 4; ++bilin) {
            ivec2 bilin_offset = bilinear_offset(base, upper_rays.num_probes, bilin);
            ProbeIndex upper_centre_probe{
                .coord = base.corner + bilin_offset,
                .dir = upper_ray_start_idx + num_rays_per_ray / 2,
                .incl = this_probe.incl,
                .wave = this_probe.wave
            };
            RayProps central_ray = ray_props(
                upper_rays,
                casc_state.state.num_cascades,
                casc_state.state.n+1,
                upper_centre_probe
            );
            ray.start = central_ray.start;
            ray.dir = ray.end - ray.start;
            const fp_t dir_len = std::sqrt(square(ray.dir(0)) + square(ray.dir(1)));
            ray.dir(0) /= dir_len;
            ray.dir(1) /= dir_len;
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
            RadianceInterval<Alo> upper_interp{};
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

                upper_interp.I += ray_weight * probe_fetch<RcMode>(upper_I, upper_rays, upper_probe);
                upper_interp.tau += ray_weight * probe_fetch<RcMode>(upper_tau, upper_rays, upper_probe);
            }
            RadianceInterval<Alo> merged = merge_intervals(ri, upper_interp);
            interp.I += weights(bilin) * merged.I;
            interp.tau += weights(bilin) * merged.tau;
        }
    } else {
        interp = dda_raymarch_2d<RcMode, Bc>(
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
    }
    return interp;
}

template <
    int RcMode=0
>
YAKL_INLINE RadianceInterval<DexEmpty> interp_probe_dir(
    const DeviceCascadeState& casc_state,
    const CascadeRays& rays,
    const ProbeIndex& probe_idx,
    fp_t dir_frac_idx,
    bool upper
) {
    ProbeIndex probe_0(probe_idx);
    const int base_idx = int(std::floor(dir_frac_idx));
    probe_0.dir = (base_idx + rays.num_flat_dirs) % rays.num_flat_dirs;
    i64 idx_0 = probe_linear_index<RcMode>(rays, probe_0);

    ProbeIndex probe_1(probe_idx);
    // TODO(cmo): How does this interact with dir_by_dir? Clamp?
    probe_1.dir = (base_idx + 1 + rays.num_flat_dirs) % rays.num_flat_dirs;
    i64 idx_1 = probe_linear_index<RcMode>(rays, probe_1);
    fp_t w1 = dir_frac_idx - base_idx;
    fp_t w0 = FP(1.0) - w1;

    RadianceInterval<DexEmpty> ri{};
    if (upper) {
        ri.I += w0 * casc_state.upper_I(idx_0);
        ri.tau += w0 * std::exp(-casc_state.upper_tau(idx_0));
        ri.I += w1 * casc_state.upper_I(idx_1);
        ri.tau += w1 * std::exp(-casc_state.upper_tau(idx_1));
    } else {
        ri.I += w0 * casc_state.cascade_I(idx_0);
        ri.tau += w0 * std::exp(-casc_state.cascade_tau(idx_0));
        ri.I += w1 * casc_state.cascade_I(idx_1);
        ri.tau += w1 * std::exp(-casc_state.cascade_tau(idx_1));
    }
    ri.tau = -std::log(ri.tau);
    ri.tau = std::min(ri.tau, FP(1e4));

    // NOTE(cmo): Can ignore the alo, as it's never merged (comes from C0 only).
    return ri;
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>>
YAKL_INLINE RadianceInterval<Alo> march_and_merge_parallax_fix(
    const CascadeStateAndBc<Bc>& casc_state,
    const CascadeRays& rays,
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

    constexpr int num_rays_per_ray = upper_texels_per_ray<RcMode>();
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval<Alo> interp{};
    if (casc_state.state.upper_I.initialized()) {
        BilinearCorner base = bilinear_corner(this_probe.coord);
        vec4 weights = bilinear_weights(base);
        JasUnpack(casc_state.state, upper_I, upper_tau);
        // NOTE(cmo): The cascade_compute_size function works with any cascade and a relative level offset for n.
        ProbeIndex prev_probe(this_probe);
        prev_probe.dir -= 1;
        if (prev_probe.dir < 0) {
            prev_probe.dir = rays.num_flat_dirs - 1;
        }
        RayProps prev_ray = ray_props(
            rays,
            casc_state.state.num_cascades,
            casc_state.state.n,
            prev_probe
        );

        ProbeIndex next_probe(this_probe);
        next_probe.dir += 1;
        if (next_probe.dir >= rays.num_flat_dirs) {
            next_probe.dir = 0;
        }
        RayProps next_ray = ray_props(
            rays,
            casc_state.state.num_cascades,
            casc_state.state.n,
            next_probe
        );
        // NOTE(cmo): These new rays haven't been inverted, so we use .end...
        vec2 cone_start_pos = ray.start + FP(0.5) * (prev_ray.end - ray.start);
        vec2 cone_end_pos = ray.start + FP(0.5) * (next_ray.end - ray.start);

        CascadeRays upper_rays = cascade_compute_size(rays, 1);
        for (int bilin = 0; bilin < 4; ++bilin) {
            ivec2 bilin_offset = bilinear_offset(base, upper_rays.num_probes, bilin);
            ProbeIndex upper_probe{
                .coord = base.corner + bilin_offset,
                .dir = 0,
                .incl = this_probe.incl,
                .wave = this_probe.wave
            };
            vec2 upper_probe_pos = probe_pos(upper_probe.coord, casc_state.state.n+1);

            for (
                int upper_ray_idx = 0;
                upper_ray_idx < num_rays_per_ray;
                ++upper_ray_idx
            ) {
                vec2 upper_ray_start = (upper_ray_idx + FP(0.5)) * (cone_end_pos - cone_start_pos) * ray_weight + cone_start_pos;
                vec2 parallax_dir = upper_ray_start - upper_probe_pos;
                fp_t frac_idx = probe_frac_dir_idx(upper_rays, parallax_dir);

                RadianceInterval<DexEmpty> upper_ri = interp_probe_dir(
                    casc_state.state,
                    upper_rays,
                    upper_probe,
                    frac_idx,
                    true
                );
                interp.I += ray_weight * weights(bilin) * upper_ri.I;
                interp.tau += ray_weight * weights(bilin) * upper_ri.tau;
            }
        }
    }
    return merge_intervals(ri, interp);
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>>
YAKL_INLINE RadianceInterval<Alo> march_parallax_fix_inner(
    const CascadeStateAndBc<Bc>& casc_state,
    const CascadeRays& rays,
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

    // NOTE(cmo): Can't merge in-place with this fix.
    return ri;
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>>
YAKL_INLINE RadianceInterval<Alo> march_and_merge_dispatch(
    const CascadeStateAndBc<Bc>& casc_state,
    const CascadeRays& rays,
    const ProbeIndex& this_probe,
    RayProps ray,
    const RaymarchParams<DynamicState>& params
) {
    if constexpr (RC_CONFIG == RcConfiguration::Vanilla) {
        return march_and_merge_average_interval<RcMode>(
            casc_state,
            rays,
            this_probe,
            ray,
            params
        );
    } else if constexpr (RC_CONFIG == RcConfiguration::BilinearFix) {
        return march_and_merge_bilinear_fix<RcMode>(
            casc_state,
            rays,
            this_probe,
            ray,
            params
        );
    } else if constexpr (RC_CONFIG == RcConfiguration::ParallaxFix) {
        if (casc_state.state.n <= PARALLAX_MERGE_ABOVE_CASCADE) {
            return march_and_merge_average_interval<RcMode>(
                casc_state,
                rays,
                this_probe,
                ray,
                params
            );
        }
        return march_and_merge_parallax_fix<RcMode>(
            casc_state,
            rays,
            this_probe,
            ray,
            params
        );
    } else if constexpr (RC_CONFIG == RcConfiguration::ParallaxFixInner) {
        if (casc_state.state.n <= INNER_PARALLAX_MERGE_ABOVE_CASCADE) {
            return march_and_merge_average_interval<RcMode>(
                casc_state,
                rays,
                this_probe,
                ray,
                params
            );
        }
        return march_parallax_fix_inner<RcMode>(
            casc_state,
            rays,
            this_probe,
            ray,
            params
        );
    } else {
        []<bool f=false> () {
            static_assert(f, "Unknown RcConfiguration in dispatch function");
        }();
    }
}

template <
    int RcMode=0
>
inline void parallax_fix_inner_merge(
    const State& state,
    const DeviceCascadeState& dev_casc_state,
    const yakl::Array<i32, 2, yakl::memDevice>& probe_space_lookup,
    const CascadeRays& rays,
    const CascadeCalcSubset& subset
) {
    JasUnpack(subset, la_start, la_end, subset_idx);
    const int cascade_idx = dev_casc_state.n;
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(rays, subset_idx);
    assert(ray_subset.start_probes(0) == 0 && ray_subset.start_probes(1) == 0 && "Subset splitting probes is not handled");
    i64 spatial_bounds = ray_subset.num_probes(1) * ray_subset.num_probes(0);
    constexpr int num_rays_per_texel = rays_per_stored_texel<RcMode>();
    int wave_batch = la_end - la_start;

    CascadeStorage dims = cascade_size(state.c0_size, cascade_idx);
    const bool sparse_calc = state.config.sparse_calculation;
    if (sparse_calc) {
        spatial_bounds = probe_space_lookup.extent(0);
    }

    // TODO(cmo): Pre-allocate these somewhere.
    Fp1d I_temp = dev_casc_state.cascade_I.createDeviceObject();
    Fp1d tau_temp = dev_casc_state.cascade_tau.createDeviceObject();
    parallel_for(
        "RC Separate Merge Loop",
        SimpleBounds<4>(
            spatial_bounds,
            ray_subset.num_flat_dirs / num_rays_per_texel,
            wave_batch,
            ray_subset.num_incl
        ),
        YAKL_LAMBDA (i64 k, /* int v, int u, */ int phi_idx, int wave, int theta_idx) {
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
            phi_idx += ray_subset.start_flat_dirs;
            theta_idx += ray_subset.start_incl;

            constexpr bool debug = false;
            constexpr int uu = 123;
            constexpr int vv = 115;
            constexpr int pphi = 0;
            constexpr int casc_idx = 2;
            const bool print_debug = debug && (uu == u && vv == v && pphi == phi_idx && casc_idx == cascade_idx && wave == 0);

            RadianceInterval<DexEmpty> average_ri{};
            const fp_t sample_weight = FP(1.0) / fp_t(num_rays_per_texel);
            for (int i = 0; i < num_rays_per_texel; ++i) {
                ProbeIndex probe_idx{
                    .coord=probe_coord,
                    // NOTE(cmo): Handles preaveraging case
                    .dir=phi_idx * num_rays_per_texel + i,
                    .incl=theta_idx,
                    .wave=wave
                };
                RayProps this_ray = ray_props(rays, dev_casc_state.num_cascades, dev_casc_state.n, probe_idx);
                const i64 lin_idx = probe_linear_index<RcMode>(rays, probe_idx);
                const fp_t base_I = dev_casc_state.cascade_I(lin_idx);

                constexpr int num_rays_per_ray = upper_texels_per_ray<RcMode>();
                const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

                RadianceInterval<DexEmpty> ri{};
                BilinearCorner base = bilinear_corner(probe_idx.coord);
                vec4 weights = bilinear_weights(base);
                JasUnpack(dev_casc_state, upper_I, upper_tau);
                // NOTE(cmo): The cascade_compute_size function works with any cascade and a relative level offset for n.
                CascadeRays upper_rays = cascade_compute_size(rays, 1);
                const int upper_ray_start_idx = probe_idx.dir * (1 << CASCADE_BRANCHING_FACTOR);
                const int upper_ray_mid_idx = upper_ray_start_idx + num_rays_per_ray / 2;

                for (int bilin = 0; bilin < 4; ++bilin) {
                    ivec2 bilin_offset = bilinear_offset(base, upper_rays.num_probes, bilin);
                    ProbeIndex upper_probe{
                        .coord = base.corner + bilin_offset,
                        .dir = upper_ray_mid_idx,
                        .incl = probe_idx.incl,
                        .wave = probe_idx.wave
                    };
                    vec2 this_probe_pos = probe_pos(probe_idx.coord, dev_casc_state.n);
                    RayProps upper_central_ray = ray_props(
                        upper_rays,
                        dev_casc_state.num_cascades,
                        dev_casc_state.n+1,
                        upper_probe
                    );
                    vec2 lower_parallax_dir = upper_central_ray.start - this_probe_pos;
                    // NOTE(cmo): Adds ringing.
                    // vec2 lower_parallax_dir = upper_central_ray.start - this_ray.start;
                    fp_t frac_idx = probe_frac_dir_idx(rays, lower_parallax_dir);
                    RadianceInterval<DexEmpty> lower_ri_sample = interp_probe_dir<RcMode>(
                        dev_casc_state,
                        rays,
                        probe_idx,
                        frac_idx,
                        false
                    );

                    RadianceInterval<DexEmpty> interp{};
                    for (
                        int upper_ray_idx = upper_ray_start_idx;
                        upper_ray_idx < upper_ray_start_idx + num_rays_per_ray;
                        ++upper_ray_idx
                    ) {
                        ProbeIndex upper_probe{
                            .coord = base.corner + bilin_offset,
                            .dir = upper_ray_idx,
                            .incl = probe_idx.incl,
                            .wave = probe_idx.wave
                        };
                        if (print_debug) {
                            printf("-- Interp I: %f, Interp tau: %f, frac_idx: %f\n", lower_ri_sample.I,  lower_ri_sample.tau, frac_idx);
                        }
                        RadianceInterval<DexEmpty> upper_sample{};
                        upper_sample.I = probe_fetch<RcMode>(upper_I, upper_rays, upper_probe);
                        upper_sample.tau = probe_fetch<RcMode>(upper_tau, upper_rays, upper_probe);
                        interp.I += ray_weight * upper_sample.I;
                        interp.tau += ray_weight * upper_sample.tau;
                        // RadianceInterval<DexEmpty> merged = merge_intervals(lower_ri_sample, upper_sample);
                        // interp.I += ray_weight * merged.I;
                        // interp.tau += ray_weight * merged.tau;
                        // if (print_debug) {
                        //     printf("--- Upper I: %f, Merged I: %f\n", upper_sample.I, merged.I);
                        // }
                    }
                    RadianceInterval<DexEmpty> merged = merge_intervals(lower_ri_sample, interp);
                    ri.I += weights(bilin) * merged.I;
                    ri.tau += weights(bilin) * merged.tau;
                    // ri.I += weights(bilin) * interp.I;
                    // ri.tau += weights(bilin) * interp.tau;
                    if (print_debug) {
                        printf("---- Final I: %f\n", ri.I);
                    }
                }

                average_ri.I += sample_weight * ri.I;
                average_ri.tau += sample_weight * ri.tau;
            }
            ProbeIndex probe_idx{
                .coord=probe_coord,
                // NOTE(cmo): Access the "first" entry stored in a texel, if we
                // have more than one ray per texel
                .dir=phi_idx * num_rays_per_texel,
                .incl=theta_idx,
                .wave=wave
            };
            i64 lin_idx = probe_linear_index<RcMode>(dims, probe_idx);
            I_temp(lin_idx) = average_ri.I;
            tau_temp(lin_idx) = average_ri.tau;
        }
    );
    yakl::fence();

    parallel_for(
        "RC Post-Merge Copy",
        SimpleBounds<4>(
            spatial_bounds,
            ray_subset.num_flat_dirs / num_rays_per_texel,
            wave_batch,
            ray_subset.num_incl
        ),
        YAKL_LAMBDA (i64 k, /* int v, int u, */ int phi_idx, int wave, int theta_idx) {
            int u, v;
            if (sparse_calc) {
                u = probe_space_lookup(k, 0);
                v = probe_space_lookup(k, 1);
            } else {
                u = k % dims.num_probes(0);
                v = k / dims.num_probes(0);
            }
            ivec2 probe_coord;
            probe_coord(0) = u;
            probe_coord(1) = v;
            phi_idx += ray_subset.start_flat_dirs;
            theta_idx += ray_subset.start_incl;
            ProbeIndex probe_idx{
                .coord=probe_coord,
                // NOTE(cmo): Access the "first" entry stored in a texel, if we
                // have more than one ray per texel
                .dir=phi_idx * num_rays_per_texel,
                .incl=theta_idx,
                .wave=wave
            };
            i64 lin_idx = probe_linear_index<RcMode>(dims, probe_idx);
            dev_casc_state.cascade_I(lin_idx) = I_temp(lin_idx);
            dev_casc_state.cascade_tau(lin_idx) = tau_temp(lin_idx);
        }
    );
    yakl::fence();
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
    const yakl::Array<bool, 3, yakl::memDevice>& dynamic_opac,
    const yakl::Array<i64, 2, yakl::memDevice>& active_map,
    const DirectionalEmisOpacInterp& dir_interp
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
    const yakl::Array<bool, 3, yakl::memDevice>& dynamic_opac,
    const yakl::Array<i64, 2, yakl::memDevice>& active_map,
    const DirectionalEmisOpacInterp& dir_interp
) {
    const fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
    vec3 mu;
    mu(0) = -ray.dir(0) * sin_theta;
    mu(1) = -incl;
    mu(2) = -ray.dir(1) * sin_theta;
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

template <>
YAKL_INLINE
Raymarch2dDynamicInterpState get_dyn_state(
    int la,
    const RayProps& ray,
    const fp_t incl,
    const Atmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const yakl::Array<bool, 3, yakl::memDevice>& dynamic_opac,
    const yakl::Array<i64, 2, yakl::memDevice>& active_map,
    const DirectionalEmisOpacInterp& dir_interp
)
{
    const fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
    vec3 mu;
    mu(0) = -ray.dir(0) * sin_theta;
    mu(1) = -incl;
    mu(2) = -ray.dir(1) * sin_theta;

    return Raymarch2dDynamicInterpState{
        .mu = mu,
        .vx = atmos.vx,
        .vy = atmos.vy,
        .vz = atmos.vz,
        .dynamic_opac = dynamic_opac,
        .active_map = active_map,
        .dir_interp = dir_interp
    };
}

template <int RcMode=0>
void cascade_i_25d(
    const State& state,
    const CascadeState& casc_state,
    int cascade_idx,
    const CascadeCalcSubset& subset,
    const DirectionalEmisOpacInterp& dir_interp = DirectionalEmisOpacInterp()
) {
    JasUnpack(state, atmos, incl_quad, adata, pops, dynamic_opac, active, active_map);
    JasUnpack(subset, la_start, la_end, subset_idx);
    const auto& profile = state.phi;
    constexpr bool compute_alo = RcMode & RC_COMPUTE_ALO;
    using AloType = std::conditional_t<compute_alo, fp_t, DexEmpty>;
    constexpr bool dynamic = RcMode & RC_DYNAMIC;
    constexpr bool dynamic_interp = (RcMode & RC_DYNAMIC) && (RcMode & RC_DYNAMIC_INTERP);
    using DynamicState = std::conditional_t<
        dynamic_interp,
        Raymarch2dDynamicInterpState,
        std::conditional_t<dynamic, Raymarch2dDynamicState, DexEmpty>
    >;
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
    constexpr int num_subsets = subset_tasks_per_cascade<RcMode>();
    assert(subset_idx < num_subsets);

    constexpr int num_rays_per_texel = rays_per_stored_texel<RcMode>();
    int wave_batch = la_end - la_start;

    DeviceBoundaries boundaries_h{
        .boundary = state.boundary,
        .zero_bc = state.zero_bc,
        .pw_bc = state.pw_bc
    };
    Fp2d flat_pops;
    if (pops.initialized()) {
        flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    }
    auto offset = get_offsets(atmos);

    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    assert(ray_subset.start_probes(0) == 0 && ray_subset.start_probes(1) == 0 && "Subset splitting probes is not handled");

    i64 spatial_bounds = ray_subset.num_probes(1) * ray_subset.num_probes(0);
    wave_batch = std::min(wave_batch, ray_subset.wave_batch);
    yakl::Array<i32, 2, yakl::memDevice> probe_space_lookup;
    if (sparse_calc) {
        probe_space_lookup = casc_state.probes_to_compute[cascade_idx];
        spatial_bounds = probe_space_lookup.extent(0);
    }

    std::string name = fmt::format("Cascade {}", cascade_idx);
    yakl::timer_start(name.c_str());
    parallel_for(
        "RC Loop",
        SimpleBounds<4>(
            spatial_bounds,
            // dims.num_probes(1),
            // dims.num_probes(0),
            ray_subset.num_flat_dirs / num_rays_per_texel,
            wave_batch,
            ray_subset.num_incl
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
            phi_idx += ray_subset.start_flat_dirs;
            int la = la_start + wave + ray_subset.start_wave_batch;
            theta_idx += ray_subset.start_incl;

            RadianceInterval<AloType> average_ri{};
            const fp_t sample_weight = FP(1.0) / fp_t(num_rays_per_texel);
            for (int i = 0; i < num_rays_per_texel; ++i) {
                ProbeIndex probe_idx{
                    .coord=probe_coord,
                    // NOTE(cmo): Handles preaveraging case
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
                    dynamic_opac,
                    active_map,
                    dir_interp
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
                auto& casc_rays = ray_set;
#if defined(YAKL_ARCH_CUDA) || defined(YAKL_ARCH_HIP) || defined(YAKL_ARCH_SYCL)
                const auto boundaries = boundaries_h;
#else
                const auto& boundaries = boundaries_h;
#endif
                if constexpr (RcMode & RC_SAMPLE_BC) {
                    switch (boundaries.boundary) {
                        case BoundaryType::Zero: {
                            auto casc_and_bc = get_bc<ZeroBc>(dev_casc_state, boundaries);
                            ri = march_and_merge_dispatch<RcMode>(
                                casc_and_bc,
                                casc_rays,
                                probe_idx,
                                ray,
                                params
                            );
                        } break;
                        case BoundaryType::Promweaver: {
                            auto casc_and_bc = get_bc<PwBc<>>(dev_casc_state, boundaries);
                            ri = march_and_merge_dispatch<RcMode>(
                                casc_and_bc,
                                casc_rays,
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
                    ri = march_and_merge_dispatch<RcMode>(
                        casc_and_bc,
                        casc_rays,
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

            ProbeIndex probe_idx{
                .coord=probe_coord,
                // NOTE(cmo): Access the "first" entry stored in a texel, if we
                // have more than one ray per texel
                .dir=phi_idx * num_rays_per_texel,
                .incl=theta_idx,
                .wave=wave
            };
            i64 lin_idx = probe_linear_index<RcMode>(dims, probe_idx);
            dev_casc_state.cascade_I(lin_idx) = average_ri.I;
            dev_casc_state.cascade_tau(lin_idx) = average_ri.tau;
            if constexpr (dev_compute_alo) {
                // dev_casc_state.alo(v, u, phi_idx, wave, theta_idx) = average_ri.alo;
                dev_casc_state.alo(lin_idx) = average_ri.alo;
            }
        }
    );
    yakl::fence();
    if constexpr (RC_CONFIG == RcConfiguration::ParallaxFixInner) {
        if (cascade_idx > INNER_PARALLAX_MERGE_ABOVE_CASCADE && dev_casc_state.upper_I.initialized()) {
            parallax_fix_inner_merge<RcMode>(state, dev_casc_state, probe_space_lookup, ray_set, subset);
        }
    }

    yakl::timer_stop(name.c_str());
}

#else
#endif