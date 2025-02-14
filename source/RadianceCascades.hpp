#if !defined(DEXRT_RADIANCE_CASCADES_2_HPP)
#define DEXRT_RADIANCE_CASCADES_2_HPP

#include "RcUtilsModes.hpp"
#include "BoundaryDispatch.hpp"
#include "RayMarching.hpp"
#include "Atmosphere.hpp"
#include "LineSweeping.hpp"

template <typename DynamicState>
struct RaymarchParams {
    fp_t distance_scale; // [m]
    vec3 mu; // mu from ray traversal perspective
    fp_t incl; // cos(theta) - polar
    fp_t incl_weight;
    int la;
    vec3 offset; // (0,0,0) corner offset from (0,0,0) in m
    int max_mip_to_sample;
    const BlockMap<BLOCK_SIZE>& block_map;
    const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE>& mr_block_map;
    const MultiResMipChain& mip_chain;
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
    RadianceInterval<Alo> ri;
    ri = multi_level_dda_raymarch_2d<RcMode, Bc>(
        Raymarch2dArgs<Bc, DynamicState>{
            .casc_state_bc = casc_state,
            .ray = ray,
            .distance_scale = params.distance_scale,
            .mu = params.mu,
            .incl = params.incl,
            .incl_weight = params.incl_weight,
            .wave = this_probe.wave,
            .la = params.la,
            .offset = params.offset,
            .max_mip_to_sample = params.max_mip_to_sample,
            .block_map = params.block_map,
            .mr_block_map = params.mr_block_map,
            .mip_chain = params.mip_chain,
            .dyn_state = params.dyn_state
        }
    );

    const int num_rays_per_ray = upper_texels_per_ray<RcMode>(casc_state.state.n);
    const int upper_ray_start_idx = upper_ray_idx(this_probe.dir, casc_state.state.n);
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval<Alo> interp{};
    if (casc_state.state.upper_I.initialized()) {
        BilinearCorner base = bilinear_corner(this_probe.coord);
        vec4 weights = bilinear_weights(base);
        JasUnpack(casc_state.state, upper_I, upper_tau, upper_dims);
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
                i64 lin_idx = probe_linear_index<RcMode>(upper_dims, upper_probe);
                interp.I += ray_weight * weights(bilin) * upper_I(lin_idx);
                if constexpr (STORE_TAU_CASCADES) {
                    interp.tau += ray_weight * weights(bilin) * upper_tau(lin_idx);
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
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>>
YAKL_INLINE RadianceInterval<Alo> march_and_merge_bilinear_fix(
    const CascadeStateAndBc<Bc>& casc_state,
    const CascadeRays& rays,
    const ProbeIndex& this_probe,
    RayProps ray,
    const RaymarchParams<DynamicState>& params
) {
    ray = invert_direction(ray);

    const int num_rays_per_ray = upper_texels_per_ray<RcMode>(casc_state.state.n);
    const int upper_ray_start_idx = upper_ray_idx(this_probe.dir, casc_state.state.n);
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval<Alo> interp{};
    if (casc_state.state.upper_I.initialized()) {
        BilinearCorner base = bilinear_corner(this_probe.coord);
        vec4 weights = bilinear_weights(base);
        JasUnpack(casc_state.state, upper_I, upper_tau);
        CascadeRays upper_rays = cascade_storage_to_rays<RcMode>(casc_state.state.upper_dims);
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
            RadianceInterval<Alo> ri = multi_level_dda_raymarch_2d<RcMode, Bc>(
                Raymarch2dArgs<Bc, DynamicState>{
                    .casc_state_bc = casc_state,
                    .ray = ray,
                    .distance_scale = params.distance_scale,
                    .mu = params.mu,
                    .incl = params.incl,
                    .incl_weight = params.incl_weight,
                    .wave = this_probe.wave,
                    .la = params.la,
                    .offset = params.offset,
                    .max_mip_to_sample = params.max_mip_to_sample,
                    .block_map = params.block_map,
                    .mr_block_map = params.mr_block_map,
                    .mip_chain = params.mip_chain,
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

                i64 lin_idx = probe_linear_index<RcMode>(casc_state.state.upper_dims, upper_probe);
                upper_interp.I += ray_weight * upper_I(lin_idx);
                if constexpr (STORE_TAU_CASCADES) {
                    upper_interp.tau += ray_weight * upper_tau(lin_idx);
                }
            }
            RadianceInterval<Alo> merged = merge_intervals(ri, upper_interp);
            interp.I += weights(bilin) * merged.I;
            interp.tau += weights(bilin) * merged.tau;
        }
    } else {
        interp = multi_level_dda_raymarch_2d<RcMode, Bc>(
            Raymarch2dArgs<Bc, DynamicState>{
                .casc_state_bc = casc_state,
                .ray = ray,
                .distance_scale = params.distance_scale,
                .mu = params.mu,
                .incl = params.incl,
                .incl_weight = params.incl_weight,
                .wave = this_probe.wave,
                .la = params.la,
                .offset = params.offset,
                .max_mip_to_sample = params.max_mip_to_sample,
                .block_map = params.block_map,
                .mr_block_map = params.mr_block_map,
                .mip_chain = params.mip_chain,
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
    // TODO(cmo): How does this interact with dir_by_dir? Clamp? -- we just don't allow it currently
    probe_1.dir = (base_idx + 1 + rays.num_flat_dirs) % rays.num_flat_dirs;
    i64 idx_1 = probe_linear_index<RcMode>(rays, probe_1);
    fp_t w1 = dir_frac_idx - base_idx;
    fp_t w0 = FP(1.0) - w1;

    RadianceInterval<DexEmpty> ri{};
    if (upper) {
        ri.I += w0 * casc_state.upper_I(idx_0);
        ri.I += w1 * casc_state.upper_I(idx_1);
        if constexpr (STORE_TAU_CASCADES) {
            ri.tau += w0 * std::exp(-casc_state.upper_tau(idx_0));
            ri.tau += w1 * std::exp(-casc_state.upper_tau(idx_1));
        }
    } else {
        ri.I += w0 * casc_state.cascade_I(idx_0);
        ri.I += w1 * casc_state.cascade_I(idx_1);
        if constexpr (STORE_TAU_CASCADES) {
            ri.tau += w0 * std::exp(-casc_state.cascade_tau(idx_0));
            ri.tau += w1 * std::exp(-casc_state.cascade_tau(idx_1));
        }
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
    RadianceInterval<Alo> ri = multi_level_dda_raymarch_2d<RcMode, Bc>(
        Raymarch2dArgs<Bc, DynamicState>{
            .casc_state_bc = casc_state,
            .ray = ray,
            .distance_scale = params.distance_scale,
            .mu = params.mu,
            .incl = params.incl,
            .incl_weight = params.incl_weight,
            .wave = this_probe.wave,
            .la = params.la,
            .offset = params.offset,
            .max_mip_to_sample = params.max_mip_to_sample,
            .block_map = params.block_map,
            .mr_block_map = params.mr_block_map,
            .mip_chain = params.mip_chain,
            .dyn_state = params.dyn_state
        }
    );

    const int num_rays_per_ray = upper_texels_per_ray<RcMode>(casc_state.state.n);
    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

    RadianceInterval<Alo> interp{};
    if (casc_state.state.upper_I.initialized()) {
        BilinearCorner base = bilinear_corner(this_probe.coord);
        vec4 weights = bilinear_weights(base);
        JasUnpack(casc_state.state, upper_I, upper_tau);
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

        CascadeRays upper_rays = cascade_storage_to_rays<RcMode>(casc_state.state.upper_dims);
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
    RadianceInterval<Alo> ri = multi_level_dda_raymarch_2d<RcMode, Bc>(
        Raymarch2dArgs<Bc, DynamicState>{
            .casc_state_bc = casc_state,
            .ray = ray,
            .distance_scale = params.distance_scale,
            .mu = params.mu,
            .incl = params.incl,
            .incl_weight = params.incl_weight,
            .wave = this_probe.wave,
            .la = params.la,
            .offset = params.offset,
            .max_mip_to_sample = params.max_mip_to_sample,
            .block_map = params.block_map,
            .mr_block_map = params.mr_block_map,
            .mip_chain = params.mip_chain,
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
    const DeviceProbesToCompute& probe_coord_lookup,
    const CascadeRays& rays,
    const CascadeCalcSubset& subset
) {
    JasUnpack(state, mr_block_map);
    JasUnpack(subset, la_start, la_end, subset_idx);
    const int cascade_idx = dev_casc_state.n;
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(rays, subset_idx);
    assert(ray_subset.start_probes(0) == 0 && ray_subset.start_probes(1) == 0 && "Subset splitting probes is not handled");
    constexpr int num_rays_per_texel = rays_per_stored_texel<RcMode>();
    int wave_batch = la_end - la_start;

    CascadeStorage dims = cascade_size(state.c0_size, cascade_idx);
    i64 spatial_bounds = probe_coord_lookup.num_active_probes();

    // TODO(cmo): Pre-allocate these somewhere.
    Fp1d I_temp = dev_casc_state.cascade_I.createDeviceObject();
    Fp1d tau_temp = dev_casc_state.cascade_tau.createDeviceObject();
    dex_parallel_for(
        "RC Separate Merge Loop",
        FlatLoop<4>(
            spatial_bounds,
            ray_subset.num_flat_dirs / num_rays_per_texel,
            wave_batch,
            ray_subset.num_incl
        ),
        YAKL_LAMBDA (i64 ks, int phi_idx, int wave, int theta_idx) {
            ivec2 probe_coord = probe_coord_lookup(ks);
            phi_idx += ray_subset.start_flat_dirs;
            theta_idx += ray_subset.start_incl;

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

                const int num_rays_per_ray = upper_texels_per_ray<RcMode>(dev_casc_state.n);
                const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);

                RadianceInterval<DexEmpty> ri{};
                BilinearCorner base = bilinear_corner(probe_idx.coord);
                vec4 weights = bilinear_weights(base);
                JasUnpack(dev_casc_state, upper_I, upper_tau);

                CascadeRays upper_rays = cascade_storage_to_rays<RcMode>(dev_casc_state.upper_dims);
                const int upper_ray_start_idx = upper_ray_idx(probe_idx.dir, dev_casc_state.n);
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
                        RadianceInterval<DexEmpty> upper_sample{};
                        upper_sample.I = probe_fetch<RcMode>(upper_I, upper_rays, upper_probe);
                        if constexpr (STORE_TAU_CASCADES) {
                            upper_sample.tau = probe_fetch<RcMode>(upper_tau, upper_rays, upper_probe);
                        }
                        interp.I += ray_weight * upper_sample.I;
                        interp.tau += ray_weight * upper_sample.tau;
                    }
                    RadianceInterval<DexEmpty> merged = merge_intervals(lower_ri_sample, interp);
                    ri.I += weights(bilin) * merged.I;
                    ri.tau += weights(bilin) * merged.tau;
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

    dex_parallel_for(
        "RC Post-Merge Copy",
        FlatLoop<4>(
            spatial_bounds,
            ray_subset.num_flat_dirs / num_rays_per_texel,
            wave_batch,
            ray_subset.num_incl
        ),
        YAKL_LAMBDA (i64 ks, int phi_idx, int wave, int theta_idx) {
            ivec2 probe_coord = probe_coord_lookup(ks);
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
            // NOTE(cmo): Constexpr capture nonsense -- needs to be "used" first
            JasUse(tau_temp);
            if constexpr (STORE_TAU_CASCADES) {
                dev_casc_state.cascade_tau(lin_idx) = tau_temp(lin_idx);
            }
        }
    );
    yakl::fence();
}

template <int RcMode=0>
void cascade_i_25d(
    const State& state,
    const CascadeState& casc_state,
    int cascade_idx,
    const CascadeCalcSubset& subset,
    const MultiResMipChain& mip_chain = MultiResMipChain()
) {
    JasUnpack(state, atmos, incl_quad, adata, pops);
    JasUnpack(subset, la_start, la_end, subset_idx);
    const auto& profile = state.phi;
    constexpr bool compute_alo = RcMode & RC_COMPUTE_ALO;
    using AloType = std::conditional_t<compute_alo, fp_t, DexEmpty>;
    typedef typename RcDynamicState<RcMode>::type DynamicState;

    CascadeIdxs lookup = cascade_indices(casc_state, cascade_idx);
    Fp1d i_cascade_i = casc_state.i_cascades[lookup.i];
    Fp1d tau_cascade_i = casc_state.tau_cascades[lookup.i];
    FpConst1d i_cascade_ip, tau_cascade_ip;
    if (lookup.ip != -1) {
        i_cascade_ip = casc_state.i_cascades[lookup.ip];
        tau_cascade_ip = casc_state.tau_cascades[lookup.ip];
    }
    CascadeStorage dims = cascade_size(state.c0_size, cascade_idx);
    CascadeStorage upper_dims = cascade_size(state.c0_size, cascade_idx+1);
    DeviceCascadeState dev_casc_state {
        .num_cascades = casc_state.num_cascades,
        .n = cascade_idx,
        .casc_dims = dims,
        .upper_dims = upper_dims,
        .cascade_I = i_cascade_i,
        .cascade_tau = tau_cascade_i,
        .upper_I = i_cascade_ip,
        .upper_tau = tau_cascade_ip
    };
    if constexpr (compute_alo) {
        dev_casc_state.alo = casc_state.alo;
    }

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
    auto offset = get_offsets(atmos);

    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    assert(ray_subset.start_probes(0) == 0 && ray_subset.start_probes(1) == 0 && "Subset splitting probes is not handled");

    wave_batch = std::min(wave_batch, ray_subset.wave_batch);
    i64 spatial_bounds = casc_state.probes_to_compute.num_active_probes(cascade_idx);
    DeviceProbesToCompute probe_coord_lookup = casc_state.probes_to_compute.bind(cascade_idx);

    JasUnpack(state, mr_block_map);
    const auto& block_map = mr_block_map.block_map;
    const int max_mip_to_sample = std::min(
        state.config.mip_config.mip_levels[cascade_idx],
        mip_chain.max_mip_factor
    );
    std::string name = fmt::format("Cascade {}", cascade_idx);
    yakl::timer_start(name.c_str());
    if (RAYMARCH_TYPE == RaymarchType::Raymarch || cascade_idx < LINE_SWEEP_START_CASCADE) {
        dex_parallel_for(
            "RC Loop",
            FlatLoop<4>(
                spatial_bounds,
                ray_subset.num_flat_dirs / num_rays_per_texel,
                wave_batch,
                ray_subset.num_incl
            ),
            YAKL_LAMBDA (i64 ks, int phi_idx, int wave, int theta_idx) {
                constexpr bool dev_compute_alo = RcMode & RC_COMPUTE_ALO;
                ivec2 probe_coord = probe_coord_lookup(ks);

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
                        atmos,
                        adata,
                        profile,
                        pops,
                        mip_chain
                    );
                    RaymarchParams<DynamicState> params {
                        .distance_scale = atmos.voxel_scale,
                        .mu = inverted_mu(ray, incl_quad.muy(theta_idx)),
                        .incl = incl_quad.muy(theta_idx),
                        .incl_weight = incl_quad.wmuy(theta_idx),
                        .la = la,
                        .offset = offset,
                        .max_mip_to_sample = max_mip_to_sample,
                        .block_map = block_map,
                        .mr_block_map = mr_block_map,
                        .mip_chain = mip_chain,
                        .dyn_state = dyn_state
                    };


                    RadianceInterval<AloType> ri;
                    auto& casc_rays = ray_set;
                    const auto& boundaries = boundaries_h;

                    auto dispatch_outer = [&]<typename BcType>(BcType bc_type){
                        auto casc_and_bc = get_bc<BcType>(dev_casc_state, boundaries);
                        ri = march_and_merge_dispatch<RcMode>(
                            casc_and_bc,
                            casc_rays,
                            probe_idx,
                            ray,
                            params
                        );
                    };

                    if constexpr (RcMode & RC_SAMPLE_BC) {
                        switch (boundaries.boundary) {
                            case BoundaryType::Zero: {
                                // NOTE(cmo): lambdas aren't templated... they're
                                // essentially structs with a template on
                                // operator(), so we have to put the "template" in
                                // as an arg
                                dispatch_outer(ZeroBc{});
                            } break;
                            case BoundaryType::Promweaver: {
                                dispatch_outer(PwBc<>{});
                            } break;
                            default: {
                                yakl::yakl_throw("Unknown BC type");
                            }
                        }
                    } else {
                        dispatch_outer(ZeroBc{});
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
                if constexpr (STORE_TAU_CASCADES) {
                    dev_casc_state.cascade_tau(lin_idx) = average_ri.tau;
                }
                if constexpr (dev_compute_alo) {
                    dev_casc_state.alo(lin_idx) = average_ri.alo;
                }
            }
        );
        yakl::fence();
        if constexpr (RC_CONFIG == RcConfiguration::ParallaxFixInner) {
            if (cascade_idx > INNER_PARALLAX_MERGE_ABOVE_CASCADE && dev_casc_state.upper_I.initialized()) {
                parallax_fix_inner_merge<RcMode>(state, dev_casc_state, probe_coord_lookup, ray_set, subset);
            }
        }
    } else if (RAYMARCH_TYPE == RaymarchType::LineSweep) {
        for (int wave = 0; wave < wave_batch; ++wave) {
            compute_line_sweep_samples<RcMode>(state, casc_state, cascade_idx, subset, wave, mip_chain);
            interpolate_line_sweep_samples_to_cascade<RcMode>(state, casc_state, cascade_idx, subset, wave);

            if (lookup.ip == -1) {
                continue;
            }

            dex_parallel_for(
                "Merge samples with upper cascade",
                FlatLoop<3>(
                    spatial_bounds,
                    ray_subset.num_flat_dirs,
                    ray_subset.num_incl
                ),
                YAKL_LAMBDA (i64 ks, int phi_idx, int theta_idx) {
                    ivec2 probe_coord = probe_coord_lookup(ks);
                    phi_idx += ray_subset.start_flat_dirs;
                    theta_idx += ray_subset.start_incl;

                    ProbeIndex this_probe{
                        .coord=probe_coord,
                        .dir=phi_idx,
                        .incl=theta_idx,
                        .wave=wave
                    };
                    const int upper_ray_start_idx = upper_ray_idx(this_probe.dir, dev_casc_state.n);
                    const int num_rays_per_ray = upper_texels_per_ray<RcMode>(dev_casc_state.n);
                    const fp_t ray_weight = FP(1.0) / fp_t(num_rays_per_ray);
                    BilinearCorner base = bilinear_corner(this_probe.coord);
                    vec4 weights = bilinear_weights(base);

                    RadianceInterval<DexEmpty> interp;
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
                            const i64 upper_lin_idx = probe_linear_index<RcMode>(upper_dims, upper_probe);
                            interp.I += ray_weight * weights(bilin) * dev_casc_state.upper_I(upper_lin_idx);
                            interp.tau += ray_weight * weights(bilin) * dev_casc_state.upper_tau(upper_lin_idx);
                        }
                    }

                    const i64 lin_idx = probe_linear_index<RcMode>(dims, this_probe);
                    RadianceInterval ri;
                    ri.I = dev_casc_state.cascade_I(lin_idx);
                    ri.tau = dev_casc_state.cascade_tau(lin_idx);

                    auto merged_ri = merge_intervals(ri, interp);

                    dev_casc_state.cascade_I(lin_idx) = merged_ri.I;
                    dev_casc_state.cascade_tau(lin_idx) = merged_ri.tau;
                }
            );
            Kokkos::fence();
        }
    }

    yakl::timer_stop(name.c_str());
}

#else
#endif
