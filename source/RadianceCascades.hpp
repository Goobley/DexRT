#if !defined(DEXRT_RADIANCE_CASCADES_HPP)
#define DEXRT_RADIANCE_CASCADES_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "Utils.hpp"
#include "RayMarching.hpp"
#include "RadianceIntervals.hpp"
#include <fmt/core.h>

template <bool UseMipmaps=USE_MIPMAPS, int NumWavelengths=NUM_WAVELENGTHS, int NumAz=NUM_AZ, int NumComponents=NUM_COMPONENTS>
void compute_cascade_i_2d (
    State* state,
    int cascade_idx,
    bool compute_alo = false
) {
    const auto march_state = state->raymarch_state;
    int cascade_lookup_idx = cascade_idx;
    int cascade_lookup_idx_p = cascade_idx + 1;
    if constexpr (PINGPONG_BUFFERS) {
        if (cascade_idx & 1) {
            cascade_lookup_idx = 1;
            cascade_lookup_idx_p = 0;
        } else {
            cascade_lookup_idx = 0;
            cascade_lookup_idx_p = 1;
        }
    }
    int x_dim = march_state.emission.extent(0) / (1 << cascade_idx);
    int z_dim = march_state.emission.extent(1) / (1 << cascade_idx);
    int ray_dim = PROBE0_NUM_RAYS * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
    const auto& cascade_i = state->cascades[cascade_lookup_idx].reshape<5>({
        x_dim,
        z_dim,
        ray_dim,
        NumComponents,
        NumAz
    });
    FpConst5d cascade_ip = cascade_i;
    if (cascade_idx != MAX_LEVEL) {
        cascade_ip = state->cascades[cascade_lookup_idx_p].reshape<5>({
            x_dim / 2,
            z_dim / 2,
            ray_dim * (1 << CASCADE_BRANCHING_FACTOR),
            NumComponents,
            NumAz
        });
    }
    auto dims = cascade_i.get_dimensions();
    auto upper_dims = cascade_ip.get_dimensions();
    auto& atmos = state->atmos;

    CascadeRTState rt_state;
    if constexpr (UseMipmaps) {
        rt_state.eta = state->raymarch_state.emission_mipmaps[cascade_idx];
        rt_state.chi = state->raymarch_state.absorption_mipmaps[cascade_idx];
        rt_state.mipmap_factor = state->raymarch_state.cumulative_mipmap_factor(cascade_idx);
    } else {
        rt_state.eta = march_state.emission;
        rt_state.chi = march_state.absorption;
        rt_state.mipmap_factor = 0;
    }

    auto az_rays = get_az_rays();
    auto az_weights = get_az_weights();

    Fp3d alo;
    if (compute_alo) {
        alo = state->alo;
    }

    std::string cascade_name = fmt::format("Cascade {}", cascade_idx);
    yakl::timer_start(cascade_name.c_str());
    parallel_for(
        SimpleBounds<3>(dims(2), dims(0), dims(1)),
        YAKL_LAMBDA (int ray_idx, int u, int v) {
            // NOTE(cmo): u, v probe indices
            // NOTE(cmo): x, y world coords
            for (int i = 0; i < NumComponents; ++i) {
                for (int j = 0; j < NumAz; ++j) {
                    cascade_i(u, v, ray_idx, i, j) = FP(0.0);
                }
            }

            int upper_cascade_idx = cascade_idx + 1;
            fp_t spacing = PROBE0_SPACING * (1 << cascade_idx);
            fp_t upper_spacing = PROBE0_SPACING * (1 << upper_cascade_idx);
            int num_rays = PROBE0_NUM_RAYS * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            int upper_num_rays = PROBE0_NUM_RAYS * (1 << (upper_cascade_idx * CASCADE_BRANCHING_FACTOR));
            fp_t radius = PROBE0_LENGTH * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            if (LAST_CASCADE_TO_INFTY && cascade_idx == MAX_LEVEL) {
                radius = LAST_CASCADE_MAX_DIST;
            }
            fp_t prev_radius = FP(0.0);
            if (cascade_idx != 0) {
                prev_radius = PROBE0_LENGTH * (1 << ((cascade_idx-1) * CASCADE_BRANCHING_FACTOR));
            }

            fp_t cx = (u + FP(0.5)) * spacing;
            fp_t cy = (v + FP(0.5)) * spacing;
            fp_t distance = radius - prev_radius;

            // NOTE(cmo): bc: bottom corner of 2x2 used for bilinear interp on next cascade
            // uc: upper corner
            int upper_u_bc = int((u - 1) / 2);
            int upper_v_bc = int((v - 1) / 2);
            fp_t u_bc_weight = FP(0.25) + FP(0.5) * (u % 2);
            fp_t v_bc_weight = FP(0.25) + FP(0.5) * (v % 2);
            fp_t u_uc_weight = FP(1.0) - u_bc_weight;
            fp_t v_uc_weight = FP(1.0) - v_bc_weight;
            int u_bc = yakl::max(upper_u_bc, 0);
            int v_bc = yakl::max(upper_v_bc, 0);
            int u_uc = yakl::min(u_bc+1, (int)upper_dims(0)-1);
            int v_uc = yakl::min(v_bc+1, (int)upper_dims(1)-1);
            // NOTE(cmo): Edge-most probes should only interpolate the edge-most
            // probes of the cascade above them, otherwise 3 probes per dimension
            // (e.g. u = 0, 1, 2) have u_bc = 0, vs 2 for every other u_bc
            // value. Additionally, this sample in the upper probe "texture"
            // isn't in the support of u_bc + 1
            if (u == 0) {
                u_uc = 0;
            }
            if (v == 0) {
                v_uc = 0;
            }

            fp_t angle = FP(2.0) * FP(M_PI) / num_rays * (ray_idx + FP(0.5));
            vec2 direction;
            direction(0) = yakl::cos(angle);
            direction(1) = yakl::sin(angle);
            vec2 start;
            vec2 end;

            int upper_ray_start_idx = ray_idx * (1 << CASCADE_BRANCHING_FACTOR);
            int num_rays_per_ray = 1 << CASCADE_BRANCHING_FACTOR;
            fp_t ray_weight = FP(1.0) / num_rays_per_ray;

            if (BRANCH_RAYS) {
                int prev_idx = ray_idx / (1 << CASCADE_BRANCHING_FACTOR);
                int prev_num_rays = 1;
                if (cascade_idx != 0) {
                    prev_num_rays = num_rays / (1 << CASCADE_BRANCHING_FACTOR);
                }
                fp_t prev_angle = FP(2.0) * FP(M_PI) / prev_num_rays * (prev_idx + FP(0.5));
                vec2 prev_direction;
                prev_direction(0) = yakl::cos(prev_angle);
                prev_direction(1) = yakl::sin(prev_angle);
                start(0) = cx + prev_radius * prev_direction(0);
                start(1) = cy + prev_radius * prev_direction(1);
                end(0) = cx + radius * direction(0);
                end(1) = cy + radius * direction(1);
            } else {
                start(0) = cx + prev_radius * direction(0);
                start(1) = cy + prev_radius * direction(1);
                end = start + direction * distance;
            }

            fp_t length_scale = FP(1.0);
            if (USE_ATMOSPHERE) {
                length_scale = atmos.voxel_scale;
            }
            auto sample = raymarch_2d<UseMipmaps, NumWavelengths, NumAz, NumComponents>(
                rt_state,
                Raymarch2dStaticArgs<NumAz>{
                    .ray_start = start,
                    .ray_end = end,
                    .az_rays = az_rays,
                    .az_weights = az_weights,
                    .alo = alo,
                    .distance_scale = length_scale
                }
            );
            decltype(sample) upper_sample(FP(0.0));
            // NOTE(cmo): Sample upper cascade.
            if (cascade_idx != MAX_LEVEL) {
                for (
                    int upper_ray_idx = upper_ray_start_idx;
                    upper_ray_idx < upper_ray_start_idx + num_rays_per_ray;
                    ++upper_ray_idx
                ) {
                    for (int i = 0; i < NumComponents; ++i) {
                        for (int r = 0; r < NumAz; ++r) {
                            if (az_rays(r) == FP(0.0)) {
                                // NOTE(cmo): Can't merge the in-out of page ray.
                                continue;
                            }
                            fp_t u_11 = cascade_ip(u_bc, v_bc, upper_ray_idx, i, r);
                            fp_t u_21 = cascade_ip(u_uc, v_bc, upper_ray_idx, i, r);
                            fp_t u_12 = cascade_ip(u_bc, v_uc, upper_ray_idx, i, r);
                            fp_t u_22 = cascade_ip(u_uc, v_uc, upper_ray_idx, i, r);
                            fp_t term = (
                                v_bc_weight * (u_bc_weight * u_11 + u_uc_weight * u_21) +
                                v_uc_weight * (u_bc_weight * u_12 + u_uc_weight * u_22)
                            );
                            upper_sample(i, r) += ray_weight * term;
                        }
                    }
                }
            }
            auto merged = merge_intervals(sample, upper_sample, az_rays);
            for (int i = 0; i < NumComponents; ++i) {
                for (int r = 0; r < NumAz; ++r) {
                    cascade_i(u, v, ray_idx, i, r) = merged(i, r);
                }
            }
        }
    );
    yakl::timer_stop(cascade_name.c_str());
}

template <bool UseMipmaps=USE_MIPMAPS, int NumWavelengths=NUM_WAVELENGTHS, int NumAz=NUM_AZ, int NumComponents=NUM_COMPONENTS>
void compute_cascade_i_bilinear_fix_2d (
    State* state,
    int cascade_idx,
    bool compute_alo = false
) {
    if (cascade_idx == MAX_LEVEL) {
        return compute_cascade_i_2d(state, cascade_idx);
    }

    const auto march_state = state->raymarch_state;
    int cascade_lookup_idx = cascade_idx;
    int cascade_lookup_idx_p = cascade_idx + 1;
    if constexpr (PINGPONG_BUFFERS) {
        if (cascade_idx & 1) {
            cascade_lookup_idx = 1;
            cascade_lookup_idx_p = 0;
        } else {
            cascade_lookup_idx = 0;
            cascade_lookup_idx_p = 1;
        }
    }
    int x_dim = march_state.emission.extent(0) / (1 << cascade_idx);
    int z_dim = march_state.emission.extent(1) / (1 << cascade_idx);
    int ray_dim = PROBE0_NUM_RAYS * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
    const auto& cascade_i = state->cascades[cascade_lookup_idx].reshape<5>({
        x_dim,
        z_dim,
        ray_dim,
        NumComponents,
        NumAz
    });
    FpConst5d cascade_ip = cascade_i;
    if (cascade_idx != MAX_LEVEL) {
        cascade_ip = state->cascades[cascade_lookup_idx_p].reshape<5>({
            x_dim / 2,
            z_dim / 2,
            ray_dim * CASCADE_BRANCHING_FACTOR,
            NumComponents,
            NumAz
        });
    }
    auto dims = cascade_i.get_dimensions();
    auto upper_dims = cascade_ip.get_dimensions();

    CascadeRTState rt_state;
    if constexpr (UseMipmaps) {
        rt_state.eta = state->raymarch_state.emission_mipmaps[cascade_idx];
        rt_state.chi = state->raymarch_state.absorption_mipmaps[cascade_idx];
        rt_state.mipmap_factor = state->raymarch_state.cumulative_mipmap_factor(cascade_idx);
    } else {
        rt_state.eta = march_state.emission;
        rt_state.chi = march_state.absorption;
        rt_state.mipmap_factor = 0;
    }

    auto& atmos = state->atmos;
    auto az_rays = get_az_rays();
    auto az_weights = get_az_weights();

    Fp3d alo;
    if (compute_alo) {
        alo = state->alo;
    }

    std::string cascade_name = fmt::format("Cascade {}", cascade_idx);
    yakl::timer_start(cascade_name.c_str());
    parallel_for(
        SimpleBounds<3>(dims(2), dims(0), dims(1)),
        YAKL_LAMBDA (int ray_idx, int u, int v) {
            // NOTE(cmo): u, v probe indices
            // NOTE(cmo): x, y world coords
            for (int i = 0; i < NumComponents; ++i) {
                for (int j = 0; j < NumAz; ++j) {
                    cascade_i(u, v, ray_idx, i, j) = FP(0.0);
                }
            }

            int upper_cascade_idx = cascade_idx + 1;
            fp_t spacing = PROBE0_SPACING * (1 << cascade_idx);
            fp_t upper_spacing = PROBE0_SPACING * (1 << upper_cascade_idx);
            int num_rays = PROBE0_NUM_RAYS * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            int upper_num_rays = PROBE0_NUM_RAYS * (1 << (upper_cascade_idx * CASCADE_BRANCHING_FACTOR));
            fp_t radius = PROBE0_LENGTH * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            if (LAST_CASCADE_TO_INFTY && cascade_idx == MAX_LEVEL) {
                radius = LAST_CASCADE_MAX_DIST;
            }
            fp_t prev_radius = FP(0.0);
            if (cascade_idx != 0) {
                prev_radius = PROBE0_LENGTH * (1 << ((cascade_idx-1) * CASCADE_BRANCHING_FACTOR));
            }

            fp_t cx = (u + FP(0.5)) * spacing;
            fp_t cy = (v + FP(0.5)) * spacing;
            fp_t distance = radius - prev_radius;

            // NOTE(cmo): bc: bottom corner of 2x2 used for bilinear interp on next cascade
            // uc: upper corner
            int upper_u_bc = int((u - 1) / 2);
            int upper_v_bc = int((v - 1) / 2);
            fp_t u_bc_weight = FP(0.25) + FP(0.5) * (u % 2);
            fp_t v_bc_weight = FP(0.25) + FP(0.5) * (v % 2);
            fp_t u_uc_weight = FP(1.0) - u_bc_weight;
            fp_t v_uc_weight = FP(1.0) - v_bc_weight;
            int u_bc = yakl::max(upper_u_bc, 0);
            int v_bc = yakl::max(upper_v_bc, 0);
            int u_uc = yakl::min(u_bc+1, (int)upper_dims(0)-1);
            int v_uc = yakl::min(v_bc+1, (int)upper_dims(1)-1);
            // NOTE(cmo): Edge-most probes should only interpolate the edge-most
            // probes of the cascade above them, otherwise 3 probes per dimension
            // (e.g. u = 0, 1, 2) have u_bc = 0, vs 2 for every other u_bc
            // value. Additionally, this sample in the upper probe "texture"
            // isn't in the support of u_bc + 1
            if (u == 0) {
                u_uc = 0;
            }
            if (v == 0) {
                v_uc = 0;
            }


            fp_t angle = FP(2.0) * FP(M_PI) / num_rays * (ray_idx + FP(0.5));
            vec2 direction;
            direction(0) = yakl::cos(angle);
            direction(1) = yakl::sin(angle);
            vec2 start;

            int upper_ray_start_idx = ray_idx * (1 << CASCADE_BRANCHING_FACTOR);
            int num_rays_per_ray = 1 << CASCADE_BRANCHING_FACTOR;
            fp_t ray_weight = FP(1.0) / num_rays_per_ray;

            if (BRANCH_RAYS) {
                int prev_idx = ray_idx / (1 << CASCADE_BRANCHING_FACTOR);
                int prev_num_rays = 1;
                if (cascade_idx != 0) {
                    prev_num_rays = num_rays / (1 << CASCADE_BRANCHING_FACTOR);
                }
                fp_t prev_angle = FP(2.0) * FP(M_PI) / prev_num_rays * (prev_idx + FP(0.5));
                vec2 prev_direction;
                prev_direction(0) = yakl::cos(prev_angle);
                prev_direction(1) = yakl::sin(prev_angle);
                start(0) = cx + prev_radius * prev_direction(0);
                start(1) = cy + prev_radius * prev_direction(1);
            } else {
                start(0) = cx + prev_radius * direction(0);
                start(1) = cy + prev_radius * direction(1);
            }
            // NOTE(cmo): Centre of upper cascade probes
            vec2 c11, c21, c12, c22;
            c11(0) = (u_bc + FP(0.5)) * upper_spacing;
            c11(1) = (v_bc + FP(0.5)) * upper_spacing;

            c21(0) = (u_uc + FP(0.5)) * upper_spacing;
            c21(1) = (v_bc + FP(0.5)) * upper_spacing;

            c12(0) = (u_bc + FP(0.5)) * upper_spacing;
            c12(1) = (v_uc + FP(0.5)) * upper_spacing;

            c22(0) = (u_uc + FP(0.5)) * upper_spacing;
            c22(1) = (v_uc + FP(0.5)) * upper_spacing;
            // NOTE(cmo): Start of interval in upper cascade (end of this ray)
            vec2 u11_start, u21_start, u12_start, u22_start;
            yakl::SArray<fp_t, 2, NumComponents, NumAz> u11_contrib, u21_contrib, u12_contrib, u22_contrib, upper_sample;

            fp_t length_scale = FP(1.0);
            if (USE_ATMOSPHERE) {
                length_scale = atmos.voxel_scale;
            }
            auto trace_and_merge_with_upper = [&upper_sample, &rt_state, &az_rays, &cascade_ip, &length_scale, &az_weights, &alo](
                vec2 start,
                vec2 end,
                int u,
                int v,
                int upper_ray_idx,
                decltype(upper_sample)& storage
            ) {
                storage = raymarch_2d<UseMipmaps, NumWavelengths, NumAz, NumComponents>(
                    rt_state,
                    Raymarch2dStaticArgs<NumAz>{
                        .ray_start = start,
                        .ray_end = end,
                        .az_rays = az_rays,
                        .az_weights = az_weights,
                        .alo = alo,
                        .distance_scale = length_scale
                    }
                );
                for (int i = 0; i < NumComponents; ++i) {
                    for (int r = 0; r < NumAz; ++r) {
                        upper_sample(i, r) = cascade_ip(u, v, upper_ray_idx, i, r);
                    }
                }
                storage = merge_intervals(storage, upper_sample, az_rays);
                return storage;
            };

            decltype(upper_sample) merged(FP(0.0));
            // NOTE(cmo): Sample upper cascade.
            if (cascade_idx != MAX_LEVEL) {
                for (
                    int upper_ray_idx = upper_ray_start_idx;
                    upper_ray_idx < upper_ray_start_idx + num_rays_per_ray;
                    ++upper_ray_idx
                ) {

                    vec2 upper_direction;
                    if (BRANCH_RAYS) {
                        upper_direction = direction;
                    } else {
                        fp_t upper_angle = FP(2.0) * FP(M_PI) / upper_num_rays * (upper_ray_idx + FP(0.5));
                        upper_direction(0) = std::cos(upper_angle);
                        upper_direction(1) = std::sin(upper_angle);
                    }
                    u11_start = c11 + radius * upper_direction;
                    u21_start = c21 + radius * upper_direction;
                    u12_start = c12 + radius * upper_direction;
                    u22_start = c22 + radius * upper_direction;


                    u11_contrib = trace_and_merge_with_upper(start, u11_start, u_bc, v_bc, upper_ray_idx, u11_contrib);
                    u21_contrib = trace_and_merge_with_upper(start, u21_start, u_uc, v_bc, upper_ray_idx, u21_contrib);
                    u12_contrib = trace_and_merge_with_upper(start, u12_start, u_bc, v_uc, upper_ray_idx, u12_contrib);
                    u22_contrib = trace_and_merge_with_upper(start, u22_start, u_uc, v_uc, upper_ray_idx, u22_contrib);

                    for (int i = 0; i < NumComponents; ++i) {
                        for (int r = 0; r < NumAz; ++r) {
                            fp_t term = (
                                v_bc_weight * (u_bc_weight * u11_contrib(i, r) + u_uc_weight * u21_contrib(i, r)) +
                                v_uc_weight * (u_bc_weight * u12_contrib(i, r) + u_uc_weight * u22_contrib(i, r))
                            );
                            merged(i, r) += ray_weight * term;
                        }
                    }
                }
            }
            for (int i = 0; i < NumComponents; ++i) {
                for (int r = 0; r < NumAz; ++r) {
                    cascade_i(u, v, ray_idx, i, r) = merged(i, r);
                }
            }
        }
    );
    yakl::timer_stop(cascade_name.c_str());
}

#else
#endif