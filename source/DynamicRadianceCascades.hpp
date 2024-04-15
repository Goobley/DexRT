#if !defined(DEXRT_DYNAMIC_RADIANCE_CASCADES_HPP)
#define DEXRT_DYNAMIC_RADIANCE_CASCADES_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "Utils.hpp"
#include "DynamicRayMarching.hpp"
#include "RadianceIntervals.hpp"
#include <fmt/core.h>

template <bool compute_alo=false, int NumWavelengths=NUM_WAVELENGTHS, int NumComponents=NUM_COMPONENTS>
void compute_dynamic_cascade_i_2d (
    State* state,
    const Fp3d& lte_scratch,
    const Fp2d& nh0,
    int cascade_idx,
    int la,
    const yakl::Array<i32, 1, yakl::memDevice>& active_set,
    fp_t wl_ray_weight
) {
    static_assert(USE_ATMOSPHERE);
    static_assert(NumComponents == 2);

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
    const auto& cascade_i = state->cascades[cascade_lookup_idx].reshape<5>(Dims(
        x_dim,
        z_dim,
        ray_dim,
        NumComponents,
        NUM_AZ
    ));
    FpConst5d cascade_ip = cascade_i;
    if (cascade_idx != MAX_LEVEL) {
        cascade_ip = state->cascades[cascade_lookup_idx_p].reshape<5>(Dims(
            x_dim / 2,
            z_dim / 2,
            ray_dim * (1 << CASCADE_BRANCHING_FACTOR),
            NumComponents,
            NUM_AZ
        ));
    }
    auto dims = cascade_i.get_dimensions();
    auto upper_dims = cascade_ip.get_dimensions();

    CascadeRTState rt_state{
        .mipmap_factor = 0,
        .eta = march_state.emission,
        .chi = march_state.absorption
    };
    FpConst2d eta(
        "eta",
        rt_state.eta.get_data(),
        rt_state.eta.extent(0),
        rt_state.eta.extent(1)
    );
    FpConst2d chi(
        "chi",
        rt_state.chi.get_data(),
        rt_state.chi.extent(0),
        rt_state.chi.extent(1)
    );

    auto az_rays = get_az_rays();
    auto az_weights = get_az_weights();

    const auto& atom = state->atom;
    const auto& phi = state->phi;
    const auto& pops = state->pops;
    const auto n_flat = state->pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    const auto lte_scratch_flat = lte_scratch.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    const auto Gamma_flat = state->Gamma.reshape<3>(Dims(
        state->Gamma.extent(0),
        state->Gamma.extent(1),
        state->Gamma.extent(2) * state->Gamma.extent(3)
    ));
    const auto& atmos = state->atmos;

    std::string cascade_name = fmt::format("Dynamic Cascade {}", cascade_idx);
    yakl::timer_start(cascade_name.c_str());
    parallel_for(
        SimpleBounds<4>(dims(0), dims(1), dims(2), dims(4)),
        YAKL_LAMBDA (int v, int u, int ray_idx, int r) {
            // NOTE(cmo): u, v probe indices
            // NOTE(cmo): x, z world coords
            cascade_i(v, u, ray_idx, 0, r) = FP(0.0);
            cascade_i(v, u, ray_idx, 1, r) = FP(0.0);

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
            fp_t cz = (v + FP(0.5)) * spacing;
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
            int u_uc = yakl::min(u_bc+1, (int)upper_dims(1)-1);
            int v_uc = yakl::min(v_bc+1, (int)upper_dims(0)-1);
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
            // NOTE(cmo): The multiplicative term for mux and muz in full 3D
            fp_t incl_factor = std::sqrt(FP(1.0) - az_rays(r));
            // NOTE(cmo): This is flatland direction; the y effect is added in the raymarcher
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
                start(1) = cz + prev_radius * prev_direction(1);
                end(0) = cx + radius * direction(0);
                end(1) = cz + radius * direction(1);
            } else {
                start(0) = cx + prev_radius * direction(0);
                start(1) = cz + prev_radius * direction(1);
                end = start + direction * distance;
            }

            fp_t length_scale = FP(1.0);
            if (USE_ATMOSPHERE) {
                length_scale = atmos.voxel_scale;
            }

            DynamicRadianceInterval upper_sample{};
            // NOTE(cmo): Sample upper cascade. This is done _before_ the
            // raymarch here, so we can compute the entries to Gamma using the
            // correct radiation field.
            if (cascade_idx != MAX_LEVEL && az_rays(r) != FP(0.0)) {
                for (
                    int upper_ray_idx = upper_ray_start_idx;
                    upper_ray_idx < upper_ray_start_idx + num_rays_per_ray;
                    ++upper_ray_idx
                ) {
                    fp_t u_11 = cascade_ip(v_bc, u_bc, upper_ray_idx, 0, r);
                    fp_t u_12 = cascade_ip(v_uc, u_bc, upper_ray_idx, 0, r);
                    fp_t u_21 = cascade_ip(v_bc, u_uc, upper_ray_idx, 0, r);
                    fp_t u_22 = cascade_ip(v_uc, u_uc, upper_ray_idx, 0, r);
                    fp_t I_interp = (
                        v_bc_weight * (u_bc_weight * u_11 + u_uc_weight * u_21) +
                        v_uc_weight * (u_bc_weight * u_12 + u_uc_weight * u_22)
                    );
                    u_11 = cascade_ip(v_bc, u_bc, upper_ray_idx, 1, r);
                    u_12 = cascade_ip(v_uc, u_bc, upper_ray_idx, 1, r);
                    u_21 = cascade_ip(v_bc, u_uc, upper_ray_idx, 1, r);
                    u_22 = cascade_ip(v_uc, u_uc, upper_ray_idx, 1, r);
                    fp_t tau_interp = (
                        v_bc_weight * (u_bc_weight * u_11 + u_uc_weight * u_21) +
                        v_uc_weight * (u_bc_weight * u_12 + u_uc_weight * u_22)
                    );
                    upper_sample.I += ray_weight * I_interp;
                    upper_sample.tau += ray_weight * tau_interp;
                }
            }

            // NOTE(cmo): This variant of the raymarcher does the merge internally
            DynamicRadianceInterval merged = dynamic_raymarch_2d<compute_alo>(
                Raymarch2dDynamicArgs{
                    .eta = eta,
                    .chi = chi,
                    .ray_start = start,
                    .ray_end = end,
                    .mux = direction(0) * incl_factor,
                    .muy = az_rays(r),
                    .muz = direction(1) * incl_factor,
                    .muy_weight = az_weights(r),
                    .distance_scale = length_scale,
                    .atmos = atmos,
                    .atom = atom,
                    .active_set = active_set,
                    .phi = phi,
                    .nh0 = nh0,
                    .n = n_flat,
                    .n_star_scratch = lte_scratch_flat,
                    .upper_sample = upper_sample,
                    .Gamma = Gamma_flat,
                    .wl_ray_weight = wl_ray_weight,
                    .la = la
                }
            );
            cascade_i(v, u, ray_idx, 0, r) = merged.I;
            cascade_i(v, u, ray_idx, 1, r) = merged.tau;
        }
    );
    yakl::timer_stop(cascade_name.c_str());
}

#else
#endif