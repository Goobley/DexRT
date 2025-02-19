#if !defined(DEXRT_LINE_SWEEPING_HPP)
#define DEXRT_LINE_SWEEPING_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "RayMarching.hpp"

/// Extend using values in sc/tc and store in sc_desc/tc_dest
KOKKOS_FORCEINLINE_FUNCTION void log2_extension(
    const KTeam& team,
    const ScratchView<fp_t*> sc_dest,
    const ScratchView<fp_t*> tc_dest,
    const ScratchView<fp_t*> sc,
    const ScratchView<fp_t*> tc,
    i32 num_samples,
    i32 log2_offset
) {
    const i32 offset = 1 << log2_offset;
    const i32 prev_offset = 1 << std::max(log2_offset - 1, 0);
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team, num_samples),
        [&] (const int t) {
            i32 far_idx = t - offset;

            const fp_t sc_sample = sc(t);
            const fp_t tc_sample = tc(t);

            // if (far_idx >= -offset && far_idx < prev_offset) {
            if (far_idx >= -offset) {
                far_idx = std::max(far_idx, 0);
            } else {
                sc_dest(t) = sc_sample;
                tc_dest(t) = tc_sample;
                return;
            }

            const fp_t far_sc = sc(far_idx);
            const fp_t far_tc = tc(far_idx);

            sc_dest(t) = far_sc * tc_sample + sc_sample;
            tc_dest(t) = tc_sample * far_tc;
        }
    );
}

template <int RcMode>
inline void compute_line_sweep_samples(
    const State& state,
    const CascadeState& casc_state,
    int cascade_idx,
    const CascadeCalcSubset& subset,
    int wave,
    const MultiResMipChain& mip_chain = MultiResMipChain()
) {
    JasUnpack(state, atmos, incl_quad, adata, pops);
    JasUnpack(subset, la_start, subset_idx);
    JasUnpack(casc_state, line_sweep_data);
    const auto& profile = state.phi;
    constexpr bool compute_alo = RcMode & RC_COMPUTE_ALO;
    using AloType = std::conditional_t<compute_alo, fp_t, DexEmpty>;
    typedef typename RcDynamicState<RcMode>::type DynamicState;

    assert(!PREAVERAGE && "Line sweeping does not support preaveraging");

    int ls_idx = line_sweep_data.get_cascade_subset_idx(cascade_idx, subset_idx);
    auto& ls_data = line_sweep_data.cascade_sets[ls_idx];
    const auto& ls_storage = line_sweep_data.storage;
    CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, cascade_idx);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);

    DeviceBoundaries boundaries_h{
        .boundary = state.boundary,
        .zero_bc = state.zero_bc,
        .pw_bc = state.pw_bc
    };
    auto offset = get_offsets(atmos);

    JasUnpack(state, mr_block_map);
    const auto& block_map = mr_block_map.block_map;
    const int max_mip_to_sample = std::min(
        state.config.mip_config.mip_levels[cascade_idx],
        mip_chain.max_mip_factor
    );
    const auto& dir_set_desc = ls_data.dir_set_desc;

    // Do a normal cascade-like dispatch that traces from the previous probe to each probe (ensuring to extend to capture the bc for first_sample). N.B. The line sweep rays are already inverted!
    dex_parallel_for(
        "Initial probe-probe raymarch",
        FlatLoop<2>(
            dir_set_desc.total_steps,
            ray_subset.num_incl
        ),
        KOKKOS_LAMBDA (int line_step, int theta_idx) {
            // NOTE(cmo): find line associated with sample.
            const i32 line_idx = upper_bound(ls_data.line_storage_start_idx, line_step) - 1;
            // NOTE(cmo): find line set associated with line
            const i32 dir_idx = upper_bound(ls_data.line_set_start_idx, line_idx) - 1;
            // step along line
            const i32 step_idx = line_step - ls_data.line_storage_start_idx(line_idx);
            const auto& line = ls_data.lines(line_idx);
            const auto& ls_desc = ls_data.line_set_desc(dir_idx);
            const fp_t step = ls_desc.step;
            const vec2 dir = ls_desc.d;

            fp_t t0, t1;
            if (step_idx == 0) {
                t0 = line.t0 - FP(10.0); // NOTE(cmo): Force to sample boundary
                t1 = line.first_sample;
            } else {
                t0 = line.first_sample + (step_idx - 1) * step;
                t0 = std::max(t0, line.first_sample);
                t1 = t0 + step;
            }
            vec2 start = line.o + t0 * dir;
            vec2 end = line.o + t1 * dir;
            vec2 centre = end;

            RayProps ray{
                .start = start,
                .end = end,
                .dir = dir,
                .centre = centre
            };
            int la = la_start + wave + ray_subset.start_wave_batch;
            DynamicState dyn_state = get_dyn_state<DynamicState>(
                    la,
                    atmos,
                    adata,
                    profile,
                    pops,
                    mip_chain
                );

            const fp_t incl = incl_quad.muy(theta_idx);
            const fp_t incl_weight = incl_quad.wmuy(theta_idx);
            // NOTE(cmo): Explicitly _not_ using inverted_mu here, because this ray doesn't need inverting
            // Although we keep the "incl" term in the -y for consistency when doing one hemisphere.
            const fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
            vec3 mu(FP(0.0));
            mu(0) = ray.dir(0) * sin_theta;
            mu(1) = -incl;
            mu(2) = ray.dir(1) * sin_theta;
            const auto& boundaries = boundaries_h;

            RadianceInterval<DexEmpty> ri;
            auto dispatch_raymarch = [&]<typename BcType>(BcType bc_type) {
                // NOTE(cmo): We don't actually use the cascade state in the raymarcher
                auto casc_and_bc = get_bc<BcType>(DeviceCascadeState{}, boundaries);
                ri = multi_level_dda_raymarch_2d<RcMode | RC_LINE_SWEEP, BcType, DynamicState, DexEmpty>(
                    Raymarch2dArgs<BcType, DynamicState> {
                        .casc_state_bc = casc_and_bc,
                        .ray = ray,
                        .distance_scale = atmos.voxel_scale,
                        .mu = mu,
                        .incl = incl,
                        .incl_weight = incl_weight,
                        .wave = wave,
                        .la = la,
                        .offset = offset,
                        .max_mip_to_sample = max_mip_to_sample,
                        .block_map = block_map,
                        .mr_block_map = mr_block_map,
                        .mip_chain = mip_chain,
                        .dyn_state = dyn_state
                    }
                );
            };
            if constexpr (RcMode & RC_SAMPLE_BC) {
                switch (boundaries.boundary) {
                    case BoundaryType::Zero: {
                        dispatch_raymarch(ZeroBc{});
                    } break;
                    case BoundaryType::Promweaver: {
                        dispatch_raymarch(PwBc<>{});
                    } break;
                    default: {
                        yakl::yakl_throw("Unknown BC type");
                    }
                }
            } else {
                dispatch_raymarch(ZeroBc{});
            }

            // NOTE(cmo): Writing to these isn't coalesced because we need
            // contiguous access to each line later, but still want to benefit
            // from coalesced access whilst tracing above.
            ls_storage.source_term(theta_idx, line_step) = ri.I;
            ls_storage.transmittance(theta_idx, line_step) = std::exp(-ri.tau);
        }
    );
    Kokkos::fence();

    // Launch co-operative groups -- one per line, and do the extensions
    // NOTE(cmo): Whilst a good idea, for sane model sizes, these don't seem to
    // profile much faster, and there's an issue with the treatment of the
    // uppermost cascade due to its interval length being super long (to ensure
    // the boundary is hit), which isn't clipped here.
    constexpr bool do_log2_extensions = false;
    const IntervalLength int_length = cascade_interval_length(casc_state.num_cascades, cascade_idx);
    const fp_t interval_length = int_length.to - int_length.from;
    const fp_t step_length = dir_set_desc.step;

    i32 max_pow2 = 0;
    if constexpr (do_log2_extensions) {
        for (int i = 0; i < 14; ++i) {
            i32 extension_length = 1 << i;
            if ((i32(interval_length / step_length) % extension_length) == 0) {
                max_pow2 = i;
                continue;
            }
            break;
        }
    }
    const fp_t log2_extended_length = step_length * (1 << max_pow2);
    const i32 max_steps_to_merge = i32(interval_length / log2_extended_length);
    size_t scratch_size = 2 * ScratchView<fp_t*>::shmem_size(dir_set_desc.max_line_steps);
    if constexpr (do_log2_extensions) {
        scratch_size += 2 * ScratchView<fp_t*>::shmem_size(dir_set_desc.max_line_steps);
    }
    const int num_incl = state.c0_size.num_incl;
    yakl::timer_start("Extend lines");
    Kokkos::parallel_for(
        "Extend lines",
        Kokkos::TeamPolicy(dir_set_desc.total_lines, std::min(DEXRT_WARP_SIZE, 32)).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA (const KTeam& team) {
            ScratchView<fp_t*> source_copy_0(team.team_scratch(0), dir_set_desc.max_line_steps);
            ScratchView<fp_t*> trans_copy_0(team.team_scratch(0), dir_set_desc.max_line_steps);
            ScratchView<fp_t*> source_copy_1;
            ScratchView<fp_t*> trans_copy_1;
            JasUse(do_log2_extensions, max_pow2);
            if constexpr (do_log2_extensions) {
                source_copy_1 = decltype(source_copy_1)(team.team_scratch(0), dir_set_desc.max_line_steps);
                trans_copy_1 = decltype(trans_copy_1)(team.team_scratch(0), dir_set_desc.max_line_steps);
            }
            const auto& source_copy = source_copy_0;
            const auto& trans_copy = trans_copy_0;
            const i32 line_idx = team.league_rank();
            const auto& line = ls_data.lines(line_idx);
            const i32 line_start = ls_data.line_storage_start_idx(line_idx);

            for (int theta_idx = 0; theta_idx < num_incl; ++theta_idx) {
                // NOTE(cmo): Copy array to scratch
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, line.num_samples),
                    [&] (const int t) {
                        source_copy(t) = ls_storage.source_term(theta_idx, line_start + t);
                        trans_copy(t) = ls_storage.transmittance(theta_idx, line_start + t);
                    }
                );
                team.team_barrier();

                if constexpr (do_log2_extensions) {
                    // NOTE(cmo): We need to pingpong between sc/1
                    const auto& sc = source_copy;
                    const auto& tc = trans_copy;
                    const auto& sc1 = source_copy_1;
                    const auto& tc1 = trans_copy_1;
                    for (int e = 0; e < max_pow2; ++e) {
                        if (e & 1) {
                            log2_extension(
                                team,
                                sc,
                                tc,
                                sc1,
                                tc1,
                                line.num_samples,
                                e
                            );
                        } else {
                            log2_extension(
                                team,
                                sc1,
                                tc1,
                                sc,
                                tc,
                                line.num_samples,
                                e
                            );
                        }
                        team.team_barrier();
                    }
                }

                // select final log2_extension dest
                auto& source_copy_dest = (do_log2_extensions && (max_pow2 & 1)) ? source_copy_1 : source_copy;
                auto& trans_copy_dest = (do_log2_extensions && (max_pow2 & 1)) ? trans_copy_1 : trans_copy;

                // NOTE(cmo): Integrate RIs -- assumption that interval_length is a multiple of step
                const i32 steps_to_merge = interval_length / log2_extended_length;
                const i32 grid_step_size = do_log2_extensions ? (1 << max_pow2) : 1;
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, line.num_samples),
                    [&] (const int t) {
                        i32 starting_sample_idx = t - ((steps_to_merge - 1) * grid_step_size);
                        if constexpr (!do_log2_extensions) {
                            starting_sample_idx = std::max(starting_sample_idx, 0);
                        }

                        fp_t source_acc = FP(0.0);
                        fp_t trans_acc = FP(1.0);
                        for (int sample_idx = starting_sample_idx; sample_idx < (t + 1); sample_idx += grid_step_size) {
                            if constexpr (do_log2_extensions) {
                                if (sample_idx < 0) {
                                    continue;
                                }
                            }
                            source_acc = source_copy_dest(sample_idx) + source_acc * trans_copy_dest(sample_idx);
                            trans_acc *= trans_copy_dest(sample_idx);
                        }
                        ls_storage.source_term(theta_idx, line_start + t) = source_acc;
                        ls_storage.transmittance(theta_idx, line_start + t) = trans_acc;
                    }
                );
                team.team_barrier();
            }
        }
    );
    Kokkos::fence();
    yakl::timer_stop("Extend lines");
}

template <int p_ax>
KOKKOS_FORCEINLINE_FUNCTION vec2 world_to_line_grid_pos_impl(const LineSetDescriptor& line_set, const LsLine& line, vec2 p) {
    static_assert(p_ax < 2, "Must call with p_ax 0 or 1");
    return vec2(FP(0.0));
}

template <>
KOKKOS_FORCEINLINE_FUNCTION vec2 world_to_line_grid_pos_impl<0>(const LineSetDescriptor& line_set, const LsLine& line, vec2 p) {
    const auto& d = line_set.d;
    p = p - line_set.origin;
    mat2x2 rot_mat(FP(0.0));

    constexpr int p_ax = 0;
    constexpr int s_ax = 1;

    rot_mat(p_ax, p_ax) = FP(1.0);
    rot_mat(p_ax, s_ax) = -d(p_ax) / d(s_ax);
    rot_mat(s_ax, s_ax) = FP(1.0) / d(s_ax);

    vec2 grid_coord;
    grid_coord(0) = rot_mat(0, 0) * p(0) + rot_mat(0, 1) * p(1);
    grid_coord(1) = rot_mat(1, 0) * p(0) + rot_mat(1, 1) * p(1);
    grid_coord(p_ax) -= line.o(p_ax) - line_set.origin(p_ax);
    grid_coord(0) /= line_set.step;
    grid_coord(1) /= line_set.step;
    return grid_coord;
}

template <>
KOKKOS_FORCEINLINE_FUNCTION vec2 world_to_line_grid_pos_impl<1>(const LineSetDescriptor& line_set, const LsLine& line, vec2 p) {
    const auto& d = line_set.d;
    p = p - line_set.origin;
    mat2x2 rot_mat(FP(0.0));

    constexpr int p_ax = 1;
    constexpr int s_ax = 0;

    rot_mat(p_ax, p_ax) = FP(1.0);
    rot_mat(p_ax, s_ax) = -d(p_ax) / d(s_ax);
    rot_mat(s_ax, s_ax) = FP(1.0) / d(s_ax);

    vec2 grid_coord;
    grid_coord(0) = rot_mat(0, 0) * p(0) + rot_mat(0, 1) * p(1);
    grid_coord(1) = rot_mat(1, 0) * p(0) + rot_mat(1, 1) * p(1);
    grid_coord(p_ax) -= line.o(p_ax) - line_set.origin(p_ax);
    grid_coord(0) /= line_set.step;
    grid_coord(1) /= line_set.step;
    return grid_coord;
}

KOKKOS_FORCEINLINE_FUNCTION vec2 world_to_line_grid_pos(const LineSetDescriptor& line_set, const LsLine& line, vec2 p) {
    const int p_ax = line_set.primary_axis;
    switch (p_ax) {
        case 0: {
            return world_to_line_grid_pos_impl<0>(line_set, line, p);
        } break;
        case 1: {
            return world_to_line_grid_pos_impl<1>(line_set, line, p);
        } break;

        default: {
            assert(false);
        }
    }
    return vec2(FP(0.0));

}

template <int RcMode>
inline void interpolate_line_sweep_samples_to_cascade(
    const State& state,
    const CascadeState& casc_state,
    int cascade_idx,
    const CascadeCalcSubset& subset,
    int wave
) {
    // Do the bilinear interpolation
    JasUnpack(subset, subset_idx);
    JasUnpack(casc_state, line_sweep_data);

    CascadeIdxs lookup = cascade_indices(casc_state, cascade_idx);
    Fp1d i_cascade_i = casc_state.i_cascades[lookup.i];
    Fp1d tau_cascade_i = casc_state.tau_cascades[lookup.i];
    CascadeStorage dims = cascade_size(state.c0_size, cascade_idx);
    DeviceCascadeState dev_casc_state {
        .num_cascades = casc_state.num_cascades,
        .n = cascade_idx,
        .casc_dims = dims,
        .cascade_I = i_cascade_i,
        .cascade_tau = tau_cascade_i
    };

    int ls_idx = line_sweep_data.get_cascade_subset_idx(cascade_idx, subset_idx);
    const auto& ls_data = line_sweep_data.cascade_sets[ls_idx];
    const auto& ls_storage = line_sweep_data.storage;
    CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, cascade_idx);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);

    i64 spatial_bounds = casc_state.probes_to_compute.num_active_probes(cascade_idx);
    DeviceProbesToCompute probe_coord_lookup = casc_state.probes_to_compute.bind(cascade_idx);
    const IntervalLength interval_length = cascade_interval_length(casc_state.num_cascades, cascade_idx);
    // NOTE(cmo): This ensures the boundary conditions get properly incorporated.
    constexpr bool clamp_inside = true;
    const fp_t inv_step_size = FP(1.0) / ls_data.dir_set_desc.step;
    // constexpr bool clamp_inside = RcMode & RC_SAMPLE_BC;
    // constexpr bool clamp_inside = false;

    // NOTE(cmo): We could actually support preaveraging by not using it in the
    // above, but merging multiple directions into the same texel during the
    // interpolation... but we don't currently
    dex_parallel_for(
        "Line Sweep Interp",
        FlatLoop<2>(
            ray_subset.num_flat_dirs,
            spatial_bounds
            // ray_subset.num_incl
        ),
        // KOKKOS_LAMBDA (int theta_idx, i64 ks, int phi_idx) {
        KOKKOS_LAMBDA (int phi_idx, i64 ks) {
            JasUse(clamp_inside);
            ivec2 probe_coord = probe_coord_lookup(ks);
            const int dir_by_dir_phi_idx = phi_idx;
            phi_idx += ray_subset.start_flat_dirs;

            const auto& line_set = ls_data.line_set_desc(dir_by_dir_phi_idx);
            const int p_ax = line_set.primary_axis;
            vec2 p = probe_pos(probe_coord, cascade_idx);

            vec2 grid_pos = world_to_line_grid_pos(line_set, ls_data.lines(line_set.line_start_idx), p);
            fp_t line_idx, t_idx;
            if (p_ax == 0) {
                line_idx = grid_pos(0);
                t_idx = grid_pos(1);
            } else {
                line_idx = grid_pos(1);
                t_idx = grid_pos(0);
            }
            // NOTE(cmo): Offset for start of interval
            t_idx -= interval_length.from * inv_step_size;

            if constexpr (!clamp_inside) {
                if (line_idx < FP(0.0)) {
                    return;
                }
            }

            constexpr int neg_sentinel = -60000;
            auto clamp_or_set_neg = [&](int v, int max_val) {
                if constexpr (clamp_inside) {
                    // clamp to range [0, n)
                    return std::min(std::max(v, 0), max_val);
                } else {
                    return neg_sentinel;
                }
            };

            const int li = i32(line_idx);
            const int lip = li + 1;
            const fp_t lipw = fp_t(lip) - line_idx;
            ivec2 lis;
            lis(0) = clamp_or_set_neg(li, line_set.num_lines-1);
            lis(1) = clamp_or_set_neg(lip, line_set.num_lines-1);
            vec2 liw;
            liw(0) = lipw;
            liw(1) = FP(1.0) - lipw;

            ivec4 sis;
            vec4 siw;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int l_idx = lis(i);
                fp_t l_wgt = liw(i);
                if constexpr (!clamp_inside) {
                    if (l_idx == neg_sentinel) {
                        continue;
                    }
                }

                int line_base_idx = line_set.line_start_idx + l_idx;
                const fp_t step_idx = t_idx - ls_data.lines(line_base_idx).first_sample * inv_step_size;
                const int si = i32(std::floor(step_idx));
                const int sip = si + 1;
                const fp_t sipw = fp_t(sip) - step_idx;
                const int max_sample = ls_data.lines(line_base_idx).num_samples - 1;
                sis(i * 2 + 0) = clamp_or_set_neg(si, max_sample);
                sis(i * 2 + 1) = clamp_or_set_neg(sip, max_sample);
                siw(i * 2 + 0) = sipw;
                siw(i * 2 + 1) = FP(1.0) - sipw;
            }

            for (int theta_idx = ray_subset.start_incl; theta_idx < ray_subset.start_incl + ray_subset.num_incl; ++theta_idx) {
                ProbeIndex probe_idx{
                    .coord = probe_coord,
                    .dir = phi_idx,
                    .incl = theta_idx,
                    .wave = wave
                };
                // NOTE(cmo): In this loop "tau" is still transmittance. Starting
                // with a transmittance of one and then subtract absorption means
                // that points that don't sample outside the grid have no opacity
                // (and so merge with upper cascades/boundaries)
                RadianceInterval<DexEmpty> interp_ri{
                    .tau = FP(1.0)
                };


                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int l_idx = lis(i);
                    fp_t l_wgt = liw(i);
                    if constexpr (!clamp_inside) {
                        if (l_idx == neg_sentinel) {
                            continue;
                        }
                    }

                    int line_base_idx = line_set.line_start_idx + l_idx;
                    int line_storage_base_idx = ls_data.line_storage_start_idx(line_base_idx);

                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int s_idx = sis(2 * i + j);
                        fp_t s_wgt = siw(2 * i + j);

                        if constexpr (!clamp_inside) {
                            if (s_idx == neg_sentinel) {
                                continue;
                            }
                        }

                        const fp_t I_sample = ls_storage.source_term(theta_idx, line_storage_base_idx + s_idx);
                        const fp_t trans_sample = ls_storage.transmittance(theta_idx, line_storage_base_idx + s_idx);
                        interp_ri.I += l_wgt * s_wgt * I_sample;
                        interp_ri.tau -= l_wgt * s_wgt * (FP(1.0) - trans_sample);
                    }
                }

                i64 lin_idx = probe_linear_index<RcMode>(dims, probe_idx);
                interp_ri.tau = std::min(-std::log(std::max(interp_ri.tau, FP(1e-15))), FP(1e3));
                dev_casc_state.cascade_I(lin_idx) = interp_ri.I;
                dev_casc_state.cascade_tau(lin_idx) = interp_ri.tau;
            }
        }
    );
    Kokkos::fence();

}


#else
#endif