#if !defined(DEXRT_LINE_SWEEPING_HPP)
#define DEXRT_LINE_SWEEPING_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "RayMarching.hpp"

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
    JasUnpack(subset, la_start, la_end, subset_idx);
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
        YAKL_LAMBDA (int line_step, int theta_idx) {
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
                t0 = line.t0 - FP(1.0); // NOTE(cmo): Force to sample boundary
                t1 = line.first_sample;
            } else {
                t0 = line.first_sample + (step_idx - 1) * step;
                t0 = std::max(t0, line.first_sample);
                t1 = t0 + step;
            }
            vec2 start = line.o + t0 * dir;
            vec2 end = line.o + t1 * dir;
            vec2 centre = start;

            RayProps ray{
                .start = start,
                .end = end,
                .dir = dir,
                .centre = centre
            };
            int la = la_start + wave;
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
                ri = multi_level_dda_raymarch_2d(
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
    // TODO(cmo): This isn't proper log2 extensions, but it'll let us test
    const IntervalLength int_length = cascade_interval_length(cascade_idx, casc_state.num_cascades);
    const fp_t interval_length = int_length.to - int_length.from;
    size_t scratch_size = 2 * ScratchView<fp_t*>::shmem_size(dir_set_desc.max_line_steps);
    const int num_incl = state.c0_size.num_incl;
    Kokkos::parallel_for(
        "Extend lines",
        Kokkos::TeamPolicy(dir_set_desc.total_lines, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA (const KTeam& team) {
            ScratchView<fp_t*> source_copy(team.team_scratch(0), dir_set_desc.max_line_steps);
            ScratchView<fp_t*> trans_copy(team.team_scratch(0), dir_set_desc.max_line_steps);
            const i32 line_idx = team.league_rank();
            const auto& line = ls_data.lines(line_idx);
            const auto& dir_set_desc = ls_data.dir_set_desc;
            const fp_t step = dir_set_desc.step;
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

                // NOTE(cmo): Integrate RIs
                i32 steps_to_merge = interval_length / step;
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, line.num_samples),
                    [&] (const int t) {
                        i32 starting_sample_idx = t - (steps_to_merge - 1);
                        starting_sample_idx = std::max(starting_sample_idx, 0);

                        fp_t source_acc = FP(0.0);
                        fp_t trans_acc = FP(0.0);
                        for (int sample_idx = starting_sample_idx; sample_idx < (t + 1); ++sample_idx) {
                            source_acc = source_copy(sample_idx) + source_acc * trans_copy(sample_idx);
                            trans_acc *= trans_copy(sample_idx);
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
}

inline void interpolate_line_sweep_samples_to_cascade(
    const State& state,
    const CascadeState& casc_state,
    int cascade_idx,
    const CascadeCalcSubset& subset) {
    // Do the bilinear interpolation

}


#else
#endif