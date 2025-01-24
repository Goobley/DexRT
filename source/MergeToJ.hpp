#if !defined(DEXRT_MERGE_TO_J_HPP)
#define DEXRT_MERGE_TO_J_HPP

#include "Config.hpp"
#include "Constants.hpp"
#include "Types.hpp"
#include "RcUtilsModes.hpp"
#include "BlockMap.hpp"

#include "LoopUtils.hpp"

inline FpConst2d merge_c0_to_J(
    const CascadeState& casc_state,
    const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE>& block_map,
    const Fp2d& J,
    const InclQuadrature incl_quad,
    int la_start,
    int la_end
) {
    const bool sparse = casc_state.probes_to_compute.sparse;
    const auto& c0_dims = casc_state.probes_to_compute.c0_size;
    const auto& c0 = casc_state.i_cascades[0];

    constexpr int RcMode = RC_flags_storage();
    const fp_t phi_weight = FP(1.0) / fp_t(c0_dirs_to_average<RcMode>());
    int wave_batch = la_end - la_start;

    // NOTE(cmo): Handle case of J on the GPU only being a single plane (config.store_J_on_cpu)
    if (c0_dims.wave_batch == J.extent(0)) {
        la_start = 0;
    }
    DeviceProbesToCompute probes_to_compute = casc_state.probes_to_compute.bind(0);

    // For a dense J, this is effectively filling in the flattened array
    const i64 num_probes = probes_to_compute.num_active_probes();
    const auto work_div = balance_parallel_work_division(BalanceLoopArgs{
        .loop = FlatLoop<1>(num_probes)
    });
    parallel_for(
        "final_cascade_to_J",
        TeamPolicy(work_div.team_count, Kokkos::AUTO()),
        KOKKOS_LAMBDA (const KTeam& team) {
            const i64 k_base = team.league_rank() * work_div.inner_work_count;
            const i64 max_k = std::min(k_base + work_div.inner_work_count, num_probes);
            for (i64 k = k_base; k < max_k; ++k) {
                FlatLoop<2> inner_loop(c0_dims.num_flat_dirs, c0_dims.num_incl);

                ivec2 coord = probes_to_compute(k);
                i64 ks;
                if (sparse) {
                    IdxGen idx_gen(block_map);
                    ks = idx_gen.idx(coord(0), coord(1));
                } else {
                    ks = coord(1) * c0_dims.num_probes(0) + coord(0);
                }

                for (int wave = 0; wave < wave_batch; ++wave) {
                    int la = la_start + wave;
                    fp_t j_sum = FP(0.0);
                    Kokkos::parallel_reduce(
                        Kokkos::TeamVectorRange(
                            team,
                            inner_loop.num_iter
                        ),
                        [&] (int loop_idx, fp_t& j_entry) {
                            auto idxs = inner_loop.unpack(loop_idx);
                            int phi_idx = idxs[0];
                            int theta_idx = idxs[1];
                            const fp_t ray_weight = phi_weight * incl_quad.wmuy(theta_idx);

                            ProbeStorageIndex idx{
                                .coord=coord,
                                .dir=phi_idx,
                                .incl=theta_idx,
                                .wave=wave
                            };

                            const fp_t sample = probe_fetch<RcMode>(c0, c0_dims, idx);
                            j_entry += ray_weight * sample;
                        },
                        j_sum
                    );
                    Kokkos::single(Kokkos::PerTeam(team), [&] () {
                        J(la, ks) += j_sum;
                    });
                }
            }
        }
    );

    return J;
}


#else
#endif