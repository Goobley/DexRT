#include "CascadeState.hpp"
#include "State.hpp"
#include "RcUtilsModes.hpp"
#include "MiscSparse.hpp"

bool CascadeState::init(const State& state, int max_cascades) {
    const bool sparse_calc = state.config.sparse_calculation;
    CascadeStorage c0 = state.c0_size;
    std::vector<yakl::Array<i32, 2, yakl::memDevice>> active_probes;
    if (sparse_calc) {
        active_probes = compute_active_probe_lists(state, max_cascades);
    }
    probes_to_compute.init(c0, sparse_calc, active_probes);
    num_cascades = max_cascades;

    if constexpr (PINGPONG_BUFFERS) {
        i64 max_entries = 0;
        for (int i = 0; i <= max_cascades; ++i) {
            auto dims = cascade_size(c0, i);
            IntervalLength length = cascade_interval_length(max_cascades, i);
            state.println("Cascade {}: {}x{} {} directions [{}->{}]", i, dims.num_probes(0), dims.num_probes(1), dims.num_flat_dirs * PROBE0_NUM_RAYS, length.from, length.to);
            max_entries = std::max(max_entries, cascade_entries(dims));
        }
        for (int i = 0; i < 2; ++i) {
            i_cascades.push_back(
                Fp1d(
                    "i_cascade",
                    max_entries
                )
            );
            Fp1d tau_entry;
            if constexpr (STORE_TAU_CASCADES) {
                tau_entry = Fp1d("tau_cascade", max_entries);
            }
            tau_cascades.push_back(tau_entry);
        }
    } else {
        for (int i = 0; i <= max_cascades; ++i) {
            auto dims = cascade_size(c0, i);
            i64 entries = cascade_entries(dims);
            i_cascades.push_back(
                Fp1d(
                    "i_cascade",
                    entries
                )
            );
            Fp1d tau_entry;
            if constexpr (STORE_TAU_CASCADES) {
                tau_entry = Fp1d("tau_cascade", entries);
            }
            tau_cascades.push_back(tau_entry);
        }
    }
    if (state.config.mode == DexrtMode::NonLte) {
        alo = Fp1d("ALO", i_cascades[0].extent(0));
    }
    mip_chain.init(state, state.mr_block_map.buffer_len(), c0.wave_batch);

    if constexpr (RAYMARCH_TYPE == RaymarchType::LineSweep) {
        line_sweep_data = construct_line_sweep_data(state, max_cascades);
    }

    return true;
}