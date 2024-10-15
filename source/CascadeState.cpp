#include "Types.hpp"
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
    return true;
}