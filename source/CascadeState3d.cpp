#include "State3d.hpp"
#include "CascadeState3d.hpp"
#include "RcUtilsModes3d.hpp"

bool CascadeState3d::init(const State3d& state, int max_cascades) {
    // TODO(cmo): Sparsity/active probes
    CascadeStorage3d c0 = state.c0_size;
    num_cascades = max_cascades;
    i64 max_entries = 0;
    for (int i = 0; i <= max_cascades; ++i) {
        auto dims = cascade_size(c0, i);
        auto rays = cascade_storage_to_rays<RC_flags_storage_3d()>(dims);
        IntervalLength length = cascade_interval_length_3d(max_cascades, i);
        fmt::println("Cascade {}: {}x{}x{} {}x{} directions [{}->{}]", i, rays.num_probes(0), rays.num_probes(1), rays.num_probes(2), rays.num_polar_rays, rays.num_az_rays, length.from, length.to);
        max_entries = std::max(max_entries, cascade_entries(dims));
    }

    if constexpr (PINGPONG_BUFFERS) {
        for (int i = 0; i < 2; ++i) {
            i_cascades.push_back(Fp1d("i_cascade", max_entries));
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
            i_cascades.push_back(Fp1d("i_cascade", entries));
            Fp1d tau_entry;
            if constexpr (STORE_TAU_CASCADES) {
                tau_entry = Fp1d("tau_cascade", entries);
            }
            tau_cascades.push_back(tau_entry);
        }
    }

    return true;
}

