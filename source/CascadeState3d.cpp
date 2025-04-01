#include "State3d.hpp"
#include "CascadeState3d.hpp"
#include "RcUtilsModes3d.hpp"
#include "MiscSparse.hpp"

bool CascadeState3d::init(const State3d& state, int max_cascades) {
    CascadeStorage3d c0 = state.c0_size;
    const bool sparse_calc = state.config.sparse_calculation;
    std::vector<ActiveProbeView<3>> active_probes;
    if (sparse_calc) {
        active_probes = compute_active_probe_lists(state, max_cascades);
    }
    probes_to_compute.init(c0, sparse_calc, active_probes);
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
            i_cascades.push_back(Fp1d("i_cascade", yakl::DimsT<i64>(max_entries)));
            Fp1d tau_entry;
            if constexpr (STORE_TAU_CASCADES) {
                tau_entry = Fp1d("tau_cascade", yakl::DimsT<i64>(max_entries));
            }
            tau_cascades.push_back(tau_entry);
        }
    } else {
        for (int i = 0; i <= max_cascades; ++i) {
            auto dims = cascade_size(c0, i);
            i64 entries = cascade_entries(dims);
            i_cascades.push_back(Fp1d("i_cascade", yakl::DimsT<i64>(entries)));
            Fp1d tau_entry;
            if constexpr (STORE_TAU_CASCADES) {
                tau_entry = Fp1d("tau_cascade", yakl::DimsT<i64>(entries));
            }
            tau_cascades.push_back(tau_entry);
        }
    }
    if (state.config.mode == DexrtMode::NonLte) {
        alo = Fp1d("ALO", yakl::DimsT<i64>(i_cascades[0].extent(0)));
    }
    mip_chain.init(state, state.mr_block_map.buffer_len());

    return true;
}

