#include "Types.hpp"
#include "State.hpp"
#include "RcUtilsModes.hpp"
#include "MiscSparse.hpp"

void ProbesToCompute2d::init(
    const CascadeStorage& c0,
    bool sparse_,
    std::vector<yakl::Array<i32, 2, yakl::memDevice>> probes_to_compute
) {
    c0_size = c0;
    sparse = sparse_;
    if (sparse && probes_to_compute.size() == 0) {
        throw std::runtime_error("Sparse config selected, but set of probes_to_compute not provided");
    }
    active_probes = probes_to_compute;
}

void ProbesToCompute2d::init(
    const State& state,
    int max_cascades
) {
    const bool sparse_calc = state.config.sparse_calculation;
    CascadeStorage c0 = state.c0_size;
    std::vector<yakl::Array<i32, 2, yakl::memDevice>> active_probes;
    if (sparse_calc) {
        active_probes = compute_active_probe_lists(state, max_cascades);
    }
    this->init(c0, sparse_calc, active_probes);
}

i64 ProbesToCompute2d::num_active_probes(int cascade_idx) const {
    if (sparse) {
        // Throw an exception on out of bounds for easier debugging -- this isn't a fast path
        return active_probes.at(cascade_idx).extent(0);
    }
    CascadeStorage casc = cascade_size(c0_size, cascade_idx);
    return i64(casc.num_probes(0)) * i64(casc.num_probes(1));
}

DeviceProbesToCompute<2> ProbesToCompute2d::bind(int cascade_idx) const {
    if (sparse) {
        return DeviceProbesToCompute{
            .sparse = true,
            .active_probes = active_probes.at(cascade_idx)
        };
    }

    CascadeStorage casc = cascade_size(c0_size, cascade_idx);
    return DeviceProbesToCompute<2>{
        .sparse = false,
        .num_probes = casc.num_probes
    };
}
