#include "ScDecomposition.hpp"
#include "Config.hpp"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fmt/core.h>

ScSlab sc_decompose_z(i32 global_num_z, const MpiState& mpi, i32 unit) {
    ScSlab slab;
    slab.global_num_z = global_num_z;
    slab.rank = mpi.rank;
    slab.num_ranks = mpi.num_ranks;
    if (unit < 1 || global_num_z % unit != 0) {
        throw std::runtime_error(fmt::format("SC z extent {} is not a multiple of the slab unit {}", global_num_z, unit));
    }
    const i32 num_units = global_num_z / unit;
    if (global_num_z < 2 || num_units < mpi.num_ranks || mpi.num_ranks < 1 || mpi.rank < 0 ||
        mpi.rank >= mpi.num_ranks) {
        throw std::runtime_error(fmt::format("SC needs at least one slab unit ({} rows) per rank", unit));
    }
    const i32 base = num_units / mpi.num_ranks;
    const i32 extra = num_units % mpi.num_ranks;
    slab.z_start = (mpi.rank * base + std::min(mpi.rank, extra)) * unit;
    slab.z_end = slab.z_start + (base + (mpi.rank < extra ? 1 : 0)) * unit;
    return slab;
}

fp_t sc_allreduce_max(const MpiState& mpi, fp_t value) {
#ifdef HAVE_MPI
    if (mpi.num_ranks > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, get_FpMpi(), MPI_MAX, mpi.comm);
    }
#endif
    return value;
}
