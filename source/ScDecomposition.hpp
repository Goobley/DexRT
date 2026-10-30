#if !defined(DEXRT_SC_DECOMPOSITION_HPP)
#define DEXRT_SC_DECOMPOSITION_HPP

#include "Types.hpp"
#include "MpiWrapper.hpp"
#include <type_traits>
#include <vector>

/// z-slab domain decomposition for the SC solver: rank r owns full-width rows
/// [z_start, z_end), ranks ordered upwards in z.
struct ScSlab {
    i32 global_num_z = 0;
    i32 z_start = 0;
    i32 z_end = 0;
    i32 rank = 0;
    i32 num_ranks = 1;

    bool has_lower() const { return rank > 0; }
    bool has_upper() const { return rank + 1 < num_ranks; }
    i32 num_z() const { return z_end - z_start; }
};

/// Equal slabs (to a unit) of whole units of rows, e.g. the activity tile
/// height, so that each rank's tiles are its own.
ScSlab sc_decompose_z(i32 global_num_z, const MpiState& mpi, i32 unit = 1);

fp_t sc_allreduce_max(const MpiState& mpi, fp_t value);

#else
#endif
