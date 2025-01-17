#if !defined(DEXRT_MISC_SPARSE_HPP)
#define DEXRT_MISC_SPARSE_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "BlockMap.hpp"

SparseAtmosphere sparsify_atmosphere(const Atmosphere& atmos, const BlockMap<BLOCK_SIZE>& block_map);
KView<u8**> reify_active_c0(const BlockMap<BLOCK_SIZE>& block_map);
/// Creates the full array associated with a sparse quantity on the host (for
/// output), input shape [n, ks], where n is arbitrary >= 1. Constructed by
/// paging each n at a time to reduce memory consumption.
KView<fp_t***, HostSpace> rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Kokkos::View<const fp_t**>& quantity);

/// For a single 2d quantity (e.g. atmosphere)
KView<fp_t**, HostSpace> rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const FpConst1d& quantity);

// template <typename ViewType>
// auto rehydrate_sparse_quanitity(const BlockMap<BLOCK_SIZE>& block_map, const ViewType& qty)

/// Compute the sets of active probes for each cascade
std::vector<KView<i32*[2]>> compute_active_probe_lists(const State& state, int max_cascades);


#else
#endif