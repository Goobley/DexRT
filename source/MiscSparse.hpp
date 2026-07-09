#if !defined(DEXRT_MISC_SPARSE_HPP)
#define DEXRT_MISC_SPARSE_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "BlockMap.hpp"

/// Sparisify 2d atmosphere (on GPU)
SparseAtmosphere sparsify_atmosphere(const Atmosphere& atmos, const BlockMap<BLOCK_SIZE>& block_map);
/// Sparsify 3d atmosphere on host and return sparse GPU atmosphere (copies one field to GPU at a time).
SparseAtmosphere sparsify_atmosphere(
    const AtmosphereNd<3, yakl::memHost>& atmos,
    const BlockMap<BLOCK_SIZE_3D, 3>& block_map
);
yakl::Array<u8, 2, yakl::memDevice> reify_active_c0(const BlockMap<BLOCK_SIZE>& block_map);
/// Creates the full array associated with a sparse quantity on the host (for
/// output), input shape [n, ks], where n is arbitrary >= 1. Constructed by
/// paging each n at a time to reduce memory consumption.
// Fp3dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp2d& quantity);
// Fp4dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE_3D, 3>& block_map, const Fp2d& quantity);
template <typename T>
yakl::Array<T, 3, yakl::memHost> rehydrate_sparse_quantity(
    const BlockMap<BLOCK_SIZE>& block_map,
    const yakl::Array<T, 2, yakl::memDevice>& quantity
);
template <typename T>
yakl::Array<T, 4, yakl::memHost> rehydrate_sparse_quantity(
    const BlockMap<BLOCK_SIZE_3D, 3>& block_map,
    const yakl::Array<T, 2, yakl::memDevice>& quantity
);

/// Same as previous but for a host array of shape [n, ks]
// Fp3dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp2dHost& quantity);
// Fp4dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE_3D, 3>& block_map, const Fp2dHost& quantity);
template <typename T>
yakl::Array<T, 3, yakl::memHost> rehydrate_sparse_quantity(
    const BlockMap<BLOCK_SIZE>& block_map,
    const yakl::Array<T, 2, yakl::memHost>& quantity
);
template <typename T>
yakl::Array<T, 4, yakl::memHost> rehydrate_sparse_quantity(
    const BlockMap<BLOCK_SIZE_3D, 3>& block_map,
    const yakl::Array<T, 2, yakl::memHost>& quantity
);

/// For a single 2d quantity (e.g. atmosphere)
// Fp2dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp1d& quantity);
// Fp3dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE_3D, 3>& block_map, const Fp1d& quantity);
template <typename T>
yakl::Array<T, 2, yakl::memHost> rehydrate_sparse_quantity(
    const BlockMap<BLOCK_SIZE>& block_map,
    const yakl::Array<T, 1, yakl::memDevice>& quantity
);
template <typename T>
yakl::Array<T, 3, yakl::memHost> rehydrate_sparse_quantity(
    const BlockMap<BLOCK_SIZE_3D, 3>& block_map,
    const yakl::Array<T, 1, yakl::memDevice>& quantity
);

/// Compute the sets of active probes for each cascade
std::vector<yakl::Array<i32, 2, yakl::memDevice>> compute_active_probe_lists(const State& state, int max_cascades);
std::vector<yakl::Array<Coord3, 1, yakl::memDevice>> compute_active_probe_lists(const State3d& state, int max_cascades);


#else
#endif