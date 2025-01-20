#if !defined(DEXRT_MISC_SPARSE_HPP)
#define DEXRT_MISC_SPARSE_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "BlockMap.hpp"

SparseAtmosphere sparsify_atmosphere(const Atmosphere& atmos, const BlockMap<BLOCK_SIZE>& block_map);
KView<u8**> reify_active_c0(const BlockMap<BLOCK_SIZE>& block_map);
void rehydrate_page(
    const BlockMap<BLOCK_SIZE>& block_map,
    const FpConst2d& quantity,
    const Fp2d& qty_page,
    int page_idx
);

KView<fp_t***, HostSpace> rehydrate_sparse_quantity_host(
    const BlockMap<BLOCK_SIZE>& block_map,
    const Kokkos::View<const fp_t**, Layout, HostSpace>& quantity
);
/// Creates the full array associated with a sparse quantity on the host (for
/// output), input shape [n, ks], where n is arbitrary >= 1. Constructed by
/// paging each n at a time to reduce memory consumption.
template <typename ViewType>
inline auto rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const ViewType& quantity) -> KView<std::add_pointer_t<typename ViewType::non_const_data_type>, HostSpace>
 {
    static_assert(quantity.rank() == 2, "Must be rehydrating a 2D array");
    if constexpr(std::is_same_v<typename ViewType::memory_space, HostSpace> && !HostDevSameSpace) {
        return rehydrate_sparse_quantity_host(block_map, quantity);
    } else {
        const int num_x = block_map.num_x_tiles * BLOCK_SIZE;
        const int num_z = block_map.num_z_tiles * BLOCK_SIZE;
        KView<fp_t***, HostSpace> result(
            quantity.label(),
            quantity.extent(0),
            num_z,
            num_x
        );
        Fp2d qty_page("qty_page", num_z, num_x);

        for (int n = 0; n < quantity.extent(0); ++n) {
            rehydrate_page(block_map, quantity, qty_page, n);
            auto qty_page_host = Kokkos::create_mirror_view_and_copy(HostSpace{}, qty_page);

            for (int z = 0; z < num_z; ++z) {
                for (int x = 0; x < num_x; ++x) {
                    result(n, z, x) = qty_page_host(z, x);
                }
            }
        }
        return result;
    }
}

// template <typename ViewType>
// KView<fp_t***, HostSpace> rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const ViewType& quantity);

/// For a single 2d quantity (e.g. atmosphere)
// template <>
// KView<fp_t**, HostSpace> rehydrate_sparse_quantity<FpConst1d>(const BlockMap<BLOCK_SIZE>& block_map, const FpConst1d& quantity);

template <>
inline auto rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const FpConst1d& quantity) -> KView<fp_t**, HostSpace> {
    FpConst2d qtyx1(quantity.data(), 1, quantity.extent(0));
    Fp2d qty_page("qty_page", block_map.num_z_tiles * BLOCK_SIZE, block_map.num_x_tiles * BLOCK_SIZE);
    rehydrate_page(block_map, qtyx1, qty_page, 0);
    auto result = Kokkos::create_mirror_view_and_copy(HostSpace{}, qty_page);
    return result;
}

template <>
inline auto rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp1d& quantity) -> KView<fp_t**, HostSpace> {
    return rehydrate_sparse_quantity<FpConst1d>(block_map, quantity);
}

// template <typename ViewType>
// auto rehydrate_sparse_quanitity(const BlockMap<BLOCK_SIZE>& block_map, const ViewType& qty)

/// Compute the sets of active probes for each cascade
std::vector<KView<i32*[2]>> compute_active_probe_lists(const State& state, int max_cascades);


#else
#endif