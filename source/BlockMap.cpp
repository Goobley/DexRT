#include "BlockMap.hpp"

template <>
template <class BlockMap>
void BlockMapInit<2>::setup_dense<BlockMap>(BlockMap* map, Dims<2> dims) {
    constexpr i32 BLOCK_SIZE = map->BLOCK_SIZE;
    i32 x_size = dims.x;
    i32 z_size = dims.z;
    if (x_size % BLOCK_SIZE != 0 || z_size % BLOCK_SIZE != 0) {
        throw std::runtime_error("Grid is not a multiple of BLOCK_SIZE");
    }
    map->num_x_tiles() = x_size / BLOCK_SIZE;
    map->num_z_tiles() = z_size / BLOCK_SIZE;
    map->num_active_tiles = map->num_x_tiles() * map->num_z_tiles();
    map->bbox.min(0) = 0;
    map->bbox.min(1) = 0;
    map->bbox.max(0) = x_size;
    map->bbox.max(1) = z_size;

    const i32 num_x_tiles = map->num_x_tiles();
    const i32 num_z_tiles = map->num_z_tiles();

    yakl::Array<uint32_t, 1, yakl::memHost> morton_order("morton_traversal_order", num_x_tiles * num_z_tiles);
    for (int z = 0; z < num_z_tiles; ++z) {
        for (int x = 0; x < num_x_tiles; ++x) {
            morton_order(z * num_x_tiles + x) = encode_morton<NumDim>(Coord2{.x = x, .z = z});
        }
    }
    std::sort(morton_order.begin(), morton_order.end());
    map->morton_traversal_order = morton_order.createDeviceCopy();
    map->active_tiles = map->morton_traversal_order;

    map->lookup.init(Dims<2>{.x = num_x_tiles, .z = num_z_tiles});
    auto lookup_host = map->lookup.createHostCopy();
    i64 grid_idx = 0;
    for (int m_idx = 0; m_idx < morton_order.extent(0); ++m_idx) {
        Coord2 tile_index = decode_morton<NumDim>(morton_order(m_idx));
        lookup_host(tile_index) = grid_idx++;
    }
    map->lookup = lookup_host.createDeviceCopy();
}

template <>
template <class BlockMap>
void BlockMapInit<2>::setup_sparse<BlockMap>(BlockMap* map, const AtmosphereNd<2>& atmos, fp_t cutoff_temperature) {
    constexpr i32 BLOCK_SIZE = map->BLOCK_SIZE;
    if (atmos.temperature.extent(0) % BLOCK_SIZE != 0 || atmos.temperature.extent(1) % BLOCK_SIZE != 0) {
        throw std::runtime_error("Grid is not a multiple of BLOCK_SIZE");
    }
    map->num_x_tiles() = atmos.temperature.extent(1) / BLOCK_SIZE;
    map->num_z_tiles() = atmos.temperature.extent(0) / BLOCK_SIZE;
    map->num_active_tiles = 0;
    map->bbox.min(0) = 0;
    map->bbox.min(1) = 0;
    map->bbox.max(0) = atmos.temperature.extent(1);
    map->bbox.max(1) = atmos.temperature.extent(0);

    const int num_x_tiles = map->num_x_tiles();
    const int num_z_tiles = map->num_z_tiles();

    yakl::Array<uint32_t, 1, yakl::memHost> morton_order("morton_traversal_order", num_x_tiles * num_z_tiles);
    for (int z = 0; z < num_z_tiles; ++z) {
        for (int x = 0; x < num_x_tiles; ++x) {
            morton_order(z * num_x_tiles + x) = encode_morton<NumDim>(Coord2{.x = x, .z = z});
        }
    }
    std::sort(morton_order.begin(), morton_order.end());
    map->morton_traversal_order = morton_order.createDeviceCopy();

    map->lookup.init(Dims<2>{.x = num_x_tiles, .z = num_z_tiles});

    yakl::Array<bool, 2, yakl::memDevice> active("active tiles", num_z_tiles, num_x_tiles);
    bool all_active = cutoff_temperature == FP(0.0);
    if (all_active) {
        active = true;
        map->num_active_tiles = num_x_tiles * num_z_tiles;
    } else {
        auto& temperature = atmos.temperature;
        dex_parallel_for(
            "Compute active cells",
            FlatLoop<2>(num_z_tiles, num_x_tiles),
            YAKL_LAMBDA (int zt, int xt) {
                active(zt, xt) = false;
                for (int z = zt * BLOCK_SIZE; z < (zt + 1) * BLOCK_SIZE; ++z) {
                    for (int x = xt * BLOCK_SIZE; x < (xt + 1) * BLOCK_SIZE; ++x) {
                        if (temperature(z, x) <= cutoff_temperature) {
                            active(zt, xt) = true;
                            return;
                        }
                    }
                }
            }
        );
    }
    yakl::fence();
    auto active_host = active.createHostCopy();
    auto lookup_host = map->lookup.createHostCopy();
    i64 grid_idx = 0;

    for (int m_idx = 0; m_idx < morton_order.extent(0); ++m_idx) {
        Coord2 tile_index = decode_morton<NumDim>(morton_order(m_idx));
        if (active_host(tile_index.z, tile_index.x)) {
            // TODO(cmo): This is awful that the order needs to be swapped!
            lookup_host(Coord2{.x = tile_index.x, .z = tile_index.z}) = grid_idx++;
        }
    }
    map->num_active_tiles = grid_idx;
    fmt::println("Num active tiles: {}/{} ({:.1f} %)", map->num_active_tiles, num_z_tiles * num_x_tiles, fp_t(map->num_active_tiles) / fp_t(num_z_tiles * num_x_tiles) * FP(100.0));
    map->lookup = lookup_host.createDeviceCopy();

    if (all_active) {
        map->active_tiles = map->morton_traversal_order;
    } else {
        yakl::Array<uint32_t, 1, yakl::memHost> active_tiles_host("morton_traversal_active", map->num_active_tiles);
        int entry = 0;
        for (int m_idx = 0; m_idx < morton_order.extent(0); ++m_idx) {
            uint32_t code = morton_order(m_idx);
            Coord2 tile_index = decode_morton<NumDim>(morton_order(m_idx));
            if (active_host(tile_index.z, tile_index.x)) {
                active_tiles_host(entry++) = code;
            }
        }
        map->active_tiles = active_tiles_host.createDeviceCopy();
    }
}

template <>
template <class BlockMap>
void BlockMapInit<3>::setup_dense<BlockMap>(BlockMap* map, Dims<3> dims) {
}

template <>
template <class BlockMap>
void BlockMapInit<3>::setup_sparse<BlockMap>(BlockMap* map, const AtmosphereNd<3>& atmos, fp_t cutoff_temperature) {
}

// Ask the compiler to generate these specialisations only
template void BlockMapInit<2>::setup_sparse<BlockMap<BLOCK_SIZE, 2>>(BlockMap<BLOCK_SIZE, 2>*, const AtmosphereNd<2>&, fp_t);
template void BlockMapInit<2>::setup_dense<BlockMap<BLOCK_SIZE, 2>>(BlockMap<BLOCK_SIZE, 2>*, Dims<2>);
template void BlockMapInit<3>::setup_sparse<BlockMap<BLOCK_SIZE, 3>>(BlockMap<BLOCK_SIZE, 3>*, const AtmosphereNd<3>&, fp_t);
template void BlockMapInit<3>::setup_dense<BlockMap<BLOCK_SIZE, 3>>(BlockMap<BLOCK_SIZE, 3>*, Dims<3>);