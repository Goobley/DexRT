#if !defined(DEXRT_BLOCKMAP_HPP)
#define DEXRT_BLOCKMAP_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include "MortonCodes.hpp"
#include <fmt/core.h>

// NOTE(cmo): Whether to assume that GridBbox.min is always 0 (in practice, yes).
constexpr bool assume_lower_tile_bound_0 = true;

template <int NumDim=2, int mem_space=yakl::memDevice>
struct BlockMapLookup {
    // NOTE(cmo): This is a separate class so it can be switch to e.g. Morton ordering
    yakl::Array<i64, NumDim, mem_space> entries;

    void init(const Dims<NumDim>& d);

    YAKL_INLINE i64& operator()(const Coord<NumDim>& c) const;

    BlockMapLookup<NumDim, yakl::memHost> createHostCopy() {
        if constexpr (mem_space == yakl::memHost) {
            return *this;
        }

        BlockMapLookup<NumDim, yakl::memHost> result;
        result.entries = entries.createHostCopy();
        return result;
    }

    BlockMapLookup<NumDim, yakl::memDevice> createDeviceCopy() {
        if constexpr (mem_space == yakl::memDevice) {
            return *this;
        }

        BlockMapLookup<NumDim, yakl::memDevice> result;
        result.entries = entries.createDeviceCopy();
        return result;
    }
};

// NOTE(cmo): We can't do partial template specialisation here, so we need copies for both mem_spaces.
template <>
inline void BlockMapLookup<2, yakl::memHost>::init(const Dims<2>& c) {
    entries = decltype(entries)("BlockMapEntries", c.z, c.x);
    entries = -1;
    yakl::fence();
}

template <>
inline void BlockMapLookup<2, yakl::memDevice>::init(const Dims<2>& c) {
    entries = decltype(entries)("BlockMapEntries", c.z, c.x);
    entries = -1;
    yakl::fence();
}

template <>
YAKL_INLINE i64& BlockMapLookup<2, yakl::memHost>::operator()(const Coord<2>& c) const {
    return entries(c.z, c.x);
}

template <>
YAKL_INLINE i64& BlockMapLookup<2, yakl::memDevice>::operator()(const Coord<2>& c) const {
    return entries(c.z, c.x);
}

template <>
inline void BlockMapLookup<3, yakl::memHost>::init(const Dims<3>& c) {
    entries = decltype(entries)("BlockMapEntries", c.z, c.y, c.x);
    entries = -1;
    yakl::fence();
}

template <>
inline void BlockMapLookup<3, yakl::memDevice>::init(const Dims<3>& c) {
    entries = decltype(entries)("BlockMapEntries", c.z, c.y, c.x);
    entries = -1;
    yakl::fence();
}

template <>
YAKL_INLINE i64& BlockMapLookup<3, yakl::memHost>::operator()(const Coord<3>& c) const {
    return entries(c.z, c.y, c.x);
}

template <>
YAKL_INLINE i64& BlockMapLookup<3, yakl::memDevice>::operator()(const Coord<3>& c) const {
    return entries(c.z, c.y, c.x);
}

/// Used to index the num_tiles member of BlockMap etc
template <i32 NumDim>
struct DimIndex {
    constexpr static i32 x = -1;
    constexpr static i32 y = -1;
    constexpr static i32 z = -1;
};

template<>
struct DimIndex<2> {
    constexpr static i32 x = 0;
    constexpr static i32 y = -1;
    constexpr static i32 z = 1;
};

template<>
struct DimIndex<3> {
    constexpr static i32 x = 0;
    constexpr static i32 y = 1;
    constexpr static i32 z = 2;
};

namespace DexImpl {
    template <int N=0>
    KOKKOS_FORCEINLINE_FUNCTION int int_pow(int x) {
        static_assert(N==0, "int_pow only defined for N = 1, 2, 3");
        return 1;
    }

    template <>
    KOKKOS_FORCEINLINE_FUNCTION int int_pow<1>(int x) {
        return x;
    }

    template <>
    KOKKOS_FORCEINLINE_FUNCTION int int_pow<2>(int x) {
        return x * x;
    }

    template <>
    KOKKOS_FORCEINLINE_FUNCTION int int_pow<3>(int x) {
        return x * x * x;
    }
}

// NOTE(cmo): This forward declare is a bit messy, but needed to let us define these in a different TU;
template<i32 BLOCK_SIZE, i32 NumDim, class Lookup>
struct BlockMap;

template <i32 NumDim>
struct BlockMapInit {
    template <class BlockMap>
    static void setup_dense(BlockMap*, Dims<NumDim>);

    template <class BlockMap, int mem_space>
    static void setup_sparse(BlockMap*, const AtmosphereNd<NumDim, mem_space>&, fp_t);
};
extern template struct BlockMapInit<2>;
extern template struct BlockMapInit<3>;

template <i32 BlockSize, i32 NumDim=2, class Lookup=BlockMapLookup<NumDim>>
struct BlockMap {
    static constexpr i32 BLOCK_SIZE = BlockSize;
    yakl::SArray<i32, 1, NumDim> num_tiles;
    i32 num_active_tiles;
    GridBbox<NumDim> bbox;

    Lookup lookup;
    /// All tiles in Morton traversal order
    yakl::Array<uint32_t, 1, yakl::memDevice> morton_traversal_order;
    /// All active tiles in Morton traversal order
    yakl::Array<uint32_t, 1, yakl::memDevice> active_tiles;

    template <int mem_space>
    void init(const AtmosphereNd<NumDim, mem_space>& atmos, fp_t cutoff_temperature) {
        BlockMapInit<NumDim>::setup_sparse(this, atmos, cutoff_temperature);
    }

    /// Everything active, no atmosphere, used for given fs
    void init(Dims<NumDim> dims) {
        BlockMapInit<NumDim>::setup_dense(this, dims);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32& num_x_tiles() {
        return num_tiles(DimIndex<NumDim>::x);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32 num_x_tiles() const {
        return num_tiles(DimIndex<NumDim>::x);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32& num_y_tiles() {
        if constexpr (NumDim == 2) {
            KOKKOS_ASSERT(false);
        }
        return num_tiles(DimIndex<NumDim>::y);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32 num_y_tiles() const {
        if constexpr (NumDim == 2) {
            KOKKOS_ASSERT(false);
        }
        return num_tiles(DimIndex<NumDim>::y);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32& num_z_tiles() {
        return num_tiles(DimIndex<NumDim>::z);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32 num_z_tiles() const {
        return num_tiles(DimIndex<NumDim>::z);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i64 get_num_active_cells() const {
        return i64(num_active_tiles) * DexImpl::int_pow<NumDim>(BLOCK_SIZE);
    }

    KOKKOS_FORCEINLINE_FUNCTION
     i64 buffer_len(i32 mip_px_size=1) const {
        using DexImpl::int_pow;
        return i64(num_active_tiles) * i64(int_pow<NumDim>(BLOCK_SIZE) / int_pow<NumDim>(mip_px_size));
    }

    FlatLoop<2> loop_bounds(i32 mip_px_size=1) const {
        using DexImpl::int_pow;
        return FlatLoop<2>(
            num_active_tiles,
            int_pow<NumDim>(BLOCK_SIZE) / int_pow<NumDim>(mip_px_size)
        );
    }
};

template <int NumDim=2, u8 entry_size=3, int mem_space=yakl::memDevice>
struct MultiLevelLookup {
    static constexpr u8 packed_entries_per_u64 = (sizeof(u64) * CHAR_BIT) / entry_size;
    static constexpr u64 lowest_entry_mask = ((1 << entry_size) - 1);
    static_assert(entry_size <= 8, "Entry size must be <= 8");
    /// The number of hyper-tiles if HYPERBLOCK2x2 is true
    yakl::SArray<i32, 1, NumDim> num_tiles;
    // NOTE(cmo): We're still laying out these tiles linearly here (unless
    // hyper_blocks are used)... not ideal, but it has a very small footprint --
    // should remain resident in cache.
    yakl::Array<u64, 1, mem_space> entries;

    template <class BlockMap, int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    void init(const BlockMap& block_map) {
        num_x_tiles() = block_map.num_x_tiles();
        num_z_tiles() = block_map.num_z_tiles();
        if constexpr (HYPERBLOCK2x2) {
            if (num_z_tiles() % 2 == 1 || num_x_tiles() % 2 == 1) {
                throw std::runtime_error(fmt::format("Must have an even number of tiles when using hyperblocking, i.e. make your model a multiple of {} cells.", BLOCK_SIZE*2));
            }
            num_x_tiles() /= 2;
            num_z_tiles() /= 2;
        }
        const i32 num_entries = block_map.num_x_tiles() * block_map.num_z_tiles();
        const i32 storage_for_entries = (num_entries + packed_entries_per_u64 - 1) / packed_entries_per_u64;
        entries = decltype(entries)("MultiLevel Entries", storage_for_entries);
    }

    template <class BlockMap, int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    void init(const BlockMap& block_map) {
        num_x_tiles() = block_map.num_x_tiles();
        num_y_tiles() = block_map.num_y_tiles();
        num_z_tiles() = block_map.num_z_tiles();
        const i32 num_entries = block_map.num_x_tiles() * block_map.num_y_tiles() * block_map.num_z_tiles();
        const i32 storage_for_entries = (num_entries + packed_entries_per_u64 - 1) / packed_entries_per_u64;
        entries = decltype(entries)("MultiLevel Entries", storage_for_entries);
    }

    // 2D
    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE i32 flat_tile_index(Coord<NumDim> c) const {
        i32 flat_idx;
        if constexpr (HYPERBLOCK2x2) {
            const i32 hyper_tile_idx = (c.z >> 1) * num_x_tiles() + (c.x >> 1);
            constexpr i32 hyper_tile_size = 4;
            flat_idx = hyper_tile_idx * hyper_tile_size + ((c.z & 1) << 1) + (c.x & 1);
        } else {
            flat_idx = c.z * num_x_tiles() + c.x;
        }
        return flat_idx;
    }

    // 3D
    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE i32 flat_tile_index(Coord<NumDim> c) const {
        // NOTE(cmo): Hyperblocking not supported in 3D
        i32 flat_idx = c.z * (num_x_tiles() * num_y_tiles()) + c.y * (num_x_tiles()) + c.x;
        return flat_idx;
    }

    /// Returns the mip level + 1 for a particular coordinate (i.e. 0 means no data)
    YAKL_INLINE u8 get(Coord<NumDim> c) const {
        const i32 flat_idx = flat_tile_index(c);
        const i32 block_idx = flat_idx / packed_entries_per_u64;
        const i32 entry_idx = flat_idx % packed_entries_per_u64;

        const u64 block = entries(block_idx);
        u8 entry = (block >> (entry_size * entry_idx)) & lowest_entry_mask;
        return entry;
    }

    // NOTE(cmo): This function is never used, but its definition is quite clear
    // and could be useful in the future
    // /// NOTE(cmo): NOT THREADSAFE -- should probably be host only?
    // YAKL_INLINE void set(i32 x, i32 z, u8 val) {
    //     const i32 flat_idx = flat_tile_index(x, z);
    //     const i32 block_idx = flat_idx / packed_entries_per_u64;
    //     const i32 entry_idx = flat_idx % packed_entries_per_u64;

    //     const u64 shifted_entry = u64(val) << (entry_idx * entry_size); // ---- --ab ----
    //     const u64 shifted_mask = ~(lowest_entry_mask << (entry_idx * entry_size)); // ffff ff00 ffff
    //     const u64 block = entries(block_idx); // 0123 4567 89ab
    //     const u64 updated_block = (block & shifted_mask) | shifted_entry; // 0123 45ab 89ab
    //     entries(block_idx) = updated_block;
    // }

    /// Pack an array of entries into the u64 storage backing array. This isn't
    /// really const, but it only changes the array contents... ehhhhh
    template <typename T>
    void pack_entries(const yakl::Array<T, 1, memDevice>& single_entries) const {
        JasUnpack((*this), entries);
        dex_parallel_for(
            "Pack T entries into u64",
            FlatLoop<1>(entries.extent(0)),
            YAKL_LAMBDA (int block_idx) {
                u64 block = 0;
                const int max_entry = std::min(
                    i32(single_entries.extent(0) - block_idx * packed_entries_per_u64),
                    i32(packed_entries_per_u64)
                );
                for (int entry_idx = 0; entry_idx < max_entry; ++entry_idx) {
                    const int flat_entry_idx = block_idx * packed_entries_per_u64 + entry_idx;
                    block |= (u64(single_entries(flat_entry_idx)) & lowest_entry_mask) << (entry_idx * entry_size);
                }
                entries(block_idx) = block;
            }
        );
        yakl::fence();
    }

    MultiLevelLookup<NumDim, entry_size, yakl::memHost> createHostCopy() {
        if constexpr (mem_space == yakl::memHost) {
            return *this;
        }

        MultiLevelLookup<NumDim, entry_size, yakl::memHost> result;
        result.num_tiles = num_tiles;
        result.entries = entries.createHostCopy();
        return result;
    }

    MultiLevelLookup<NumDim, entry_size, yakl::memDevice> createDeviceCopy() {
        if constexpr (mem_space == yakl::memDevice) {
            return *this;
        }

        MultiLevelLookup<NumDim, entry_size, yakl::memDevice> result;
        result.num_tiles = num_tiles;
        result.entries = entries.createDeviceCopy();
        return result;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32& num_x_tiles() {
        return num_tiles(DimIndex<NumDim>::x);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32 num_x_tiles() const {
        return num_tiles(DimIndex<NumDim>::x);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32& num_y_tiles() {
        if constexpr (NumDim == 2) {
            KOKKOS_ASSERT(false);
        }
        return num_tiles(DimIndex<NumDim>::y);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32 num_y_tiles() const {
        if constexpr (NumDim == 2) {
            KOKKOS_ASSERT(false);
        }
        return num_tiles(DimIndex<NumDim>::y);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32& num_z_tiles() {
        return num_tiles(DimIndex<NumDim>::z);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    i32 num_z_tiles() const {
        return num_tiles(DimIndex<NumDim>::z);
    }
};

/// NOTE(cmo): Instead of storing the tile index (which is stored in the
/// blockmap), this stores a MultiLevelLookup of packed tile information.
/// When unpacked to a u8 via lookup.get, if 0, the tile is empty, otherwise
/// val-1 is its associated mip level (log2 voxel block size). The max mip level
/// is then clamped when traversing in MultiLevelDda. This struct provides the
/// length of an array for storing all of the mips of the sparse tiles.
template <int BLOCK_SIZE, int ENTRY_SIZE=3, int NumDim=2, class Lookup=BlockMapLookup<NumDim>, class BlockMap=BlockMap<BLOCK_SIZE, NumDim, Lookup>>
struct MultiResBlockMap {
    static constexpr i32 max_storable_entry = (1 << ENTRY_SIZE) - 1;
    static constexpr i32 block_size = BLOCK_SIZE;
    i32 max_mip_level;
    yakl::SArray<i64, 1, max_storable_entry> mip_offsets;
    BlockMap block_map;
    MultiLevelLookup<NumDim, ENTRY_SIZE> lookup;

    void init(const BlockMap& block_map_, i32 max_mip_level_) {
        block_map = block_map_;
        lookup.init(block_map);
        max_mip_level = max_mip_level_;
        fmt::println("Using a max mip level of {}", max_mip_level);
        if ((max_mip_level + 1) > max_storable_entry) {
            throw std::runtime_error("More mip levels requested than storable in packed data. Increase ENTRY_SIZE");
        }

        for (int i = 0; i < mip_offsets.size(); ++i) {
            mip_offsets(i) = 0;
        }
        i64 buffer_len_acc = 0;
        for (int i = 0; i <= max_mip_level; ++i) {
            mip_offsets(i) = buffer_len_acc;
            buffer_len_acc += block_map.buffer_len(1 << i);
        }
    }

    YAKL_INLINE i64 buffer_len() const {
        return mip_offsets(max_mip_level) + block_map.buffer_len(1 << max_mip_level);
    }

    YAKL_INLINE
    i64 get_num_active_cells() const {
        return block_map.get_num_active_cells();
    }
};

constexpr bool INNER_MORTON_LOOKUP = false;
template<i32 BLOCK_SIZE, i32 NumDim=2>
struct IndexGen {
    Coord<NumDim> tile_key;
    i64 tile_base_idx;
    const BlockMap<BLOCK_SIZE, NumDim>& block_map;

    static constexpr i32 L2_BLOCK_SIZE = std::bit_width(u32(BLOCK_SIZE)) - 1;

    YAKL_INLINE
    IndexGen(const BlockMap<BLOCK_SIZE, NumDim>& block_map_) :
        tile_key(), // Now auto initialises to -1
        tile_base_idx(),
        block_map(block_map_)
    {}

    template <int entry_size>
    YAKL_INLINE
    IndexGen(const MultiResBlockMap<BLOCK_SIZE, entry_size, NumDim>& mr_block_map) :
        tile_key(),
        tile_base_idx(),
        block_map(mr_block_map.block_map)
    {}

    YAKL_INLINE
    Coord<NumDim> compute_tile_coord(i64 tile_idx) const {
        return decode_morton<NumDim>(block_map.active_tiles(tile_idx));
    }

    /// Returns the index of the storage associated with the first element of
    /// the nth tile
    YAKL_INLINE
    i64 compute_base_idx(i64 tile_idx) const {
        return tile_idx * DexImpl::int_pow<NumDim>(BLOCK_SIZE);
    }

    /// Convert from inner tile flat index to tile-local coordinates (these
    /// include the effects of being on a mip level)
    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    Coord<NumDim> compute_tile_inner_offset(i32 tile_offset) const {
        Coord2 coord;
        if constexpr (INNER_MORTON_LOOKUP) {
            coord = decode_morton<NumDim>(uint32_t(tile_offset));
        } else {
            coord.z = tile_offset / BLOCK_SIZE;
            coord.x = tile_offset - BLOCK_SIZE * coord.z;
        }
        return coord;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    Coord<NumDim> compute_tile_inner_offset(i32 tile_offset) const {
        Coord3 coord;
        i32 stride = square(BLOCK_SIZE);
        coord.z = tile_offset / stride;
        i32 lower_dims = coord.z * stride;
        stride = BLOCK_SIZE;
        coord.y = (tile_offset - lower_dims) / stride;
        lower_dims += coord.y * stride;
        coord.x = tile_offset - lower_dims;

        return coord;
    }

    /// Convert from tile-local coordinates to flat index
    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    i32 compute_inner_offset(Coord<NumDim> c) const {
        if constexpr (INNER_MORTON_LOOKUP) {
            return encode_morton<NumDim>(c);
        } else {
            return (c.z * BLOCK_SIZE + c.x);
        }
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    i32 compute_inner_offset(Coord<NumDim> c) const {
        return (c.z * BLOCK_SIZE + c.y) * BLOCK_SIZE + c.x;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    i64 idx(Coord<NumDim> c) {
        i32 tile_x = c.x >> L2_BLOCK_SIZE;
        i32 tile_z = c.z >> L2_BLOCK_SIZE;
        i32 inner_x = c.x - tile_x * BLOCK_SIZE;
        i32 inner_z = c.z - tile_z * BLOCK_SIZE;
        // i32 inner_x = c.x % BLOCK_SIZE;
        // i32 inner_z = c.z % BLOCK_SIZE;
        Coord2 tile_key_lookup{
            .x = tile_x,
            .z = tile_z
        };

        if (tile_key == tile_key_lookup) {
            return tile_base_idx + compute_inner_offset(Coord2{.x = inner_x, .z = inner_z});
        }

        i64 tile_idx = block_map.lookup(tile_key_lookup);
#ifdef DEXRT_DEBUG
        if (tile_idx < 0) {
            yakl::yakl_throw("OOB block requested!");
        }
#endif
        if (tile_idx >= 0) {
            tile_base_idx = compute_base_idx(tile_idx);
            tile_key = tile_key_lookup;
            return tile_base_idx + compute_inner_offset(Coord2{.x = inner_x, .z = inner_z});
        }
        return -1;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    i64 idx(Coord<NumDim> c) {
        i32 tile_x = c.x >> L2_BLOCK_SIZE;
        i32 tile_y = c.y >> L2_BLOCK_SIZE;
        i32 tile_z = c.z >> L2_BLOCK_SIZE;
        i32 inner_x = c.x - tile_x * BLOCK_SIZE;
        i32 inner_y = c.y - tile_y * BLOCK_SIZE;
        i32 inner_z = c.z - tile_z * BLOCK_SIZE;

        Coord3 tile_key_lookup{
            .x = tile_x,
            .y = tile_y,
            .z = tile_z
        };

        if (tile_key == tile_key_lookup) {
            return tile_base_idx + compute_inner_offset(Coord3{.x = inner_x, .y = inner_y, .z = inner_z});
        }

        i64 tile_idx = block_map.lookup(tile_key_lookup);
#ifdef DEXRT_DEBUG
        if (tile_idx < 0) {
            yakl::yakl_throw("OOB block requested!");
        }
#endif
        if (tile_idx >= 0) {
            tile_base_idx = compute_base_idx(tile_idx);
            tile_key = tile_key_lookup;
            return tile_base_idx + compute_inner_offset(Coord3{.x = inner_x, .y = inner_y, .z = inner_z});
        }
        return -1;
    }


    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    bool has_leaves(Coord<NumDim> c) {
        i32 tile_x = c.x >> L2_BLOCK_SIZE;
        i32 tile_z = c.z >> L2_BLOCK_SIZE;
        Coord2 tile_key_lookup{
            .x = tile_x,
            .z = tile_z
        };

        if (tile_key == tile_key_lookup) {
            return true;
        }
        if constexpr (assume_lower_tile_bound_0) {
            if (
                (u32(c.x) >= block_map.bbox.max(DimIndex<NumDim>::x)) ||
                (u32(c.z) >= block_map.bbox.max(DimIndex<NumDim>::z))
            ) {
                return false;
            }
        } else {
            if (
                (c.x < 0) || (c.x >= block_map.bbox.max(DimIndex<NumDim>::x)) ||
                (c.z < 0) || (c.z >= block_map.bbox.max(DimIndex<NumDim>::z))
            ) {
                return false;
            }
        }
        i64 tile_idx = block_map.lookup(tile_key_lookup);
        bool result = tile_idx != -1;
        if (result) {
            tile_base_idx = compute_base_idx(tile_idx);
            tile_key = tile_key_lookup;
        }
        return result;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    bool has_leaves(Coord<NumDim> c) {
        i32 tile_x = c.x >> L2_BLOCK_SIZE;
        i32 tile_y = c.y >> L2_BLOCK_SIZE;
        i32 tile_z = c.z >> L2_BLOCK_SIZE;
        Coord3 tile_key_lookup{
            .x = tile_x,
            .y = tile_y,
            .z = tile_z
        };

        if (tile_key == tile_key_lookup) {
            return true;
        }
        if constexpr (assume_lower_tile_bound_0) {
            if (
                u32(c.x) >= block_map.bbox.max(DimIndex<NumDim>::x) ||
                u32(c.y) >= block_map.bbox.max(DimIndex<NumDim>::y) ||
                u32(c.z) >= block_map.bbox.max(DimIndex<NumDim>::z)
            ) {
                return false;
            }
        } else {
            if (
                (c.x < block_map.bbox.min(DimIndex<NumDim>::x)) || (c.x >= block_map.bbox.max(DimIndex<NumDim>::x)) ||
                (c.y < block_map.bbox.min(DimIndex<NumDim>::y)) || (c.y >= block_map.bbox.max(DimIndex<NumDim>::y)) ||
                (c.z < block_map.bbox.min(DimIndex<NumDim>::z)) || (c.z >= block_map.bbox.max(DimIndex<NumDim>::z))
            ) {
                return false;
            }
        }

        i64 tile_idx = block_map.lookup(tile_key_lookup);
        bool result = tile_idx != -1;
        if (result) {
            tile_base_idx = compute_base_idx(tile_idx);
            tile_key = tile_key_lookup;
        }
        return result;
    }

    YAKL_INLINE
    i64 loop_idx(i64 tile_idx, i32 block_idx) const {
        return compute_base_idx(tile_idx) + block_idx;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    Coord2 loop_coord(i64 tile_idx, i32 block_idx) const {
        Coord2 tile_coord = compute_tile_coord(tile_idx);
        Coord2 tile_offset = compute_tile_inner_offset(block_idx);
        return Coord2 {
            .x = tile_coord.x * BLOCK_SIZE + tile_offset.x,
            .z = tile_coord.z * BLOCK_SIZE + tile_offset.z
        };
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    Coord3 loop_coord(i64 tile_idx, i32 block_idx) const {
        Coord3 tile_coord = compute_tile_coord(tile_idx);
        Coord3 tile_offset = compute_tile_inner_offset(block_idx);
        return Coord3 {
            .x = tile_coord.x * BLOCK_SIZE + tile_offset.x,
            .y = tile_coord.y * BLOCK_SIZE + tile_offset.y,
            .z = tile_coord.z * BLOCK_SIZE + tile_offset.z
        };
    }
};

template <int NumDim=2>
struct MultiLevelTileKey {
    i32 mip_level;
    Coord<NumDim> coord;

    YAKL_INLINE bool operator==(const MultiLevelTileKey& other) const {
        return (mip_level == other.mip_level) && (coord == other.coord);
    }
};

// TODO(cmo): This approach won't work when distributed... we'll likely need to
// add another layer of indirection, or, if we distribute by morton order, a
// base index that is subtracted per level per node
template<i32 BLOCK_SIZE, u8 ENTRY_SIZE, i32 NumDim=2>
struct MultiLevelIndexGen {
    MultiLevelTileKey<NumDim> tile_key;
    i64 tile_base_idx;
    const BlockMap<BLOCK_SIZE, NumDim>& block_map;
    const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE, NumDim>& mip_block_map;

    static constexpr i32 L2_BLOCK_SIZE = std::bit_width(u32(BLOCK_SIZE)) - 1;

    YAKL_INLINE
    MultiLevelIndexGen(
        const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE, NumDim>& mip_block_map_
    ) :
        tile_key(MultiLevelTileKey<NumDim>{.mip_level = -1, .coord=Coord<NumDim>{}}),
        tile_base_idx(),
        block_map(mip_block_map_.block_map),
        mip_block_map(mip_block_map_)
    {}

    YAKL_INLINE
    Coord<NumDim> compute_tile_coord(i64 tile_idx) const {
        return decode_morton<NumDim>(block_map.active_tiles(tile_idx));
    }

    YAKL_INLINE
    i64 compute_base_idx(i32 mip_level, i64 tile_idx) const {
        return mip_block_map.mip_offsets(mip_level) + tile_idx * DexImpl::int_pow<NumDim>(BLOCK_SIZE >> mip_level);
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    Coord<NumDim> compute_tile_inner_offset(i32 mip_level, i32 tile_offset) const {
        Coord2 coord;
        if constexpr (INNER_MORTON_LOOKUP) {
            coord = decode_morton<NumDim>(uint32_t(tile_offset));
        } else {
            coord.z = tile_offset / (BLOCK_SIZE >> mip_level);
            coord.x = tile_offset - (BLOCK_SIZE >> mip_level) * coord.z;
            // coord = Coord2 {
            //     .x = tile_offset % (BLOCK_SIZE >> mip_level),
            //     .z = tile_offset / (BLOCK_SIZE >> mip_level)
            // };
        }
        coord.x <<= mip_level;
        coord.z <<= mip_level;
        return coord;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    Coord<NumDim> compute_tile_inner_offset(i32 mip_level, i32 tile_offset) const {
        Coord3 coord;
        i32 stride = square(BLOCK_SIZE >> mip_level);
        coord.z = tile_offset / stride;
        i32 lower_dims = coord.z * stride;
        stride = BLOCK_SIZE >> mip_level;
        coord.y = (tile_offset - lower_dims) / stride;
        lower_dims += coord.y * stride;
        coord.x = tile_offset - lower_dims;

        coord.x <<= mip_level;
        coord.y <<= mip_level;
        coord.z <<= mip_level;
        return coord;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    i32 compute_inner_offset(i32 mip_level, Coord<NumDim> inner) const {
        Coord<NumDim> mip_coord {
            .x = inner.x >> mip_level,
            .z = inner.z >> mip_level
        };
        if constexpr (INNER_MORTON_LOOKUP) {
            return encode_morton<NumDim>(mip_coord);
        } else {
            return mip_coord.z * (BLOCK_SIZE >> mip_level) + mip_coord.x;
        }
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    i32 compute_inner_offset(i32 mip_level, Coord<NumDim> inner) const {
        Coord<NumDim> mip_coord {
            .x = inner.x >> mip_level,
            .y = inner.y >> mip_level,
            .z = inner.z >> mip_level
        };
        const i32 reduced_block_size = BLOCK_SIZE >> mip_level;
        return (mip_coord.z * reduced_block_size + mip_coord.y) * reduced_block_size + mip_coord.x;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    i64 idx(i32 mip_level, Coord<NumDim> c) {
        // NOTE(cmo): The shift vs integer division may seem like premature
        // optimisation here. I assure you it isn't. It properly captures the
        // case of x and z being small negatives. Due to integer rules -1 / 16
        // == 0, but -1 >> 4 == -1
        i32 tile_x = c.x >> L2_BLOCK_SIZE;
        i32 tile_z = c.z >> L2_BLOCK_SIZE;
        i32 inner_x = c.x - tile_x * BLOCK_SIZE;
        i32 inner_z = c.z - tile_z * BLOCK_SIZE;
        // i32 inner_x = c.x % BLOCK_SIZE;
        // i32 inner_z = c.z % BLOCK_SIZE;
        MultiLevelTileKey<NumDim> tile_key_lookup{
            .mip_level = mip_level,
            .coord {
                .x = tile_x,
                .z = tile_z
            }
        };

        if (tile_key == tile_key_lookup) {
            return tile_base_idx + compute_inner_offset(mip_level, Coord2{.x = inner_x, .z = inner_z});
        }

        i64 tile_idx = block_map.lookup(Coord2{.x = tile_x, .z = tile_z});
#ifdef DEXRT_DEBUG
        if (tile_idx < 0) {
            yakl::yakl_throw("OOB block requested!");
        }
#endif
        if (tile_idx >= 0) {
            tile_base_idx = compute_base_idx(mip_level, tile_idx);
            tile_key = tile_key_lookup;
            return tile_base_idx + compute_inner_offset(mip_level, Coord2{.x = inner_x, .z = inner_z});
        }
        return -1;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    i64 idx(i32 mip_level, Coord<NumDim> c) {
        // NOTE(cmo): The shift vs integer division may seem like premature
        // optimisation here. I assure you it isn't. It properly captures the
        // case of x and z being small negatives. Due to integer rules -1 / 16
        // == 0, but -1 >> 4 == -1
        i32 tile_x = c.x >> L2_BLOCK_SIZE;
        i32 tile_y = c.y >> L2_BLOCK_SIZE;
        i32 tile_z = c.z >> L2_BLOCK_SIZE;
        i32 inner_x = c.x - tile_x * BLOCK_SIZE;
        i32 inner_y = c.y - tile_y * BLOCK_SIZE;
        i32 inner_z = c.z - tile_z * BLOCK_SIZE;

        MultiLevelTileKey<NumDim> tile_key_lookup{
            .mip_level = mip_level,
            .coord {
                .x = tile_x,
                .y = tile_y,
                .z = tile_z
            }
        };

        if (tile_key == tile_key_lookup) {
            return tile_base_idx + compute_inner_offset(
                mip_level,
                Coord3{.x = inner_x, .y = inner_y, .z = inner_z}
            );
        }

        i64 tile_idx = block_map.lookup(tile_key_lookup.coord);
#ifdef DEXRT_DEBUG
        if (tile_idx < 0) {
            yakl::yakl_throw("OOB block requested!");
        }
#endif
        if (tile_idx >= 0) {
            tile_base_idx = compute_base_idx(mip_level, tile_idx);
            tile_key = tile_key_lookup;
            return tile_base_idx + compute_inner_offset(
                mip_level,
                Coord3{.x = inner_x, .y = inner_y, .z = inner_z}
            );
        }
        return -1;
    }

    /// Returns the mip level to sample on, or -1 if empty
    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    i32 get_sample_level(Coord<NumDim> c) {
        i32 tile_x = c.x >> L2_BLOCK_SIZE;
        i32 tile_z = c.z >> L2_BLOCK_SIZE;
        Coord2 tile_coord {
            .x = tile_x,
            .z = tile_z
        };
        /// NOTE(cmo): This makes the assumption that if one is requesting the
        /// mip level of the current tile, then whatever level is there is the
        /// one returned.
        if (tile_coord == tile_key.coord) {
            return tile_key.mip_level;
        }
        if constexpr (assume_lower_tile_bound_0) {
            if (
                (u32(c.x) >= block_map.bbox.max(DimIndex<NumDim>::x)) ||
                (u32(c.z) >= block_map.bbox.max(DimIndex<NumDim>::z))
            ) {
                return -1;
            }
        } else {
            if (
                (c.x < block_map.bbox.min(DimIndex<NumDim>::x)) || (c.x >= block_map.bbox.max(DimIndex<NumDim>::x)) ||
                (c.z < block_map.bbox.min(DimIndex<NumDim>::z)) || (c.z >= block_map.bbox.max(DimIndex<NumDim>::z))
            ) {
                return -1;
            }
        }

        i32 tile_mip_level = i32(mip_block_map.lookup.get(tile_coord)) - 1;
        bool sampleable = tile_mip_level != -1;
        if (sampleable) {
            i64 tile_idx = block_map.lookup(tile_coord);
            tile_base_idx = compute_base_idx(tile_mip_level, tile_idx);
            tile_key = MultiLevelTileKey<NumDim> {
                .mip_level = tile_mip_level,
                .coord = tile_coord
            };
        }
        return tile_mip_level;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    i32 get_sample_level(Coord<NumDim> c) {
        i32 tile_x = c.x >> L2_BLOCK_SIZE;
        i32 tile_y = c.y >> L2_BLOCK_SIZE;
        i32 tile_z = c.z >> L2_BLOCK_SIZE;
        Coord3 tile_coord {
            .x = tile_x,
            .y = tile_y,
            .z = tile_z
        };
        /// NOTE(cmo): This makes the assumption that if one is requesting the
        /// mip level of the current tile, then whatever level is there is the
        /// one returned.
        if (tile_coord == tile_key.coord) {
            return tile_key.mip_level;
        }
        if constexpr (assume_lower_tile_bound_0) {
            if (
                u32(c.x) >= block_map.bbox.max(DimIndex<NumDim>::x) ||
                u32(c.y) >= block_map.bbox.max(DimIndex<NumDim>::y) ||
                u32(c.z) >= block_map.bbox.max(DimIndex<NumDim>::z)
            ) {
                return -1;
            }
        } else {
            if (
                (c.x < block_map.bbox.min(DimIndex<NumDim>::x)) || (c.x >= block_map.bbox.max(DimIndex<NumDim>::x)) ||
                (c.y < block_map.bbox.min(DimIndex<NumDim>::y)) || (c.y >= block_map.bbox.max(DimIndex<NumDim>::y)) ||
                (c.z < block_map.bbox.min(DimIndex<NumDim>::z)) || (c.z >= block_map.bbox.max(DimIndex<NumDim>::z))
            ) {
                return -1;
            }
        }

        i32 tile_mip_level = i32(mip_block_map.lookup.get(tile_coord)) - 1;
        bool sampleable = tile_mip_level != -1;
        if (sampleable) {
            i64 tile_idx = block_map.lookup(tile_coord);
            tile_base_idx = compute_base_idx(tile_mip_level, tile_idx);
            tile_key = MultiLevelTileKey<NumDim> {
                .mip_level = tile_mip_level,
                .coord = tile_coord
            };
        }
        return tile_mip_level;
    }

    YAKL_INLINE
    i64 loop_idx(i32 mip_level, i64 tile_idx, i32 block_idx) const {
        return compute_base_idx(mip_level, tile_idx) + block_idx;
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 2, int> = 0>
    YAKL_INLINE
    Coord2 loop_coord(i32 mip_level, i64 tile_idx, i32 block_idx) const {
        Coord2 tile_coord = compute_tile_coord(tile_idx);
        Coord2 tile_offset = compute_tile_inner_offset(mip_level, block_idx);
        return Coord2 {
            .x = tile_coord.x * BLOCK_SIZE + tile_offset.x,
            .z = tile_coord.z * BLOCK_SIZE + tile_offset.z
        };
    }

    template <int num_dim = NumDim, std::enable_if_t<num_dim == 3, int> = 0>
    YAKL_INLINE
    Coord3 loop_coord(i32 mip_level, i64 tile_idx, i32 block_idx) const {
        Coord3 tile_coord = compute_tile_coord(tile_idx);
        Coord3 tile_offset = compute_tile_inner_offset(mip_level, block_idx);
        return Coord3 {
            .x = tile_coord.x * BLOCK_SIZE + tile_offset.x,
            .y = tile_coord.y * BLOCK_SIZE + tile_offset.y,
            .z = tile_coord.z * BLOCK_SIZE + tile_offset.z
        };
    }
};

template <int NumDim>
YAKL_INLINE ivec<NumDim> round_down(vec<NumDim> pt) {
    ivec<NumDim> result;
    for (int i = 0; i < NumDim; ++i) {
        result(i) = i32(std::floor(pt(i)));
    }
    return result;
}

template <i32 BlockSize, u8 EntrySize, i32 NumDim=2>
struct MultiLevelDDA {
    u8 step_axis;
    i8 max_mip_level;
    i8 current_mip_level;
    fp_t t;
    fp_t dt;
    i32 step_size;
    ivec<NumDim> curr_coord;
    vec<NumDim> next_hit;
    ivec<NumDim> step;
    vec<NumDim> delta;

    RaySegment<NumDim> ray;
    MultiLevelIndexGen<BlockSize, EntrySize, NumDim>& idx_gen;

    YAKL_INLINE MultiLevelDDA(MultiLevelIndexGen<BlockSize, EntrySize, NumDim>& idx_gen_) : idx_gen(idx_gen_) {};

    YAKL_INLINE
    bool init(const RaySegment<NumDim>& ray_, i32 max_mip_level_, bool* start_clipped=nullptr) {
        ray = ray_;
        if (!ray.clip(idx_gen.block_map.bbox, start_clipped)) {
            return false;
        }

        max_mip_level = max_mip_level_;

        constexpr fp_t eps = FP(1e-6);
        ray.update_origin(ray.t0);
        if (std::abs(ray.t1 - ray.t0) < FP(1e-4)) {
            // NOTE(cmo): Ray length has collapsed to essentially 0
            return false;
        }
        vec<NumDim> end_pos = ray(ray.t1 - eps);
        vec<NumDim> start_pos = ray(ray.t0 + eps);
        curr_coord = round_down<NumDim>(start_pos);
        step_size = 1;
        current_mip_level = 0;
        // NOTE(cmo): Most of the time we start in bounds, but occasionally not,
        // and clamping was the wrong behaviour. If we're not in bounds, work on
        // a step size of 1.
        if (in_bounds() && get_sample_level() < 0) {
            step_size = BlockSize;
        }
        for (int ax = 0; ax < NumDim; ++ax) {
            curr_coord(ax) = curr_coord(ax) & (~uint32_t(step_size-1));
        }
        t = ray.t0;

        vec<NumDim> inv_d = ray.inv_d;
        for (int ax = 0; ax < NumDim; ++ax) {
            if (ray.d(ax) == FP(0.0)) {
                step(ax) = 0;
                next_hit(ax) = FP(1e24);
            } else if (inv_d(ax) > FP(0.0)) {
                step(ax) = 1;
                next_hit(ax) = t + (curr_coord(ax) + step_size - start_pos(ax)) * inv_d(ax);
                delta(ax) = inv_d(ax);
            } else {
                step(ax) = -1;
                next_hit(ax) = t + (curr_coord(ax) - start_pos(ax)) * inv_d(ax);
                delta(ax) = -inv_d(ax);
            }
        }
        compute_axis_and_dt();
        // NOTE(cmo): If we didn't start in bounds, advance until we're in, and set the step size
        constexpr i32 max_steps = 32;
        i32 steps = 0;
        while (!in_bounds() || (in_bounds() && dt < FP(1e-6))) {
            next_intersection();
            if (++steps > max_steps) {
#ifdef DEXRT_DEBUG
                yakl::yakl_throw("Failed to walk into grid");
#endif
                return false;
            }
        }
        update_mip_level(true);
        return true;
    }

    YAKL_INLINE i32 get_sample_level() const {
        i32 mip_level;
        if constexpr (NumDim == 2) {
            mip_level = idx_gen.get_sample_level(Coord2{.x = curr_coord(0), .z = curr_coord(1)});
        } else {
            mip_level = idx_gen.get_sample_level(Coord3{.x = curr_coord(0), .y = curr_coord(1), .z = curr_coord(2)});
        }
        return std::min(mip_level, i32(max_mip_level));
    }

    YAKL_INLINE i32 get_step_size() const {
        i32 mip_level = get_sample_level();
        if (mip_level < 0) {
            return BlockSize;
        }
        return 1 << mip_level;
    }

    YAKL_INLINE i32 get_step_size(i32 mip_level) const {
        if (mip_level < 0) {
            return BlockSize;
        }
        return 1 << mip_level;
    }

    template <int ax>
    YAKL_INLINE void compute_axis_and_dt_impl() {
        fp_t next_t = next_hit(ax);

        if (next_t <= t) {
            next_hit(ax) += t - FP(0.999999) * next_hit(ax) + FP(1.0e-6);
            next_t = next_hit(ax);
        }

        if (next_t > ray.t1) {
            dt = ray.t1 - t;
        } else {
            dt = next_t - t;
        }
        dt = std::max(dt, FP(0.0));
    }

    YAKL_INLINE void compute_axis_and_dt() {
        if constexpr (NumDim == 2) {
            step_axis = 0;
            if (next_hit(1) < next_hit(0)) {
                step_axis = 1;
            }
            switch (step_axis) {
                case 0: {
                    compute_axis_and_dt_impl<0>();
                } break;
                case 1: {
                    compute_axis_and_dt_impl<1>();
                } break;
            }
        } else {
            if (next_hit(0) <= next_hit(1) && next_hit(0) <= next_hit(2)) {
                step_axis = 0;
            } else if (next_hit(1) <= next_hit(2)) {
                step_axis = 1;
            } else {
                step_axis = 2;
            }
            switch (step_axis) {
                case 0: {
                    compute_axis_and_dt_impl<0>();
                } break;
                case 1: {
                    compute_axis_and_dt_impl<1>();
                } break;
                case 2: {
                    compute_axis_and_dt_impl<2>();
                }
            }
        }
    }

    /// NOTE(cmo): With lock_to_block curr_coord is not updated based on t,
    /// although curr_pos still uses this. In this instance the pos can _on the
    /// boundary_ of the next block, but we really don't want to start out of
    /// bounds... the ray can end up marginally (O(1e-6) * voxel_scale) larger
    /// than correct.  This is mostly important for the initial call to this
    /// from init.
    /// Now returns new mip level, computed from grid position
    YAKL_INLINE i32 update_mip_level(bool lock_to_block=false) {
        // NOTE(cmo): If we're not locking to block then push us slightly
        // forward along the ray before checking where we are to minimise the
        // effects of rounding into the previous box due to floating point
        // issues.
        if (!lock_to_block) {
            t += FP(1e-3);
        }
        vec<NumDim> curr_pos = ray(t);
        if (!lock_to_block) {
            curr_coord = round_down<NumDim>(curr_pos);
        }
        current_mip_level = get_sample_level();
        step_size = get_step_size();
        for (int ax = 0; ax < NumDim; ++ax) {
            curr_coord(ax) = curr_coord(ax) & (~uint32_t(step_size-1));
        }

        vec<NumDim> inv_d = ray.inv_d;
        for (int ax = 0; ax < NumDim; ++ax) {
            if (step(ax) == 0) {
                continue;
            }

            next_hit(ax) = t + (curr_coord(ax) - curr_pos(ax)) * inv_d(ax);
            if (step(ax) > 0) {
                next_hit(ax) += step_size * inv_d(ax);
            }
        }
        compute_axis_and_dt();
        return current_mip_level;
    }

    template <int ax>
    YAKL_INLINE void next_intersection_impl() {
        t = next_hit(ax);
        next_hit(ax) += step_size * delta(ax);
        curr_coord(ax) += step_size * step(ax);
        compute_axis_and_dt();
    }

    YAKL_INLINE bool next_intersection() {
        if constexpr (NumDim == 2) {
            switch (step_axis) {
                case 0: {
                    next_intersection_impl<0>();
                } break;
                case 1: {
                    next_intersection_impl<1>();
                } break;
            }
        } else {
            switch (step_axis) {
                case 0: {
                    next_intersection_impl<0>();
                } break;
                case 1: {
                    next_intersection_impl<1>();
                } break;
                case 2: {
                    next_intersection_impl<2>();
                }
            }
        }
        return t < ray.t1;
    }

    YAKL_INLINE bool step_through_grid() {
        // NOTE(cmo): Designed to be used with a do-while, i.e. the first
        // intersection is set up before this has been called.
        while (next_intersection()) {
            i32 mip_level = get_sample_level();
            auto has_leaves = [&mip_level]() {
                return mip_level >= 0;
            };
            if (mip_level == current_mip_level && has_leaves()) {
                // NOTE(cmo): Already marching at expected step size through region with data
                return true;
            }

            // NOTE(cmo): DDA predicts we're moving into a derefined region.
            if (mip_level == -1 && step_size != BlockSize) {
                // NOTE(cmo): Boop us away from the boundary to avoid getting
                // stuck. Occasionally this may cut very small corners.
                // Shouldn't be important, but may need tuning.
                t += FP(0.01);
            }

            mip_level = update_mip_level();
            if (has_leaves()) {
                return true;
            }
        }
        return false;
    }

    YAKL_INLINE bool can_sample() const {
        return current_mip_level != -1;
    }

    YAKL_INLINE bool in_bounds() const {
        if constexpr (NumDim == 2) {
            if constexpr (assume_lower_tile_bound_0) {
                return (
                    u32(curr_coord(DimIndex<NumDim>::x)) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::x) &&
                    u32(curr_coord(DimIndex<NumDim>::z)) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::z)
                );
            } else {
                return (
                    curr_coord(DimIndex<NumDim>::x) >= idx_gen.block_map.bbox.min(DimIndex<NumDim>::x) &&
                    curr_coord(DimIndex<NumDim>::z) >= idx_gen.block_map.bbox.min(DimIndex<NumDim>::z) &&
                    curr_coord(DimIndex<NumDim>::x) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::x) &&
                    curr_coord(DimIndex<NumDim>::z) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::z)
                );
            }
        } else {
            if constexpr (assume_lower_tile_bound_0) {
                return (
                    u32(curr_coord(DimIndex<NumDim>::x)) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::x) &&
                    u32(curr_coord(DimIndex<NumDim>::y)) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::y) &&
                    u32(curr_coord(DimIndex<NumDim>::z)) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::z)
                );
            } else {
                return (
                    curr_coord(DimIndex<NumDim>::x) >= idx_gen.block_map.bbox.min(DimIndex<NumDim>::x) &&
                    curr_coord(DimIndex<NumDim>::y) >= idx_gen.block_map.bbox.min(DimIndex<NumDim>::y) &&
                    curr_coord(DimIndex<NumDim>::z) >= idx_gen.block_map.bbox.min(DimIndex<NumDim>::z) &&
                    curr_coord(DimIndex<NumDim>::x) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::x) &&
                    curr_coord(DimIndex<NumDim>::y) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::y) &&
                    curr_coord(DimIndex<NumDim>::z) < idx_gen.block_map.bbox.max(DimIndex<NumDim>::z)
                );
            }
        }
    }
};

typedef IndexGen<BLOCK_SIZE, 2> IdxGen;
typedef MultiLevelIndexGen<BLOCK_SIZE, ENTRY_SIZE, 2> MRIdxGen;
typedef IndexGen<BLOCK_SIZE_3D, 3> IdxGen3d;
typedef MultiLevelIndexGen<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3> MRIdxGen3d;

#else
#endif