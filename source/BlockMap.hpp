#if !defined(DEXRT_BLOCKMAP_HPP)
#define DEXRT_BLOCKMAP_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include "MortonCodes.hpp"

struct GridBbox {
    yakl::SArray<i32, 1, NUM_DIM> min;
    yakl::SArray<i32, 1, NUM_DIM> max;
};

template <int mem_space=yakl::memDevice>
struct BlockMapLookup {
    // NOTE(cmo): This is a separate class so it can be switch to e.g. Morton ordering
    yakl::Array<i64, 2, mem_space> entries;

    void init(i32 num_x, i32 num_z) {
        entries = decltype(entries)("BlockMapEntries", num_z, num_x);
        entries = -1;
        yakl::fence();
    }

    YAKL_INLINE i64& operator()(i32 x, i32 z) const {
        return entries(z, x);
    }

    BlockMapLookup<yakl::memHost> createHostCopy() {
        if constexpr (mem_space == yakl::memHost) {
            return *this;
        }

        BlockMapLookup<yakl::memHost> result;
        result.entries = entries.createHostCopy();
        return result;
    }

    BlockMapLookup<yakl::memDevice> createDeviceCopy() {
        if constexpr (mem_space == yakl::memDevice) {
            return *this;
        }

        BlockMapLookup<yakl::memDevice> result;
        result.entries = entries.createDeviceCopy();
        return result;
    }

};

template <i32 BLOCK_SIZE, class Lookup=BlockMapLookup<>>
struct BlockMap {
    i32 num_x_tiles;
    i32 num_z_tiles;
    i32 num_active_tiles;
    GridBbox bbox;

    Lookup lookup;
    yakl::Array<uint32_t, 1, yakl::memDevice> morton_traversal_order;
    yakl::Array<uint32_t, 1, yakl::memDevice> active_tiles;

    void init(const Atmosphere& atmos, fp_t cutoff_temperature) {
        if (atmos.temperature.extent(0) % BLOCK_SIZE != 0 || atmos.temperature.extent(1) % BLOCK_SIZE != 0) {
            throw std::runtime_error("Grid is not a multiple of BLOCK_SIZE");
        }
        num_x_tiles = atmos.temperature.extent(1) / BLOCK_SIZE;
        num_z_tiles = atmos.temperature.extent(0) / BLOCK_SIZE;
        num_active_tiles = 0;
        bbox.min(0) = 0;
        bbox.min(1) = 0;
        bbox.max(0) = atmos.temperature.extent(1);
        bbox.max(1) = atmos.temperature.extent(0);

        yakl::Array<uint32_t, 1, yakl::memHost> morton_order("morton_traversal_order", num_x_tiles * num_z_tiles);
        for (int z = 0; z < num_z_tiles; ++z) {
            for (int x = 0; x < num_x_tiles; ++x) {
                morton_order(z * num_x_tiles + x) = encode_morton_2(Coord2{.x = x, .z = z});
            }
        }
        std::sort(morton_order.begin(), morton_order.end());
        morton_traversal_order = morton_order.createDeviceCopy();

        lookup.init(num_x_tiles, num_z_tiles);

        yakl::Array<bool, 2, yakl::memDevice> active("active tiles", num_z_tiles, num_x_tiles);
        bool all_active = cutoff_temperature == FP(0.0);
        if (all_active) {
            active = true;
            num_active_tiles = num_x_tiles * num_z_tiles;
        } else {
            auto& temperature = atmos.temperature;
            parallel_for(
                "Compute active cells",
                SimpleBounds<2>(num_z_tiles, num_x_tiles),
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
        auto lookup_host = lookup.createHostCopy();
        i64 grid_idx = 0;

        for (int m_idx = 0; m_idx < morton_order.extent(0); ++m_idx) {
            Coord2 tile_index = decode_morton_2(morton_order(m_idx));
            if (active_host(tile_index.z, tile_index.x)) {
                // TODO(cmo): This is awful that the order needs to be swapped!
                lookup_host(tile_index.x, tile_index.z) = grid_idx++;
            }
        }
        num_active_tiles = grid_idx;
        fmt::println("Num active tiles: {}/{} ({:.1f} %)", num_active_tiles, num_z_tiles * num_x_tiles, fp_t(num_active_tiles) / fp_t(num_z_tiles * num_x_tiles) * FP(100.0));
        lookup = lookup_host.createDeviceCopy();

        if (all_active) {
            active_tiles = morton_traversal_order;
        } else {
            yakl::Array<uint32_t, 1, yakl::memHost> active_tiles_host("morton_traversal_active", num_active_tiles);
            int entry = 0;
            for (int m_idx = 0; m_idx < morton_order.extent(0); ++m_idx) {
                uint32_t code = morton_order(m_idx);
                Coord2 tile_index = decode_morton_2(morton_order(m_idx));
                if (active_host(tile_index.z, tile_index.x)) {
                    active_tiles_host(entry++) = code;
                }
            }
            active_tiles = active_tiles_host.createDeviceCopy();
        }
    }

    /// Everything active, no atmosphere, used for given fs
    void init(i32 x_size, i32 z_size) {
        if (x_size % BLOCK_SIZE != 0 || z_size % BLOCK_SIZE != 0) {
            throw std::runtime_error("Grid is not a multiple of BLOCK_SIZE");
        }
        num_x_tiles = x_size / BLOCK_SIZE;
        num_z_tiles = z_size / BLOCK_SIZE;
        num_active_tiles = num_x_tiles * num_z_tiles;
        bbox.min(0) = 0;
        bbox.min(1) = 0;
        bbox.max(0) = x_size;
        bbox.max(1) = z_size;


        yakl::Array<uint32_t, 1, yakl::memHost> morton_order("morton_traversal_order", num_x_tiles * num_z_tiles);
        for (int z = 0; z < num_z_tiles; ++z) {
            for (int x = 0; x < num_x_tiles; ++x) {
                morton_order(z * num_x_tiles + x) = encode_morton_2(Coord2{.x = x, .z = z});
            }
        }
        std::sort(morton_order.begin(), morton_order.end());
        morton_traversal_order = morton_order.createDeviceCopy();
        active_tiles = morton_traversal_order;

        lookup.init(num_x_tiles, num_z_tiles);
        auto lookup_host = lookup.createHostCopy();
        i64 grid_idx = 0;
        for (int m_idx = 0; m_idx < morton_order.extent(0); ++m_idx) {
            Coord2 tile_index = decode_morton_2(morton_order(m_idx));
            lookup_host(tile_index.x, tile_index.z) = grid_idx++;
        }
        lookup = lookup_host.createDeviceCopy();
    }

    YAKL_INLINE i64 buffer_len(i32 mip_px_size=1) const {
        return i64(num_active_tiles) * i64(square(BLOCK_SIZE) / square(mip_px_size));
    }

    SimpleBounds<2> loop_bounds(i32 mip_px_size=1) const {
        return SimpleBounds<2>(
            num_active_tiles,
            square(BLOCK_SIZE) / square(mip_px_size)
        );
    }
};

template <u8 entry_size=3, int mem_space=yakl::memDevice>
struct MultiLevelLookup {
    static constexpr u8 packed_entries_per_u64 = (sizeof(u64) * CHAR_BIT) / entry_size;
    static constexpr u64 lowest_entry_mask = ((1 << entry_size) - 1);
    static_assert(entry_size <= 8, "Entry size must be <= 8");
    /// The number of hyper-tiles if HYPERBLOCK2x2 is true
    i32 num_x_tiles;
    i32 num_z_tiles;
    // NOTE(cmo): We're still laying out these tiles linearly here (unless
    // hyper_blocks are used)... not ideal, but it has a very small footprint --
    // should remain resident in cache.
    yakl::Array<u64, 1, mem_space> entries;

    template <class BlockMap>
    void init(const BlockMap& block_map) {
        num_x_tiles = block_map.num_x_tiles;
        num_z_tiles = block_map.num_z_tiles;
        if constexpr (HYPERBLOCK2x2) {
            num_x_tiles /= 2;
            num_z_tiles /= 2;
        }
        const i32 num_entries = block_map.num_x_tiles * block_map.num_z_tiles;
        const i32 storage_for_entries = (num_entries + packed_entries_per_u64 - 1) / packed_entries_per_u64;
        entries = decltype(entries)("MultiLevel Entries", storage_for_entries);
    }

    YAKL_INLINE i32 flat_tile_index(i32 x, i32 z) const {
        i32 flat_idx;
        if constexpr (HYPERBLOCK2x2) {
            const i32 hyper_tile_idx = (z >> 1) * num_x_tiles + (x >> 1);
            constexpr i32 hyper_tile_size = 4;
            flat_idx = hyper_tile_idx * hyper_tile_size + ((z & 1) << 1) + (x & 1);
        } else {
            flat_idx = z * num_x_tiles + x;
        }
        return flat_idx;
    }

    YAKL_INLINE u8 get(i32 x, i32 z) const {
        const i32 flat_idx = flat_tile_index(x, z);
        const i32 block_idx = flat_idx / packed_entries_per_u64;
        const i32 entry_idx = flat_idx % packed_entries_per_u64;

        const u64 block = entries(block_idx);
        u8 entry = (block >> (entry_size * entry_idx)) & lowest_entry_mask;
        return entry;
    }

    /// NOTE(cmo): NOT THREADSAFE -- should probably be host only?
    YAKL_INLINE void set(i32 x, i32 z, u8 val) {
        const i32 flat_idx = flat_tile_index(x, z);
        const i32 block_idx = flat_idx / packed_entries_per_u64;
        const i32 entry_idx = flat_idx % packed_entries_per_u64;

        const u64 shifted_entry = u64(val) << (entry_idx * entry_size); // ---- --ab ----
        const u64 shifted_mask = ~(lowest_entry_mask << (entry_idx * entry_size)); // ffff ff00 ffff
        const u64 block = entries(block_idx); // 0123 4567 89ab
        const u64 updated_block = (block & shifted_mask) | shifted_entry; // 0123 45ab 89ab
        entries(block_idx) = updated_block;
    }

    /// Pack an array of entries into the u64 storage backing array. This isn't
    /// really const, but it only changes the array contents... ehhhhh
    template <typename T>
    void pack_entries(const yakl::Array<T, 1, memDevice>& single_entries) const {
        JasUnpack((*this), entries);
        parallel_for(
            "Pack T entries into u64",
            SimpleBounds<1>(entries.extent(0)),
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

    MultiLevelLookup<entry_size, yakl::memHost> createHostCopy() {
        if constexpr (mem_space == yakl::memHost) {
            return *this;
        }

        MultiLevelLookup<entry_size, yakl::memHost> result;
        result.num_x_tiles = num_x_tiles;
        result.num_z_tiles = num_z_tiles;
        result.entries = entries.createHostCopy();
        return result;
    }

    MultiLevelLookup<entry_size, yakl::memDevice> createDeviceCopy() {
        if constexpr (mem_space == yakl::memDevice) {
            return *this;
        }

        MultiLevelLookup<entry_size, yakl::memDevice> result;
        result.num_x_tiles = num_x_tiles;
        result.num_z_tiles = num_z_tiles;
        result.entries = entries.createDeviceCopy();
        return result;
    }
};

/// NOTE(cmo): Instead of storing the tile index (which is stored in the
/// blockmap), this stores a MultiLevelLookup of packed tile information.
/// When unpacked to a u8 via lookup.get, if 0, the tile is empty, otherwise
/// val-1 is its associated mip level (log2 voxel block size). The max mip level
/// is then clamped when traversing in MultiLevelDda. This struct provides the
/// length of an array for storing all of the mips of the sparse tiles.
template <int BLOCK_SIZE, int ENTRY_SIZE=3, class Lookup=BlockMapLookup<>, class BlockMap=BlockMap<BLOCK_SIZE, Lookup>>
struct MultiResBlockMap {
    static constexpr i32 max_storable_entry = (1 << ENTRY_SIZE) - 1;
    i32 max_mip_level;
    yakl::SArray<i64, 1, max_storable_entry> mip_offsets;
    BlockMap block_map;
    MultiLevelLookup<ENTRY_SIZE> lookup;

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
};


constexpr bool INNER_MORTON_LOOKUP = false;
template<i32 BLOCK_SIZE>
struct IndexGen {
    Coord2 tile_key;
    i64 tile_base_idx;
    i32 refined_size;
    const BlockMap<BLOCK_SIZE>& block_map;

    YAKL_INLINE
    IndexGen(const BlockMap<BLOCK_SIZE>& block_map_, i32 refined_size_=1) :
        tile_key({.x = -1, .z = -1}),
        tile_base_idx(),
        block_map(block_map_),
        refined_size(refined_size_)
    {}

    template <u8 entry_size>
    YAKL_INLINE
    IndexGen(const MultiResBlockMap<BLOCK_SIZE, entry_size>& mr_block_map, i32 refined_size_=1) :
        tile_key({.x = -1, .z = -1}),
        tile_base_idx(),
        block_map(mr_block_map.block_map),
        refined_size(refined_size_)
    {}

    YAKL_INLINE
    Coord2 compute_tile_coord(i64 tile_idx) const {
        return decode_morton_2(block_map.active_tiles(tile_idx));
    }

    YAKL_INLINE
    i64 compute_base_idx(i64 tile_idx) const {
        return tile_idx * square(BLOCK_SIZE / refined_size);
    }

    YAKL_INLINE
    Coord2 compute_tile_inner_offset(i32 tile_offset) const {
        Coord2 coord;
        if constexpr (INNER_MORTON_LOOKUP) {
            coord = decode_morton_2(uint32_t(tile_offset));
        } else {
            coord = Coord2 {
                .x = tile_offset % (BLOCK_SIZE / refined_size),
                .z = tile_offset / (BLOCK_SIZE / refined_size)
            };
        }
        coord.x *= refined_size;
        coord.z *= refined_size;
        return coord;
    }

    YAKL_INLINE
    i32 compute_inner_offset(i32 inner_x, i32 inner_z) const {
        if constexpr (INNER_MORTON_LOOKUP) {
            Coord2 coord {
                .x = inner_x / refined_size,
                .z = inner_z / refined_size
            };
            return encode_morton_2(coord);
        } else {
            return (inner_z * (BLOCK_SIZE / refined_size) + inner_x) / refined_size;
        }
    }

    YAKL_INLINE
    i64 full_flat_idx(i32 x, i32 z) {
        return z * block_map.num_x_tiles * BLOCK_SIZE + x;
    }

    YAKL_INLINE
    i64 idx(i32 x, i32 z) {
        i32 tile_x = x / BLOCK_SIZE;
        i32 tile_z = z / BLOCK_SIZE;
        i32 inner_x = x % BLOCK_SIZE;
        i32 inner_z = z % BLOCK_SIZE;
        Coord2 tile_key_lookup{
            .x = tile_x,
            .z = tile_z
        };

        if (tile_key == tile_key_lookup) {
            return tile_base_idx + compute_inner_offset(inner_x, inner_z);
        }

        i64 tile_idx = block_map.lookup(tile_x, tile_z);
#ifdef DEXRT_DEBUG
        if (tile_idx < 0) {
            yakl::yakl_throw("OOB block requested!");
        }
#endif
        if (tile_idx >= 0) {
            tile_base_idx = compute_base_idx(tile_idx);
            tile_key = tile_key_lookup;
            return tile_base_idx + compute_inner_offset(inner_x, inner_z);
        }
        return -1;
    }

    YAKL_INLINE
    bool has_leaves(i32 x, i32 z) {
        i32 tile_x = x / BLOCK_SIZE;
        i32 tile_z = z / BLOCK_SIZE;
        Coord2 tile_key_lookup{
            .x = tile_x,
            .z = tile_z
        };

        if (tile_key == tile_key_lookup) {
            return true;
        }
        if (
            (tile_x < 0) || (tile_x >= block_map.bbox.max(0)) ||
            (tile_z < 0) || (tile_z >= block_map.bbox.max(1))
        ) {
            return false;
        }
        i64 tile_idx = block_map.lookup(tile_x, tile_z);
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

    YAKL_INLINE
    Coord2 loop_coord(i64 tile_idx, i32 block_idx) const {
        Coord2 tile_coord = compute_tile_coord(tile_idx);
        Coord2 tile_offset = compute_tile_inner_offset(block_idx);
        return Coord2 {
            .x = tile_coord.x * BLOCK_SIZE + tile_offset.x,
            .z = tile_coord.z * BLOCK_SIZE + tile_offset.z
        };
    }
};

struct MultiLevelTileKey {
    i32 mip_level;
    i32 x;
    i32 z;


    YAKL_INLINE bool operator==(const MultiLevelTileKey& other) const {
        return (mip_level == other.mip_level) && (x == other.x) && (z == other.z);
    }
};

// TODO(cmo): This approach won't work when distributed... we'll likely need to
// add another layer of indirection, or, if we distribute by morton order, a
// base index that is subtracted per level per node
template<i32 BLOCK_SIZE, u8 ENTRY_SIZE>
struct MultiLevelIndexGen {
    MultiLevelTileKey tile_key;
    i64 tile_base_idx;
    const BlockMap<BLOCK_SIZE>& block_map;
    const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE>& mip_block_map;

    YAKL_INLINE
    MultiLevelIndexGen(
        const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE>& mip_block_map_
    ) :
        tile_key({.mip_level = -1, .x = -1, .z = -1}),
        tile_base_idx(),
        block_map(mip_block_map_.block_map),
        mip_block_map(mip_block_map_)
    {}

    YAKL_INLINE
    Coord2 compute_tile_coord(i64 tile_idx) const {
        return decode_morton_2(block_map.active_tiles(tile_idx));
    }

    YAKL_INLINE
    i64 compute_base_idx(i32 mip_level, i64 tile_idx) const {
        return mip_block_map.mip_offsets(mip_level) + tile_idx * square(BLOCK_SIZE >> mip_level);
    }

    YAKL_INLINE
    Coord2 compute_tile_inner_offset(i32 mip_level, i32 tile_offset) const {
        Coord2 coord;
        if constexpr (INNER_MORTON_LOOKUP) {
            coord = decode_morton_2(uint32_t(tile_offset));
        } else {
            coord = Coord2 {
                .x = tile_offset % (BLOCK_SIZE >> mip_level),
                .z = tile_offset / (BLOCK_SIZE >> mip_level)
            };
        }
        coord.x <<= mip_level;
        coord.z <<= mip_level;
        return coord;
    }

    YAKL_INLINE
    i32 compute_inner_offset(i32 mip_level, i32 inner_x, i32 inner_z) const {
        if constexpr (INNER_MORTON_LOOKUP) {
            Coord2 coord {
                .x = inner_x >> mip_level,
                .z = inner_z >> mip_level
            };
            return encode_morton_2(coord);
        } else {
            return (inner_z * (BLOCK_SIZE >> mip_level) + inner_x) >> mip_level;
        }
    }

    YAKL_INLINE
    i64 full_flat_idx(i32 mip_level, i32 x, i32 z) {
#ifdef DEXRT_DEBUG
        if (mip_level != 0) {
            yakl::yakl_throw("No concept of a flat index for mip_level != 0");
        }
#endif
        return z * block_map.num_x_tiles * BLOCK_SIZE + x;
    }

    YAKL_INLINE
    i64 idx(i32 mip_level, i32 x, i32 z) {
        i32 tile_x = x / BLOCK_SIZE;
        i32 tile_z = z / BLOCK_SIZE;
        i32 inner_x = x % BLOCK_SIZE;
        i32 inner_z = z % BLOCK_SIZE;
        MultiLevelTileKey tile_key_lookup{
            .mip_level = mip_level,
            .x = tile_x,
            .z = tile_z
        };

        if (tile_key == tile_key_lookup) {
            return tile_base_idx + compute_inner_offset(mip_level, inner_x, inner_z);
        }

        i64 tile_idx = block_map.lookup(tile_x, tile_z);
#ifdef DEXRT_DEBUG
        if (tile_idx < 0) {
            yakl::yakl_throw("OOB block requested!");
        }
#endif
        if (tile_idx >= 0) {
            tile_base_idx = compute_base_idx(mip_level, tile_idx);
            tile_key = tile_key_lookup;
            return tile_base_idx + compute_inner_offset(mip_level, inner_x, inner_z);
        }
        return -1;
    }

    /// Returns the mip level to sample on, or -1 if empty
    YAKL_INLINE
    i32 get_sample_level(i32 x, i32 z) {
        i32 tile_x = x / BLOCK_SIZE;
        i32 tile_z = z / BLOCK_SIZE;
        /// NOTE(cmo): This makes the assumption that if one is requesting the
        /// mip level of the current tile, then whatever level is there is the
        /// one returned.
        if (tile_x == tile_key.x && tile_z == tile_key.z) {
            return tile_key.mip_level;
        }

        if (
            (tile_x < 0) || (tile_x >= block_map.bbox.max(0)) ||
            (tile_z < 0) || (tile_z >= block_map.bbox.max(1))
        ) {
            return -1;
        }

        i32 tile_mip_level = i32(mip_block_map.lookup.get(tile_x, tile_z)) - 1;
        bool sampleable = tile_mip_level != -1;
        if (sampleable) {
            i64 tile_idx = block_map.lookup(tile_x, tile_z);
            tile_base_idx = compute_base_idx(tile_mip_level, tile_idx);
            tile_key = MultiLevelTileKey {
                .mip_level = tile_mip_level,
                .x = tile_x,
                .z = tile_z
            };
        }
        return tile_mip_level;
    }

    YAKL_INLINE
    i64 loop_idx(i32 mip_level, i64 tile_idx, i32 block_idx) const {
        return compute_base_idx(mip_level, tile_idx) + block_idx;
    }

    YAKL_INLINE
    Coord2 loop_coord(i32 mip_level, i64 tile_idx, i32 block_idx) const {
        Coord2 tile_coord = compute_tile_coord(tile_idx);
        Coord2 tile_offset = compute_tile_inner_offset(mip_level, block_idx);
        return Coord2 {
            .x = tile_coord.x * BLOCK_SIZE + tile_offset.x,
            .z = tile_coord.z * BLOCK_SIZE + tile_offset.z
        };
    }
};

struct IntersectionResult {
    /// i.e. at least partially inside.
    bool intersects;
    fp_t t0;
    fp_t t1;
};

struct RaySegment {
    vec2 o;
    vec2 d;
    vec2 inv_d;
    fp_t t0;
    fp_t t1;


    YAKL_INLINE
    RaySegment() :
        o(),
        d(),
        inv_d(),
        t0(),
        t1()
    {}

    YAKL_INLINE
    RaySegment(vec2 o_, vec2 d_, fp_t t0_=FP(0.0), fp_t t1_=FP(1e24)) :
        o(o_),
        d(d_),
        t0(t0_),
        t1(t1_)
    {
        inv_d(0) = FP(1.0) / d(0);
        inv_d(1) = FP(1.0) / d(1);
    }

    YAKL_INLINE vec2 operator()(fp_t t) const {
        vec2 result;
        result(0) = o(0) + t * d(0);
        result(1) = o(1) + t * d(1);
        return result;
    }

    YAKL_INLINE IntersectionResult intersects(const GridBbox& bbox) const {
        fp_t t0_ = t0;
        fp_t t1_ = t1;

        for (int ax = 0; ax < NUM_DIM; ++ax) {
            fp_t a = fp_t(bbox.min(ax));
            fp_t b = fp_t(bbox.max(ax));
            if (a >= b) {
                return IntersectionResult{
                    .intersects = false,
                    .t0 = t0_,
                    .t1 = t1_
                };
            }

            a = (a - o(ax)) * inv_d(ax);
            b = (b - o(ax)) * inv_d(ax);
            if (a > b) {
                fp_t temp = b;
                b = a;
                a = temp;
            }

            if (a > t0_) {
                t0_ = a;
            }
            if (b < t1_) {
                t1_ = b;
            }
            if (t0_ > t1_) {
                return IntersectionResult{
                    .intersects = false,
                    .t0 = t0_,
                    .t1 = t1_
                };
            }
        }
        return IntersectionResult{
            .intersects = true,
            .t0 = t0_,
            .t1 = t1_
        };
    }

    YAKL_INLINE bool clip(const GridBbox& bbox, bool* start_clipped=nullptr) {
        IntersectionResult result = intersects(bbox);
        if (start_clipped) {
            *start_clipped = false;
        }

        if (result.intersects) {
            if (start_clipped && result.t0 > t0) {
                *start_clipped = true;
            }

            t0 = result.t0;
            t1 = result.t1;
        }
        return result.intersects;
    }

    /// Updates the origin to be ray(t) and the start/end t accordingly
    YAKL_INLINE void update_origin(fp_t t) {
        o = (*this)(t);
        t0 -= t;
        t1 -= t;
    }

    /// Computes t for a particular position. Does not verify if actually on the
    /// line! Uses x unless dir(x) == 0, in which case it uses z
    YAKL_INLINE fp_t compute_t(vec2 pos) {
        fp_t t;
        if (d(0) != FP(0.0)) {
            t = (pos(0) - o(0)) * inv_d(0);
        } else {
            t = (pos(1) - o(1)) * inv_d(1);
        }
        return t;
    }
};


YAKL_INLINE ivec2 round_down(vec2 pt) {
    ivec2 result;
    result(0) = i32(std::floor(pt(0)));
    result(1) = i32(std::floor(pt(1)));
    return result;
}

template <i32 BLOCK_SIZE, u8 ENTRY_SIZE>
struct MultiLevelDDA {
    u8 step_axis;
    i8 max_mip_level;
    i8 current_mip_level;
    fp_t t;
    fp_t dt;
    i32 step_size;
    ivec2 curr_coord;
    vec2 next_hit;
    ivec2 step;
    vec2 delta;

    RaySegment ray;
    MultiLevelIndexGen<BLOCK_SIZE, ENTRY_SIZE>& idx_gen;

    YAKL_INLINE MultiLevelDDA(MultiLevelIndexGen<BLOCK_SIZE, ENTRY_SIZE>& idx_gen_) : idx_gen(idx_gen_) {};

    YAKL_INLINE
    bool init(RaySegment& ray_, i32 max_mip_level_, bool* start_clipped=nullptr) {
        if (!ray_.clip(idx_gen.block_map.bbox, start_clipped)) {
            return false;
        }

        ray = ray_;
        max_mip_level = max_mip_level_;

        constexpr fp_t eps = FP(1e-6);
        ray.update_origin(ray.t0);
        vec2 end_pos = ray(ray.t1 - eps);
        vec2 start_pos = ray(ray.t0 + eps);
        curr_coord = round_down(start_pos);
        step_size = 1;
        current_mip_level = 0;
        // NOTE(cmo): Most of the time we start in bounds, but occasionally not,
        // and clamping was the wrong behaviour. If we're not in bounds, work on
        // a step size of 1.
        if (in_bounds() && get_sample_level() < 0) {
            step_size = BLOCK_SIZE;
        }
        for (int ax = 0; ax < NUM_DIM; ++ax) {
            curr_coord(ax) = curr_coord(ax) & (~uint32_t(step_size-1));
        }
        t = ray.t0;

        vec2 inv_d = ray.inv_d;
        for (int ax = 0; ax < NUM_DIM; ++ax) {
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
        while (!in_bounds() || (in_bounds() && dt < FP(1e-6))) {
            next_intersection();
        }
        update_mip_level(get_sample_level(), true);
        return true;
    }

    YAKL_INLINE i32 get_sample_level() const {
        i32 mip_level = idx_gen.get_sample_level(curr_coord(0), curr_coord(1));
        return std::min(mip_level, i32(max_mip_level));
    }

    YAKL_INLINE i32 get_step_size() const {
        i32 mip_level = get_sample_level();
        if (mip_level < 0) {
            return BLOCK_SIZE;
        }
        return 1 << mip_level;
    }

    YAKL_INLINE i32 get_step_size(i32 mip_level) const {
        if (mip_level < 0) {
            return BLOCK_SIZE;
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
    }

    /// NOTE(cmo): With lock_to_block curr_coord is not updated based on t,
    /// although curr_pos still uses this. In this instance the pos can _on the
    /// boundary_ of the next block, but we really don't want to start out of
    /// bounds... the ray can end up marginally (O(1e-6) * voxel_scale) larger
    /// than correct.  This is mostly important for the initial call to this
    /// from init.
    YAKL_INLINE void update_mip_level(i32 new_mip_level, bool lock_to_block=false) {
        if (new_mip_level == current_mip_level) {
            return;
        }
        const i32 new_step_size = get_step_size(new_mip_level);
        current_mip_level = new_mip_level;
        step_size = new_step_size;

        vec2 curr_pos = ray(t);
        if (!lock_to_block) {
            curr_coord = round_down(curr_pos);
        }
        for (int ax = 0; ax < NUM_DIM; ++ax) {
            curr_coord(ax) = curr_coord(ax) & (~uint32_t(step_size-1));
        }

        vec2 inv_d = ray.inv_d;
        for (int ax = 0; ax < NUM_DIM; ++ax) {
            if (step(ax) == 0) {
                continue;
            }

            next_hit(ax) = t + (curr_coord(ax) - curr_pos(ax)) * inv_d(ax);
            if (step(ax) > 0) {
                next_hit(ax) += step_size * inv_d(ax);
            }
        }
        compute_axis_and_dt();
    }

    template <int ax>
    YAKL_INLINE void next_intersection_impl() {
        t = next_hit(ax);
        next_hit(ax) += step_size * delta(ax);
        curr_coord(ax) += step_size * step(ax);
        compute_axis_and_dt();
    }

    YAKL_INLINE bool next_intersection() {
        switch (step_axis) {
            case 0: {
                next_intersection_impl<0>();
            } break;
            case 1: {
                next_intersection_impl<1>();
            } break;
        }
        return t < ray.t1;
    }

    YAKL_INLINE bool step_through_grid() {
        // NOTE(cmo): Designed to be used with a do-while, i.e. the first
        // intersection is set up before this has been called.
        while (next_intersection()) {
            const i32 mip_level = get_sample_level();
            const bool has_leaves = (mip_level >= 0);
            if (mip_level == current_mip_level && has_leaves) {
                // NOTE(cmo): Already marching at expected step size through region with data
                return true;
            }

            if (has_leaves) {
                // NOTE(cmo): Refine, then return first intersection in refined region with data.
                // TODO(cmo): Push t into the expected box so we can remove the check after?
                update_mip_level(mip_level);
                if (get_sample_level() != mip_level) {
                    // NOTE(cmo): Sometimes we round to just outside the refined
                    // region on resolution change, and may need to take a very small step
                    continue;
                }
                return true;
            }

            if (step_size != BLOCK_SIZE) {
                // NOTE(cmo): Not in a refined region (no leaves), so go to big steps

                // NOTE(cmo): Boop us away from the boundary to avoid getting
                // stuck. Occasionally this may cut very small corners.
                // Shouldn't be important, but may need tuning.
                t += FP(0.01);
                update_mip_level(mip_level);
            }
        }
        return false;
    }

    YAKL_INLINE bool can_sample() const {
        return current_mip_level != -1;
    }

    YAKL_INLINE bool in_bounds() const {
        return (
            curr_coord(0) >= idx_gen.block_map.bbox.min(0)
            && curr_coord(1) >= idx_gen.block_map.bbox.min(1)
            && curr_coord(0) < idx_gen.block_map.bbox.max(0)
            && curr_coord(1) < idx_gen.block_map.bbox.max(1)
        );
    }
};

typedef IndexGen<BLOCK_SIZE> IdxGen;
typedef MultiLevelIndexGen<BLOCK_SIZE, ENTRY_SIZE> MRIdxGen;

#else
#endif