#if !defined(DEXRT_BLOCKMAP_HPP)
#include "Config.hpp"
#include "Atmosphere.hpp"

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// "Insert" a 0 bit after each of the 16 low bits of x
YAKL_INLINE uint32_t part_1_by_1(uint32_t x)
{
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}

// Inverse of Part1By1 - "delete" all odd-indexed bits
YAKL_INLINE uint32_t compact_1_by_1(uint32_t x)
{
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

struct Coord2 {
    i32 x;
    i32 z;

    bool operator==(const Coord2& other) const {
        return (x == other.x) && (z == other.z);
    }
};

struct GridBbox {
    yakl::SArray<i32, 1, NUM_DIM> min;
    yakl::SArray<i32, 1, NUM_DIM> max;
};

YAKL_INLINE uint32_t encode_morton_2(const Coord2& coord)
{
  return (part_1_by_1(uint32_t(coord.z)) << 1) + part_1_by_1(uint32_t(coord.x));
}

YAKL_INLINE Coord2 decode_morton_2(uint32_t code) {
    return Coord2 {
        .x = i32(compact_1_by_1(code)),
        .z = i32(compact_1_by_1(code >> 1))
    };
}


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
        return result
    }

    BlockMapLookup<yakl::memDevice> createDeviceCopy() {
        if constexpr (mem_space == yakl::memDevice) {
            return *this;
        }

        BlockMapLookup<yakl::memDevice> result;
        result.entries = entries.createDeviceCopy();
        return result
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

        yakl::Array<bool, 2, yakl::memDevice> active;
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
                    for (int z = zt; z < zt + BLOCK_SIZE; ++z) {
                        for (int x = xt; x < xt + BLOCK_SIZE; ++x) {
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
                lookup_host(tile_index.z, tile_index.x) = grid_idx++;
            }
        }
        num_active_tiles = grid_idx;
        lookup = lookup_host.createDeviceCopy();

        if (all_active) {
            active_tiles = morton_traversal_order;
        } else {
            yakl::Array<uint32_t, 1, yakl::memHost> active_tiles_host("morton_traversal_active", num_active_tiles);
            int entry = 0;
            for (int m_idx = 0; m_idx < morton_order.extent(0); ++m_idx) {
                uint32_t code = morton_order(m_idx);
                if (active_host(tile_index.z, tile_index.x)) {
                    active_tiles_host(entry++) = code;
                }
            }
            active_tiles = active_tiles_host.createDeviceCopy();
        }
    }

    YAKL_INLINE i64 buffer_len() const {
        return i64(num_active_tiles) * i64(square(BLOCK_SIZE));
    }

    SimpleBounds<2> loop_bounds() const {
        return SimpleBounds<2>(
            num_active_tiles,
            square(BLOCK_SIZE)
        );
    }
};

// TODO(cmo): REFINED_SIZE NYI
template<i32 BLOCK_SIZE, i32 REFINED_SIZE=1, class BlockMap=BlockMap<BLOCK_SIZE>>
struct IndexGen {
    Coord2 tile_key;
    i64 tile_base_idx;
    const BlockMap& block_map;

    IndexGen(const BlockMap& block_map_) : tile_key({.x = -1, .z = -1}), tile_base_idx(), block_map(block_map_)
    {}

    Coord2 compute_tile_coord(i64 tile_idx) const {
        return decode_morton_2(block_map.active_tiles(tile_idx));
    }

    i64 compute_base_idx(i64 tile_idx) const {
        return tile_idx * square(BLOCK_SIZE / REFINED_SIZE);
    }

    Coord2 compute_tile_inner_offset(i32 tile_offset) const {
        return Coord2 {
            .x = tile_offset % BLOCK_SIZE,
            .z = tile_offset / BLOCK_SIZE
        };
    }

    i32 compute_inner_offset(i32 inner_x, i32 inner_z) const {
        return inner_z * BLOCK_SIZE + inner_x;
    }

    i64& get(i32 x, i32 z) {
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
            throw std::runtime_error(fmt::format("Out of bounds block requested {} ({}, {}), check with has_leaves", tile_idx, x, z));
        }
#endif
        if (tile_idx > 0) {
            tile_base_idx = compute_base_idx(tile_idx);
            tile_key = tile_key_lookup;
            return tile_base_idx + compute_inner_offset(inner_x, inner_z);
        }
        return -1;
    }


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
        i64 tile_idx = block_map.lookup(tile_x, tile_z);
        bool result = tile_idx != -1;
        if (result) {
            tile_base_idx = compute_base_idx(tile_idx);
            tile_key = tile_key_lookup;
        }
        return result;
    }

    i64 loop_idx(i64 tile_idx, i32 block_idx) const {
        return compute_base_idx(tile_idx) + compute_inner_offset(block_idx);
    }

    Coord2 loop_coord(i64 tile_idx, i32 block_idx) const {
        Coord2 tile_coord = compute_tile_coord(tile_idx);
        Coord2 tile_offset = compute_tile_inner_offset(block_idx);
        return Coord2 {
            .x = tile_coord.x + tile_offset.x,
            .z = tile_coord.z + tile_offset.z
        };
    }
};

struct RaySegment {
    vec2 o;
    vec2 d;
    vec2 inv_d;
    fp_t t0;
    fp_t t1;

    struct IntersectionResult {
        /// i.e. at least partially inside.
        bool intersects;
        fp_t t0;
        fp_t t1;
    };

    RaySegment(vec2 o_, vec2 d_, fp_t t0_=FP(0.0), fp_t t1_=FP(1e24)) :
        o(o_),
        d(d_),
        t0(t0_),
        t1(t1_)
    {
        inv_d(0) = FP(1.0) / d(0);
        inv_d(1) = FP(1.0) / d(1);
    }

    vec2 operator()(fp_t t) const {
        vec2 result;
        result(0) = o(0) + t * d(0);
        result(1) = o(1) + t * d(1);
        return result;
    }

    IntersectionResult intersects(const GridBbox& bbox) const {
        IntersectionResult result{
            .intersects = true,
            .t0 = t0,
            .t1 = t1
        };
        fp_t& t0_ = result.t0;
        fp_t& t1_ = result.t1;

        for (int ax = 0; ax < o.size(); ++ax) {
            fp_t a = fp_t(bbox.min(ax));
            fp_t b = fp_t(bbox.max(ax));
            if (a >= b) {
                result.intersects = false;
                return result;
            }

            a = (a - o(ax)) * inv_d(ax);
            b = (b - o(ax)) * inv_d(ax);
            if (a > b) {
                fp_t temp = a;
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
                result.intersects = false;
                return result;
            }
        }
        return result;
    }

    bool clip(const GridBbox& bbox, bool* start_clipped) {
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
};

Coord2 round_down(vec2 pt) {
    return Coord2 {
        .x = i32(std::floor(pt(0))),
        .z = i32(std::floor(pt(1)))
    };
}

template <i32 BLOCK_SIZE, i32 REFINED_SIZE=1, class IndexGen=IndexGen<BLOCK_SIZE, REFINED_SIZE, BlockMap<BLOCK_SIZE>>>
struct TwoLevelDDA {

};

#else
#endif