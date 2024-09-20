#if !defined(DEXRT_BLOCKMAP_HPP)
#define DEXRT_BLOCKMAP_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include <optional>

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

    YAKL_INLINE bool operator==(const Coord2& other) const {
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
        fmt::println("Num active tiles: {}/{} ({})", num_active_tiles, num_z_tiles * num_x_tiles, cutoff_temperature);
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
};


YAKL_INLINE ivec2 round_down(vec2 pt) {
    ivec2 result;
    result(0) = i32(std::floor(pt(0)));
    result(1) = i32(std::floor(pt(1)));
    return result;
}

template <i32 BLOCK_SIZE>
struct TwoLevelDDA {
    uint8_t step_axis;
    fp_t t;
    fp_t dt;
    i32 step_size;
    ivec2 curr_coord;
    vec2 next_hit;
    ivec2 step;
    vec2 delta;

    RaySegment ray;
    IndexGen<BLOCK_SIZE>& idx_gen;

    YAKL_INLINE TwoLevelDDA(IndexGen<BLOCK_SIZE>& idx_gen_) : idx_gen(idx_gen_) {};

    YAKL_INLINE
    bool init(RaySegment& ray_, bool* start_clipped=nullptr) {
        if (!ray_.clip(idx_gen.block_map.bbox, start_clipped)) {
            return false;
        }

        ray = ray_;

        constexpr fp_t eps = FP(1e-6);
        vec2 end_pos = ray(ray.t1 - eps);
        vec2 start_pos = ray(ray.t0 + eps);
        curr_coord = round_down(start_pos);
        for (int ax = 0; ax < NUM_DIM; ++ax) {
            curr_coord(ax) = std::min(curr_coord(ax), idx_gen.block_map.bbox.max(ax)-1);
        }
        step_size = idx_gen.refined_size;
        if (!idx_gen.has_leaves(curr_coord(0), curr_coord(1))) {
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
        return true;
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

    YAKL_INLINE void update_step_size(i32 new_step_size) {
        if (new_step_size == step_size) {
            return;
        }

        step_size = new_step_size;
        vec2 curr_pos = ray(t);
        curr_coord = round_down(curr_pos);
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
            const bool has_leaves = idx_gen.has_leaves(curr_coord(0), curr_coord(1));
            if (step_size == idx_gen.refined_size && has_leaves) {
                // NOTE(cmo): Already marching at refined_size through region with data
                return true;
            }

            if (has_leaves) {
                // NOTE(cmo): Refine, then return first intersection in refined region with data.
                update_step_size(idx_gen.refined_size);
                if (!idx_gen.has_leaves(curr_coord(0), curr_coord(1))) {
                    // NOTE(cmo): Sometimes we round to just outside the refined
                    // region on resolution change, and may need to take a step
                    continue;
                }
                return true;
            }

            if (step_size == idx_gen.refined_size) {
                // NOTE(cmo): Not in a refined region (no leaves), so go to big steps

                // NOTE(cmo): Boop us away from the boundary to avoid getting
                // stuck. Occasionally this may cut very small corners.
                // Shouldn't be important, but may need tuning.
                t += FP(0.01);
                update_step_size(BLOCK_SIZE);
            }
        }
        return false;
    }

    YAKL_INLINE bool can_sample() const {
        if (BLOCK_SIZE != idx_gen.refined_size) {
            return step_size == idx_gen.refined_size;
        } else {
            return idx_gen.has_leaves(curr_coord(0), curr_coord(1));
        }
    }
};


#else
#endif