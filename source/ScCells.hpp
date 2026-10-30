#if !defined(DEXRT_SC_CELLS_HPP)
#define DEXRT_SC_CELLS_HPP

#include "Types.hpp"
#include "ScConfig.hpp"

using ScActiveMaskHost = yakl::Array<u8, 2, yakl::memHost>;

/// Expand already-global tile activity into an arbitrary owned z window.
/// Tile flags must include every intersected global tile, not flags computed
/// independently from a rank's partial tile. Outer dimensions remain tiled.
/// Tiles are [z tile, y tile * x tiles + x tile] of the given size, over
/// num_y rows in y (2D: 1, a single y tile); the mask is [z, y * num_x + x].
ScActiveMaskHost sc_active_mask_from_tiles(const ScActiveMaskHost& global_tiles,
    i32 z_start,
    i32 num_z,
    i32 tile = BLOCK_SIZE,
    i32 num_y = 1);

/// A run of active cells along x within one row (y, z local, halo rows
/// included); its first cell is ScCells::ks(x0, y, z).
struct ScRunPiece {
    i32 z;
    i32 y;
    i32 x0;
    i32 n;
    /// Compile-time 1 when groups are single cells (GPU), so per-cell arrays
    /// stay in registers.
    YAKL_INLINE i32 length() const {
        if constexpr (SC_CELL_GROUP == 1) {
            return 1;
        } else {
            return n;
        }
    }
};

/// Active cells of a dense index space. A templated per-cell routine loops i in
/// [0, num()) with ks = cells(i), via cell_list(state).
template <int Mem = yakl::memDevice>
struct ScCellList {
    yakl::Array<i64, 1, Mem> ks;
    i64 n = 0;

    YAKL_INLINE i64 num() const {
        return n;
    }
    YAKL_INLINE i64 operator()(i64 i) const {
        return ks(i);
    }
    ScCellList<yakl::memHost> host() const {
        return ScCellList<yakl::memHost>{.ks = n ? ks.createHostCopy() : yakl::Array<i64, 1, yakl::memHost>(), .n = n};
    }
};

/// This rank's dense slab index space: ks = (z + H) * plane + y * num_x + x,
/// for local z in [-H, num_z + H), H = SC_MEDIUM_HALO (the halo rows), plane
/// = num_y * num_x (2D: num_y = 1). Per-cell arrays shared with the templated
/// physics have num_cells() entries; SC-private ones (J) cover the owned rows
/// only, at ks - owned_offset(). Void cells are skipped through the owned
/// list and runs. Built on the host at setup; kernels capture it by value.
struct ScCells {
    i32 num_x = 0;
    i32 num_y = 1;
    i32 num_z = 0;
    i32 z_start = 0;
    i32 group_size = SC_CELL_GROUP;
    i64 num_active = 0; /// owned
    i64 num_active_with_halo = 0;
    i64 num_runs = 0; /// owned (the first num_runs of runs)
    i64 num_runs_with_halo = 0;
    yakl::Array<u8, 2, yakl::memDevice> active; /// [num_rows, plane]; halo rows from the neighbours (ScHalo), else 0
    yakl::Array<i64, 1, yakl::memDevice> owned; /// active owned ks, ascending
    yakl::Array<i64, 1, yakl::memDevice> owned_and_halo; /// active ks of every row, ascending
    yakl::Array<ScRunPiece, 1, yakl::memDevice> runs; /// owned runs, then the halo rows'

    YAKL_INLINE i64 plane_size() const {
        return i64(num_x) * num_y;
    }
    YAKL_INLINE i32 num_rows() const {
        return num_z + 2 * SC_MEDIUM_HALO;
    }
    YAKL_INLINE i64 num_cells() const {
        return i64(num_rows()) * plane_size();
    }
    YAKL_INLINE i64 num_owned_cells() const {
        return i64(num_z) * plane_size();
    }
    YAKL_INLINE i64 owned_offset() const {
        return i64(SC_MEDIUM_HALO) * plane_size();
    }
    /// Local z (halo rows included).
    YAKL_INLINE i64 ks(i32 x, i32 y, i32 z) const {
        return (i64(z) + SC_MEDIUM_HALO) * plane_size() + i64(y) * num_x + x;
    }
    YAKL_INLINE Coord3 coord(i64 k) const {
        const i64 plane = plane_size();
        const i64 row = k / plane;
        const i64 xy = k - row * plane;
        const i32 y = xy / num_x;
        return Coord3{.x = i32(xy - i64(y) * num_x), .y = y, .z = i32(row) - SC_MEDIUM_HALO};
    }
    ScCellList<> list() const {
        return ScCellList<>{.ks = owned, .n = num_active};
    }
    /// For the medium inputs (bases), which the sweep reads in the halo rows.
    ScCellList<> list_with_halo() const {
        return ScCellList<>{.ks = owned_and_halo, .n = num_active_with_halo};
    }

    /// mask is [num_z, plane] over the owned rows, entries 0 or 1 (the halo
    /// rows inactive). Empty active sets are valid. group_size may be
    /// overridden at setup for tests; kernels must match its bound.
    void init(const ScActiveMaskHost& mask, i32 global_z_start = 0, i32 group = SC_CELL_GROUP, i32 num_y = 1);
    /// Rebuild the lists and runs from active (after its halo rows change).
    void build_lists();

    /// Dense host [lead, num_cells] from owned rows [lead, num_z, plane],
    /// with the void and halo cells zeroed.
    template <typename T>
    yakl::Array<T, 2, yakl::memHost> from_owned_rows(const yakl::Array<T, 3, yakl::memHost>& rows) const {
        const auto mask = active.createHostCopy();
        yakl::Array<T, 2, yakl::memHost> result("sc_cells_dense", rows.extent(0), num_cells());
        result = T(0);
        const i64 off = owned_offset();
        for (i64 l = 0; l < rows.extent(0); ++l) {
            for (i64 k = 0; k < num_owned_cells(); ++k) {
                if (mask.data()[off + k]) {
                    result(l, off + k) = rows.data()[l * num_owned_cells() + k];
                }
            }
        }
        return result;
    }

    /// Owned rows [lead, num_z, plane] of a dense [lead, num_cells] array, with
    /// the void cells zeroed.
    template <typename T>
    yakl::Array<T, 3, yakl::memHost> owned_rows(const yakl::Array<T, 2, yakl::memDevice>& dense) const {
        const auto values = dense.createHostCopy();
        const auto mask = active.createHostCopy();
        yakl::Array<T, 3, yakl::memHost> result("sc_cells_rows", dense.extent(0), num_z, plane_size());
        const i64 off = owned_offset();
        for (i64 l = 0; l < dense.extent(0); ++l) {
            for (i64 k = 0; k < num_owned_cells(); ++k) {
                result.data()[l * num_owned_cells() + k] = mask.data()[off + k] ? values(l, off + k) : T(0);
            }
        }
        return result;
    }
};

#endif
