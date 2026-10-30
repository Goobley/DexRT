#if !defined(DEXRT_SC_GEOMETRY_HPP)
#define DEXRT_SC_GEOMETRY_HPP

#include "Types.hpp"
#include "ScConfig.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <fmt/core.h>

/// Grid of cell-centred nodes: node (x, y, z) sits at ((x + 0.5) dx, (y + 0.5)
/// dy, (z + 0.5) dz) (plus offsets), or at height z_node(z + 1) for a
/// non-uniform z grid. 2D: num_y = 1 (y is the invariant direction). A plane of
/// constant z holds num_y * num_x cells, row-major in (y, x).
struct ScGrid {
    i32 num_x;
    i32 num_y = 1;
    i32 num_z;
    fp_t dx;
    fp_t dy = FP(0.0);
    fp_t dz; /// uniform z, else the smallest row step
    fp_t offset_x;
    fp_t offset_y;
    fp_t offset_z;
    /// Non-uniform z only: node heights [m] of rows -1 .. num_z (neighbouring
    /// slab rows, or ghosts mirroring the end spacing at the domain edges).
    Fp1d z_node;
    i32 dim = 2;

    YAKL_INLINE i64 plane_size() const { return i64(num_x) * num_y; }
};

/// Height of global row k from the global cell centres z, with ghost rows -1
/// and size() mirroring the end spacing (so the domain faces lie half a step
/// beyond the end nodes).
inline f64 sc_global_row_height(const std::vector<f64>& z, i32 k) {
    const i32 n = z.size();
    if (k < 0) {
        return z[0] - (z[1] - z[0]) * f64(-k);
    }
    if (k >= n) {
        return z[n - 1] + (z[n - 1] - z[n - 2]) * f64(k - n + 1);
    }
    return z[k];
}

/// Step from local row k - 1 to row k, for k in [0, num_z] (rows -1 and num_z
/// are the neighbouring slab rows or ghosts). z empty: uniform dz.
inline std::vector<f64> sc_row_steps(const ScGrid& grid, const std::vector<f64>& z, i32 z_start) {
    std::vector<f64> steps(grid.num_z + 1, f64(grid.dz));
    if (!z.empty()) {
        for (int k = 0; k <= grid.num_z; ++k) {
            steps[k] = sc_global_row_height(z, z_start + k) - sc_global_row_height(z, z_start + k - 1);
        }
    }
    return steps;
}

enum class ScRayMode : i32 {
    /// Swept row by row in z; the characteristic into each node is traced back
    /// to the previous row, crossing t columns.
    Sweep,
    /// 2D: no projection onto the x-z plane (along y).
    AlongY,
};

/// Canonical frame of one direction: rows (planes) p along z, columns s along
/// x and sy along y, flipped so the sweep runs in increasing p and the ray
/// moves towards increasing s and sy. 2D: num_sy = 1.
struct ScFrame {
    ScRayMode mode;
    bool flip_z;         /// physical z = flip_z ? num_p - 1 - p : p
    bool flip_x;         /// physical x = flip_x ? num_s - 1 - s : s
    bool flip_y = false; /// physical y = flip_y ? num_sy - 1 - sy : sy
    i32 num_p;
    i32 num_s;
    i32 num_sy = 1;
    fp_t t;            /// Sweep: columns crossed per row (largest row step, see ScDirBatch::row_t)
    fp_t ty = FP(0.0); /// 3D sweep: as t, along y
    fp_t ds;           /// Sweep: path length per row [m] (as t)
};

/// Cells per canonical plane (the medium halo is in whole planes).
YAKL_INLINE i64 sc_plane_size(const ScFrame& f) {
    return i64(f.num_s) * f.num_sy;
}

/// Flat canonical index of physical (x, y, z).
YAKL_INLINE i64 sc_canonical_idx(const ScFrame& f, i32 x, i32 y, i32 z) {
    const i32 p = f.flip_z ? (f.num_p - 1 - z) : z;
    const i32 s = f.flip_x ? (f.num_s - 1 - x) : x;
    const i32 sy = f.flip_y ? (f.num_sy - 1 - y) : y;
    return (i64(p) * f.num_sy + sy) * f.num_s + s;
}

/// Columns crossed and path length for a row step dz.
inline void sc_row_geometry(
    const ScGrid& grid,
    f64 mux,
    f64 muz,
    f64 dz,
    fp_t* t,
    fp_t* ds
) {
    const f64 ax = std::abs(mux);
    const f64 az = std::abs(muz);
    *t = fp_t((ax * dz) / (az * grid.dx));
    *ds = fp_t(dz / az);
}

/// max_dz: the largest row step (uniform: dz). Rays along y (2.5D) are AlongY;
/// horizontal rays (mu_z = 0) and those crossing more than
/// SC_MAX_ROW_CROSSINGS columns per row are rejected.
inline ScFrame sc_frame_2d(
    const ScGrid& grid,
    f64 mux,
    f64 muz,
    bool periodic_x,
    f64 max_dz
) {
    ScFrame f{};
    const f64 ax = std::abs(mux);
    const f64 az = std::abs(muz);
    f.num_p = grid.num_z;
    f.num_s = grid.num_x;
    f.flip_z = muz < 0.0;
    f.flip_x = mux < 0.0;
    if (ax < SC_DEGENERATE_MU && az < SC_DEGENERATE_MU) {
        f.mode = ScRayMode::AlongY;
        f.t = FP(0.0);
        f.ds = FP(0.0);
        return f;
    }
    if (az < SC_DEGENERATE_MU) {
        throw std::runtime_error(
            "SC: rays with mu_z = 0 are not supported (use Stepan, or the rotated Healpix)");
    }
    const f64 t = (ax * max_dz) / (az * grid.dx);
    if (t > SC_MAX_ROW_CROSSINGS) {
        throw std::runtime_error(fmt::format(
            "SC: a ray crosses {:.0f} columns per row (more than {:.0f})",
            t,
            SC_MAX_ROW_CROSSINGS
        ));
    }
    f.mode = ScRayMode::Sweep;
    sc_row_geometry(
        grid,
        mux,
        muz,
        max_dz,
        &f.t,
        &f.ds
    );
    return f;
}

inline ScFrame sc_frame_2d(const ScGrid& grid, f64 mux, f64 muz, bool periodic_x) {
    return sc_frame_2d(
        grid,
        mux,
        muz,
        periodic_x,
        f64(grid.dz)
    );
}

/// Physical (x, z) of canonical (p, s). Indices outside the grid map to the
/// corresponding ghost positions.
YAKL_INLINE Coord2 sc_physical_coord(const ScFrame& f, i32 p, i32 s) {
    return Coord2{.x = f.flip_x ? (f.num_s - 1 - s) : s, .z = f.flip_z ? (f.num_p - 1 - p) : p};
}

/// Physical position of a (fractional) canonical location.
YAKL_INLINE vec3 sc_canonical_position(const ScGrid& grid, const ScFrame& f, fp_t p, fp_t s) {
    const fp_t x = f.flip_x ? (fp_t(f.num_s - 1) - s) : s;
    const fp_t z = f.flip_z ? (fp_t(f.num_p - 1) - p) : p;
    vec3 pos;
    pos(0) = (x + FP(0.5)) * grid.dx + grid.offset_x;
    pos(1) = grid.offset_y;
    pos(2) = (z + FP(0.5)) * grid.dz + grid.offset_z;
    if (grid.z_node.initialized()) {
        // NOTE(cmo): Linear between node heights (rows -1 .. num_p).
        const fp_t zc = std::min(std::max(z, FP(-1.0)), fp_t(f.num_p));
        const i32 k = std::min(i32(std::floor(zc)), f.num_p - 1);
        const fp_t frac = zc - fp_t(k);
        pos(2) = grid.z_node(k + 1) + frac * (grid.z_node(k + 2) - grid.z_node(k + 1));
    }
    return pos;
}

/// Columns crossed in x and y, and path length, for a plane step dz.
inline void
sc_plane_geometry(
    const ScGrid& grid,
    f64 mux,
    f64 muy,
    f64 muz,
    f64 dz,
    fp_t* tx,
    fp_t* ty,
    fp_t* ds
) {
    const f64 az = std::abs(muz);
    *tx = fp_t((std::abs(mux) * dz) / (az * grid.dx));
    *ty = fp_t((std::abs(muy) * dz) / (az * grid.dy));
    *ds = fp_t(dz / az);
}

/// 3D frame: swept plane by plane in z. Horizontal rays (mu_z = 0) and those
/// crossing more than SC_MAX_ROW_CROSSINGS columns per plane step are
/// rejected. max_dz: the largest plane step.
inline ScFrame sc_frame_3d(
    const ScGrid& grid,
    f64 mux,
    f64 muy,
    f64 muz,
    f64 max_dz
) {
    ScFrame f{};
    f.num_p = grid.num_z;
    f.num_s = grid.num_x;
    f.num_sy = grid.num_y;
    f.flip_z = muz < 0.0;
    f.flip_x = mux < 0.0;
    f.flip_y = muy < 0.0;
    const f64 az = std::abs(muz);
    if (az < SC_DEGENERATE_MU) {
        throw std::runtime_error(
            "SC 3D: rays with mu_z = 0 are not supported (use Stepan, or the rotated Healpix)");
    }
    const f64 tx = (std::abs(mux) * max_dz) / (az * grid.dx);
    const f64 ty = (std::abs(muy) * max_dz) / (az * grid.dy);
    if (tx + ty > SC_MAX_ROW_CROSSINGS) {
        throw std::runtime_error(fmt::format(
            "SC 3D: a ray crosses {:.0f} columns per plane (more than {:.0f})",
            tx + ty,
            SC_MAX_ROW_CROSSINGS
        ));
    }
    f.mode = ScRayMode::Sweep;
    sc_plane_geometry(
        grid,
        mux,
        muy,
        muz,
        max_dz,
        &f.t,
        &f.ty,
        &f.ds
    );
    return f;
}

/// Physical position of a (fractional) canonical location (plane p, column s,
/// row sy).
YAKL_INLINE vec3 sc_canonical_position_3d(
    const ScGrid& grid,
    const ScFrame& f,
    fp_t p,
    fp_t s,
    fp_t sy
) {
    const fp_t x = f.flip_x ? (fp_t(f.num_s - 1) - s) : s;
    const fp_t y = f.flip_y ? (fp_t(f.num_sy - 1) - sy) : sy;
    const fp_t z = f.flip_z ? (fp_t(f.num_p - 1) - p) : p;
    vec3 pos;
    pos(0) = (x + FP(0.5)) * grid.dx + grid.offset_x;
    pos(1) = (y + FP(0.5)) * grid.dy + grid.offset_y;
    pos(2) = (z + FP(0.5)) * grid.dz + grid.offset_z;
    if (grid.z_node.initialized()) {
        // NOTE(cmo): Linear between node heights (planes -1 .. num_p).
        const fp_t zc = std::min(std::max(z, FP(-1.0)), fp_t(f.num_p));
        const i32 k = std::min(i32(std::floor(zc)), f.num_p - 1);
        const fp_t frac = zc - fp_t(k);
        pos(2) = grid.z_node(k + 1) + frac * (grid.z_node(k + 2) - grid.z_node(k + 1));
    }
    return pos;
}

/// Longer characteristics (3D): as in 2D (ScLongerCharacteristics2d.hpp), the
/// characteristics form a global lattice, one per cell of every face, and in
/// each plane all of them cross the same pattern of x and y node planes. A
/// plane is described by the entry offsets within the entry cell, the shifts
/// of the entry cell to the next plane, and the segments: path length and cell
/// offset (ox, oy) from the entry cell, in order along the ray.
struct ScLongerPlane {
    f64 u0x;
    f64 u0y;
    i32 dkx;
    i32 dky;
    std::vector<f64> seglen;
    std::vector<i32> ox;
    std::vector<i32> oy;
};

/// Plane crossing canonical faces Z0 -> Z1, for a direction crossing tau_x
/// (tau_y) columns per metre of z, with path length path_per_z per metre of z.
inline ScLongerPlane sc_longer_plane(
    f64 Z0,
    f64 Z1,
    f64 tau_x,
    f64 tau_y,
    f64 path_per_z
) {
    ScLongerPlane plane;
    const f64 A0x = Z0 * tau_x;
    const f64 A0y = Z0 * tau_y;
    const f64 n0x = std::floor(A0x);
    const f64 n0y = std::floor(A0y);
    plane.u0x = A0x - n0x;
    plane.u0y = A0y - n0y;
    plane.dkx = i32(std::floor(Z1 * tau_x) - n0x);
    plane.dky = i32(std::floor(Z1 * tau_y) - n0y);
    // NOTE(cmo): Node planes strictly inside (A0, A1) split the path, at the
    // heights Z = m / tau; merged in order (a tie gives an empty segment).
    struct Event {
        f64 Z;
        bool x;
    };
    std::vector<Event> events;
    if (tau_x > 0.0) {
        for (f64 m = n0x + 1.0; m < Z1 * tau_x; m += 1.0) {
            events.push_back(Event{.Z = m / tau_x, .x = true});
        }
    }
    if (tau_y > 0.0) {
        for (f64 m = n0y + 1.0; m < Z1 * tau_y; m += 1.0) {
            events.push_back(Event{.Z = m / tau_y, .x = false});
        }
    }
    std::stable_sort(events.begin(), events.end(), [](const Event& a, const Event& b) { return a.Z < b.Z; });
    f64 Z = Z0;
    i32 ox = 0;
    i32 oy = 0;
    for (const auto& e : events) {
        const f64 Ze = std::min(std::max(e.Z, Z), Z1);
        plane.seglen.push_back((Ze - Z) * path_per_z);
        plane.ox.push_back(ox);
        plane.oy.push_back(oy);
        Z = Ze;
        ox += e.x;
        oy += !e.x;
    }
    plane.seglen.push_back((Z1 - Z) * path_per_z);
    plane.ox.push_back(ox);
    plane.oy.push_back(oy);
    return plane;
}

#endif
