#ifndef DEXRT_SC_LONGER_GEOMETRY_HPP
#define DEXRT_SC_LONGER_GEOMETRY_HPP
#include "Types.hpp"
#include "ScGeometry.hpp"
#include <cmath>
#include <limits>

struct ScLongerGrid {
    i32 nx = 0, ny = 0;
    bool periodic_x = false, periodic_y = false;
};

/// Global-lattice entry labels and bounded rolling-line geometry. Slab-face
/// labels precede each plane's side labels; physical inflow uses separate arrays.
struct ScLongerLayout {
    ScLongerGrid grid;
    i32 nz = 0, margin_x = 0, margin_y = 0, line_width = 0;
    i64 plane_size = 0, side_size = 0, side_rows = 0;
    i64 line_count = 0, inflow_count = 0;
};

inline ScLongerLayout
sc_longer_layout(
    i32 nx,
    i32 ny,
    i32 nz,
    i32 mx,
    i32 my,
    bool periodic_x,
    bool periodic_y
) {
    ScLongerLayout layout;
    layout.grid = {nx, ny, periodic_x, periodic_y};
    layout.nz = nz;
    layout.margin_x = mx;
    layout.margin_y = my;
    layout.line_width = nx + mx;
    layout.plane_size = i64(nx) * ny;
    layout.side_rows = i64(my) * layout.line_width;
    layout.side_size = layout.side_rows + i64(ny) * mx;
    layout.line_count = layout.plane_size + layout.side_size;
    layout.inflow_count = layout.plane_size + nz * layout.side_size;
    return layout;
}

struct ScLongerEntry {
    i32 plane, x, y; // entry-cell labels; a negative x/y denotes a side entry
};

YAKL_INLINE i64 sc_longer_face_index(const ScLongerLayout& layout, i32 x, i32 y) {
    return i64(y) * layout.grid.nx + x;
}
YAKL_INLINE i64 sc_longer_line_index(const ScLongerLayout& layout, i32 x, i32 y) {
    return i64(y + layout.margin_y) * layout.line_width + x + layout.margin_x;
}
YAKL_INLINE i64 sc_longer_inflow_index(const ScLongerLayout& layout, ScLongerEntry entry) {
    if (entry.x >= 0 && entry.y >= 0) {
        return sc_longer_face_index(layout, entry.x, entry.y);
    }
    const i64 side = entry.y < 0
                         ? sc_longer_line_index(layout, entry.x, entry.y)
                         : layout.side_rows + i64(entry.y) * layout.margin_x + entry.x + layout.margin_x;
    return layout.plane_size + entry.plane * layout.side_size + side;
}
YAKL_INLINE ScLongerEntry sc_longer_entry(const ScLongerLayout& layout, i64 ray) {
    if (ray < layout.plane_size) {
        const i32 y = ray / layout.grid.nx;
        return {0, i32(ray - i64(y) * layout.grid.nx), y};
    }
    const i32 p = (ray - layout.plane_size) / layout.side_size;
    const i64 side = ray - layout.plane_size - p * layout.side_size;
    if (side < layout.side_rows) {
        const i32 y = side / layout.line_width - layout.margin_y;
        return {p, i32(side - i64(y + layout.margin_y) * layout.line_width) - layout.margin_x, y};
    }
    const i64 rest = side - layout.side_rows;
    const i32 y = rest / layout.margin_x;
    return {p, i32(rest - i64(y) * layout.margin_x) - layout.margin_x, y};
}

struct ScLatticePath {
    fp_t x, y, dx, dy, length;
    i32 shift_x, shift_y;
};

/// Direction constants. Global lattice slopes seed a ray once; DDA crossing
/// distances then carry its phase. No per-layer phase/path tables are stored.
struct ScLongerDirection {
    f64 slope_x, slope_y, inv_mu_z;
    fp_t delta_x, delta_y; // distance along the ray between x/y cell faces
    fp_t uniform_dx, uniform_dy, uniform_length;
};
struct ScLongerGeometry {
    ScLongerLayout layout;
    bool uniform_z = true; // CPU selects the compiled spacing variant
    f64 z_first = 0, dz = 0;
    yakl::Array<ScLongerDirection, 1, yakl::memDevice> directions;
    yakl::Array<f64, 1, yakl::memDevice> z_faces; // variable z only, canonical slab faces
};

template <bool UniformZ> YAKL_INLINE f64 sc_longer_face_height(const ScLongerGeometry& geometry, i32 p) {
    if constexpr (UniformZ) {
        return geometry.z_first + f64(p) * geometry.dz;
    } else {
        return geometry.z_faces(p);
    }
}

/// Shared plane pattern for CPU SIMD and the independent cell-response pass.
/// This is initialized from the global lattice, never from a slab-local phase.
template <bool UniformZ>
YAKL_INLINE ScLatticePath sc_longer_plane_path(const ScLongerGeometry& geometry, const ScLongerDirection& direction, i32 p) {
    const f64 z0 = sc_longer_face_height<UniformZ>(geometry, p);
    const f64 z1 = sc_longer_face_height<UniformZ>(geometry, p + 1);
    const f64 ax = z0 * direction.slope_x, bx = z1 * direction.slope_x;
    const f64 ay = z0 * direction.slope_y, by = z1 * direction.slope_y;
    constexpr fp_t below_one = FP(1.0) - std::numeric_limits<fp_t>::epsilon() * FP(0.5);
    ScLatticePath path{
        .x = std::min(fp_t(ax - std::floor(ax)), below_one),
        .y = std::min(fp_t(ay - std::floor(ay)), below_one),
        .dx = 0,
        .dy = 0,
        .length = 0,
        .shift_x = i32(std::floor(bx) - std::floor(ax)),
        .shift_y = i32(std::floor(by) - std::floor(ay)),
    };
    if constexpr (UniformZ) {
        path.dx = direction.uniform_dx;
        path.dy = direction.uniform_dy;
        path.length = direction.uniform_length;
    } else {
        path.dx = fp_t((z1 - z0) * direction.slope_x);
        path.dy = fp_t((z1 - z0) * direction.slope_y);
        path.length = fp_t((z1 - z0) * direction.inv_mu_z);
    }
    return path;
}

/// Locate the first physical side-face intersection in the entry layer.
YAKL_INLINE fp_t sc_longer_entry_fraction(const ScLongerLayout& layout, const ScLatticePath& path, ScLongerEntry entry) {
    fp_t fraction = FP(0.0);
    if (entry.x < 0 && !layout.grid.periodic_x) {
        fraction = std::max(fraction, (-fp_t(entry.x) - path.x) / path.dx);
    }
    if (entry.y < 0 && !layout.grid.periodic_y) {
        fraction = std::max(fraction, (-fp_t(entry.y) - path.y) / path.dy);
    }
    return fraction;
}
/// Canonical cell coordinates, in boundary-sampler order (z, x, y).
struct ScLongerEntryPosition {
    fp_t z, x, y;
};
YAKL_INLINE ScLongerEntryPosition sc_longer_entry_position(const ScLongerLayout& layout, const ScLatticePath& path, ScLongerEntry entry) {
    const fp_t h = std::min(sc_longer_entry_fraction(layout, path, entry), FP(1.0));
    return {
        .z = fp_t(entry.plane) - FP(0.5) + h,
        .x = std::max(fp_t(entry.x) + path.x + h * path.dx - FP(0.5), -FP(0.5)),
        .y = std::max(fp_t(entry.y) + path.y + h * path.dy - FP(0.5), -FP(0.5)),
    };
}
/// Longer characteristics (2D, swept along z). In canonical column units with
/// cell s spanning [s, s + 1], the characteristics are the lines a = Z tau - k
/// (Z: canonical height of the ray above the canonical start face, tau:
/// columns crossed per metre of z): one per column on every face, a global
/// lattice (the prototype's integer lattice). In each row all of them cross the
/// same pattern of vertical lines, so a row is described once: the entry
/// offset u0 within the entry cell, the segment path lengths through the cells
/// entry, entry + 1, ..., and the shift dk of the entry cell to the next row.
/// A characteristic is labelled by its entry cell in the row it is crossing.

/// Canonical face heights from the global cell centres (increasing upwards):
/// faces midway between centres, the end faces mirrored half a step out.
/// Measured from the canonical start face (the bottom, or the top for
/// downward sweeps).
inline std::vector<f64> sc_canonical_faces(const std::vector<f64>& z, bool flip_z) {
    const i32 n = z.size();
    std::vector<f64> faces(n + 1);
    for (int k = 0; k <= n; ++k) {
        faces[k] = 0.5 * (sc_global_row_height(z, k - 1) + sc_global_row_height(z, k));
    }
    std::vector<f64> result(n + 1);
    for (int j = 0; j <= n; ++j) {
        result[j] = flip_z ? faces[n] - faces[n - j] : faces[j] - faces[0];
    }
    return result;
}

/// One row of one direction.
struct ScLongerRow {
    f64 u0;                  /// entry offset in [0, 1) within the entry cell
    i32 dk;                  /// entry cell in the next row - entry cell in this row
    std::vector<f64> seglen; /// path length [m] in cells entry, entry + 1, ...
};

/// Row from lattice coordinate A0 to A1 (columns: the entry offset is the
/// fractional part of A0), over a path length path [m].
inline ScLongerRow sc_longer_row_a(f64 A0, f64 A1, f64 path) {
    ScLongerRow row;
    const f64 n0 = std::floor(A0);
    row.u0 = A0 - n0;
    row.dk = i32(std::floor(A1) - n0);
    // NOTE(cmo): Vertical lines strictly inside (A0, A1) split the path.
    const i32 crossings = std::max(i32(std::ceil(A1) - n0) - 1, 0);
    const f64 span = A1 - A0;
    if (span <= 0.0) {
        row.seglen.push_back(path);
        return row;
    }
    for (int i = 0; i <= crossings; ++i) {
        const f64 a_start = std::max(row.u0, f64(i));
        const f64 a_end = std::min(A1 - n0, f64(i + 1));
        row.seglen.push_back(std::max(a_end - a_start, 0.0) / span * path);
    }
    return row;
}

/// Row crossing canonical faces Z0 -> Z1 for a direction crossing tau columns
/// per metre of z, with path length path_per_z per metre of z.
inline ScLongerRow sc_longer_row(f64 Z0, f64 Z1, f64 tau, f64 path_per_z) {
    return sc_longer_row_a(Z0 * tau, Z1 * tau, (Z1 - Z0) * path_per_z);
}

#endif
