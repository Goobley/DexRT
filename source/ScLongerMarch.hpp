#ifndef DEXRT_SC_LONGER_MARCH_HPP
#define DEXRT_SC_LONGER_MARCH_HPP
#include "ScLongerGeometry.hpp"
#include "ScRayMarch.hpp"
#include "ScRayEntry.hpp"

struct ScLongerGridSegment {
    i64 cell;
    i32 x, y, plane;
    fp_t length, fraction, plane_length;
};

/// Whole-characteristic DDA in the canonical half-octant. Remaining crossing
/// distances carry the transverse phase through every z layer. Only crossed
/// coordinates change; periodic wrapping happens here, never in integration.
template <int Dim, bool UniformZ> struct ScLongerGridMarch {
    // Borrow the kernel capture; this marcher never outlives its lane.
    // Copying the array-bearing owner here forces addressable local state.
    const ScLongerGeometry& geometry;
    const ScLongerDirection& direction;
    i32 x = 0, y = 0, plane = 0;
    fp_t next_x = 0, next_y = 0, next_z = 0, plane_length = 0;
    i64 outgoing = 0;
    bool active = false, reaches_outflow = false;

    YAKL_INLINE ScLongerGridSegment segment() const {
        const fp_t length = std::min(next_z, std::min(next_x, next_y));
        JasUnpack(geometry, layout);
        return {
            .cell = plane * layout.plane_size + sc_longer_face_index(layout, x, y),
            .x = x,
            .y = y,
            .plane = plane,
            .length = length,
            .fraction = length / plane_length,
            .plane_length = plane_length,
        };
    }
};

template <int Dim, bool UniformZ>
YAKL_INLINE bool sc_step_longer_grid(ScLongerGridMarch<Dim, UniformZ>& march) {
    const fp_t length = march.segment().length;
    const bool cross_z = march.next_z <= length;
    march.next_x -= length;
    march.next_y -= length;
    march.next_z -= length;
    JasUnpack(march.geometry.layout, grid);
    // Consume coincident transverse crossings too, including at the slab exit.
    if (march.next_x <= FP(0.0)) {
        ++march.x;
        march.next_x += march.direction.delta_x;
        if (grid.periodic_x && march.x == grid.nx) {
            march.x = 0;
        }
    }
    if constexpr (Dim == 3) {
        if (march.next_y <= FP(0.0)) {
            ++march.y;
            march.next_y += march.direction.delta_y;
            if (grid.periodic_y && march.y == grid.ny) {
                march.y = 0;
            }
        }
    }
    if (cross_z) {
        ++march.plane;
        if (march.plane == march.geometry.layout.nz) {
            march.active = false;
            march.reaches_outflow = true;
            return false;
        }
        if constexpr (!UniformZ) {
            march.plane_length = fp_t(
                (march.geometry.z_faces(march.plane + 1) - march.geometry.z_faces(march.plane)) *
                march.direction.inv_mu_z);
        }
        march.next_z = march.plane_length;
    }
    march.active = march.x < grid.nx && march.y < grid.ny;
    return march.active;
}

template <int Dim, bool UniformZ>
YAKL_INLINE ScLongerGridMarch<Dim, UniformZ> sc_begin_longer_grid(
    const ScLongerGeometry& geometry,
    const ScLongerDirection& direction,
    ScLongerEntry entry
) {
    ScLongerGridMarch<Dim, UniformZ> march{.geometry = geometry, .direction = direction};
    march.plane = entry.plane;
    JasUnpack(geometry, layout);
    JasUnpack(layout, grid);
    const auto path = sc_longer_plane_path<UniformZ>(geometry, direction, entry.plane);
    const fp_t h = sc_longer_entry_fraction(layout, path, entry);
    // Side lists are conservative. A ray entering after this layer belongs to
    // a later layer's list and must not be traced twice.
    if (h > FP(1.0)) {
        return march;
    }
    const fp_t px = std::max(FP(0.0), fp_t(entry.x) + path.x + h * path.dx);
    const fp_t py = std::max(FP(0.0), fp_t(entry.y) + path.y + h * path.dy);
    march.x = i32(std::floor(px));
    march.y = Dim == 3 ? i32(std::floor(py)) : 0;
    march.next_x = (FP(1.0) - (px - std::floor(px))) * direction.delta_x;
    march.next_y = Dim == 3 ? (FP(1.0) - (py - std::floor(py))) * direction.delta_y : FP(1e30);
    march.plane_length = path.length;
    march.next_z = (FP(1.0) - h) * path.length;
    if (grid.periodic_x) {
        march.x = sc_wrap(march.x, grid.nx);
    }
    if (grid.periodic_y) {
        march.y = sc_wrap(march.y, grid.ny);
    }
    // Preserve the global lattice labels used on MPI slab faces. This is a
    // one-time mapping, independent of roundoff accumulated by the DDA.
    const f64 start_z = sc_longer_face_height<UniformZ>(geometry, entry.plane);
    const f64 end_z = sc_longer_face_height<UniformZ>(geometry, layout.nz);
    i32 end_x =
        entry.x + i32(std::floor(end_z * direction.slope_x) - std::floor(start_z * direction.slope_x));
    i32 end_y =
        entry.y + i32(std::floor(end_z * direction.slope_y) - std::floor(start_z * direction.slope_y));
    if (grid.periodic_x) {
        end_x = sc_wrap(end_x, grid.nx);
    }
    if (grid.periodic_y) {
        end_y = sc_wrap(end_y, grid.ny);
    }
    march.outgoing = end_x >= 0 && end_x < grid.nx && end_y >= 0 && end_y < grid.ny
                         ? sc_longer_face_index(layout, end_x, end_y)
                         : -1;
    march.active = march.x < grid.nx && march.y < grid.ny;
    if (h == FP(1.0) && march.active) {
        sc_step_longer_grid(march);
    }
    return march;
}

/// CPU geometry shares one DDA pattern across all characteristics. Its
/// bounded offsets map whole contiguous spans, rather than folding each lane.
template <int Dim> struct ScLongerPatternMarch {
    ScRayMarch<Dim - 1> pattern;
    const ScLongerLayout& layout;
    const ScLongerGrid& grid;
    i32 plane;
    i32 offset_x = 0, offset_y = 0;
    i32 next_offset_x, next_offset_y;

    YAKL_INLINE ScLongerPatternMarch(const ScLatticePath& path, const ScLongerLayout& layout_, i32 p)
        : pattern(path.x, path.y, path.dx, path.dy), layout(layout_), grid(layout_.grid), plane(p),
          next_offset_x(grid.periodic_x ? sc_wrap(path.shift_x, grid.nx) : path.shift_x),
          next_offset_y(grid.periodic_y ? sc_wrap(path.shift_y, grid.ny) : path.shift_y) {}

    YAKL_INLINE fp_t fraction() const { return pattern.segment().fraction(); }
};

template <int Dim> YAKL_INLINE bool sc_step_longer_pattern(ScLongerPatternMarch<Dim>& march) {
    const i32 axis = march.pattern.segment().axis;
    if (!march.pattern.step()) {
        return false;
    }
    if (axis == 0) {
        ++march.offset_x;
        if (march.grid.periodic_x && march.offset_x == march.grid.nx) {
            march.offset_x = 0;
        }
    } else {
        ++march.offset_y;
        if (march.grid.periodic_y && march.offset_y == march.grid.ny) {
            march.offset_y = 0;
        }
    }
    return true;
}

/// A contiguous translation from characteristic state into canonical cells.
/// Periodic x produces at most two spans; physical boundaries produce one
/// clipped span. A lane advances both indices by one, without edge handling.
struct ScLongerSpan {
    i64 line = 0, cell = 0;
    i32 x = 0, count = 0;
};
struct ScLongerSpans {
    ScLongerSpan runs[2];
};

YAKL_INLINE ScLongerSpans sc_longer_translated_spans(
    const ScLongerLayout& layout,
    i32 p,
    i32 y,
    i32 offset_x,
    i32 offset_y
) {
    ScLongerSpans spans{};
    JasUnpack(layout, grid);
    const i32 mx = layout.margin_x, my = layout.margin_y;
    i32 entry_y = y - offset_y;
    if (grid.periodic_y && entry_y < 0) {
        entry_y += grid.ny;
    }
    if (entry_y < -my || entry_y >= grid.ny) {
        return spans;
    }
    const i64 line_row = sc_longer_line_index(layout, -mx, entry_y);
    const i64 cell_row = (i64(p) * grid.ny + y) * grid.nx;
    if (grid.periodic_x) {
        spans.runs[0] = {line_row + mx + grid.nx - offset_x, cell_row, 0, offset_x};
        spans.runs[1] = {line_row + mx, cell_row + offset_x, offset_x, grid.nx - offset_x};
    } else {
        const i32 x = std::min(grid.nx, std::max(0, offset_x - mx));
        spans.runs[0] = {line_row + x - offset_x + mx, cell_row + x, x, grid.nx - x};
    }
    return spans;
}

template <int Dim>
YAKL_INLINE ScLongerSpans sc_longer_cell_spans(const ScLongerPatternMarch<Dim>& march, i32 y) {
    return sc_longer_translated_spans(
        march.layout,
        march.plane,
        y,
        march.offset_x,
        march.offset_y
    );
}

template <int Dim>
YAKL_INLINE ScLongerSpans sc_longer_outgoing_spans(const ScLongerPatternMarch<Dim>& march, i32 y) {
    return sc_longer_translated_spans(
        march.layout,
        0,
        y,
        march.next_offset_x,
        march.next_offset_y
    );
}
#endif
