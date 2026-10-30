#ifndef DEXRT_SC_SHORT_NODE_3D_HPP
#define DEXRT_SC_SHORT_NODE_3D_HPP
#include "ScShortView.hpp"
#include "ScInterp.hpp"

/// Medium on a z plane using the original tensor-product stencil.
template <ScInterp Interp>
YAKL_INLINE ScMediumPoint sc_short_plane_medium(const ScShortViews<3>& acc, i32 q, fp_t x, fp_t y) {
    constexpr i32 lo = (Interp == ScInterp::Linear) ? -1 : -2;
    constexpr i32 hi = (Interp == ScInterp::Linear) ? 0 : 1;
    constexpr i32 width = hi - lo + 1;
    i32 cx, cy;
    fp_t ta, tb;
    sc_interp_position(x, &cx, &ta);
    sc_interp_position(y, &cy, &tb);
    const i32 z = acc.medium.lev(q);
    i32 cols[width];
    i32 rows[width];
    for (int k = 0; k < width; ++k) {
        cols[k] = acc.medium.col(cx + lo + k);
        rows[k] = acc.medium.row(cy + lo + k);
    }
    return sc_interp_medium_2d<Interp>(
        [&](i32 i, i32 j) {
            return acc.medium.cell(z + rows[j - lo] + cols[i - lo]);
        },
        ta,
        tb);
}

/// Medium on an x/y face using the original horizontal/vertical stencil.
template <ScInterp Interp>
YAKL_INLINE ScMediumPoint
sc_short_face_medium(
    const ScShortViews<3>& acc,
    i32 p,
    bool x_face,
    i32 c,
    fp_t u,
    fp_t h
) {
    constexpr i32 lo = (Interp == ScInterp::Linear) ? -1 : -2;
    constexpr i32 hi = (Interp == ScInterp::Linear) ? 0 : 1;
    constexpr i32 width = hi - lo + 1;
    i32 cu;
    fp_t ta;
    sc_interp_position(u, &cu, &ta);
    const fp_t tb = FP(1.0) - h;
    const i32 fixed = x_face ? acc.medium.col(c) : acc.medium.row(c);
    i32 horiz[width];
    i32 levs[width];
    for (int k = 0; k < width; ++k) {
        horiz[k] = x_face ? acc.medium.row(cu + lo + k) : acc.medium.col(cu + lo + k);
        levs[k] = acc.medium.lev(p + lo + k);
    }
    return sc_interp_medium_2d<Interp>(
        [&](i32 i, i32 j) {
            return acc.medium.cell(levs[j - lo] + fixed + horiz[i - lo]);
        },
        ta,
        tb);
}

/// Start at physical side inflow and its edge medium.
template <ScInterp Interp, class Upwind, class Entry, class Step>
YAKL_INLINE void sc_short_upwind_from_side(
    Upwind& upwind,
    const Entry& entry,
    const Step& step,
    i32 p,
    i32 s,
    i32 sy,
    const ScShortViews<3>& acc) {
    const fp_t tx = step.tx, ty = step.ty;
    // NOTE(cmo): Enters through a side face; the medium is that of the
    // edge node plane there.
    upwind.h = entry.h_side;
    upwind.I = acc.inflow.side(acc.medium, p, s, sy);
    upwind.medium = entry.side_x
                        ? sc_short_face_medium<Interp>(
                            acc,
                            p,
                            true,
                            0,
                            entry.y0 + upwind.h * ty,
                            upwind.h
                        )
                        : sc_short_face_medium<Interp>(
                            acc,
                            p,
                            false,
                            0,
                            entry.x0 + upwind.h * tx,
                            upwind.h
                        );
}

/// Interpolate between the physical face and the first interior node.
template <ScInterp Interp, class Upwind, class Entry, class Step>
YAKL_INLINE void sc_short_upwind_from_edge_half_cell(
    Upwind& upwind,
    const Entry& entry,
    const Step& step,
    i32 p,
    i32 s,
    i32 sy,
    const ScShortViews<3>& acc) {
    const fp_t tx = step.tx, ty = step.ty;
    auto fetch_I = [&](i32 p, i32 s, i32 sy) {
        return acc.intensity.I(
            acc.medium,
            acc.inflow,
            p,
            s,
            sy
        );
    };
    const bool half_x = !acc.medium.periodic_x && entry.x0 < FP(0.0);
    const bool half_y = !acc.medium.periodic_y && entry.y0 < FP(0.0);
    // NOTE(cmo): Inside an edge half cell: linear between the face
    // value and node 0, as a ghost node 2 I_face - I_0 (bilinear).
    i32 cx, cy;
    fp_t ta, tb;
    sc_interp_position(entry.x0, &cx, &ta);
    sc_interp_position(entry.y0, &cy, &tb);
    auto value = [&](i32 i, i32 j) -> fp_t {
        if (half_x && i < 0 && half_y && j < 0) {
            return std::max(FP(2.0) * acc.inflow.xface(acc.medium, p, -1) - fetch_I(p - 1, 0, 0), FP(0.0));
        }
        if (half_x && i < 0) {
            return std::max(FP(2.0) * acc.inflow.xface(acc.medium, p, j) - fetch_I(p - 1, 0, j), FP(0.0));
        }
        if (half_y && j < 0) {
            return std::max(FP(2.0) * acc.inflow.yface(acc.medium, p, i) - fetch_I(p - 1, i, 0), FP(0.0));
        }
        return fetch_I(p - 1, i, j);
    };
    upwind.I = sc_interp_2d<ScInterp::Linear>(
        [&](i32 i, i32 j) {
            return value(cx + i, cy + j);
        },
        ta,
        tb);
    upwind.medium = sc_short_plane_medium<Interp>(acc, p - 1, entry.x0, entry.y0);
}

/// Interpolate completed intensity and medium on the previous plane.
template <ScInterp Interp, class Upwind, class Entry, class Step>
YAKL_INLINE void sc_short_upwind_from_previous_plane(
    Upwind& upwind,
    const Entry& entry,
    const Step& step,
    i32 p,
    i32 s,
    i32 sy,
    const ScShortViews<3>& acc) {
    const fp_t tx = step.tx, ty = step.ty;
    auto fetch_I = [&](i32 p, i32 s, i32 sy) {
        return acc.intensity.I(
            acc.medium,
            acc.inflow,
            p,
            s,
            sy
        );
    };
    constexpr i32 lo = (Interp == ScInterp::Linear) ? -1 : -2;
    constexpr i32 hi = (Interp == ScInterp::Linear) ? 0 : 1;
    constexpr i32 width = hi - lo + 1;
    i32 cx, cy;
    fp_t ta, tb;
    sc_interp_position(entry.x0, &cx, &ta);
    sc_interp_position(entry.y0, &cy, &tb);
    if (acc.intensity.I_inside(
        acc.medium,
        p - 1,
        cx + lo,
        cx + hi,
        cy + lo,
        cy + hi
    )) {
        upwind.I = sc_interp_2d<Interp>(
            [&](i32 i, i32 j) {
                return acc.intensity.I_fast(acc.medium, p - 1, cx + i, cy + j);
            },
            ta,
            tb);
    } else {
        upwind.I = sc_interp_2d<Interp>(
            [&](i32 i, i32 j) {
                return fetch_I(p - 1, cx + i, cy + j);
            },
            ta,
            tb);
    }
    upwind.medium = sc_short_plane_medium<Interp>(acc, p - 1, entry.x0, entry.y0);
}

#endif
