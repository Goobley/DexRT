#ifndef DEXRT_SC_SHORT_NODE_2D_HPP
#define DEXRT_SC_SHORT_NODE_2D_HPP
#include "ScShortView.hpp"
#include "ScInterp.hpp"

/// Medium on a completed z plane: keep the one-dimensional interpolation path.
template <ScInterp Interp>
YAKL_INLINE ScMediumPoint sc_short_plane_medium(const ScShortViews<2>& acc, i32 p, fp_t x, fp_t y) {
    const i32 c = i32(std::floor(x)) + 1;
    const fp_t fraction = fp_t(c) - x;
    return sc_interp_medium_1d<Interp>(
        [&](i32 i) {
            return acc.medium.at(p, c + i);
        },
        fraction);
}

/// Medium on an x face, interpolated vertically between neighbouring planes.
template <ScInterp Interp>
YAKL_INLINE ScMediumPoint
sc_short_face_medium(
    const ScShortViews<2>& acc,
    i32 p,
    bool x_face,
    i32 c,
    fp_t u,
    fp_t h
) {
    const fp_t fraction = FP(1.0) - h;
    return sc_interp_medium_1d<Interp>(
        [&](i32 i) {
            return acc.medium.at(p + i, c);
        },
        fraction);
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
    const ScShortViews<2>& acc) {
    // NOTE(cmo): Enters through the side edge.
    upwind.h = entry.h_side;
    upwind.I = acc.inflow.side(acc.medium, p, s);
    upwind.medium = sc_short_face_medium<Interp>(
        acc,
        p,
        true,
        0,
        FP(0.0),
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
    const ScShortViews<2>& acc) {
    // NOTE(cmo): Inside the edge half cell: between the edge (boundary
    // value) and node 0.
    const fp_t frac = (entry.x0 + FP(0.5)) * FP(2.0);
    const fp_t I_edge = acc.inflow.edge(p);
    upwind.I = I_edge + frac * (acc.intensity.I(acc.medium, acc.inflow, p - 1, 0) - I_edge);
    upwind.medium = acc.medium.at(p - 1, 0);
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
    const ScShortViews<2>& acc) {
    const i32 c = i32(std::floor(entry.x0)) + 1;
    const fp_t tf = fp_t(c) - entry.x0;
    constexpr bool linear = (Interp == ScInterp::Linear);
    const fp_t Im1 = acc.intensity.I(acc.medium, acc.inflow, p - 1, c - 1);
    const fp_t I0 = acc.intensity.I(acc.medium, acc.inflow, p - 1, c);
    if (linear) {
        upwind.I = sc_interp<Interp>(
            FP(0.0),
            Im1,
            I0,
            FP(0.0),
            tf
        );
    } else {
        upwind.I = sc_interp<Interp>(
            acc.intensity.I(acc.medium, acc.inflow, p - 1, c - 2),
            Im1,
            I0,
            acc.intensity.I(acc.medium, acc.inflow, p - 1, c + 1),
            tf);
    }
    upwind.medium = sc_short_plane_medium<Interp>(acc, p - 1, entry.x0, FP(0.0));
}

#endif
