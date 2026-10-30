#ifndef DEXRT_SC_SHORT_NODE_HPP
#define DEXRT_SC_SHORT_NODE_HPP
#include "ScShortNode2d.hpp"
#include "ScShortNode3d.hpp"
#include "ScRayMarch.hpp"
#include "ScFormalSolver.hpp"
#include <limits>

/// Plane geometry shared by every node; 2D has no transverse y crossing.
struct ScRowStep {
    fp_t tx, ty, ds;
};

/// Position and medium of the radiation being advanced from plane p-1 to p.
struct ScUpwind {
    fp_t I;
    ScMediumPoint medium;
    fp_t h;
};

struct ScShortNodeResult {
    ScStepResult result;
    fp_t I_uw;
};

/// Canonical entry on the previous plane, physical inflow plane or side face.
/// x_start is retained explicitly to preserve the original 2D expression.
template <int Dim> struct ScNodeEntry {
    fp_t x0 = 0, y0 = 0, x_start = 0, h0 = 0, h_side = 0;
    bool z_edge = false, side = false, side_x = false;
};

/// Locate physical entry without changing the dimensional arithmetic.
template <int Dim>
YAKL_INLINE ScNodeEntry<Dim>
sc_node_entry(
    fp_t tx,
    fp_t ty,
    i32 p,
    i32 s,
    i32 sy,
    bool periodic_x,
    bool periodic_y,
    bool has_upstream
) {
    ScNodeEntry<Dim> entry;
    entry.x0 = fp_t(s) - tx;
    entry.z_edge = (p == 0) && !has_upstream;
    entry.h0 = entry.z_edge ? FP(0.5) : FP(0.0);
    if constexpr (Dim == 2) {
        const fp_t inv_t = tx > FP(0.0) ? FP(1.0) / tx : FP(0.0);
        entry.x_start = entry.x0 + entry.h0 * tx;
        entry.side = !periodic_x && entry.x_start < -FP(0.5);
        entry.h_side = (-FP(0.5) - entry.x0) * inv_t;
    } else {
        entry.y0 = fp_t(sy) - ty;
        const bool enter_x = !periodic_x && entry.x0 + entry.h0 * tx < -FP(0.5);
        const bool enter_y = !periodic_y && entry.y0 + entry.h0 * ty < -FP(0.5);
        const fp_t hx = enter_x ? (-FP(0.5) - entry.x0) / tx : -FP(1.0);
        const fp_t hy = enter_y ? (-FP(0.5) - entry.y0) / ty : -FP(1.0);
        entry.side = enter_x || enter_y;
        entry.side_x = enter_x && hx >= hy;
        entry.h_side = std::max(hx, hy);
    }
    return entry;
}

/// Seed radiation and medium from the physical inflow plane.
template <int Dim, ScInterp Interp>
YAKL_INLINE void sc_short_upwind_from_inflow_plane(
    ScUpwind& upwind,
    const ScNodeEntry<Dim>& entry,
    const ScRowStep& step,
    i32 s,
    i32 sy,
    const ScShortViews<Dim>& view
) {
    upwind.I = view.inflow.bottom(view.medium, s, sy);
    const fp_t x = Dim == 2 ? entry.x_start : entry.x0 + upwind.h * step.tx;
    const fp_t y = Dim == 2 ? FP(0.0) : entry.y0 + upwind.h * step.ty;
    upwind.medium = sc_short_plane_medium<Interp>(view, 0, x, y);
}

/// Choose the physical face, edge half-cell or completed previous plane.
template <int Dim, ScInterp Interp>
YAKL_INLINE ScUpwind sc_short_upwind(
    const ScNodeEntry<Dim>& entry,
    const ScRowStep& step,
    i32 p,
    i32 s,
    i32 sy,
    const ScShortViews<Dim>& view) {
    ScUpwind upwind{.h = entry.h0};
    if (entry.side) {
        sc_short_upwind_from_side<Interp>(
            upwind,
            entry,
            step,
            p,
            s,
            sy,
            view
        );
    } else if (entry.z_edge) {
        sc_short_upwind_from_inflow_plane<Dim, Interp>(
            upwind,
            entry,
            step,
            s,
            sy,
            view
        );
    } else if (
        (!view.medium.periodic_x && entry.x0 < FP(0.0)) ||
        (Dim == 3 && !view.medium.periodic_y && entry.y0 < FP(0.0))) {
        sc_short_upwind_from_edge_half_cell<Interp>(
            upwind,
            entry,
            step,
            p,
            s,
            sy,
            view
        );
    } else {
        sc_short_upwind_from_previous_plane<Interp>(
            upwind,
            entry,
            step,
            p,
            s,
            sy,
            view
        );
    }
    return upwind;
}

/// Advance only intermediate crossings; finishing the node is a separate operation.
template <int Dim, ScFormalSolver Solver, ScInterp Interp>
YAKL_INLINE ScUpwind sc_march_crossings(
    ScUpwind upwind,
    const ScNodeEntry<Dim>& entry,
    const ScRowStep& step,
    i32 p,
    i32 s,
    i32 sy,
    const ScShortViews<Dim>& view) {
    auto march = ScRayMarch<Dim - 1>(
        entry.x0,
        Dim == 2 ? FP(0.0) : entry.y0,
        step.tx,
        step.ty,
        upwind.h,
        s,
        Dim == 2 ? std::numeric_limits<i32>::max() : sy);
    if (march.valid()) {
        do {
            const auto segment = march.segment();
            if (segment.last()) {
                break;
            }
            const auto medium =
                segment.axis == 0
                    ? sc_short_face_medium<Interp>(
                        view,
                        p,
                        true,
                        segment.face,
                        segment.y,
                        segment.end
                    )
                    : sc_short_face_medium<Interp>(
                        view,
                        p,
                        false,
                        segment.face,
                        segment.x,
                        segment.end
                    );
            const auto result = sc_formal_step<Solver, false>(
                upwind.I,
                upwind.medium.S,
                upwind.medium.chi,
                medium.S,
                medium.chi,
                segment.fraction() * step.ds);
            upwind.I = result.I;
            upwind.medium = medium;
            if constexpr (Dim == 2) {
                upwind.h = segment.end;
            } else {
                upwind.h = std::max(upwind.h, segment.end);
            }
        } while (march.step());
    }
    return upwind;
}

/// Integrate the remaining segment into the target node.
template <ScFormalSolver Solver>
YAKL_INLINE ScStepResult sc_finish_node(const ScUpwind& upwind, ScMediumPoint node, fp_t ds) {
    return sc_formal_step<Solver>(
        upwind.I,
        upwind.medium.S,
        upwind.medium.chi,
        node.S,
        node.chi,
        (FP(1.0) - upwind.h) * ds);
}

/// One node: entry -> upwind -> march -> finish, in either dimension.
template <int Dim, ScFormalSolver Solver, ScInterp Interp, bool Loss>
YAKL_INLINE ScShortNodeResult
sc_solve_short_node(
    const ScRowStep& step,
    i32 p,
    i32 s,
    i32 sy,
    const ScShortViews<Dim>& view
) {
    const auto entry = sc_node_entry<Dim>(
        step.tx,
        step.ty,
        p,
        s,
        sy,
        view.medium.periodic_x,
        view.medium.periodic_y,
        view.medium.has_upstream);
    auto upwind = sc_short_upwind<Dim, Interp>(
        entry,
        step,
        p,
        s,
        sy,
        view
    );
    const fp_t I_uw = Loss ? upwind.I : FP(0.0);
    upwind = sc_march_crossings<Dim, Solver, Interp>(
        upwind,
        entry,
        step,
        p,
        s,
        sy,
        view
    );
    return {
        .result = sc_finish_node<Solver>(upwind, view.medium.at(p, s, sy), step.ds),
        .I_uw = I_uw,
    };
}
#endif
