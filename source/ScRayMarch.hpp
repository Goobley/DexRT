#ifndef DEXRT_SC_RAY_MARCH_HPP
#define DEXRT_SC_RAY_MARCH_HPP
#include "Types.hpp"
#include <cmath>

/// A straight path in the canonical half-octant. Transverse node/cell planes
/// are integer coordinates; the longitudinal coordinate is a fraction [0,1].
/// Bounded state, independent of path length. No stored chain or wrapping:
/// domain-aware marchers or stencil views supply cell/boundary mapping.
template <int Transverse> struct ScRayMarch {
    static_assert(Transverse == 1 || Transverse == 2);
    fp_t x0, y0, dx, dy, inv_x, inv_y;
    i32 next_x, next_y, limit_x, limit_y;
    fp_t begin, end;
    i32 axis;
    bool active;

    struct Segment {
        fp_t begin, end;    // fractions of the complete path
        fp_t x, y;          // endpoint in canonical coordinates
        i32 axis;           // crossed plane: 0=x, 1=y, 2=path endpoint
        i32 face;           // integer coordinate of crossed x/y plane
        i32 cell_x, cell_y; // cell occupied before this crossing
        YAKL_INLINE fp_t fraction() const { return std::max(end - begin, FP(0.0)); }
        YAKL_INLINE bool last() const { return axis == 2; }
    };

    YAKL_INLINE ScRayMarch(
        fp_t x,
        fp_t y,
        fp_t tx,
        fp_t ty,
        fp_t start = FP(0.0),
        i32 last_x = 2147483647,
        i32 last_y = 2147483647
    )
        : x0(x), y0(y), dx(tx), dy(ty), inv_x(tx > FP(0.0) ? FP(1.0) / tx : FP(0.0)),
          inv_y(ty > FP(0.0) ? FP(1.0) / ty : FP(0.0)), next_x(i32(std::floor(x + start * tx)) + 1),
          next_y(Transverse == 2 ? i32(std::floor(y + start * ty)) + 1 : 1), limit_x(last_x), limit_y(last_y),
          begin(start), end(start), axis(2), active(start < FP(1.0)) {
        select_end();
    }
    YAKL_INLINE bool valid() const { return active; }
    YAKL_INLINE Segment segment() const {
        return {
            begin,
            end,
            axis == 0 ? fp_t(next_x) : x0 + end * dx,
            axis == 1 ? fp_t(next_y) : y0 + end * dy,
            axis,
            axis == 0 ? next_x : next_y,
            next_x - 1,
            next_y - 1,
        };
    }
    YAKL_INLINE bool step() {
        if (axis == 2) {
            active = false;
            return false;
        }
        if (axis == 0) {
            ++next_x;
        } else {
            ++next_y;
        }
        begin = std::max(begin, end);
        select_end();
        return true;
    }

    YAKL_INLINE void select_end() {
        const fp_t hx = dx > FP(0.0) && next_x < limit_x ? (fp_t(next_x) - x0) * inv_x : FP(2.0);
        const fp_t hy =
            Transverse == 2 && dy > FP(0.0) && next_y < limit_y ? (fp_t(next_y) - y0) * inv_y : FP(2.0);
        axis = hx <= hy ? 0 : 1; // deterministic x-first corner ties
        end = std::min(hx, hy);
        if (end >= FP(1.0)) {
            end = FP(1.0);
            axis = 2;
        }
    }
};
#endif
