#ifndef DEXRT_SC_SHORT_VIEW_HPP
#define DEXRT_SC_SHORT_VIEW_HPP
#include "ScPreparedGeometry.hpp"
#include "ScFormal.hpp"
#include "ScFormalSolver.hpp"
#include "ScRayEntry.hpp"
#include <type_traits>

/// Canonical medium indexing and chi/S reads. Preserve the 3D i32 offsets
/// and restricted pointers; fixed edges clamp only medium samples.
template <int Dim> struct ScShortMedium {
    using Index = std::conditional_t<Dim == 2, i64, i32>;
    const fp_t* __restrict chi = nullptr;
    const fp_t* __restrict S = nullptr;
    i32 Nx, Ny, Np;
    Index plane;
    bool periodic_x, periodic_y, has_upstream, has_downstream;

    YAKL_INLINE static i32 wrap_near(i32 s, i32 n) {
        s = s < 0 ? s + n : s;
        s = s >= n ? s - n : s;
        return s < 0 || s >= n ? sc_wrap(s, n) : s;
    }
    YAKL_INLINE i32 wrap_x(i32 s) const {
        if constexpr (Dim == 2) {
            if (s >= 0 && s < Nx) {
                return s;
            }
            return periodic_x ? sc_wrap(s, Nx) : std::min(s, Nx - 1);
        } else {
            return periodic_x ? wrap_near(s, Nx) : std::min(s, Nx - 1);
        }
    }
    YAKL_INLINE i32 wrap_y(i32 sy) const { return periodic_y ? wrap_near(sy, Ny) : std::min(sy, Ny - 1); }
    YAKL_INLINE Index col(i32 s) const { return std::max(wrap_x(s), 0); }
    YAKL_INLINE Index row(i32 sy) const { return Index(std::max(wrap_y(sy), 0)) * Nx; }
    YAKL_INLINE Index lev(i32 p) const {
        if (p < 0 && !has_upstream) {
            p = 0;
        } else if (p >= Np && !has_downstream) {
            p = Np - 1;
        }
        return Index(p + SC_MEDIUM_HALO) * plane;
    }
    YAKL_INLINE ScMediumPoint cell(Index idx) const { return {.chi = chi[idx], .S = S[idx]}; }
    YAKL_INLINE ScMediumPoint at(i32 p, i32 s, i32 sy = 0) const {
        return cell(lev(p) + (Dim == 3 ? row(sy) : Index(0)) + col(s));
    }
};

/// Named physical inflow samples; each pointer borrows its own boundary array.
template <int Dim> struct ScShortInflow {
    using Index = typename ScShortMedium<Dim>::Index;
    const fp_t* __restrict bottom_values = nullptr;
    const fp_t* __restrict edge_values = nullptr;
    const fp_t* __restrict ghost_values = nullptr;
    const fp_t* __restrict side_values = nullptr;
    const fp_t* __restrict xface_values = nullptr;
    const fp_t* __restrict yface_values = nullptr;
    const fp_t* __restrict ghost_x_values = nullptr;
    const fp_t* __restrict ghost_y_values = nullptr;
    i32 side_x, side_y;

    YAKL_INLINE fp_t ghost(const ScShortMedium<Dim>& m, i32 p, i32 s, i32 sy) const {
        if constexpr (Dim == 2) {
            return ghost_values[2 * (p + 1) + std::min(-s - 1, 1)];
        } else {
            if (s < 0) {
                const i64 row = 2 * (i64(m.Ny) + 2);
                const i64 cell = i64(-std::max(s, -2) - 1) * (m.Ny + 2) + std::max(sy, -2) + 2;
                return ghost_x_values[(p + 1) * row + cell];
            }
            const i64 cell = i64(-std::max(sy, -2) - 1) * m.Nx + s;
            return ghost_y_values[(p + 1) * (2 * i64(m.Nx)) + cell];
        }
    }
    YAKL_INLINE fp_t bottom(const ScShortMedium<Dim>& m, i32 s, i32 sy = 0) const {
        return bottom_values[Index(sy) * m.Nx + s];
    }
    YAKL_INLINE fp_t side(const ScShortMedium<Dim>& m, i32 p, i32 s, i32 sy = 0) const {
        if constexpr (Dim == 2) {
            return side_values[i64(p) * side_x + s];
        } else {
            const i64 row = i64(m.Ny) * side_x + i64(side_y) * m.Nx;
            const i64 cell = s < side_x ? i64(sy) * side_x + s : i64(m.Ny) * side_x + i64(sy) * m.Nx + s;
            return side_values[p * row + cell];
        }
    }
    YAKL_INLINE fp_t edge(i32 p) const { return edge_values[p]; }
    YAKL_INLINE fp_t xface(const ScShortMedium<Dim>& m, i32 p, i32 sy) const {
        if (sy >= 0) {
            sy = m.periodic_y ? sc_wrap(sy, m.Ny) : std::min(sy, m.Ny - 1);
        }
        return xface_values[p * (m.Ny + 1) + sy + 1];
    }
    YAKL_INLINE fp_t yface(const ScShortMedium<Dim>& m, i32 p, i32 s) const {
        s = m.periodic_x ? sc_wrap(s, m.Nx) : std::min(std::max(s, 0), m.Nx - 1);
        return yface_values[p * m.Nx + s];
    }
};

/// Completed intensity uses a two-plane ring in both dimensions. The full
/// split results remain available for Gamma, J and emergent accumulation.
template <int Dim> struct ScShortIntensity {
    using Index = typename ScShortMedium<Dim>::Index;
    const fp_t* __restrict input = nullptr;
    fp_t* __restrict I_ext = nullptr;
    fp_t* __restrict alo = nullptr;
    fp_t* __restrict one_minus_alo = nullptr;
    fp_t* __restrict ring = nullptr;

    YAKL_INLINE fp_t I_fast(const ScShortMedium<Dim>& m, i32 p, i32 s, i32 sy) const {
        const Index c = Index(sy) * m.Nx + s;
        if (p < 0) {
            return input[c];
        }
        return ring[(p & 1) * m.plane + c];
    }
    YAKL_INLINE fp_t
    I(
        const ScShortMedium<Dim>& m,
        const ScShortInflow<Dim>& inflow,
        i32 p,
        i32 s,
        i32 sy = 0
    ) const {
        s = m.wrap_x(s);
        sy = m.wrap_y(sy);
        if (s < 0 || sy < 0) {
            return inflow.ghost(m, p, s, sy);
        }
        return p < 0 && !m.has_upstream ? FP(0.0) : I_fast(m, p, s, sy);
    }
    YAKL_INLINE bool I_inside(
        const ScShortMedium<Dim>& m,
        i32 p,
        i32 s0,
        i32 s1,
        i32 sy0,
        i32 sy1
    ) const {
        return (p >= 0 || m.has_upstream) && s0 >= 0 && s1 < m.Nx && sy0 >= 0 && sy1 < m.Ny;
    }
    YAKL_INLINE void store(const ScShortMedium<Dim>& m, i32 p, i32 c, const ScStepResult& result) const {
        const Index idx = Index(p) * m.plane + c;
        I_ext[idx] = result.I_ext;
        alo[idx] = result.alo;
        one_minus_alo[idx] = result.one_minus_alo;
        ring[(p & 1) * m.plane + c] = result.I;
    }
};

template <int Dim> struct ScShortViews {
    ScShortMedium<Dim> medium;
    ScShortIntensity<Dim> intensity;
    ScShortInflow<Dim> inflow;
};

/// Host preparation captures only the arrays the Short kernel uses. bind()
/// selects one pair on-device; its pointers are borrowed for that kernel.
template <int Dim> struct ScShortBatch {
    Fp3d chi, S, I_ext, alo, one_minus_alo, input, ring;
    Fp3d bottom, edge, ghost, side, xface, yface, ghost_x, ghost_y;
    ScShortViews<Dim> views;

    explicit ScShortBatch(
        const ScPreparedGeometry& prepared,
        const ScBlockBuffers& work,
        const ScSweepTask& task)
        : chi(work.medium.chi),
          S(work.medium.S),
          I_ext(work.solution.I_ext),
          alo(work.solution.alo),
          one_minus_alo(work.solution.one_minus_alo),
          input(work.incoming),
          ring(work.short_storage().intensity_planes),
          bottom(work.short_storage().bottom),
          edge(work.short_storage().edge),
          ghost(work.short_storage().ghost),
          side(work.short_storage().side),
          xface(work.short_storage().xface),
          yface(work.short_storage().yface),
          ghost_x(work.short_storage().ghost_x),
          ghost_y(work.short_storage().ghost_y),
          views{
              .medium = {
                  .Nx = prepared.grid.num_x,
                  .Ny = prepared.grid.num_y,
                  .Np = prepared.grid.num_z,
                  .plane = typename ScShortMedium<Dim>::Index(prepared.grid.plane_size()),
                  .periodic_x = prepared.periodic_x,
                  .periodic_y = prepared.periodic_y,
                  .has_upstream = task.has_upstream,
                  .has_downstream = task.has_downstream,
              },
              .intensity = {},
              .inflow = {
                  .side_x = work.short_storage().side_x,
                  .side_y = work.short_storage().side_y,
              },
          } {}

    YAKL_INLINE ScShortViews<Dim> bind(i32 d, i32 w) const {
        auto result = views;
        result.medium.chi = &chi(d, w, 0);
        result.medium.S = &S(d, w, 0);
        result.intensity.I_ext = &I_ext(d, w, 0);
        result.intensity.alo = &alo(d, w, 0);
        result.intensity.one_minus_alo = &one_minus_alo(d, w, 0);
        result.intensity.input = &input(d, w, 0);
        result.inflow.bottom_values = &bottom(d, w, 0);
        result.inflow.side_values = &side(d, w, 0);
        if constexpr (Dim == 2) {
            result.inflow.edge_values = &edge(d, w, 0);
            result.inflow.ghost_values = &ghost(d, w, 0);
        } else {
            result.inflow.xface_values = &xface(d, w, 0);
            result.inflow.yface_values = &yface(d, w, 0);
            result.inflow.ghost_x_values = &ghost_x(d, w, 0);
            result.inflow.ghost_y_values = &ghost_y(d, w, 0);
        }
        result.intensity.ring = &ring(d, w, 0);
        return result;
    }
};

#endif
