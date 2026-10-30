#ifndef DEXRT_SC_FORMAL_HPP
#define DEXRT_SC_FORMAL_HPP
#include "Types.hpp"
#include "ScConfig.hpp"
#include "ScMedium.hpp"
#include "Constants.hpp"
#include "ScGeometry.hpp"
#include <variant>

/// rad_loss for a direction batch: each direction's conservative difference
/// (I_out - I_in) / ds, weighted by 4 pi dlambda w_mu, added into the
/// wavelength batch's slot [wave, owned ks] (as RC's march). Off if loss
/// isn't initialized.
struct ScLossSink {
    Fp2d loss;
    yakl::Array<fp_t const, 1, yakl::memDevice> wavelength_bin;
    Fp1d wmu;
    yakl::Array<i32, 1, yakl::memDevice> dir_idx;
    i32 la_start = 0;

    bool enabled() const { return loss.initialized(); }
    YAKL_INLINE fp_t weight(i32 d, i32 w) const {
        return ConstantsFP::four_pi * wavelength_bin(la_start + w) * wmu(dir_idx(d));
    }
    /// Canonical node (p, s, sy) of frame f.
    YAKL_INLINE void add(
        const ScFrame& f,
        i32 w,
        i32 p,
        i32 s,
        i32 sy,
        fp_t v
    ) const {
        // NOTE(cmo): The flips are involutions, so canonical -> owned is the
        // same map as owned -> canonical.
        Kokkos::atomic_add(&loss(w, sc_canonical_idx(f, s, sy, p)), v);
    }
};

struct ScDirBatch;

/// One task's inputs, dependencies and optional loss destination. Geometry
/// and reusable workspace are separate borrows with their own lifetimes.
struct ScSweepTask {
    const ScDirBatch& batch;
    i32 la_start;
    i32 wave_batch;
    bool has_upstream;
    bool has_downstream;
    ScLossSink loss;
};

struct ScSolution {
    Fp3d I_ext, alo, one_minus_alo; // canonical owned cells
    YAKL_INLINE fp_t intensity(i32 d, i32 w, i64 cell, fp_t source) const {
        return I_ext(d, w, cell) + alo(d, w, cell) * source;
    }
};
struct ScShortStorage {
    Fp3d bottom, edge, ghost, side;
    Fp3d xface, yface, ghost_x, ghost_y;
    Fp3d intensity_planes; // Last two completed planes in either dimension.
    i32 side_x = 1, side_y = 1;
};
struct ScLongerStorage {
    Fp3d bottom, side_x, side_y;
    Fp3d host_lines; // CPU only: bounded state for a plane of characteristics
};

/// One synchronous job borrows these buffers through accumulation. Wavelength
/// sums and MPI host rings have separate owners and longer lifetimes.
struct ScBlockBuffers {
    std::variant<ScShortStorage, ScLongerStorage> scratch;

    ScMedium medium;
    ScSolution solution;
    Fp3d incoming; // completed nodal plane (Short) or face characteristics (Longer)
    Fp3d outgoing; // completed downstream plane, independent of solver
    const ScShortStorage& short_storage() const { return std::get<ScShortStorage>(scratch); }
    const ScLongerStorage& longer_storage() const { return std::get<ScLongerStorage>(scratch); }
};

struct ScPreparedGeometry;
/// Integrate a prepared task; physical inflow and medium are already filled.
void sc_solve_short(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    ScFormalSolver solver,
    ScInterp interp);
void sc_solve_longer(const ScPreparedGeometry& geometry, const ScBlockBuffers& work, const ScSweepTask& task);

/// Allocate a reusable workspace from geometry's derived storage requirements.
ScBlockBuffers sc_make_buffers(const ScPreparedGeometry& geometry, i32 width);
/// LOCAL's invariant-y medium is homogeneous: I=S if opaque, zero otherwise.
void sc_along_y_2d(const ScPreparedGeometry& geometry, const ScBlockBuffers& work, const ScSweepTask& task);
/// Select Short or Longer once, after inflow and medium are ready.
void sc_solve(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScOptions& opts);
/// Resolve a host boolean into a compiled kernel variant.
template <class Run> void sc_dispatch_bool(bool value, const Run& run) {
    if (value) {
        run.template operator()<true>();
    } else {
        run.template operator()<false>();
    }
}
#endif
