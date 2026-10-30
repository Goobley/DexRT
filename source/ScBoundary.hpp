#ifndef DEXRT_SC_BOUNDARY_HPP
#define DEXRT_SC_BOUNDARY_HPP
#include "ScPreparedGeometry.hpp"
#include "ScFormal.hpp"
#include "BoundaryType.hpp"
#include "PromweaverBoundary.hpp"
#include "ZeroBoundary.hpp"
#include "ThermalisedBoundary.hpp"
#include <variant>

using ScBoundary = std::variant<ZeroBc, PwBc<>, ThermalisedBc>;
/// Precompute physical inflow samples; incoming MPI rows are unpacked by the
/// formal-solution owner. Neither operation depends on medium coefficients.
void sc_prepare_short_boundaries(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScBoundary& boundary
);
void sc_prepare_longer_boundaries(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScBoundary& boundary
);

ScBoundary sc_resolve_boundary(
    BoundaryType type,
    const PwBc<>& pw_bc,
    const ThermalisedBc& thermalised
);
/// Dispatch physical inflow independently of the task's medium.
void sc_prepare_boundaries(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScBoundary& boundary
);
/// Consume the host receive row before the scheduler reuses it.
void sc_unpack_intensity(
    const Fp1dHost& row,
    const Fp1d& staging,
    const Fp3d& incoming,
    i32 num_dirs,
    i32 wave_batch
);
/// Complete the host copy before MPI borrows the independent send row.
void sc_pack_intensity(
    const Fp3d& outgoing,
    const Fp1d& staging,
    const Fp1dHost& row,
    i32 num_dirs,
    i32 wave_batch
);
#endif
