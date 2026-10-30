#ifndef DEXRT_SC_MEDIUM_HPP
#define DEXRT_SC_MEDIUM_HPP
#include "Types.hpp"
#include "ScAngularQuadrature.hpp"

struct ScDirBatch;
struct ScState;
struct ScMediumCoefficients;

struct ScMedium {
    Fp3d chi, S; // direction, wavelength, canonical cell (including z halos)
};

/// Fill chi/S (including vacuum and atmospheric halo cells) for one job.
/// coefficients is null for GivenFs. No dependence on prior buffer contents.
void sc_prepare_medium(
    const ScState& state,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMediumCoefficients* coefficients,
    const ScMedium& medium,
    i32 la,
    i32 width
);
#endif
