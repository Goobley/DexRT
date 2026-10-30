#if !defined(DEXRT_SC_RADIATION_HPP)
#define DEXRT_SC_RADIATION_HPP

#include "Types.hpp"
#include "ScState.hpp"
#include "ScMediumCoefficients.hpp"
#include "ScPreparedGeometry.hpp"
#include "ScFormal.hpp"

/// Accumulate the batch's intensity into Gamma (and the rate diagnostics),
/// chi/S: the batch's medium. lambda_iterate: without preconditioning.
/// J: if initialized, also accumulate J over the active cells into this
/// slot (as sc_accumulate_J<false>).
void sc_accumulate_gamma(
    const ScState& state,
    const ScMediumCoefficients& medium_coefficients,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMedium& medium,
    const ScSolution& solution,
    i32 la_start,
    i32 wave_batch,
    bool lambda_iterate,
    const Fp2d& J,
    const Rad16Array& previous_J
);

/// Accumulate the batch's I into J, its wavelength batch's slot [wave,
/// owned ks], S: the batch's medium. Dense: every owned cell (void
/// included, e.g. a final dense solve), otherwise the active cells only.
template <bool Dense>
void sc_accumulate_J(
    const ScState& state,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMedium& medium,
    const ScSolution& solution,
    const Fp2d& J,
    i32 wave_batch
);

/// Store upward I at the top boundary: emergent [la, compact upward direction, plane].
void sc_store_emergent(
    const ScGrid& grid,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMedium& medium,
    const ScSolution& solution,
    const Fp3d& outgoing,
    const Fp3d& emergent,
    i32 la_start,
    i32 wave_batch
);

#else
#endif
