#if !defined(DEXRT_CHARGE_CONSERVATION_HPP)
#define DEXRT_CHARGE_CONSERVATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "Collisions.hpp"
#include "GammaMatrix.hpp"

struct NrPostUpdateOptions {
    /// When computing relative change, ignore the change in populations with a
    /// starting fraction lower than this
    fp_t ignore_change_below_ntot_frac = FP(0.0);
    bool conserve_pressure = false;
    /// Set the total baryon abundance relative to H (e.g. 1.1 for a 10% He mix). Negative for default
    fp_t total_abund = FP(-1.0);
};


/**
 * Computes the post statistical equilibrium charge conservation update for H (only, currently).
 * Internal precision configured in Config with StatEq.
 */
template <typename State>
fp_t nr_post_update(State* state, const NrPostUpdateOptions& args = NrPostUpdateOptions());

#else
#endif