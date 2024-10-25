#if !defined(DEXRT_PROFILE_NORMALISATION_HPP)
#define DEXRT_PROFILE_NORMALISATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "Atmosphere.hpp"
#include "Voigt.hpp"
#include "EmisOpac.hpp"
#include "RcUtilsModes.hpp"


void compute_profile_normalisation(const State& state, const CascadeState& casc_state);

#else
#endif
