#if !defined(DEXRT_PROFILE_NORMALISATION_HPP)
#define DEXRT_PROFILE_NORMALISATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "State3d.hpp"
#include "Atmosphere.hpp"
#include "Voigt.hpp"
#include "EmisOpac.hpp"
#include "RcUtilsModes.hpp"

struct CascadeState;
struct CascadeState3d;
struct ScState;

void compute_profile_normalisation(const State& state, const CascadeState& casc_state, bool print_worst_wphi=true);
void compute_profile_normalisation(const State3d& state, const CascadeState3d& casc_state, bool print_worst_wphi=true);
void compute_profile_normalisation(const ScState& state, const Fp2d& mu, const Fp1d& wmu, bool print_worst_wphi=true);

#else
#endif
