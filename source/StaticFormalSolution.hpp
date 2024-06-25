#if !defined(DEXRT_STATIC_FORMAL_SOLUTION_HPP)
#define DEXRT_STATIC_FORMAL_SOLUTION_HPP

#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"

// void static_compute_gamma(State* state, int la, const Fp3d& lte_scratch);
// void static_formal_sol_rc(State* state, int la);
struct CascadeState;
/// General atomic solver for static atmospheres
void static_formal_sol_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate, int la_start, int la_end);
/// Solver for provided emissivity and opacity
void static_formal_sol_given_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate, int la_start, int la_end);

#else
#endif