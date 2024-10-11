#if !defined(DEXRT_STATIC_FORMAL_SOLUTION_HPP)
#define DEXRT_STATIC_FORMAL_SOLUTION_HPP

#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"

struct CascadeState;
/// Solver for provided emissivity and opacity
void static_formal_sol_given_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate, int la_start, int la_end);

#else
#endif