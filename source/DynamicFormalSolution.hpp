#if !defined(DEXRT_DYNAMIC_FORMAL_SOLUTION_HPP)
#define DEXRT_DYNAMIC_FORMAL_SOLUTION_HPP

#include "Types.hpp"
#include "State.hpp"

// void dynamic_formal_sol_rc(State* state, int la);
struct CascadeState;
void dynamic_formal_sol_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate, int la_start, int la_end);

#else
#endif