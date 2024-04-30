#if !defined(DEXRT_STATIC_FORMAL_SOLUTION_HPP)
#define DEXRT_STATIC_FORMAL_SOLUTION_HPP

#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"

// void static_compute_gamma(State* state, int la, const Fp3d& lte_scratch);
// void static_formal_sol_rc(State* state, int la);
struct CascadeState;
void static_formal_sol_rc(const State& state, const CascadeState& casc_state, int la_start, int la_end);

#else
#endif