#if !defined(DEXRT_FEW_FREQ_FORMAL_SOLUTION_HPP)
#define DEXRT_FEW_FREQ_FORMAL_SOLUTION_HPP

#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"

struct CascadeState;
void few_freq_formal_sol_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate);

#else
#endif