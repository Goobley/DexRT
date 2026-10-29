#if !defined(DEXRT_DYNAMIC_FORMAL_SOLUTION_HPP)
#define DEXRT_DYNAMIC_FORMAL_SOLUTION_HPP

#include "Types.hpp"
#include "State.hpp"

// void dynamic_formal_sol_rc(State* state, int la);
struct CascadeState;
struct DynamicFormalSolRcOptions {
    int la_start;
    int la_end;
    bool lambda_iterate = false;
    bool compute_rad_loss = false;
    bool compute_gamma = true;
};
void dynamic_formal_sol_rc(const State& state, const CascadeState& casc_state, DynamicFormalSolRcOptions opts);

#else
#endif