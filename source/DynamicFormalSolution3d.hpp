#if !defined(DEXRT_DYNAMIC_FORMAL_SOLUTION_3D_HPP)
#define DEXRT_DYNAMIC_FORMAL_SOLUTION_3D_HPP

#include "State3d.hpp"
#include "CascadeState3d.hpp"

void dynamic_formal_sol_rc_3d(const State3d& state, const CascadeState3d& casc_state, int la);

#else
#endif