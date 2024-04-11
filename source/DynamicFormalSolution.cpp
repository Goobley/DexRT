#include "DynamicFormalSolution.hpp"

void dynamic_formal_sol_rc(State* state, int la) {
    // 1. Accumulate the static components of eta/chi; where norm(v) is low (< 1km/s?), add lines to this too
    // 1b. Don't mipmap as the next steps aren't compatible.
    // 2. Radiance cascade solver, propagating atomic state downwards to 2b.
    // 2b. Raymarch the grid with new march function that samples the constant eta/chi where norm(v) is small, and adds the local doppler shifted values where norm(v) is larger. For this it requires access to the state
    // 2c. When computing C0, accumulate necessary terms into gamma - alo array is unneeded here as it's done in one pass
}
