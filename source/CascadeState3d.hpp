#if !defined(DEXRT_CASCADE_STATE_3D_HPP)
#define DEXRT_CASCADE_STATE_3D_HPP

#include "Types.hpp"
#include "Mipmaps.hpp"

struct State3d;
struct CascadeState3d {
    int num_cascades;
    std::vector<Fp1d> i_cascades;
    std::vector<Fp1d> tau_cascades;

    bool init(const State3d& state, int max_cascade);
};

#endif
