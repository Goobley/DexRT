#if !defined(DEXRT_CASCADE_STATE_HPP)
#define DEXRT_CASCADE_STATE_HPP
#include "Types.hpp"
#include "Mipmaps.hpp"
#include "LineSweepSetup.hpp"

struct State;
struct CascadeState {
    int num_cascades;
    std::vector<Fp1d> i_cascades;
    std::vector<Fp1d> tau_cascades;
    Fp1d alo;
    ProbesToCompute2d probes_to_compute;
    MultiResMipChain mip_chain;
    LineSweepData line_sweep_data;

    bool init(const State& state, int max_cascade);
};

/// Index i for cascade n, ip for n+1. If no n+1, ip=-1
struct CascadeIdxs {
    int i;
    int ip;
};

template <typename CascadeState>
YAKL_INLINE CascadeIdxs cascade_indices(const CascadeState& casc, int n) {
    CascadeIdxs idxs;
    if constexpr (PINGPONG_BUFFERS) {
        if (n & 1) {
            idxs.i = 1;
            idxs.ip = 0;
        } else {
            idxs.i = 0;
            idxs.ip = 1;
        }
    } else {
        idxs.i = n;
        idxs.ip = n + 1;
    }

    if (n == casc.num_cascades) {
        idxs.ip = -1;
    }
    return idxs;
}

template <typename CascadeStorage>
struct DeviceCascadeStateImpl {
    int num_cascades;
    int n;
    CascadeStorage casc_dims;
    CascadeStorage upper_dims;
    Fp1d cascade_I;
    Fp1d cascade_tau;
    FpConst1d upper_I;
    FpConst1d upper_tau;
    Fp1d alo; /// [ks, phi, wave, theta], but flattened. Index using Cascade operators (probe_lin_index) -- you need to fetch intensity at the same time anyway.
};
typedef DeviceCascadeStateImpl<CascadeStorage> DeviceCascadeState;
typedef DeviceCascadeStateImpl<CascadeStorage3d> DeviceCascadeState3d;

template <typename Bc>
struct CascadeStateAndBc {
    const DeviceCascadeState& state;
    const Bc& bc;
};

#else
#endif