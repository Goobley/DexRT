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
    ProbesToCompute probes_to_compute;
    MultiResMipChain mip_chain;
    LineSweepData line_sweep_data;

    bool init(const State& state, int max_cascade);
};

struct DeviceCascadeState {
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

template <typename Bc>
struct CascadeStateAndBc {
    const DeviceCascadeState& state;
    const Bc& bc;
};

#else
#endif