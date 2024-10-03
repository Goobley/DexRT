#if !defined(DEXRT_CASCADE_STATE_HPP)
#define DEXRT_CASCADE_STATE_HPP
#include "Types.hpp"
#include "State.hpp"
#include "RcUtilsModes.hpp"

inline CascadeState CascadeState_new(const CascadeStorage& c0, int num_cascades) {
    CascadeState result;
    if constexpr (PINGPONG_BUFFERS) {
        i64 max_entries = 0;
        for (int i = 0; i <= num_cascades; ++i) {
            auto dims = cascade_size(c0, i);
            max_entries = std::max(max_entries, cascade_entries(dims));
        }
        for (int i = 0; i < 2; ++i) {
            result.i_cascades.push_back(
                Fp1d(
                    "i_cascade",
                    max_entries
                )
            );
            result.tau_cascades.push_back(
                Fp1d(
                    "tau_cascade",
                    max_entries
                )
            );
        }
    } else {
        for (int i = 0; i <= num_cascades; ++i) {
            auto dims = cascade_size(c0, i);
            i64 entries = cascade_entries(dims);
            result.i_cascades.push_back(
                Fp1d(
                    "i_cascade",
                    entries
                )
            );
            result.tau_cascades.push_back(
                Fp1d(
                    "tau_cascade",
                    entries
                )
            );
        }
    }
    result.num_cascades = num_cascades;
    return result;
}

#else
#endif