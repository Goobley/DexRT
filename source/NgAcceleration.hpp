#if !defined(DEXRT_NG_ACCELERATION_HPP)
#define DEXRT_NG_ACCELERATION_HPP

#include "Config.hpp"
#include "Types.hpp"

struct NgAccelArgs {
    i64 num_level;
    i64 num_space;
    fp_t accel_tol = FP(5e-2);
    fp_t lower_tol = FP(2e-4);
};

struct NgAccelerator {
    static constexpr i32 num_steps = 5;
    i32 step_count = 0;
    Fp3dHost pops; // [level, iter_step, ks]
    fp_t accel_tol = FP(5e-2); /// Accelerate if all of the previous steps are below this threshold
    fp_t lower_tol = FP(2e-4); /// Tolerance below which to disable acceleration
    yakl::SArray<fp_t, 1, num_steps> change_hist;

    bool init(const NgAccelArgs& args) {
        pops = decltype(pops)("ng_pops", args.num_level, num_steps, args.num_space);
        accel_tol = args.accel_tol;
        lower_tol = args.lower_tol;
        return true;
    }

    void copy_pops(i32 storage_idx, const Fp2d& pops_in) {
        JasUnpack((*this), pops);
        auto pops_in_h = pops_in.createHostCopy();
        // TODO(cmo): Tidy this up when we move to kokkos
        for (i32 l = 0; l < pops.extent(0); ++l) {
            for (i64 ks = 0; ks < pops.extent(2); ++ks) {
                pops(l, storage_idx, ks) = pops_in_h(l, ks);
            }
        }
    }

    /// Copies the populations into the Ng buffer and returns true if they were
    /// updated (i.e. accelerated)
    bool accelerate(const State& state, fp_t change);
};

#else
#endif