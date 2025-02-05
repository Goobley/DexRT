#if !defined(DEXRT_NG_ACCELERATION_HPP)
#define DEXRT_NG_ACCELERATION_HPP

#include "Config.hpp"

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
    bool accelerate(const State& state, fp_t change) {
        yakl::timer_start("Ng Acceleration");
        i32 storage_idx = step_count % num_steps;
        step_count += 1;

        const Fp2d& pops_in = state.pops;

        copy_pops(storage_idx, pops_in);
        change_hist(storage_idx) = change;

        if (storage_idx != num_steps - 1 || change < lower_tol) {
            yakl::timer_stop("Ng Acceleration");
            return false;
        }

        for (int i = 0; i < num_steps; ++i) {
            if (change_hist(i) > accel_tol) {
                yakl::timer_stop("Ng Acceleration");
                return false;
            }
        }

        // Ng matrices (1 per level)
        i64 num_space = pops.extent(2);
        i64 num_level = pops.extent(0);
        static_assert(num_steps == 5, "Need to update Ng algorithm");
        yakl::Array<f64, 3, yakl::memHost> aa("Ng A", num_level, 3, 3);
        yakl::Array<f64, 2, yakl::memHost> bb("Ng b", num_level, 3);

        for (int l = 0; l < pops.extent(0); ++l) {
            for (int i = 0; i < 3; ++i) {
                bb(l, i) = FP(0.0);
                for (int j = 0; j < 3; ++j) {
                    aa(l, i, j) = FP(0.0);
                }
            }

            for (i64 ks = 0; ks < pops.extent(2); ++ks) {
                const fp_t weight = FP(1.0) / square(pops(l, 4, ks));
                const fp_t d0 = pops(l, 4, ks) - pops(l, 3, ks);
                const fp_t d1 = d0 - (pops(l, 3, ks) - pops(l, 2, ks));
                const fp_t d2 = d0 - (pops(l, 2, ks) - pops(l, 1, ks));
                const fp_t d3 = d0 - (pops(l, 1, ks) - pops(l, 0, ks));

                aa(l, 0, 0) += weight * d1 * d1;
                aa(l, 1, 0) += weight * d1 * d2;
                aa(l, 2, 0) += weight * d1 * d3;
                aa(l, 1, 1) += weight * d2 * d2;
                aa(l, 2, 1) += weight * d2 * d3;
                aa(l, 2, 2) += weight * d3 * d3;
                bb(l, 0) += weight * d0 * d1;
                bb(l, 1) += weight * d0 * d2;
                bb(l, 2) += weight * d0 * d3;
            }
            aa(l, 0, 1) = aa(l, 1, 0);
            aa(l, 0, 2) = aa(l, 2, 0);
            aa(l, 1, 2) = aa(l, 2, 1);
        }

        auto aa_d = aa.createDeviceCopy();
        auto bb_d = bb.createDeviceCopy();

#ifdef DEXRT_USE_MAGMA
        yakl::Array<f64*, 1, yakl::memDevice> aa_ptrs("aa_ptrs", num_level);
        yakl::Array<f64*, 1, yakl::memDevice> bb_ptrs("bb_ptrs", num_level);
        yakl::Array<i32, 2, yakl::memDevice> ipivs("ipivs", num_level, bb.extent(1));
        yakl::Array<i32*, 1, yakl::memDevice> ipiv_ptrs("ipiv_ptrs", num_level);
        yakl::Array<i32, 1, yakl::memDevice> info("info", num_level);

        dex_parallel_for(
            FlatLoop<1>(num_level),
            YAKL_LAMBDA (int l) {
                aa_ptrs(l) = &aa_d(l, 0, 0);
                bb_ptrs(l) = &bb_d(l, 0);
                ipiv_ptrs(l) = &ipivs(l, 0);
            }
        );
        yakl::fence();

        magma_dgesv_batched_small(
            aa_d.extent(1),
            1,
            aa_ptrs.data(),
            aa_d.extent(1),
            ipiv_ptrs.data(),
            bb_ptrs.data(),
            bb_d.extent(1),
            info.data(),
            num_level,
            state.magma_queue
        );
        magma_queue_sync(state.magma_queue);

        dex_parallel_for(
            "info check",
            FlatLoop<1>(info.extent(0)),
            YAKL_LAMBDA (int k) {
                if (info(k) != 0) {
                    printf("LINEAR SOLVER PROBLEM k: %d, info: %d (Ng accel)\n", k, info(k));
                }
            }
        );
#else
        state.println("Need magma for Ng acceleration (or bring your own matrix solver)");
#endif

        auto pops_hist = pops.createDeviceCopy();
        dex_parallel_for(
            "Update pops",
            FlatLoop<2>(num_level, num_space),
            YAKL_LAMBDA (i32 l, i64 ks) {
                f64 new_val = (FP(1.0) - bb_d(l, 0) - bb_d(l, 1) - bb_d(l, 2)) * pops_hist(l, 4, ks);
                new_val += bb_d(l, 0) * pops_hist(l, 3, ks);
                new_val += bb_d(l, 1) * pops_hist(l, 2, ks);
                new_val += bb_d(l, 2) * pops_hist(l, 1, ks);
                pops_in(l, ks) = new_val;
            }
        );
        yakl::fence();

        yakl::timer_stop("Ng Acceleration");
        return true;
    }

};

#else
#endif