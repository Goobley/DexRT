#include "NgAcceleration.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"

bool NgAccelerator::accelerate(const State& state, fp_t change) {
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
    KView<f64***, HostSpace> aa("Ng A", num_level, 3, 3);
    KView<f64**, HostSpace> bb("Ng b", num_level, 3);

    Kokkos::parallel_for(
        "Construct Ng Matrices",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(Kokkos::DefaultHostExecutionSpace(), 0, num_level),
        KOKKOS_CLASS_LAMBDA (const int l) {
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
                aa(l, 0, 1) += weight * d1 * d2;
                aa(l, 0, 2) += weight * d1 * d3;
                aa(l, 1, 1) += weight * d2 * d2;
                aa(l, 1, 2) += weight * d2 * d3;
                aa(l, 2, 2) += weight * d3 * d3;
                bb(l, 0) += weight * d0 * d1;
                bb(l, 1) += weight * d0 * d2;
                bb(l, 2) += weight * d0 * d3;
            }
            aa(l, 1, 0) = aa(l, 0, 1);
            aa(l, 2, 0) = aa(l, 0, 2);
            aa(l, 2, 1) = aa(l, 1, 2);
        }
    );
    Kokkos::fence();

    Kokkos::parallel_for(
        "Solve Ng matrices",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(Kokkos::DefaultHostExecutionSpace(), 0, num_level),
        KOKKOS_LAMBDA (int l) {
            auto inner_a = Kokkos::subview(aa, l, Kokkos::ALL(), Kokkos::ALL());
            auto inner_b = Kokkos::subview(bb, l, Kokkos::ALL());

            KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Default>::invoke(inner_a);
            KokkosBatched::SerialSolveLU<KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Trsm::Default>::invoke(inner_a, inner_b);
        }
    );
    Kokkos::fence();

    auto pops_hist = create_device_copy(pops);
    auto bb_d = create_device_copy(bb);
    dex_parallel_for(
        "Update pops",
        FlatLoop<2>(num_level, num_space),
        KOKKOS_LAMBDA (i32 l, i64 ks) {
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