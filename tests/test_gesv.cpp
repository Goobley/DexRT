#include "catch_amalgamated.hpp"
#include "Types.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Kokkos-Kernels Solve", "[kokkos-lu]") {
        size_t scratch_size = ScratchView<f32**>::shmem_size(3, 3);
        scratch_size += ScratchView<f32*>::shmem_size(3);
        Fp1d result("result", 3);

        Kokkos::parallel_for(
            Kokkos::TeamPolicy(1, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team) {
                ScratchView<f32**> A(team.team_scratch(0), 3, 3);
                ScratchView<f32*> b(team.team_scratch(0), 3);
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    A(0, 0) = FP(100.0);
                    A(0, 1) = FP(3.0);
                    A(0, 2) = FP(-10.0);
                    A(1, 0) = FP(-1000.0);
                    A(1, 1) = FP(0.5);
                    A(1, 2) = FP(5000.0);
                    A(2, 0) = FP(-1e-4);
                    A(2, 1) = FP(5e5);
                    A(2, 2) = FP(1.0);
                    b(0) = FP(37614.0);
                    b(1) = FP(-37811.0);
                    b(2) = FP(6188999991.9996);
                });
                // LU factorise
                KokkosBatched::LU<KTeam, KokkosBatched::Mode::Team, KokkosBatched::Algo::LU::Unblocked>::invoke(
                    team, A
                );
                team.team_barrier();
                // LU Solve
                KokkosBatched::TeamSolveLU<
                    KTeam,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Trsm::Unblocked
                >::invoke(
                    team,
                    A,
                    b
                );
                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, 3),
                    [&] (const int i) {
                        result(i) = b(i);
                    }
                );
            }
        );
        Kokkos::fence();

        auto result_h = result.createHostCopy();
        REQUIRE_THAT(result_h(0), WithinRel(FP(4.0), FP(1e-4)));
        REQUIRE_THAT(result_h(1), WithinRel(FP(12378.0), FP(1e-4)));
        REQUIRE_THAT(result_h(2), WithinRel(FP(-8.0), FP(1e-4)));
}