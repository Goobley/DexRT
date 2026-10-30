#if !defined(DEXRT_PRESSURE_CONSERVATION_HPP)
#define DEXRT_PRESSURE_CONSERVATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "CellList.hpp"
#include <inttypes.h>

template <typename State>
inline fp_t simple_conserve_pressure(State* state) {
    const auto cells = cell_list(*state);
    if (cells.num() == 0 && (state->mpi_state.distributed || state->mpi_state.num_ranks == 1)) {
        return FP(0.0);
    }
    fp_t max_change;
#ifdef HAVE_MPI
    if (state->mpi_state.rank == 0 || state->mpi_state.distributed) {
#endif
        yakl::timer_start("Pressure conservation");
        assert(state->have_h && "Requires H model to be present");
        using namespace ConstantsFP;

        const fp_t total_abund = state->config.total_abund;
        const auto h_pops = slice_pops(state->pops, state->adata_host, 0);

        auto flatmos = flatten<fp_t>(state->atmos);
        // NOTE(cmo): Per visited cell i (ks = cells(i)).
        Fp1d nh_tot_correction("nhtot_corr", cells.num());
        Fp1d nh_tot_ratio("nhtot_ratio", cells.num());
        Fp1d rel_change("nhtot_rel_change", cells.num());
        dex_parallel_for(
            "Compute correction",
            FlatLoop<1>(cells.num()),
            YAKL_LAMBDA (i64 i) {
                const i64 k = cells(i);
                const fp_t N = total_abund * flatmos.nh_tot(k) + flatmos.ne(k);
                const fp_t N_error = flatmos.pressure(k) / (k_B * flatmos.temperature(k)) - N;
                // NOTE(cmo): This is the simplified pw treatment -- assuming all electrons come from H
                const fp_t N_corr = N_error / (total_abund + h_pops(h_pops.extent(0)-1, k) / flatmos.nh_tot(k));
                nh_tot_correction(i) = N_corr;
                fp_t new_nhtot = flatmos.nh_tot(k) + N_corr;
                if (new_nhtot < FP(0.0)) {
    #ifndef KOKKOS_ENABLE_SYCL
                    printf("Nhtot driven negative @ k = %" PRId64 ", clamped to 1.0e10 m-3\n", k);
    #endif
                    new_nhtot = FP(1e10);
                }
                nh_tot_ratio(i) = new_nhtot / flatmos.nh_tot(k);
                rel_change(i) = std::abs(FP(1.0) - nh_tot_ratio(i));
            }
        );
        yakl::fence();

        typedef Kokkos::MaxLoc<fp_t, i64> MaxLoc;
        MaxLoc::value_type maxloc;

        dex_parallel_reduce(
            "PressureConsMaxLoc",
            FlatLoop<1>(rel_change.extent(0)),
            KOKKOS_LAMBDA (const int i, MaxLoc::value_type& max_loc) {
                const fp_t val = rel_change(i);
                if (val > max_loc.val) {
                    max_loc.val = val;
                    max_loc.loc = i;
                }
            },
            MaxLoc(maxloc)
        );
        max_change = maxloc.val;
        i64 max_change_loc = maxloc.loc;

        dex_parallel_for(
            "Apply updates",
            FlatLoop<1>(cells.num()),
            YAKL_LAMBDA (i64 i) {
                const i64 k = cells(i);
                flatmos.ne(k) += nh_tot_correction(i) * h_pops(h_pops.extent(0)-1, k) / flatmos.nh_tot(k);
                flatmos.nh_tot(k) *= nh_tot_ratio(i);
            }
        );
        yakl::fence();
        const auto& pops = state->pops;
        dex_parallel_for(
            "Rescale pops",
            FlatLoop<2>(pops.extent(0), cells.num()),
            YAKL_LAMBDA (int l, i64 i) {
                pops(l, cells(i)) *= nh_tot_ratio(i);
            }
        );
        yakl::fence();

        state->println(
            "     Max Change nh_tot: {} (@ {})",
            max_change,
            max_change_loc
        );
        yakl::timer_stop("Pressure conservation");
#ifdef HAVE_MPI
    }
    if (!state->mpi_state.distributed) {
        MPI_Bcast(&max_change, 1, get_FpMpi(), 0, state->mpi_state.comm);
    }
#endif
    return max_change;
}

#else
#endif