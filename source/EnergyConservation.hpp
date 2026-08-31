#if !defined(DEXRT_ENERGY_CONSERVATION_HPP)
#define DEXRT_ENERGY_CONSERVATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include <inttypes.h>

template <typename State>
inline fp_t simple_conserve_energy(State* state) {
    fp_t max_change;
    if (!state->atmos.e_int.initialized()) {
        throw std::runtime_error("Calling conserve_energy without providing e_int at launch!");
    }
#ifdef HAVE_MPI
    if (state->mpi_state.rank == 0) {
#endif
        yakl::timer_start("e_int conservation");
        assert(state->have_h && "Requires H model to be present");
        using namespace ConstantsFP;
        constexpr fp_t invgammam1 = FP(1.0) / (FP(5.0) / FP(3.0) - 1.0);

        // NOTE(cmo): Set total_abund to 1.0 in the config to match Mosscap,
        // which feeds in only H.
        const fp_t total_abund = state->config.total_abund;
        JasUnpack((*state), pops, adata);
        const auto h_pops = slice_pops(pops, state->adata_host, 0);
        // NOTE(cmo): This does neglect everything other than H, but that's consistent with mosscap for now.

        constexpr fp_t min_temperature = 1e3;

        auto flatmos = flatten<fp_t>(state->atmos);
        Fp1d rel_change("temperature_rel_change", flatmos.nh_tot.extent(0));
        dex_parallel_for(
            "Compute correction",
            FlatLoop<1>(flatmos.nh_tot.extent(0)),
            YAKL_LAMBDA (i64 k) {
                const fp_t e_intk = flatmos.e_int(k);
                fp_t e_ion = FP(0.0);
                for (int j = 0; j < h_pops.extent(0); ++j) {
                    e_ion += h_pops(j, k) * adata.energy(j) * eV;
                }
                const fp_t e_th_target = e_intk - e_ion;
                const fp_t N = total_abund * flatmos.nh_tot(k) + flatmos.ne(k);
                const fp_t e_th_error = e_th_target - N * (k_B * invgammam1) * flatmos.temperature(k);
                fp_t delta_temp = e_th_error / (N * (k_B * invgammam1));

                fp_t new_temperature = flatmos.temperature(k) + delta_temp;
                if (new_temperature < min_temperature) {
    #ifndef KOKKOS_ENABLE_SYCL
                    printf("Temperature driven below floor @ k = %" PRId64 ", clamped to %f K\n", k, f64(min_temperature));
    #endif
                    new_temperature = min_temperature;
                }
                rel_change(k) = std::abs(FP(1.0) - new_temperature / flatmos.temperature(k));
                flatmos.temperature(k) = new_temperature;

                // NOTE(cmo): Make pressure consistent (in case someone else uses it)
                flatmos.pressure(k) = N * k_B * new_temperature;
            }
        );
        yakl::fence();

        typedef Kokkos::MaxLoc<fp_t, i64> MaxLoc;
        MaxLoc::value_type maxloc;

        dex_parallel_reduce(
            "EnergyConsMaxLoc",
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

        state->println(
            "     Max Change temperature (e_int): {} (@ {})",
            max_change,
            max_change_loc
        );
        yakl::timer_stop("e_int conservation");
#ifdef HAVE_MPI
    }
    MPI_Bcast(&max_change, 1, get_FpMpi(), 0, state->mpi_state.comm);
#endif
    return max_change;
}

#else
#endif