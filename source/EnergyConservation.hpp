#if !defined(DEXRT_ENERGY_CONSERVATION_HPP)
#define DEXRT_ENERGY_CONSERVATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
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

        // NOTE(cmo): These may need tuning.
        constexpr fp_t min_temperature = FP(1e3);
        // For the de_ion/dT term
        // NOTE(cmo): These may need tuning.
        constexpr fp_t min_temperature = FP(1e3);
        // For the de_ion/dT term
        constexpr fp_t damping = FP(2.0);
        constexpr fp_t damping_T_ref = FP(1e4);
        constexpr fp_t max_frac_change = FP(0.2);
        constexpr fp_t damping_T_ref = FP(1e4);
        constexpr fp_t max_frac_change = FP(0.2);

        yakl::Array<i32, 1, yakl::memDevice> limited("temperature_limited", flatmos.nh_tot.extent(0));
        auto flatmos = flatten<fp_t>(state->atmos);
        Fp1d rel_change("temperature_rel_change", flatmos.nh_tot.extent(0));
        yakl::Array<i32, 1, yakl::memDevice> limited("temperature_limited", flatmos.nh_tot.extent(0));
        dex_parallel_for(
                const fp_t temperature = flatmos.temperature(k);
            "Compute correction",
            FlatLoop<1>(flatmos.nh_tot.extent(0)),
            YAKL_LAMBDA (i64 k) {
                const fp_t temperature = flatmos.temperature(k);
                const fp_t e_intk = flatmos.e_int(k);
                fp_t e_ion = FP(0.0);
                for (int j = 0; j < h_pops.extent(0); ++j) {
                const fp_t de_th_dT = N * (k_B * invgammam1);
                const fp_t e_th_error = e_th_target - de_th_dT * temperature;
                // NOTE(cmo): This isn't anything like the true derivative
                // (which is hard to evaluate and would really require another
                // preconditioned system). It simply provides a term that scales
                // with temperature for the amount of energy taken up by
                // ionisation to damp the swing taken by temperature. This is
                // very vaguely LTE motivated assuming de_ion/dT scales very
                // approximately as 1/T^3/2. At convergence, this term won't
                // have any effect though :)
                const fp_t de_ion_dT = (
                    damping * e_ion / temperature * std::sqrt(damping_T_ref / temperature)
                );
                fp_t delta_temp = e_th_error / (de_th_dT + de_ion_dT);

                i32 was_limited = 0;
                if (max_frac_change > FP(0.0)) {
                    const fp_t max_delta = max_frac_change * temperature;
                    if (std::abs(delta_temp) > max_delta) {
                        delta_temp = std::copysign(max_delta, delta_temp);
                        was_limited = 1;
                    }
                }
                const fp_t e_th_target = e_intk - e_ion;
                fp_t new_temperature = temperature + delta_temp;
                const fp_t de_th_dT = N * (k_B * invgammam1);
                // preconditioned system). It simply provides a term that scales
                    was_limited = 1;
                // with temperature for the amount of energy taken up by
                limited(k) = was_limited;
                rel_change(k) = std::abs(FP(1.0) - new_temperature / temperature);
                // very vaguely LTE motivated assuming de_ion/dT scales very
                // approximately as 1/T^3/2. At convergence, this term won't
                // have any effect though :)
                const fp_t de_ion_dT = (
                    damping * e_ion / temperature * std::sqrt(damping_T_ref / temperature)
                );
                fp_t delta_temp = e_th_error / (de_th_dT + de_ion_dT);

        i32 num_limited = 0;
        dex_parallel_reduce(
            "EnergyConsNumLimited",
            FlatLoop<1>(limited.extent(0)),
            KOKKOS_LAMBDA (const int i, i32& running_total) {
                running_total += limited(i);
            },
            Kokkos::Sum<i32>(num_limited)
        );

                i32 was_limited = 0;
                if (max_frac_change > FP(0.0)) {
                    const fp_t max_delta = max_frac_change * temperature;
                    if (std::abs(delta_temp) > max_delta) {
                        delta_temp = std::copysign(max_delta, delta_temp);
                        was_limited = 1;
                    }
                }

                fp_t new_temperature = temperature + delta_temp;
                if (new_temperature < min_temperature) {
                    new_temperature = min_temperature;
                    was_limited = 1;
                }
                limited(k) = was_limited;
                rel_change(k) = std::abs(FP(1.0) - new_temperature / temperature);
                flatmos.temperature(k) = new_temperature;

                // NOTE(cmo): Make pressure consistent (in case someone else uses it)
            "     Max Change temperature (e_int): {} (@ {}) [{} cells limited]",
            }
            max_change_loc,
            num_limited
        yakl::fence();

        i32 num_limited = 0;
        dex_parallel_reduce(
            "EnergyConsNumLimited",
            FlatLoop<1>(limited.extent(0)),
            KOKKOS_LAMBDA (const int i, i32& running_total) {
                running_total += limited(i);
            },
            Kokkos::Sum<i32>(num_limited)
        );

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
            "     Max Change temperature (e_int): {} (@ {}) [{} cells limited]",
            max_change,
            max_change_loc,
            num_limited
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