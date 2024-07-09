#if !defined(DEXRT_PRESSURE_CONSERVATION_HPP)
#define DEXRT_PRESSURE_CONSERVATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include <inttypes.h>

inline fp_t simple_conserve_pressure(State* state) {
    assert(state->have_h && "Requires H model to be present");
    using namespace ConstantsFP;

    fp_t total_abund = FP(0.0);
    if constexpr (false) {
        for (int ia = 0; ia < state->adata_host.num_level.extent(0); ++ia) {
            total_abund += state->adata_host.abundance(ia);
        }
    } else {
        // NOTE(cmo): From Asplund 2009/Lw calc
        total_abund = FP(1.0861550335264554);
    }
    const auto& active = state->active.collapse();
    const auto h_pops = slice_pops(state->pops, state->adata_host, 0);
    const auto h_pops_flat = h_pops.reshape<2>(Dims(h_pops.extent(0), h_pops.extent(1) * h_pops.extent(2)));

    auto flatmos = flatten<fp_t>(state->atmos);
    Fp1d nh_tot_correction("nhtot_corr", flatmos.nh_tot.extent(0));
    Fp1d nh_tot_ratio("nhtot_ratio", flatmos.nh_tot.extent(0));
    Fp1d rel_change("nhtot_rel_change", flatmos.nh_tot.extent(0));
    parallel_for(
        "Compute correction",
        SimpleBounds<1>(flatmos.nh_tot.extent(0)),
        YAKL_LAMBDA (i64 k) {
            if (active(k)) {
                const fp_t N = total_abund * flatmos.nh_tot(k) + flatmos.ne(k);
                const fp_t N_error = flatmos.pressure(k) / (k_B * flatmos.temperature(k)) - N;
                // NOTE(cmo): This is the simplified pw treatment -- assuming all electrons come from H
                const fp_t N_corr = N_error / (total_abund + h_pops_flat(h_pops_flat.extent(0)-1, k) / flatmos.nh_tot(k));
                nh_tot_correction(k) = N_corr;
                fp_t new_nhtot = flatmos.nh_tot(k) + nh_tot_correction(k);
                if (new_nhtot < FP(0.0)) {
#ifndef YAKL_ARCH_SYCL
                    printf("Nhtot driven negative @ k = %" PRId64 ", clamped to 1.0e10 m-3\n", k);
#endif
                    new_nhtot = FP(1e10);
                }
                nh_tot_ratio(k) = new_nhtot / flatmos.nh_tot(k);
                rel_change(k) = std::abs(FP(1.0) - nh_tot_ratio(k));

            } else {
                nh_tot_correction(k) = FP(0.0);
                nh_tot_ratio(k) = FP(1.0);
                rel_change(k) = FP(0.0);
            }
        }
    );
    yakl::fence();
    fp_t max_change = yakl::intrinsics::maxval(rel_change);
    i64 max_change_loc = yakl::intrinsics::maxloc(rel_change);
    yakl::fence();
    int max_change_x = max_change_loc % state->atmos.nh_tot.extent(1);
    max_change_loc /= state->atmos.nh_tot.extent(1);
    int max_change_z = max_change_loc;

    parallel_for(
        "Apply updates",
        SimpleBounds<1>(flatmos.nh_tot.extent(0)),
        YAKL_LAMBDA (i64 k) {
            if (active(k)) {
                flatmos.ne(k) += nh_tot_correction(k) * h_pops_flat(h_pops_flat.extent(0)-1, k) / flatmos.nh_tot(k);
                flatmos.nh_tot(k) *= nh_tot_ratio(k);
            }
        }
    );
    yakl::fence();
    const auto& pops = state->pops.reshape<2>(Dims(state->pops.extent(0), state->pops.extent(1) * state->pops.extent(2)));
    parallel_for(
        "Rescale pops",
        SimpleBounds<2>(pops.extent(0), pops.extent(1)),
        YAKL_LAMBDA (int i, i64 k) {
            if (active(k)) {
                pops(i, k) *= nh_tot_ratio(k);
            }
        }
    );
    yakl::fence();

    fmt::println(
        "Max Change nh_tot: {} (@ {}, {})",
        max_change,
        max_change_z,
        max_change_x
    );
    return max_change;
}

#else
#endif