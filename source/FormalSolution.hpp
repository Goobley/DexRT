#if !defined(DEXRT_FORMAL_SOLUTION_HPP)
#define DEXRT_FORMAL_SOLUTION_HPP

#include "Types.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "Utils.hpp"

/**
 * Temporary function to give us the LTE ground state fraction of h for
 * broadening things. Uses tabulated partition function based on classical 5+1
 * level H.
*/
YAKL_INLINE fp_t nh0_lte(fp_t temperature, fp_t ne, fp_t nh_tot) {
    constexpr int grid_size = 31;
    constexpr fp_t log_T[grid_size] = {FP(3.000000e+00), FP(3.133333e+00), FP(3.266667e+00), FP(3.400000e+00), FP(3.533333e+00), FP(3.666667e+00), FP(3.800000e+00), FP(3.933333e+00), FP(4.066667e+00), FP(4.200000e+00), FP(4.333333e+00), FP(4.466667e+00), FP(4.600000e+00), FP(4.733333e+00), FP(4.866667e+00), FP(5.000000e+00), FP(5.133333e+00), FP(5.266667e+00), FP(5.400000e+00), FP(5.533333e+00), FP(5.666667e+00), FP(5.800000e+00), FP(5.933333e+00), FP(6.066667e+00), FP(6.200000e+00), FP(6.333333e+00), FP(6.466667e+00), FP(6.600000e+00), FP(6.733333e+00), FP(6.866667e+00), FP(7.000000e+00)};
    constexpr fp_t h_partfn[grid_size] = {FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000000e+00), FP(2.000012e+00), FP(2.000628e+00), FP(2.013445e+00), FP(2.136720e+00), FP(2.776522e+00), FP(4.825934e+00), FP(9.357075e+00), FP(1.691968e+01), FP(2.713700e+01), FP(3.892452e+01), FP(5.101649e+01), FP(6.238607e+01), FP(7.240922e+01), FP(8.083457e+01), FP(8.767237e+01), FP(9.307985e+01), FP(9.727538e+01), FP(1.004851e+02), FP(1.029155e+02), FP(1.047417e+02), FP(1.061062e+02), FP(1.071217e+02), FP(1.078750e+02), FP(1.084327e+02)};

    using namespace ConstantsFP;
    constexpr fp_t saha_const = (FP(2.0) * FP(M_PI) * k_B) / h * (h / m_e);
    constexpr fp_t ion_energy_k_B = FP(157887.51240204);
    constexpr fp_t u_hii = FP(1.0);

    const yakl::Array<fp_t const, 1, yakl::memDevice> log_t_grid("log_t_grid", (fp_t*)log_T, grid_size);
    const yakl::Array<fp_t const, 1, yakl::memDevice> partfn_grid("log_t_grid", (fp_t*)h_partfn, grid_size);
    const fp_t log_temp = std::log10(temperature);
    const fp_t u_hi = interp(log_temp, log_t_grid, partfn_grid);
    const fp_t ratio = 2 * u_hii / (ne * u_hi) * std::pow(saha_const * temperature, FP(1.5)) * std::exp(-ion_energy_k_B / temperature);
    const fp_t result = nh_tot / (FP(1.0) + ratio);
    return result;
}

void static_formal_sol_rc(State* state, int la) {
    auto& march_state = state->raymarch_state;

    auto& atmos = state->atmos;
    auto& phi = state->phi;
    auto& pops = state->pops;
    auto& atom = state->atom;
    auto& eta = march_state.emission;
    auto& chi = march_state.absorption;

    // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
    // it, for now, trust the pool allocator
    auto pops_dims = pops.get_dimensions();
    Fp3d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1), pops_dims(2));

    auto atmos_dims = atmos.temperature.get_dimensions();
    // NOTE(cmo): Compute emis/opac
    parallel_for(
        "Compute eta, chi",
        SimpleBounds<2>(atmos_dims(0), atmos_dims(1)),
        YAKL_LAMBDA (int x, int y) {
            AtmosPointParams local_atmos;
            local_atmos.temperature = atmos.temperature(x, y);
            local_atmos.ne = atmos.ne(x, y);
            local_atmos.vturb = atmos.vturb(x, y);
            local_atmos.nhtot = atmos.nh_tot(x, y);
            local_atmos.nh0 = nh0_lte(local_atmos.temperature, local_atmos.ne, local_atmos.nhtot);
            auto result = emis_opac(
                atom, 
                phi, 
                la, 
                pops.slice<1>({x, y, yakl::COLON}),
                lte_scratch.slice<1>({x, y, yakl::COLON}),
                local_atmos
            );
            eta(x, y, 0) = result.eta;
            chi(x, y, 0) = result.chi;
        }
    );
    yakl::fence();
    // NOTE(cmo): Regenerate mipmaps
    if constexpr (USE_MIPMAPS) {
        int current_mip_factor = 0;
        for (int i = 0; i < march_state.emission_mipmaps.size(); ++i) {
            if (march_state.cumulative_mipmap_factor(i) == current_mip_factor) {
                continue;
            }
            current_mip_factor = march_state.cumulative_mipmap_factor(i);
            auto new_eta = march_state.emission_mipmaps[i];
            auto new_chi = march_state.absorption_mipmaps[i];
            auto dims = new_eta.get_dimensions();
            parallel_for(
                SimpleBounds<2>(dims(0), dims(1)),
                YAKL_LAMBDA (int x, int y) {
                    mipmap_arr(eta, new_eta, current_mip_factor, x, y);
                }
            );
            parallel_for(
                SimpleBounds<2>(dims(0), dims(1)),
                YAKL_LAMBDA (int x, int y) {
                    mipmap_arr(chi, new_chi, current_mip_factor, x, y);
                }
            );
        }
        yakl::fence();
    }
    // NOTE(cmo): Compute RC FS
    for (int i = MAX_LEVEL; i >= 0; --i) {
        if constexpr (BILINEAR_FIX) {
            compute_cascade_i_bilinear_fix_2d(state, i);
        } else {
            compute_cascade_i_2d(state, i);
        }
        yakl::fence();
    }
    // NOTE(cmo): J is not computed in this function, but done in main for now
}

#else
#endif