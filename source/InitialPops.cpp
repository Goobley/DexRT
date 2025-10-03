#include "InitialPops.hpp"
#include "Types.hpp"
#include "EmisOpac.hpp"
#include "Populations.hpp"
#include "Collisions.hpp"
#include "State.hpp"
#include "State3d.hpp"

template <typename State>
void set_zero_radiation_pops(State* st, int atom) {
    const auto& state = *st;
    JasUnpack(state, adata);
    const auto& Gamma = state.Gamma[atom];
    const int level_start = state.adata_host.level_start(atom);
    const int num_level = state.adata_host.num_level(atom);
    const int line_start = state.adata_host.line_start(atom);
    const int num_line = state.adata_host.num_line(atom);
    const int cont_start = state.adata_host.cont_start(atom);
    const int num_cont = state.adata_host.num_cont(atom);
    const int num_wave = state.adata_host.wavelength.extent(0);
    const i64 num_space = Gamma.extent(2);
    const auto& wavelength = adata.wavelength;
    const auto& temperature = state.atmos.temperature;
    const auto& ne = state.atmos.ne;

    dex_parallel_for(
        "Zero radiation initialisation: Lines",
        FlatLoop<1>(num_space),
        KOKKOS_LAMBDA (i64 ks) {
            for (int kr = line_start; kr < line_start + num_line; ++kr) {
                const auto& line = adata.lines(kr);
                Kokkos::atomic_add(&Gamma(line.i, line.j, ks), line.Aji);
            }
        }
    );
    dex_parallel_for(
        "Zero radiation initialisation: Continua",
        FlatLoop<2>(num_wave, num_space),
        KOKKOS_LAMBDA (i32 la, i64 ks) {
            using namespace ConstantsFP;
            constexpr fp_t debroglie_const = fp_t(h / (FP(2.0) * pi * k_B) * (h / m_e));
            const fp_t temperature_k = temperature(ks);
            const fp_t ne_k = ne(ks);
            for (int kr = cont_start; kr < cont_start + num_cont; ++kr) {
                const auto& cont = adata.continua(kr);
                if (!cont.is_active(la)) {
                    continue;
                }

                const fp_t lambda = wavelength(la);
                const fp_t hnu_4pi = hc_kJ_nm / (four_pi * lambda);
                fp_t wl_weight = FP(1.0) / hnu_4pi * adata.wavelength_bin(la);

                const fp_t dE_kbT = (adata.energy(level_start + cont.j) - adata.energy(level_start + cont.i)) / (k_B_eV * temperature_k);
                const fp_t saha_boltzmann = adata.g(level_start + cont.i) / adata.g(level_start + cont.j) * std::pow(debroglie_const / temperature_k, FP(1.5)) * std::exp(dE_kbT); // NOTE(cmo): Not missing a minus.

                const auto& sigma_grid = get_sigma(adata, cont);
                const fp_t Uji = ne_k * saha_boltzmann * twohc2_kW_nm2 / (cube(lambda) * square(lambda) * FP(1e-18)) * sigma_grid.sigma(la - cont.blue_idx);

                Kokkos::atomic_add(&Gamma(cont.i, cont.j, ks), Uji * wl_weight);
            }
        }
    );
    Kokkos::fence();
}

template <typename State>
void set_initial_pops_special(State* state) {
    // NOTE(cmo): Check whether there's any work to do -- LTE is already done before this
    bool any_zero_rad = false;
    for (int ia = 0; ia < state->adata_host.init_pops_scheme.extent(0); ++ia) {
        const auto initial_pops = state->adata_host.init_pops_scheme(ia);
        if (initial_pops == AtomInitialPops::ZeroRadiation) {
            any_zero_rad = true;
            break;
        }
    }
    if (!any_zero_rad) {
        return;
    }

    compute_collisions_to_gamma(state);
    for (int ia = 0; ia < state->adata_host.init_pops_scheme.extent(0); ++ia) {
        const auto intial_pops = state->adata_host.init_pops_scheme(ia);

        if (intial_pops != AtomInitialPops::ZeroRadiation) {
            continue;
        }

        set_zero_radiation_pops(state, ia);
        state->println("Setting ele ({}, Z={}) to ZeroRadiation conditions.", ia, state->adata_host.Z(ia));
        stat_eq(
            state,
            StatEqOptions {
                .only_atom = ia
            }
        );
    }
}

template void set_initial_pops_special<State>(State* state);
template void set_initial_pops_special<State3d>(State3d* state);
