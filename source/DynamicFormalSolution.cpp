#include "DynamicFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "DynamicRadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "Utils.hpp"

void dynamic_formal_sol_rc(State* state, int la) {
    // 1. Accumulate the static components of eta/chi; where norm(v) is low (< 1km/s?), add lines to this too
    // 1b. Don't mipmap as the next steps aren't compatible.
    // 2. Radiance cascade solver, propagating atomic state downwards to 2b.
    // 2b. Raymarch the grid with new march function that samples the constant eta/chi where norm(v) is small, and adds the local doppler shifted values where norm(v) is larger. For this it requires access to the state
    // 2c. When computing C0, accumulate necessary terms into gamma - alo array is unneeded here as it's done in one pass

    auto& march_state = state->raymarch_state;

    auto& atmos = state->atmos;
    auto& phi = state->phi;
    auto& pops = state->pops;
    auto& atom = state->atom;
    auto& nh_lte = state->nh_lte;
    auto& eta = march_state.emission;
    auto& chi = march_state.absorption;

    // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
    // it, for now, trust the pool allocator
    auto pops_dims = pops.get_dimensions();
    Fp3d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1), pops_dims(2));
    // TODO(cmo): Same for this one... it's dumb to calculate it in the inner loop...
    Fp2d nh0("nh0", pops_dims(1), pops_dims(2));

    auto flat_temperature = atmos.temperature.collapse();
    auto flat_ne = atmos.ne.collapse();
    auto flat_vturb = atmos.vturb.collapse();
    auto flat_nhtot = atmos.nh_tot.collapse();
    auto flat_vx = atmos.vx.collapse();
    auto flat_vy = atmos.vy.collapse();
    auto flat_vz = atmos.vz.collapse();
    auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
    auto flat_n_star = lte_scratch.reshape<2>(Dims(lte_scratch.extent(0), lte_scratch.extent(1) * lte_scratch.extent(2)));
    auto flat_eta = eta.reshape<2>(Dims(eta.extent(0) * eta.extent(1), eta.extent(2)));
    auto flat_chi = chi.reshape<2>(Dims(chi.extent(0) * chi.extent(1), chi.extent(2)));
    auto flat_nh0 = nh0.collapse();
    // TODO(cmo): This bad.
    const auto& lines = atom.lines.createHostCopy();
    // NOTE(cmo): Compute emis/opac
    parallel_for(
        "Compute eta, chi",
        SimpleBounds<1>(flat_temperature.extent(0)),
        YAKL_LAMBDA (int64_t k) {
            AtmosPointParams local_atmos;
            local_atmos.temperature = flat_temperature(k);
            local_atmos.ne = flat_ne(k);
            local_atmos.vturb = flat_vturb(k);
            local_atmos.nhtot = flat_nhtot(k);
            local_atmos.nh0 = nh_lte(local_atmos.temperature, local_atmos.ne, local_atmos.nhtot);

            fp_t vel = std::sqrt(square(flat_vx(k)) + square(flat_vy(k)) + square(flat_vz(k)));
            auto mode = EmisOpacMode::StaticOnly;
            if (
                vel <= ANGLE_INVARIANT_THERMAL_VEL_FRAC * thermal_vel(atom.mass, local_atmos.temperature)
            ) {
                mode = EmisOpacMode::All;
            }

            auto result = emis_opac(
                EmisOpacState<fp_t>{
                    .atom = atom,
                    .profile = phi,
                    .la = la,
                    .n = flat_pops,
                    .n_star_scratch = flat_n_star,
                    .k = k,
                    .atmos = local_atmos,
                    .mode = mode
                }
            );
            flat_nh0(k) = local_atmos.nh0;
            flat_eta(k, 0) = result.eta;
            flat_chi(k, 0) = result.chi;
        }
    );

    yakl::Array<i32, 1, yakl::memHost> active_host("active set", lines.extent(0));
    int num_active_lines = 0;
    for (int line_idx = 0; line_idx < lines.extent(0); ++line_idx) {
        const auto& line = lines(line_idx);
        if (line.is_active(la)) {
            active_host(num_active_lines) = line_idx;
            num_active_lines += 1;
        }
    }
    yakl::Array<i32, 1, yakl::memHost>active_host_cut("active set", active_host.get_data(), num_active_lines);
    auto active_set = active_host_cut.createDeviceCopy();
    yakl::fence();

    // NOTE(cmo): Compute RC FS
    if (num_active_lines == 0) {
        // NOTE(cmo): Use the static solver
        for (int i = MAX_LEVEL; i >= 0; --i) {
            compute_cascade_i_2d(state, i, false);
            yakl::fence();
        }
    } else {
        yakl::fence();
        for (int i = MAX_LEVEL; i >= 0; --i) {
            // const bool compute_alo = ((i == 0) && state->alo.initialized());
            compute_dynamic_cascade_i_2d(state, lte_scratch, nh0, i, la, active_set, false);
            yakl::fence();
        }
    }


}
