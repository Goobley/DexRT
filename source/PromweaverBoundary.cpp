#include "PromweaverBoundary.hpp"
#include "State.hpp"


void load_bc(const std::string& path, State* state) {
    yakl::SimpleNetCDF nc;
    nc.open(path, yakl::NETCDF_MODE_READ);

    if (!nc.varExists("prom_bc_I")) {
        if (USE_BC) {
            throw std::runtime_error("No BC data present.");
        }
        return;
    }

    int mu_dim = nc.getDimSize("prom_bc_mu");
    int wl_dim = nc.getDimSize("prom_bc_wavelength");
    PwBc<> result;
    nc.read(result.mu_min, "prom_bc_mu_min");
    nc.read(result.mu_max, "prom_bc_mu_max");
    result.mu_step = (result.mu_max - result.mu_min) / fp_t(mu_dim - 1);
    Fp2d I_load = Fp2d("pw_bc_preinterp", wl_dim, mu_dim);
    nc.read(I_load, "prom_bc_I");
    Fp1d wl_load("prom_bc_wavelength", wl_dim);
    nc.read(wl_load, "prom_bc_wavelength");
    FpConst1d wl(wl_load);
    result.I = Fp2d("pw_bc", state->atom.wavelength.extent(0), mu_dim);
    auto& I = result.I;

    using yakl::c::parallel_outer;
    using yakl::c::parallel_inner;
    const auto& atom = state->atom;
    const auto& wavelength = atom.wavelength;
    parallel_outer(
        "Promweaver BC Interp",
        SimpleBounds<1>(wavelength.extent(0)),
        YAKL_LAMBDA (int la, yakl::InnerHandler inner_handler) {
            fp_t wavelength_sought = wavelength(la);
            // NOTE(cmo): Corresponding fractional index in loaded data
            int idx, idxp;
            fp_t t = FP(0.0);
            if (wavelength_sought <= wl(0)) {
                idx = 0;
                idxp = 0;
            } else if (wavelength_sought >= wl(wl.extent(0) - 1)) {
                idx = wl.extent(0) - 1;
                idxp = wl.extent(0) - 1;
            } else {
                idxp = upper_bound(wl, wavelength_sought);
                idx = idxp - 1;
                t = (wl(idxp) - wavelength_sought) / (wl(idxp) - wl(idx));
            }

            parallel_inner(
                SimpleBounds<1>(mu_dim),
                [&la, &idx, &idxp, &t, &I, &I_load] (int mu) {
                    I(la, mu) = t * I_load(idx, mu) + (FP(1.0) - t) * I_load(idxp, mu);
                },
                inner_handler
            );
        }
    );
    yakl::fence();

    state->pw_bc = result;
}