#include "PromweaverBoundary.hpp"
#include "State.hpp"


PwBc<> load_bc(const std::string& path, const FpConst1d& wavelength, BoundaryType type) {
    ExYakl::SimpleNetCDF nc;
    nc.open(path, ExYakl::NETCDF_MODE_READ);

    if (!nc.varExists("prom_bc_I")) {
        if (type == BoundaryType::Promweaver) {
            throw std::runtime_error("No BC data present.");
        }
        return PwBc<>{};
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
    Fp2d I("pw_bc", wavelength.extent(0), mu_dim);
    result.I = I;

    dex_parallel_for(
        "Promweaver BC Interp",
        FlatLoop<2>(wavelength.extent_int(0), mu_dim),
        KOKKOS_LAMBDA (int la, int mu) {
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

            I(la, mu) = t * I_load(idx, mu) + (FP(1.0) - t) * I_load(idxp, mu);
        }
    );
    yakl::fence();

    return result;
}