#include "PromweaverBoundary.hpp"
#include "State.hpp"


PwBc<> load_bc(
    const std::string& path,
    const FpConst1d& wavelength,
    BoundaryType type,
    PromweaverResampleType resample
) {
    yakl::SimpleNetCDF nc;
    nc.open(path, yakl::NETCDF_MODE_READ);

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

    if (resample == PromweaverResampleType::FluxConserving) {
        // NOTE(cmo): Approximately the same as: https://ui.adsabs.harvard.edu/abs/2017arXiv170505165C/abstract
        Fp1dHost in_edges_h("in_edges", wl_dim+1);
        Fp1dHost out_edges_h("out_edges", wavelength.extent(0)+1);
        Fp1dHost wl_in_h = wl_load.createHostCopy();
        FpConst1dHost wavelength_h = wavelength.createHostCopy();

        auto compute_edges = [](auto arr, auto edges) {
            edges(0) = arr(0) - FP(0.5) * (arr(1) - arr(0));
            for (int la = 1; la < arr.extent(0); ++la) {
                edges(la) = FP(0.5) * (arr(la - 1) + arr(la));
            }
            i32 end = arr.extent(0);
            edges(end) = arr(end-1) + FP(0.5) * (arr(end-1) - arr(end-2));
        };
        compute_edges(wl_in_h, in_edges_h);
        compute_edges(wavelength_h, out_edges_h);

        Fp2dHost I_load_h = I_load.createHostCopy();
        Fp2dHost I_h("pw_bc_h", mu_dim, wavelength_h.extent(0));
        dex_parallel_for<Kokkos::DefaultHostExecutionSpace>(
            "Flux conserving resample",
            FlatLoop<1>(I_h.extent(0)),
            KOKKOS_LAMBDA (int mu) {
                i32 start = 0;
                i32 stop = 0;
                for (int la = 0; la < wavelength_h.extent(0); ++la) {
                    // NOTE(cmo): If bin entirely outside grid, set 0
                    if (
                        out_edges_h(la + 1) <= in_edges_h(0) ||
                        out_edges_h(la) >= in_edges_h(in_edges_h.extent(0) - 1)
                    ) {
                        I_h(mu, la) = FP(0.0);
                        continue;
                    }

                    fp_t start_edge = std::max(in_edges_h(0), out_edges_h(la));
                    fp_t end_edge = std::min(in_edges_h(in_edges_h.extent(0) - 1), out_edges_h(la + 1));

                    // NOTE(cmo): Move lower bound to first bin that overlaps current
                    while (in_edges_h(start + 1) <= start_edge) {
                        start++;
                    }
                    // NOTE(cmo): Move upper bound to last bin that overlaps current (last bin is stop: i.e. [stop, stop+1] in edge space)
                    while (in_edges_h(stop + 1) < end_edge) {
                        stop++;
                    }

                    if (start == stop) {
                        // NOTE(cmo): Nothing fancy, flux is constant
                        I_h(mu, la) = I_load_h(start, mu);
                        continue;
                    }

                    // NOTE(cmo): Do flux preserving integral
                    fp_t start_overlap = in_edges_h(start + 1) - out_edges_h(la);
                    fp_t end_overlap = out_edges_h(la + 1) - in_edges_h(stop);

                    fp_t int_flux = FP(0.0);
                    fp_t int_lambda = FP(0.0);

                    int_flux += start_overlap * I_load_h(start, mu);
                    int_lambda += start_overlap;
                    for (int ll = start + 1; ll < stop; ++ll) {
                        const fp_t width = in_edges_h(ll+1) - in_edges_h(ll);
                        int_flux += width * I_load_h(ll, mu);
                        int_lambda += width;
                    }
                    int_flux += end_overlap * I_load_h(stop, mu);
                    int_lambda += end_overlap;
                    I_h(mu, la) = int_flux / int_lambda;
                }
            }
        );
        Kokkos::DefaultHostExecutionSpace{}.fence();

        Fp2d I_dev = I_h.createDeviceCopy();
        dex_parallel_for(
            "Transpose and copy resampled array",
            FlatLoop<2>(I_dev.extent(0), I_dev.extent(1)),
            KOKKOS_LAMBDA (i32 mu, i32 la) {
                I(la, mu) = I_dev(mu, la);
            }
        );
        Kokkos::fence();
    } else {
        dex_parallel_for(
            "Promweaver BC Interp",
            FlatLoop<2>(wavelength.extent(0), mu_dim),
            YAKL_LAMBDA (int la, int mu) {
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
    }
    Kokkos::fence();

    return result;
}