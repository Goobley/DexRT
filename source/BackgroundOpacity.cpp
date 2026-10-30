#include "BackgroundOpacity.hpp"
#include "State.hpp"
#include "ScState.hpp"
#include "ScIo.hpp"
#include <type_traits>
#include "LoopUtils.hpp"
#include "Atmosphere.hpp"
#include <netcdf.h>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cmath>

namespace {

void nc_check(int ierr, const std::string& what) {
    if (ierr != NC_NOERR) {
        throw std::runtime_error(fmt::format("Background tables: {}: {}", what, nc_strerror(ierr)));
    }
}

std::vector<f64> read_1d(int ncid, const char* name) {
    int varid;
    nc_check(nc_inq_varid(ncid, name, &varid), name);
    int dimid;
    nc_check(nc_inq_vardimid(ncid, varid, &dimid), name);
    size_t len;
    nc_check(nc_inq_dimlen(ncid, dimid, &len), name);
    std::vector<f64> result(len);
    nc_check(nc_get_var_double(ncid, varid, result.data()), name);
    return result;
}

std::vector<float> read_3d(int ncid, const std::string& name, size_t size) {
    int varid;
    nc_check(nc_inq_varid(ncid, name.c_str(), &varid), name);
    std::vector<float> result(size);
    nc_check(nc_get_var_float(ncid, varid, result.data()), name);
    return result;
}

/// Uniform axis (start, step) from the table coordinates.
std::pair<f64, f64> uniform_axis(const std::vector<f64>& x, const char* name) {
    const f64 step = (x.back() - x.front()) / f64(x.size() - 1);
    for (int i = 0; i < x.size(); ++i) {
        if (std::abs(x[i] - (x.front() + i * step)) > 1e-6 * std::abs(step)) {
            throw std::runtime_error(fmt::format("Background tables: {} axis isn't uniform", name));
        }
    }
    return {x.front(), step};
}

/// Resample [table_lambda, t, ne] log10 values onto the run's wavelengths,
/// linearly in log lambda (clamped at the ends).
Fp3d resample(
    const std::vector<float>& tab,
    const std::vector<f64>& table_lambda,
    const std::vector<f64>& lambda,
    i32 num_t,
    i32 num_ne,
    const char* name
) {
    const i64 plane = i64(num_t) * num_ne;
    Fp3dHost out(name, lambda.size(), num_t, num_ne);
    for (int la = 0; la < lambda.size(); ++la) {
        const f64 x = std::log(lambda[la]);
        int i = std::upper_bound(table_lambda.begin(), table_lambda.end(), lambda[la]) - table_lambda.begin() - 1;
        i = std::min(std::max(i, 0), int(table_lambda.size()) - 2);
        f64 f = (x - std::log(table_lambda[i])) / (std::log(table_lambda[i + 1]) - std::log(table_lambda[i]));
        f = std::min(std::max(f, 0.0), 1.0);
        for (i64 j = 0; j < plane; ++j) {
            const f64 v = (1.0 - f) * tab[i * plane + j] + f * tab[(i + 1) * plane + j];
            out(la, j / num_ne, j % num_ne) = fp_t(v);
        }
    }
    return out.createDeviceCopy();
}

const std::map<std::string, int> element_z = {
    {"H", 1}, {"He", 2}, {"C", 6}, {"N", 7}, {"O", 8}, {"Na", 11}, {"Mg", 12},
    {"Al", 13}, {"Si", 14}, {"S", 16}, {"Ca", 20}, {"Fe", 26}
};

} // namespace

template <typename StateT>
void load_background(StateT* state, const std::string& path) {
    auto& bg = state->background;
    const auto& adata = state->adata_host;
    // NOTE(cmo): Read by rank 0 and broadcast (opened in memory elsewhere).
    int ncid;
    std::vector<char> file_buffer;
    nc_check(sc_nc_open_replicated(state->mpi_state, path, &ncid, &file_buffer), path);

    const auto table_lambda = read_1d(ncid, "wavelength");
    const auto log_t = read_1d(ncid, "log_temperature");
    const auto log_ne = read_1d(ncid, "log_ne");
    const auto [t0, dt] = uniform_axis(log_t, "log_temperature");
    const auto [ne0, dne] = uniform_axis(log_ne, "log_ne");
    bg.log_t0 = t0;
    bg.dlog_t = dt;
    bg.num_t = log_t.size();
    bg.log_ne0 = ne0;
    bg.dlog_ne = dne;
    bg.num_ne = log_ne.size();
    const size_t size = table_lambda.size() * log_t.size() * log_ne.size();

    std::vector<f64> lambda(adata.wavelength.extent(0));
    for (int la = 0; la < lambda.size(); ++la) {
        lambda[la] = adata.wavelength(la);
    }

    size_t att_len;
    nc_check(nc_inq_attlen(ncid, NC_GLOBAL, "elements", &att_len), "elements");
    std::string elements_str(att_len, '\0');
    nc_check(nc_get_att_text(ncid, NC_GLOBAL, "elements", elements_str.data()), "elements");

    // NOTE(cmo): Elements that are atoms in this run are skipped (their
    // continua come from the atom).
    auto is_detailed_atom = [&] (const std::string& el) {
        const int Z = element_z.at(el);
        for (int ia = 0; ia < adata.Z.extent(0); ++ia) {
            if (adata.Z(ia) == Z) {
                return true;
            }
        }
        return false;
    };
    std::vector<f64> metals(size, 0.0);
    std::vector<std::string> used;
    std::istringstream elements(elements_str);
    std::string el;
    while (elements >> el) {
        if (is_detailed_atom(el)) {
            continue;
        }
        used.push_back(el);
        const auto tab = read_3d(ncid, el + "_bf", size);
        for (size_t i = 0; i < size; ++i) {
            metals[i] += std::pow(10.0, f64(tab[i]));
        }
    }
    std::vector<float> metals_log(size);
    for (size_t i = 0; i < size; ++i) {
        metals_log[i] = float(std::max(std::log10(metals[i]), -99.0));
    }
    bg.metals = resample(metals_log, table_lambda, lambda, bg.num_t, bg.num_ne, "bg_metals");
    bg.hminus = resample(read_3d(ncid, "Hminus", size), table_lambda, lambda, bg.num_t, bg.num_ne, "bg_hminus");
    bg.h2plus = resample(read_3d(ncid, "H2plus", size), table_lambda, lambda, bg.num_t, bg.num_ne, "bg_h2plus");
    bg.h_ff = resample(read_3d(ncid, "H_ff", size), table_lambda, lambda, bg.num_t, bg.num_ne, "bg_h_ff");
    if (!is_detailed_atom("He")) {
        bg.he_rayleigh = resample(read_3d(ncid, "He_rayleigh", size), table_lambda, lambda, bg.num_t, bg.num_ne, "bg_he_rayleigh");
    }
    const auto h_ray = read_1d(ncid, "H_rayleigh");
    {
        std::vector<float> h_ray_f(size);
        const i64 plane = i64(bg.num_t) * bg.num_ne;
        for (size_t la = 0; la < table_lambda.size(); ++la) {
            for (i64 j = 0; j < plane; ++j) {
                h_ray_f[la * plane + j] = h_ray[la];
            }
        }
        // NOTE(cmo): Resample through the 3D path, then keep one plane.
        const auto r = resample(h_ray_f, table_lambda, lambda, bg.num_t, bg.num_ne, "bg_h_ray").createHostCopy();
        Fp1dHost h_ray_h("bg_h_rayleigh", lambda.size());
        for (int la = 0; la < lambda.size(); ++la) {
            h_ray_h(la) = r(la, 0, 0);
        }
        bg.h_rayleigh = h_ray_h.createDeviceCopy();
    }
    nc_close(ncid);

    const i64 num_cells = state->atmos.temperature.extent(0);
    bg.n1s = Fp1d("bg_n1s", num_cells);
    bg.n_p = Fp1d("bg_n_p", num_cells);
    bg.have_J = false;
    bg.enabled = true;

    std::string used_str;
    for (const auto& e : used) {
        used_str += e + " ";
    }
    state->println("Background opacities from {} (bound-free continua of: {})", path, used_str);
}

template <typename StateT>
void update_background(StateT* state) {
    auto& bg = state->background;
    if (!bg.enabled) {
        return;
    }
    const auto& adata_host = state->adata_host;
    int h_atom = -1;
    for (int ia = 0; ia < adata_host.Z.extent(0); ++ia) {
        if (adata_host.Z(ia) == 1) {
            h_atom = ia;
        }
    }
    const auto flatmos = flatten<const fp_t>(state->atmos);
    const auto& pops = state->pops;
    JasUnpack(bg, n1s, n_p);
    const auto cells = cell_list(*state);
    if (h_atom >= 0) {
        const int ground = adata_host.level_start(h_atom);
        const int cont = ground + adata_host.num_level(h_atom) - 1;
        dex_parallel_for(
            "Background H pops",
            FlatLoop<1>(cells.num()),
            YAKL_LAMBDA (i64 i) {
                const i64 ks = cells(i);
                n1s(ks) = pops(ground, ks);
                n_p(ks) = pops(cont, ks);
            }
        );
    } else {
        // NOTE(cmo): LTE (Saha) two-level hydrogen.
        dex_parallel_for(
            "Background H pops (LTE)",
            FlatLoop<1>(cells.num()),
            YAKL_LAMBDA (i64 i) {
                using namespace ConstantsFP;
                const i64 ks = cells(i);
                const fp_t T = flatmos.temperature(ks);
                constexpr fp_t saha_const = FP(2.4146830395719654e21); // (2 pi m_e k / h^2)^1.5 [m-3 K-1.5]
                const fp_t ratio = saha_const * std::pow(T, FP(1.5)) * std::exp(-FP(13.598434) / (k_B_eV * T)) / flatmos.ne(ks);
                const fp_t nh = flatmos.nh_tot(ks);
                n1s(ks) = nh / (FP(1.0) + ratio);
                n_p(ks) = nh - n1s(ks);
            }
        );
    }

    Kokkos::fence();
}

template void load_background<State>(State* state, const std::string& path);
template void load_background<ScState>(ScState* state, const std::string& path);
template void update_background<ScState>(ScState* state);
