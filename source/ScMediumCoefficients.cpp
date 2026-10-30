#include "ScMediumCoefficients.hpp"
#include "EmisOpac.hpp"
#include "CoreAndVoigtEmisOpac.hpp"
#include "Utils.hpp"
#include "LoopUtils.hpp"
#include "Atmosphere.hpp"
#include <vector>
#include <algorithm>

/// Allocate direction-independent continua, line coefficients and LTE scratch for this cache entry.
void ScMediumCoefficients::init(const ScState& state, i32 wave_batch_max) {
    const i64 num_cells = state.cells.num_cells();
    const i32 total_lines = std::max(i32(state.adata_host.lines.extent(0)), 1);
    cont_eta = Fp2d("sc_cont_eta", wave_batch_max, num_cells);
    cont_chi = Fp2d("sc_cont_chi", wave_batch_max, num_cells);
    eta_star = Fp2d("sc_eta_star", total_lines, num_cells);
    chi_star = Fp2d("sc_chi_star", total_lines, num_cells);
    a_damp = Fp2d("sc_a_damp", total_lines, num_cells);
    inv_dop_width = Fp2d("sc_inv_dop_width", total_lines, num_cells);
    const auto pops_dims = state.pops.get_dimensions();
    n_star = Fp2d("sc_n_star", pops_dims(0), pops_dims(1));
}

/// Rebuild an entry for this wavelength interval using the admitted immutable previous-J snapshot.
void ScMediumCoefficients::fill(
    const ScState& state,
    const Rad16Array& previous_J,
    i32 la_start_,
    i32 la_end
) {
    JasUnpack(state, background);
    la_start = la_start_;
    wave_batch = la_end - la_start;
    const auto& adata_host = state.adata_host;

    // NOTE(cmo): Lines active anywhere in the batch, and per wavelength which
    // of them are active.
    std::vector<i32> lines;
    std::vector<std::vector<i32>> per_wave(wave_batch);
    for (int wave = 0; wave < wave_batch; ++wave) {
        const auto a_set = slice_active_set(adata_host, la_start + wave);
        for (int i = 0; i < a_set.extent(0); ++i) {
            const i32 kr = a_set(i);
            auto it = std::find(lines.begin(), lines.end(), kr);
            i32 local = it - lines.begin();
            if (it == lines.end()) {
                lines.push_back(kr);
            }
            per_wave[wave].push_back(local);
        }
    }
    num_lines = lines.size();
    max_lines_per_wave = 1;
    for (const auto& w : per_wave) {
        max_lines_per_wave = std::max(max_lines_per_wave, i32(w.size()) + 1);
    }
    yakl::Array<i32, 2, yakl::memHost> wave_lines_h("sc_wave_lines", wave_batch, max_lines_per_wave);
    for (int wave = 0; wave < wave_batch; ++wave) {
        for (int i = 0; i < max_lines_per_wave; ++i) {
            wave_lines_h(wave, i) = (i < per_wave[wave].size()) ? per_wave[wave][i] : -1;
        }
    }
    wave_lines = wave_lines_h.createDeviceCopy();
    Fp1dHost lambda0_h("sc_lambda0", std::max(num_lines, 1));
    yakl::Array<i32, 1, yakl::memHost> kr_h("sc_line_kr", std::max(num_lines, 1));
    yakl::Array<i32, 1, yakl::memHost> prd_h("sc_line_prd", std::max(num_lines, 1));
    prd = state.prd;
    const auto prd_of_kr =
        prd.enabled() ? prd.line_of_kr.createHostCopy() : decltype(prd.line_of_kr.createHostCopy())();
    for (int l = 0; l < num_lines; ++l) {
        lambda0_h(l) = adata_host.lines(lines[l]).lambda0;
        kr_h(l) = lines[l];
        prd_h(l) = prd.enabled() ? prd_of_kr(lines[l]) : -1;
    }
    lambda0 = lambda0_h.createDeviceCopy();
    line_kr = kr_h.createDeviceCopy();
    line_prd = prd_h.createDeviceCopy();

    JasUnpack(state, atmos, pops, phi, adata);
    JasUnpack((*this), cont_eta, cont_chi, eta_star, chi_star, a_damp, inv_dop_width, n_star, line_kr);
    const bool use_background = background.enabled;
    const auto flatmos = flatten<const fp_t>(atmos);
    const i32 la0 = la_start;
    const i32 wb = wave_batch;
    const i32 nl = num_lines;
    // NOTE(cmo): The halo rows too: the sweep's medium stencil reads them.
    const auto cells = state.cells.list_with_halo();
    const auto fill_cell = YAKL_LAMBDA(i64 i) {
        const i64 ks = cells(i);
        AtmosPointParams local_atmos{};
        local_atmos.temperature = flatmos.temperature(ks);
        local_atmos.ne = flatmos.ne(ks);
        local_atmos.vturb = flatmos.vturb(ks);
        local_atmos.nhtot = flatmos.nh_tot(ks);
        local_atmos.nh0 = flatmos.nh0(ks);
        // NOTE(cmo): The continua (n_star is updated on the first wavelength).
        for (i32 wave = 0; wave < wb; ++wave) {
            const i32 la = la0 + wave;
            const auto result = emis_opac(EmisOpacState<fp_t>{
                .adata = adata,
                .profile = phi,
                .la = la,
                .n = pops,
                .n_star_scratch = n_star,
                .k = ks,
                .atmos = local_atmos,
                .active_set = slice_active_set(adata, la),
                .active_set_cont = slice_active_cont_set(adata, la),
                .update_n_star = (wave == 0),
                .mode = EmisOpacMode::StaticOnly,
            });
            fp_t eta = result.eta;
            fp_t chi = result.chi;
            if (use_background) {
                const BackgroundTerms bg = background_terms(
                    background,
                    la,
                    ks,
                    local_atmos.temperature,
                    local_atmos.ne,
                    local_atmos.nhtot
                );
                eta += bg.chi_abs * planck_wavelength(adata.wavelength(la), local_atmos.temperature) +
                       bg.sca * previous_J(wave, ks);
                chi += bg.chi_abs + bg.sca;
            }
            cont_eta(wave, ks) = eta;
            cont_chi(wave, ks) = chi;
        }
        const CoreAndVoigt::CoreAndVoigtState line_state{
            .k = ks,
            .atmos = local_atmos,
            .adata = adata,
            .n = pops,
        };
        for (i32 l = 0; l < nl; ++l) {
            const auto line = CoreAndVoigt::compute_core_and_voigt(line_state, line_kr(l));
            eta_star(l, ks) = line.eta_star;
            chi_star(l, ks) = line.chi_star;
            a_damp(l, ks) = line.a_damp;
            inv_dop_width(l, ks) = FP(1.0) / line.dop_width;
        }
    };
    dex_parallel_for("SC emis/opac base", FlatLoop<1>(cells.num()), fill_cell);
    Kokkos::fence();
}

/// Allocate the bounded cache once; GivenFs requires no coefficient payloads.
ScMediumCoefficientCache sc_make_coefficient_cache(const ScState& state, i32 count, i32 width) {
    ScMediumCoefficientCache cache;
    if (state.config.mode != DexrtMode::GivenFs) {
        cache.entries.resize(count);
        for (auto& entry : cache.entries) {
            entry.coefficients.init(state, width);
        }
    }
    return cache;
}

/// Invalidate coefficients and selection when the populations/background change.
void sc_reset_coefficient_cache(ScMediumCoefficientCache& cache) {
    cache.clock = cache.builds = cache.hits = 0;
    cache.current_entry = -1;
    for (auto& entry : cache.entries) {
        entry.la = entry.end = -1;
        entry.touched = 0;
    }
}

/// Select a hit or rebuild the least recently used entry, retaining the selection through accumulation.
const ScMediumCoefficients& sc_get_medium_coefficients(
    ScMediumCoefficientCache& cache,
    const ScState& state,
    const Rad16Array& previous_J,
    i32 la,
    i32 end
) {
    for (i32 i = 0; i < cache.entries.size(); ++i) {
        auto& entry = cache.entries[i];
        if (entry.la == la && entry.end == end) {
            entry.touched = ++cache.clock;
            cache.current_entry = i;
            ++cache.hits;
            return entry.coefficients;
        }
    }
    auto selected = std::min_element(
        cache.entries.begin(),
        cache.entries.end(),
        [](const auto& a, const auto& b) { return a.touched < b.touched; }
    );
    auto& entry = *selected;
    yakl::timer_start("SC base emis/opac");
    entry.coefficients.fill(state, previous_J, la, end);
    yakl::timer_stop("SC base emis/opac");
    entry.la = la;
    entry.end = end;
    entry.touched = ++cache.clock;
    cache.current_entry = selected - cache.entries.begin();
    ++cache.builds;
    return entry.coefficients;
}
