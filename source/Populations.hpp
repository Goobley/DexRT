#if !defined(DEXRT_POPULATIONS_HPP)
#define DEXRT_POPULATIONS_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "Constants.hpp"
#include "Utils.hpp"

template <typename T=fp_t, typename U=fp_t, int mem_space=yakl::memDevice>
CompAtom<T, mem_space> to_comp_atom(const ModelAtom<U>& model) {
#define DFPU(X) U(FP(X))
    using namespace ConstantsF64;
    CompAtom<T, yakl::memHost> host_atom;
    host_atom.mass = model.element.mass;
    host_atom.abundance = std::exp(model.element.abundance - DFPU(12.0));
    host_atom.Z = model.element.Z;

    const int n_level = model.levels.size();
    yakl::Array<T, 1, yakl::memHost> energy("energy", n_level);
    yakl::Array<T, 1, yakl::memHost> g("g", n_level);
    yakl::Array<T, 1, yakl::memHost> stage("stage", n_level);

    for (int i = 0; i < n_level; ++i) {
        energy(i) = model.levels[i].energy;
        g(i) = model.levels[i].g;
        stage(i) = model.levels[i].stage;
    }
    host_atom.energy = energy;
    host_atom.g = g;
    host_atom.stage = stage;

    const int n_lines = model.lines.size();
    yakl::Array<CompLine<T>, 1, yakl::memHost> lines("lines", n_lines);
    int n_broadening = 0;
    int n_max_wavelengths = 0;
    for (const auto& line : model.lines) {
        n_broadening += line.broadening.size();
        n_max_wavelengths += line.wavelength.size();
    }
    yakl::Array<ScaledExponentsBroadening<T>, 1, yakl::memHost> broadening("broadening", n_broadening);
    int broad_idx = 0;
    for (int kr = 0; kr < n_lines; ++kr) {
        const auto& l = model.lines[kr];
        auto& new_l = lines(kr);
        new_l.type = l.type;
        new_l.j = l.j;
        new_l.i = l.i;
        new_l.f = l.f;
        new_l.g_natural = l.g_natural;
        new_l.Aji = l.Aji;
        // Convert to (nm m2) / kJ
        new_l.Bji = l.Bji_wavelength * DFPU(1e12);
        new_l.Bij = l.Bij_wavelength * DFPU(1e12);
        new_l.lambda0 = l.lambda0;

        new_l.broad_start = broad_idx;
        for (int local_b_idx = 0; local_b_idx < l.broadening.size(); ++local_b_idx) {
            ScaledExponentsBroadening<T> new_broad;
            auto& old_broad = l.broadening[local_b_idx];
            new_broad.scaling = old_broad.scaling;
            new_broad.electron_exponent = old_broad.electron_exponent;
            new_broad.temperature_exponent = old_broad.temperature_exponent;
            new_broad.hydrogen_exponent = old_broad.hydrogen_exponent;
            broadening(broad_idx) = new_broad;
            broad_idx += 1;
        }
        new_l.broad_end = broad_idx;
    }
    host_atom.broadening = broadening;

    const int n_cont = model.continua.size();
    for (const auto& cont : model.continua) {
        n_max_wavelengths += cont.wavelength.size();
    }
    yakl::Array<CompCont<T>, 1, yakl::memHost> continua("continua", n_cont);
    for (int kr = 0; kr < n_cont; ++kr) {
        const auto& c = model.continua[kr];
        auto& new_c = continua(kr);
        new_c.i = c.i;
        new_c.j = c.j;
    }

    // NOTE(cmo): Fancy new wavelength grid... more complex than intended!
    struct WavelengthAndOrigin {
        U wavelength;
        int origin;
    };
    std::vector<WavelengthAndOrigin> default_grid;
    default_grid.reserve(n_max_wavelengths);
    for (int kr = 0; kr < model.continua.size(); ++kr) {
        const auto& cont = model.continua[kr];
        int trans_idx = n_lines + kr;
        for (const auto la : cont.wavelength) {
            WavelengthAndOrigin entry;
            entry.wavelength = la;
            entry.origin = trans_idx;
            default_grid.emplace_back(entry);
        }
    }
    for (int kr = 0; kr < model.lines.size(); ++kr) {
        const auto& line = model.lines[kr];
        int trans_idx = kr;
        for (const auto la : line.wavelength) {
            WavelengthAndOrigin entry;
            entry.wavelength = la;
            entry.origin = trans_idx;
            default_grid.emplace_back(entry);
        }
    }
    std::sort(
        default_grid.begin(), 
        default_grid.end(),
        [](auto a, auto b) {
            return a.wavelength < b.wavelength;
        }
    );
    std::vector<int> active_trans;
    active_trans.reserve(n_lines + n_cont);
    std::vector<U> blue_wavelengths;
    std::vector<U> red_wavelengths;
    std::vector<U> lambda0;
    blue_wavelengths.reserve(n_lines + n_cont);
    red_wavelengths.reserve(n_lines + n_cont);
    lambda0.reserve(n_lines + n_cont);
    for (const auto& line : model.lines) {
        blue_wavelengths.emplace_back(line.wavelength.front());
        red_wavelengths.emplace_back(line.wavelength.back());
        lambda0.emplace_back(line.lambda0);
    }
    for (const auto& cont : model.continua) {
        blue_wavelengths.emplace_back(cont.wavelength.front());
        red_wavelengths.emplace_back(cont.wavelength.back());
        lambda0.emplace_back(cont.wavelength.back());
    }

    auto is_line = [n_lines](int trans_idx) {
        return trans_idx < n_lines;
    };

    auto is_active = [&blue_wavelengths, &red_wavelengths](int trans_idx, U wave) {
        return (wave >= blue_wavelengths[trans_idx] && wave <= red_wavelengths[trans_idx]);
    };

    auto get_active_trans = [&active_trans, &is_active, n_lines, n_cont](U wave) {
        active_trans.clear();
        // NOTE(cmo): Lines will always be first in the active_trans array
        for (int kr = 0; kr < n_lines; ++kr) {
            if (is_active(kr, wave)) {
                active_trans.emplace_back(kr);
            }
        }
        for (int kr = 0; kr < n_cont; ++kr) {
            int trans_idx = kr + n_lines;
            if (is_active(trans_idx, wave)) {
                active_trans.emplace_back(trans_idx);
            }
        }
    };

    std::vector<U> new_grid;
    new_grid.reserve(n_max_wavelengths);
    auto ptr = default_grid.begin();
    while (ptr != default_grid.end()) {
        const U wave = ptr->wavelength;
        get_active_trans(wave);
        bool lines_only = is_line(active_trans[0]);
        U min_weight = FP(1e8);
        int governing_trans = active_trans[0];

        for (int t_idx = 0; t_idx < active_trans.size(); ++t_idx) {
            const auto& t = active_trans[t_idx];
            if (!lines_only || (lines_only && is_line(t))) {
                U inv_weight = std::abs(wave - lambda0[t]);
                if (inv_weight < min_weight) {
                    min_weight = inv_weight;
                    governing_trans = t;
                }
            }
        }

        if (ptr->origin == governing_trans) {
            new_grid.emplace_back(ptr->wavelength);
        }

        ++ptr;
    }

    yakl::Array<T, 1, yakl::memHost> wavelength("wavelength", new_grid.size());
    for (int la = 0; la < new_grid.size(); ++la) {
        wavelength(la) = new_grid[la];
    }
    host_atom.wavelength = wavelength;

    int n_sigma = 0;
    for (int kr = 0; kr < blue_wavelengths.size(); ++kr) {
        auto blue_iter = std::lower_bound(
            new_grid.begin(), 
            new_grid.end(), 
            blue_wavelengths[kr]
        );
        auto red_iter = std::upper_bound(
            new_grid.begin(), 
            new_grid.end(), 
            red_wavelengths[kr]
        );
        if (kr < n_lines) {
            auto& line = lines(kr);
            line.blue_idx = blue_iter - new_grid.begin();
            line.red_idx = red_iter - new_grid.begin();
        } else {
            auto& cont = continua(kr - n_lines);
            cont.blue_idx = blue_iter - new_grid.begin();
            cont.red_idx = red_iter - new_grid.begin();
            n_sigma += cont.red_idx - cont.blue_idx;
        }
    }
    yakl::Array<T, 1, yakl::memHost> sigma("sigma", n_sigma);
    int sigma_offset = 0;
    for (int kr = 0; kr < continua.extent(0); ++kr) {
        const auto& model_cont = model.continua[kr];
        yakl::Array<U const, 1, yakl::memHost> model_wave(
            "wavelength", 
            model_cont.wavelength.data(), 
            model_cont.wavelength.size()
        );
        yakl::Array<U const, 1, yakl::memHost> model_sigma(
            "sigma", 
            model_cont.sigma.data(), 
            model_cont.sigma.size()
        );

        auto& cont = continua(kr);
        cont.sigma_start = sigma_offset;
        for (int la = cont.blue_idx; la < cont.red_idx; ++la) {
            int sigma_idx = sigma_offset + la - cont.blue_idx;
            sigma(sigma_idx) = interp(
                U(wavelength(la)),
                model_wave,
                model_sigma
            );
        }

        sigma_offset += cont.red_idx - cont.blue_idx;
        cont.sigma_end = sigma_offset;
    }
    host_atom.lines = lines;
    host_atom.continua = continua;
    host_atom.sigma = sigma;

    if constexpr (mem_space == yakl::memDevice) {
        CompAtom<T, mem_space> result;
        result.mass = host_atom.mass;
        result.abundance = host_atom.abundance;
        result.Z = host_atom.Z;

        result.energy = host_atom.energy.createDeviceCopy();
        result.g = host_atom.g.createDeviceCopy();
        result.stage = host_atom.stage.createDeviceCopy();
        result.lines = host_atom.lines.createDeviceCopy();
        result.broadening = host_atom.broadening.createDeviceCopy();
        result.wavelength = host_atom.wavelength.createDeviceCopy();
        result.continua = host_atom.continua.createDeviceCopy();
        result.sigma = host_atom.sigma.createDeviceCopy();

        return result;
    } else {
        return host_atom;
    }
#undef DFPU
}


template <typename FPT=fp_t, typename T=fp_t, int mem_space>
YAKL_INLINE
void lte_pops(
    const yakl::Array<T const, 1, mem_space>& energy,
    const yakl::Array<T const, 1, mem_space>& g,
    const yakl::Array<T const, 1, mem_space>& stage,
    fp_t temperature,
    fp_t ne,
    fp_t ntot,
    const yakl::Array<T, 1, mem_space>& pops
) {
    using namespace ConstantsF64;
    // NOTE(cmo): Rearranged for fp_t stability
    constexpr FPT debroglie_const = h / (FP(2.0) * FP(M_PI) * k_B) * (h / m_e);

    const int n_level = energy.extent(0);

    const FPT kbT = temperature * k_B_eV;
    const FPT saha_term = FP(0.5) * ne * std::pow(debroglie_const / temperature, FP(1.5));
    FPT sum = FP(1.0);

    for (int i = 1; i < n_level; ++i) {
        const FPT dE = energy(i) - energy(0);
        const FPT gi0 = g(i) / g(0);
        const int dZ = stage(i) - stage(0);

        const FPT dE_kbT = dE / kbT;
        FPT pop_i = gi0 * std::exp(-dE_kbT);
        for (int _ = 1; _ <= dZ; ++_) {
            pop_i /= saha_term;
        }
        sum += pop_i;
        pops(i) = pop_i;
    }
    const FPT pop_0 = ntot / sum;
    pops(0) = pop_0;

    for (int i = 1; i < n_level; ++i) {
        FPT pop_i = pops(i) * pop_0;
        pop_i = yakl::max(pop_i, std::numeric_limits<FPT>::min());
        pops(i) = pop_i;
    }
}

#else
#endif