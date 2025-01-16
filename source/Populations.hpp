#if !defined(DEXRT_POPULATIONS_HPP)
#define DEXRT_POPULATIONS_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "Constants.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "JasPP.hpp"

template <typename T=fp_t, typename U=fp_t, typename mem_space=DefaultMemSpace>
CompAtom<T, mem_space> to_comp_atom(const ModelAtom<U>& model) {
#define DFPU(X) U(FP(X))
    using namespace ConstantsF64;
    CompAtom<T, HostSpace> host_atom;
    host_atom.mass = model.element.mass;
    host_atom.abundance = std::pow(DFPU(10.0), model.element.abundance - DFPU(12.0));
    host_atom.Z = model.element.Z;

    const int n_level = model.levels.size();
    KView<T*, HostSpace> energy("energy", n_level);
    KView<T*, HostSpace> g("g", n_level);
    KView<T*, HostSpace> stage("stage", n_level);

    for (int i = 0; i < n_level; ++i) {
        energy(i) = model.levels[i].energy;
        g(i) = model.levels[i].g;
        stage(i) = model.levels[i].stage;
    }
    host_atom.energy = energy;
    host_atom.g = g;
    host_atom.stage = stage;

    const int n_lines = model.lines.size();
    KView<CompLine<T>*, HostSpace> lines("lines", n_lines);
    int n_broadening = 0;
    int n_max_wavelengths = 0;
    for (const auto& line : model.lines) {
        n_broadening += line.broadening.size();
        n_max_wavelengths += line.wavelength.size();
    }
    KView<ScaledExponentsBroadening<T>*, HostSpace> broadening("broadening", n_broadening);
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
    KView<CompCont<T>*, HostSpace> continua("continua", n_cont);
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

    KView<T*, HostSpace> wavelength("wavelength", new_grid.size());
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
    KView<T*, HostSpace> sigma("sigma", n_sigma);
    int sigma_offset = 0;
    for (int kr = 0; kr < continua.extent(0); ++kr) {
        const auto& model_cont = model.continua[kr];
        auto model_wave = KView(model_cont.wavelength.data(), model_cont.wavelength.size());
        auto model_sigma = KView(model_cont.sigma.data(), model_cont.sigma.size());

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

    int n_temperature = 0;
    for (int i = 0; i < model.coll_rates.size(); ++i) {
        n_temperature += model.coll_rates[i].temperature.size();
    }
    KView<T*, HostSpace> temperature("temperature grid", n_temperature);
    KView<T*, HostSpace> coll_rates("coll rates grid", n_temperature);
    KView<CompColl<T>*, HostSpace> collisions("collisions", model.coll_rates.size());
    int temp_offset = 0;
    for (int i = 0; i < model.coll_rates.size(); ++i) {
        const auto& coll = model.coll_rates[i];
        collisions(i).type = coll.type;
        collisions(i).j = coll.j;
        collisions(i).i = coll.i;
        collisions(i).start_idx = temp_offset;
        for (int temp_idx = 0; temp_idx < coll.temperature.size(); ++temp_idx) {
            temperature(temp_offset + temp_idx) = coll.temperature[temp_idx];
            coll_rates(temp_offset + temp_idx) = coll.data[temp_idx];
        }
        temp_offset += coll.temperature.size();
        collisions(i).end_idx = temp_offset;
    }
    host_atom.temperature = temperature;
    host_atom.coll_rates = coll_rates;
    host_atom.collisions = collisions;

    if constexpr (std::is_same_v<mem_space, HostSpace>) {
        return host_atom;
    } else {
        CompAtom<T, mem_space> result;
        result.mass = host_atom.mass;
        result.abundance = host_atom.abundance;
        result.Z = host_atom.Z;
        result.treatment = host_atom.treatment;

        result.energy = create_device_copy(host_atom.energy);
        result.g = create_device_copy(host_atom.g);
        result.stage = create_device_copy(host_atom.stage);
        result.lines = create_device_copy(host_atom.lines);
        result.broadening = create_device_copy(host_atom.broadening);
        result.wavelength = create_device_copy(host_atom.wavelength);
        result.continua = create_device_copy(host_atom.continua);
        result.sigma = create_device_copy(host_atom.sigma);
        result.temperature = create_device_copy(host_atom.temperature);
        result.coll_rates = create_device_copy(host_atom.coll_rates);
        result.collisions = create_device_copy(host_atom.collisions);

        return result;
    }
#undef DFPU
}

template <typename T>
struct AtomicDataHostDevice {
    AtomicData<T, HostSpace> host;
    AtomicData<T, DefaultMemSpace> device;
    bool have_h_model = false;
};

template <typename T=fp_t, typename U=fp_t>
AtomicDataHostDevice<T> to_atomic_data(std::vector<ModelAtom<U>> models) {
#define DFPU(X) U(FP(X))
    using namespace ConstantsF64;
    // Sort by mass
    std::sort(
        models.begin(),
        models.end(),
        [] (auto a, auto b) {
            return a.element.mass < b.element.mass;
        }
    );
    // TODO(cmo): Currently just set all atoms active
    AtomicData<T, HostSpace> host_data;
    const int n_atom = models.size();
    KView<T*, HostSpace> mass("mass", n_atom);
    KView<T*, HostSpace> abundance("abundance", n_atom);
    KView<int*, HostSpace> Z("Z", n_atom);
    KView<AtomicTreatment*, HostSpace> treatment("treatment", n_atom);
    for (int i = 0; i < n_atom; ++i) {
        mass(i) = models[i].element.mass;
        Z(i) = models[i].element.Z;
        abundance(i) = std::pow(DFPU(10.0), models[i].element.abundance - DFPU(12.0));
        treatment(i) = models[i].treatment;
    }
    JasPack(host_data, mass, abundance, Z, treatment);

    int total_n_level = 0;
    int total_n_line = 0;
    int total_n_cont = 0;
    int total_n_coll = 0;
    KView<int*, HostSpace> level_start("level_start", n_atom);
    KView<int*, HostSpace> num_level("num_level", n_atom);
    KView<int*, HostSpace> line_start("line_start", n_atom);
    KView<int*, HostSpace> num_line("num_line", n_atom);
    KView<int*, HostSpace> cont_start("cont_start", n_atom);
    KView<int*, HostSpace> num_cont("num_cont", n_atom);
    KView<int*, HostSpace> coll_start("coll_start", n_atom);
    KView<int*, HostSpace> num_coll("num_coll", n_atom);
    for (int ia = 0; ia < n_atom; ++ia) {
        level_start(ia) = total_n_level;
        num_level(ia) = models[ia].levels.size();
        total_n_level += num_level(ia);

        line_start(ia) = total_n_line;
        num_line(ia) = models[ia].lines.size();
        total_n_line += num_line(ia);

        cont_start(ia) = total_n_cont;
        num_cont(ia) = models[ia].continua.size();
        total_n_cont += num_cont(ia);

        coll_start(ia) = total_n_coll;
        num_coll(ia) = models[ia].coll_rates.size();
        total_n_coll += num_coll(ia);
    }
    JasPack(host_data, level_start, num_level, line_start, num_line);
    JasPack(host_data, cont_start, num_cont, coll_start, num_coll);

    KView<T*, HostSpace> energy("energy", total_n_level);
    KView<T*, HostSpace> g("g", total_n_level);
    KView<T*, HostSpace> stage("stage", total_n_level);

    for (int ia = 0; ia < n_atom; ++ia) {
        const int i_base = level_start(ia);
        for (int i = 0; i < num_level(ia); ++i) {
            energy(i_base + i) = models[ia].levels[i].energy;
            g(i_base + i) = models[ia].levels[i].g;
            stage(i_base + i) = models[ia].levels[i].stage;
        }
    }
    JasPack(host_data, energy, g, stage);

    KView<CompLine<T>*, HostSpace> lines("lines", total_n_line);
    int n_broadening = 0;
    int n_max_wavelengths = 0;
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        for (const auto& line : model.lines) {
            n_broadening += line.broadening.size();
            n_max_wavelengths += line.wavelength.size();
        }
    }
    KView<ScaledExponentsBroadening<T>*, HostSpace> broadening("broadening", n_broadening);
    int broad_idx = 0;
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        const int kr_base = line_start(ia);
        for (int kr = 0; kr < num_line(ia); ++kr) {
            const auto& l = model.lines[kr];
            auto& new_l = lines(kr_base + kr);
            new_l.type = l.type;
            new_l.atom = ia;
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
    }
    host_data.broadening = broadening;

    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        for (const auto& cont : model.continua) {
            n_max_wavelengths += cont.wavelength.size();
        }
    }
    KView<CompCont<T>*> continua("continua", total_n_cont);
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        const int kr_base = cont_start(ia);
        for (int kr = 0; kr < num_cont(ia); ++kr) {
            const auto& c = model.continua[kr];
            auto& new_c = continua(kr_base + kr);
            new_c.atom = ia;
            new_c.i = c.i;
            new_c.j = c.j;
        }
    }

    // NOTE(cmo): Fancy new wavelength grid... more complex than intended!
    struct WavelengthAndOrigin {
        U wavelength;
        int origin;
    };
    std::vector<WavelengthAndOrigin> default_grid;
    default_grid.reserve(n_max_wavelengths);
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        const int kr_base = cont_start(ia);
        for (int kr = 0; kr < model.continua.size(); ++kr) {
            const auto& cont = model.continua[kr];
            int trans_idx = total_n_line + kr_base + kr;
            for (const auto la : cont.wavelength) {
                WavelengthAndOrigin entry;
                entry.wavelength = la;
                entry.origin = trans_idx;
                default_grid.emplace_back(entry);
            }
        }
    }
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        const int kr_base = line_start(ia);
        for (int kr = 0; kr < model.lines.size(); ++kr) {
            const auto& line = model.lines[kr];
            int trans_idx = kr_base + kr;
            for (const auto la : line.wavelength) {
                WavelengthAndOrigin entry;
                entry.wavelength = la;
                entry.origin = trans_idx;
                default_grid.emplace_back(entry);
            }
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
    active_trans.reserve(total_n_line + total_n_cont);
    std::vector<U> blue_wavelengths;
    std::vector<U> red_wavelengths;
    std::vector<U> lambda0;
    blue_wavelengths.reserve(total_n_line + total_n_cont);
    red_wavelengths.reserve(total_n_line + total_n_cont);
    lambda0.reserve(total_n_line + total_n_cont);
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        for (const auto& line : model.lines) {
            blue_wavelengths.emplace_back(line.wavelength.front());
            red_wavelengths.emplace_back(line.wavelength.back());
            lambda0.emplace_back(line.lambda0);
        }
    }
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        for (const auto& cont : model.continua) {
            blue_wavelengths.emplace_back(cont.wavelength.front());
            red_wavelengths.emplace_back(cont.wavelength.back());
            lambda0.emplace_back(cont.wavelength.back());
        }
    }
    auto is_line = [total_n_line](int trans_idx) {
        return trans_idx < total_n_line;
    };

    auto is_active = [&blue_wavelengths, &red_wavelengths](int trans_idx, U wave) {
        return (wave >= blue_wavelengths[trans_idx] && wave <= red_wavelengths[trans_idx]);
    };

    auto get_active_trans = [&active_trans, &is_active, total_n_line, total_n_cont](U wave) {
        active_trans.clear();
        // NOTE(cmo): Lines will always be first in the active_trans array
        for (int kr = 0; kr < total_n_line; ++kr) {
            if (is_active(kr, wave)) {
                active_trans.emplace_back(kr);
            }
        }
        for (int kr = 0; kr < total_n_cont; ++kr) {
            int trans_idx = kr + total_n_line;
            if (is_active(trans_idx, wave)) {
                active_trans.emplace_back(trans_idx);
            }
        }
    };

    std::vector<U> new_grid;
    new_grid.reserve(n_max_wavelengths);
    std::vector<TransitionIndex> gov_trans;
    gov_trans.reserve(n_max_wavelengths);
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
            TransitionIndex trans_idx;
            if (is_line(governing_trans)) {
                trans_idx.atom = lines(governing_trans).atom;
                trans_idx.kr = governing_trans - line_start(trans_idx.atom);
                trans_idx.line = true;
            } else {
                trans_idx.atom = continua(governing_trans - total_n_line).atom;
                trans_idx.kr = governing_trans - total_n_line - cont_start(trans_idx.atom);
                trans_idx.line = false;
            }
            gov_trans.emplace_back(trans_idx);
        }

        ++ptr;
    }

    KView<T*, HostSpace> wavelength("wavelength", new_grid.size());
    KView<TransitionIndex*, HostSpace> governing_trans("governing_trans", new_grid.size());
    for (int la = 0; la < new_grid.size(); ++la) {
        wavelength(la) = new_grid[la];
        governing_trans(la) = gov_trans[la];
    }
    JasPack(host_data, wavelength, governing_trans);

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
        if (kr < total_n_line) {
            auto& line = lines(kr);
            line.blue_idx = blue_iter - new_grid.begin();
            line.red_idx = red_iter - new_grid.begin();
        } else {
            auto& cont = continua(kr - total_n_line);
            cont.blue_idx = blue_iter - new_grid.begin();
            cont.red_idx = red_iter - new_grid.begin();
            n_sigma += cont.red_idx - cont.blue_idx;
        }
    }
    KView<T*, HostSpace> sigma("sigma", n_sigma);
    int sigma_offset = 0;
    for (int kr = 0; kr < continua.extent(0); ++kr) {
        auto& cont = continua(kr);
        const auto& model_cont = models[cont.atom].continua[kr - cont_start(cont.atom)];
        auto model_wave = KView(model_cont.wavelength.data(), model_cont.wavelength.size());
        auto model_sigma = KView(model_cont.sigma.data(), model_cont.sigma.size());

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
    JasPack(host_data, lines, continua, sigma);

    int n_temperature = 0;
    int n_collisions = 0;
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        for (int i = 0; i < model.coll_rates.size(); ++i) {
            n_temperature += model.coll_rates[i].temperature.size();
            n_collisions += 1;
        }
    }
    KView<T*, HostSpace> temperature("temperature grid", n_temperature);
    KView<T*, HostSpace> coll_rates("coll rates grid", n_temperature);
    KView<CompColl<T>*, HostSpace> collisions("collisions", n_collisions);
    int temp_offset = 0;
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        const int i_base = coll_start(ia);
        for (int i = 0; i < model.coll_rates.size(); ++i) {
            const auto& coll = model.coll_rates[i];
            collisions(i_base + i).type = coll.type;
            collisions(i_base + i).atom = ia;
            collisions(i_base + i).j = coll.j;
            collisions(i_base + i).i = coll.i;
            collisions(i_base + i).start_idx = temp_offset;
            for (int temp_idx = 0; temp_idx < coll.temperature.size(); ++temp_idx) {
                temperature(temp_offset + temp_idx) = coll.temperature[temp_idx];
                coll_rates(temp_offset + temp_idx) = coll.data[temp_idx];
            }
            temp_offset += coll.temperature.size();
            collisions(i_base + i).end_idx = temp_offset;
        }
    }
    JasPack(host_data, temperature, coll_rates, collisions);

    KView<i32*, HostSpace> active_lines_start("active lines start idx", wavelength.extent(0));
    KView<i32*, HostSpace> active_lines_end("active lines end idx", wavelength.extent(0));
    KView<i32*, HostSpace> active_cont_start("active cont start idx", wavelength.extent(0));
    KView<i32*, HostSpace> active_cont_end("active cont end idx", wavelength.extent(0));
    std::vector<u16> active_lines;
    active_lines.reserve(3 * wavelength.extent(0));
    std::vector<u16> active_cont;
    active_cont.reserve(3 * wavelength.extent(0));
    for (int la = 0; la < wavelength.extent(0); ++la) {
        active_lines_start(la) = active_lines.size();
        for (int kr = 0; kr < lines.extent(0); ++kr) {
            const auto& line = lines(kr);
            if (line.is_active(la)) {
                active_lines.emplace_back(kr);
            }
        }
        active_lines_end(la) = active_lines.size();
        active_cont_start(la) = active_cont.size();
        for (int kr = 0; kr < continua.extent(0); ++kr) {
            const auto& cont = continua(kr);
            if (cont.is_active(la)) {
                active_cont.emplace_back(kr);
            }
        }
        active_cont_end(la) = active_cont.size();
    }
    KView<u16*, HostSpace> line_active_set("active lines flat buffer", active_lines.size());
    for (int i = 0; i < active_lines.size(); ++i) {
        line_active_set(i) = active_lines[i];
    }
    KView<u16*, HostSpace> cont_active_set("active lines flat buffer", active_cont.size());
    for (int i = 0; i < active_cont.size(); ++i) {
        cont_active_set(i) = active_cont[i];
    }
    host_data.active_lines = line_active_set;
    host_data.active_lines_start = active_lines_start;
    host_data.active_lines_end = active_lines_end;
    host_data.active_cont = cont_active_set;
    host_data.active_cont_start = active_cont_start;
    host_data.active_cont_end = active_cont_end;

    AtomicData<T, DefaultMemSpace> dev_data{
        .treatment = create_device_copy(host_data.treatment),
        .mass = create_device_copy(host_data.mass),
        .abundance = create_device_copy(host_data.abundance),
        .Z = create_device_copy(host_data.Z),
        .level_start = create_device_copy(host_data.level_start),
        .num_level = create_device_copy(host_data.num_level),
        .line_start = create_device_copy(host_data.line_start),
        .num_line = create_device_copy(host_data.num_line),
        .cont_start = create_device_copy(host_data.cont_start),
        .num_cont = create_device_copy(host_data.num_cont),
        .coll_start = create_device_copy(host_data.coll_start),
        .num_coll = create_device_copy(host_data.num_coll),
        .energy = create_device_copy(host_data.energy),
        .g = create_device_copy(host_data.g),
        .stage = create_device_copy(host_data.stage),
        .lines = create_device_copy(host_data.lines),
        .broadening = create_device_copy(host_data.broadening),
        .continua = create_device_copy(host_data.continua),
        .sigma = create_device_copy(host_data.sigma),
        .wavelength = create_device_copy(host_data.wavelength),
        .governing_trans = create_device_copy(host_data.governing_trans),
        .collisions = create_device_copy(host_data.collisions),
        .temperature = create_device_copy(host_data.temperature),
        .coll_rates = create_device_copy(host_data.coll_rates),
        .active_lines = create_device_copy(host_data.active_lines),
        .active_lines_start = create_device_copy(host_data.active_lines_start),
        .active_lines_end = create_device_copy(host_data.active_lines_end),
        .active_cont = create_device_copy(host_data.active_cont),
        .active_cont_start = create_device_copy(host_data.active_cont_start),
        .active_cont_end = create_device_copy(host_data.active_cont_end)
    };

    AtomicDataHostDevice<T> result{
        .host = host_data,
        .device = dev_data,
        .have_h_model = (models[0].element.Z == 1)
    };

    return result;

#undef DFPU
}

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
inline
CompAtom<T, mem_space> extract_atom(
    const AtomicData<T, mem_space>& adata,
    const AtomicData<T, HostSpace>& adata_host,
    int ia
) {
    assert(ia < adata.level_start.extent(0));

    const int level_start = adata_host.level_start(ia);
    const int n_level = adata_host.num_level(ia);
    auto level_range = std::make_pair(level_start, level_start + n_level);
    const int line_start = adata_host.line_start(ia);
    const int n_line = adata_host.num_line(ia);
    auto line_range = std::make_pair(line_start, line_start + n_line);
    const int cont_start = adata_host.cont_start(ia);
    const int n_cont = adata_host.num_cont(ia);
    auto cont_range = std::make_pair(cont_start, cont_start + n_cont);
    const int coll_start = adata_host.coll_start(ia);
    const int n_coll = adata_host.num_coll(ia);
    auto coll_range = std::make_pair(coll_start, coll_start + n_coll);

    CompAtom<T, mem_space> result{
        .mass = adata_host.mass(ia),
        .abundance = adata_host.abundance(ia),
        .Z = adata_host.Z(ia),
        .treatment = adata_host.treatment(ia),

        .energy = Kokkos::subview(adata.energy, level_range),
        .g = Kokkos::subview(adata.g, level_range),
        .stage = Kokkos::subview(adata.stage, level_range),

        .lines = Kokkos::subview(adata.lines, line_range),
        // NOTE(cmo): We hand over the whole broadening array as the lines are set up to index into this
        .broadening = adata.broadening,
        .wavelength = adata.wavelength,

        .continua = Kokkos::subview(adata.continua, cont_range),
        // NOTE(cmo): Same for sigma
        .sigma = adata.sigma,

        .collisions = Kokkos::subview(adata.collisions, coll_range),
        // NOTE(cmo): Same for collisional data
        .temperature = adata.temperature,
        .coll_rates = adata.coll_rates
    };
    return result;
}

template <typename T, typename mem_space=DefaultMemSpace>
YAKL_INLINE
LteTerms<T, mem_space>
extract_lte_terms_dev(const AtomicData<T, mem_space>& adata, int ia) {
    const int level_start = adata.level_start(ia);
    const int n_level = adata.num_level(ia);
    auto level_range = std::make_pair(level_start, level_start + n_level);
    LteTerms<T, mem_space> result = {
        .mass = adata.mass(ia),
        .abundance = adata.abundance(ia),
        .energy = Kokkos::subview(adata.energy, level_range),
        .g = Kokkos::subview(adata.g, level_range),
        .stage = Kokkos::subview(adata.stage, level_range)
    };
    return result;
}


template <typename T, typename mem_space=DefaultMemSpace>
inline
std::vector<CompAtom<T, mem_space>>
extract_atoms(
    const AtomicData<T, mem_space>& adata,
    const AtomicData<T, HostSpace> adata_host
) {
    const int n_atom = adata.num_level.extent(0);
    std::vector<CompAtom<T, mem_space>> result(n_atom);
    for (int ia = 0; ia < n_atom; ++ia) {
        result[ia] = extract_atom(adata, adata_host, ia);
    }
    return result;
}

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
struct GammaAtomsAndMapping {
    std::vector<CompAtom<T, mem_space>> atoms;
    std::vector<int> mapping;

};
template <typename T=fp_t, typename mem_space=DefaultMemSpace>
inline
GammaAtomsAndMapping<T, mem_space>
extract_atoms_with_gamma_and_mapping(
    const AtomicData<T, mem_space>& adata,
    const AtomicData<T, HostSpace>& adata_h
) {
    const int n_atom = adata.num_level.extent(0);
    GammaAtomsAndMapping<T, mem_space> result;
    for (int ia = 0; ia < n_atom; ++ia) {
        if (has_gamma(adata_h.treatment(ia))) {
            result.mapping.emplace_back(ia);
            result.atoms.emplace_back(extract_atom(adata, adata_h, ia));
        }
    }
    return result;
}

template <typename FPT=fp_t, typename T=fp_t, typename mem_space=DefaultMemSpace>
YAKL_INLINE
void lte_pops(
    const KView<T*, mem_space>& energy,
    const KView<T*, mem_space>& g,
    const KView<T*, mem_space>& stage,
    fp_t temperature,
    fp_t ne,
    fp_t ntot,
    const KView<T**, mem_space>& pops,
    int64_t x
) {
    using namespace ConstantsF64;
    // NOTE(cmo): Rearranged for fp_t stability
    constexpr FPT debroglie_const = FPT(h / (FP(2.0) * pi * k_B) * (h / m_e));

    const int n_level = energy.extent(0);

    const FPT kbT = temperature * FPT(k_B_eV);
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
        pops(i, x) = pop_i;
    }
    const FPT pop_0 = ntot / sum;
    pops(0, x) = pop_0;

    for (int i = 1; i < n_level; ++i) {
        FPT pop_i = pops(i, x) * pop_0;
        pop_i = std::max(pop_i, std::numeric_limits<FPT>::min());
        pops(i, x) = pop_i;
    }
}

void compute_lte_pops_flat(
    const CompAtom<fp_t>& atom,
    const SparseAtmosphere& atmos,
    const Fp2d& pops
);

/**
 * Computes the LTE populations in state. Assumes state->pops is already allocated.
*/
void compute_lte_pops(State* state);

/**
 * Computes the LTE populations in to a provided allocated array.
*/
void compute_lte_pops(const State* state, const Fp2d& shared_pops);

struct StatEqOptions {
    /// When computing relative change, ignore the change in populations with a
    /// starting fraction lower than this
    fp_t ignore_change_below_ntot_frac = FP(0.0);
};

void compute_nh0(const State& state);

/**
 * Computes the statistical equilibrium solution for the atoms in State.
 * Internal precision configured in Config.
 */
fp_t stat_eq(State* state, const StatEqOptions& args = StatEqOptions());

#else
#endif