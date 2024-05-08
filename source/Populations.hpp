#if !defined(DEXRT_POPULATIONS_HPP)
#define DEXRT_POPULATIONS_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "Constants.hpp"
#include "Utils.hpp"
#include "State.hpp"

template <typename T=fp_t, typename U=fp_t, int mem_space=yakl::memDevice>
CompAtom<T, mem_space> to_comp_atom(const ModelAtom<U>& model) {
#define DFPU(X) U(FP(X))
    using namespace ConstantsF64;
    CompAtom<T, yakl::memHost> host_atom;
    host_atom.mass = model.element.mass;
    host_atom.abundance = std::pow(DFPU(10.0), model.element.abundance - DFPU(12.0));
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

    int n_temperature = 0;
    for (int i = 0; i < model.coll_rates.size(); ++i) {
        n_temperature += model.coll_rates[i].temperature.size();
    }
    yakl::Array<T, 1, yakl::memHost> temperature("temperature grid", n_temperature);
    yakl::Array<T, 1, yakl::memHost> coll_rates("coll rates grid", n_temperature);
    yakl::Array<CompColl<T>, 1, yakl::memHost> collisions("collisions", model.coll_rates.size());
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

    yakl::Array<i32, 1, yakl::memHost> lines_start("active lines start idx", wavelength.extent(0));
    yakl::Array<i32, 1, yakl::memHost> lines_end("active lines end idx", wavelength.extent(0));
    yakl::Array<i32, 1, yakl::memHost> cont_start("active cont start idx", wavelength.extent(0));
    yakl::Array<i32, 1, yakl::memHost> cont_end("active cont end idx", wavelength.extent(0));
    std::vector<u16> active_lines;
    active_lines.reserve(3 * wavelength.extent(0));
    std::vector<u16> active_cont;
    active_cont.reserve(3 * wavelength.extent(0));
    for (int la = 0; la < wavelength.extent(0); ++la) {
        lines_start(la) = active_lines.size();
        for (int kr = 0; kr < lines.extent(0); ++kr) {
            const auto& line = lines(kr);
            if (line.is_active(la)) {
                active_lines.emplace_back(kr);
            }
        }
        lines_end(la) = active_lines.size();
        cont_start(la) = active_cont.size();
        for (int kr = 0; kr < continua.extent(0); ++kr) {
            const auto& cont = continua(kr);
            if (cont.is_active(la)) {
                active_cont.emplace_back(kr);
            }
        }
        cont_end(la) = active_cont.size();
    }
    yakl::Array<u16, 1, yakl::memHost> line_active_set("active lines flat buffer", active_lines.size());
    for (int i = 0; i < active_lines.size(); ++i) {
        line_active_set(i) = active_lines[i];
    }
    yakl::Array<u16, 1, yakl::memHost> cont_active_set("active lines flat buffer", active_cont.size());
    for (int i = 0; i < active_cont.size(); ++i) {
        cont_active_set(i) = active_cont[i];
    }
    host_atom.active_lines = line_active_set;
    host_atom.active_lines_start = lines_start;
    host_atom.active_lines_end = lines_end;
    host_atom.active_cont = cont_active_set;
    host_atom.active_cont_start = cont_start;
    host_atom.active_cont_end = cont_end;


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
        result.temperature = host_atom.temperature.createDeviceCopy();
        result.coll_rates = host_atom.coll_rates.createDeviceCopy();
        result.collisions = host_atom.collisions.createDeviceCopy();

        result.active_lines = host_atom.active_lines.createDeviceCopy();
        result.active_lines_start = host_atom.active_lines_start.createDeviceCopy();
        result.active_lines_end = host_atom.active_lines_end.createDeviceCopy();
        result.active_cont = host_atom.active_cont.createDeviceCopy();
        result.active_cont_start = host_atom.active_cont_start.createDeviceCopy();
        result.active_cont_end = host_atom.active_cont_end.createDeviceCopy();

        return result;
    } else {
        return host_atom;
    }
#undef DFPU
}

template <typename T=fp_t, typename U=fp_t, int mem_space=yakl::memDevice>
AtomicData<T, mem_space> to_atomic_data(const std::vector<ModelAtom<U>>& models) {
#define DFPU(X) U(FP(X))
    using namespace ConstantsF64;
    // TODO(cmo): Return both host and device from this function
    AtomicData<T, yakl::memHost> host_data;
    const int n_atom = models.size();
    yakl::Array<T, 1, yakl::memHost> mass("mass", n_atom);
    yakl::Array<T, 1, yakl::memHost> abundance("abundance", n_atom);
    yakl::Array<int, 1, yakl::memHost> Z("Z", n_atom);
    for (int i = 0; i < n_atom; ++i) {
        mass(i) = models[i].element.mass;
        Z(i) = models[i].element.Z;
        abundance(i) = std::pow(DFPU(10.0), models[i].element.abundance - DFPU(12.0));
    }
    JasPack(host_data, mass, abundance, Z);

    int total_n_level = 0;
    int total_n_line = 0;
    int total_n_cont = 0;
    int total_n_coll = 0;
    yakl::Array<int, 1, yakl::memHost> level_start("level_start", n_atom);
    yakl::Array<int, 1, yakl::memHost> num_level("num_level", n_atom);
    yakl::Array<int, 1, yakl::memHost> line_start("line_start", n_atom);
    yakl::Array<int, 1, yakl::memHost> num_line("num_line", n_atom);
    yakl::Array<int, 1, yakl::memHost> cont_start("cont_start", n_atom);
    yakl::Array<int, 1, yakl::memHost> num_cont("num_cont", n_atom);
    yakl::Array<int, 1, yakl::memHost> coll_start("coll_start", n_atom);
    yakl::Array<int, 1, yakl::memHost> num_coll("num_coll", n_atom);
    for (int ia = 0; ia < n_atom; ++ia) {
        level_start(ia) = total_n_level;
        num_level(ia) = models[ia].levels().size();
        total_n_level += num_level(ia);

        line_start(ia) = total_n_line;
        num_line(ia) = models[ia].lines.size();
        total_n_line += num_line(ia);

        cont_start(ia) = total_n_cont;
        num_cont(ia) = models[ia].continua.size();
        total_n_cont += num_cont(ia);

        coll_start(ia) = total_n_coll;
        num_coll(ia) = models[ia].collisions.size();
        total_n_coll += num_coll(ia);
    }
    JasPack(host_data, level_start, num_level, line_start, num_line);
    JasPack(host_data, cont_start, num_cont, coll_start, num_coll);

    yakl::Array<T, 1, yakl::memHost> energy("energy", total_n_level);
    yakl::Array<T, 1, yakl::memHost> g("g", total_n_level);
    yakl::Array<T, 1, yakl::memHost> stage("stage", total_n_level);

    for (int ia = 0; ia < n_atom; ++ia) {
        const int i_base = level_start(ia);
        for (int i = 0; i < num_level(ia); ++i) {
            energy(i_base + i) = models[ia].levels[i].energy;
            g(i_base + i) = models[ia].levels[i].g;
            stage(i_base + i) = models[ia].levels[i].stage;
        }
    }
    JasPack(host_data, energy, g, stage);

    yakl::Array<CompLine<T>, 1, yakl::memHost> lines("lines", total_n_line);
    int n_broadening = 0;
    int n_max_wavelengths = 0;
    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        for (const auto& line : model.lines) {
            n_broadening += line.broadening.size();
            n_max_wavelengths += line.wavelength.size();
        }
    }
    yakl::Array<ScaledExponentsBroadening<T>, 1, yakl::memHost> broadening("broadening", n_broadening);
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
    host_atom.broadening = broadening;

    for (int ia = 0; ia < n_atom; ++ia) {
        const auto& model = models[ia];
        for (const auto& cont : model.continua) {
            n_max_wavelengths += cont.wavelength.size();
        }
    }
    yakl::Array<CompCont<T>, 1, yakl::memHost> continua("continua", total_n_cont);
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

    yakl::Array<T, 1, yakl::memHost> wavelength("wavelength", new_grid.size());
    yakl::Array<TransitionIndex, 1, yakl::memHost> governing_trans("governing_trans", new_grid.size());
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
    yakl::Array<T, 1, yakl::memHost> sigma("sigma", n_sigma);
    int sigma_offset = 0;
    for (int kr = 0; kr < continua.extent(0); ++kr) {
        auto& cont = continua(kr);
        const auto& model_cont = models[cont.atom].continua[kr - cont_start(cont.atom)];
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

    assert(false && "COntinue updating from here");
    int n_temperature = 0;
    for (int i = 0; i < model.coll_rates.size(); ++i) {
        n_temperature += model.coll_rates[i].temperature.size();
    }
    yakl::Array<T, 1, yakl::memHost> temperature("temperature grid", n_temperature);
    yakl::Array<T, 1, yakl::memHost> coll_rates("coll rates grid", n_temperature);
    yakl::Array<CompColl<T>, 1, yakl::memHost> collisions("collisions", model.coll_rates.size());
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

    yakl::Array<i32, 1, yakl::memHost> lines_start("active lines start idx", wavelength.extent(0));
    yakl::Array<i32, 1, yakl::memHost> lines_end("active lines end idx", wavelength.extent(0));
    yakl::Array<i32, 1, yakl::memHost> cont_start("active cont start idx", wavelength.extent(0));
    yakl::Array<i32, 1, yakl::memHost> cont_end("active cont end idx", wavelength.extent(0));
    std::vector<u16> active_lines;
    active_lines.reserve(3 * wavelength.extent(0));
    std::vector<u16> active_cont;
    active_cont.reserve(3 * wavelength.extent(0));
    for (int la = 0; la < wavelength.extent(0); ++la) {
        lines_start(la) = active_lines.size();
        for (int kr = 0; kr < lines.extent(0); ++kr) {
            const auto& line = lines(kr);
            if (line.is_active(la)) {
                active_lines.emplace_back(kr);
            }
        }
        lines_end(la) = active_lines.size();
        cont_start(la) = active_cont.size();
        for (int kr = 0; kr < continua.extent(0); ++kr) {
            const auto& cont = continua(kr);
            if (cont.is_active(la)) {
                active_cont.emplace_back(kr);
            }
        }
        cont_end(la) = active_cont.size();
    }
    yakl::Array<u16, 1, yakl::memHost> line_active_set("active lines flat buffer", active_lines.size());
    for (int i = 0; i < active_lines.size(); ++i) {
        line_active_set(i) = active_lines[i];
    }
    yakl::Array<u16, 1, yakl::memHost> cont_active_set("active lines flat buffer", active_cont.size());
    for (int i = 0; i < active_cont.size(); ++i) {
        cont_active_set(i) = active_cont[i];
    }
    host_atom.active_lines = line_active_set;
    host_atom.active_lines_start = lines_start;
    host_atom.active_lines_end = lines_end;
    host_atom.active_cont = cont_active_set;
    host_atom.active_cont_start = cont_start;
    host_atom.active_cont_end = cont_end;


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
        result.temperature = host_atom.temperature.createDeviceCopy();
        result.coll_rates = host_atom.coll_rates.createDeviceCopy();
        result.collisions = host_atom.collisions.createDeviceCopy();

        result.active_lines = host_atom.active_lines.createDeviceCopy();
        result.active_lines_start = host_atom.active_lines_start.createDeviceCopy();
        result.active_lines_end = host_atom.active_lines_end.createDeviceCopy();
        result.active_cont = host_atom.active_cont.createDeviceCopy();
        result.active_cont_start = host_atom.active_cont_start.createDeviceCopy();
        result.active_cont_end = host_atom.active_cont_end.createDeviceCopy();

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
    const yakl::Array<T, 2, mem_space>& pops,
    int64_t x
) {
    using namespace ConstantsF64;
    // NOTE(cmo): Rearranged for fp_t stability
    constexpr FPT debroglie_const = FPT(h / (FP(2.0) * FP(M_PI) * k_B) * (h / m_e));

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
        pop_i = yakl::max(pop_i, std::numeric_limits<FPT>::min());
        pops(i, x) = pop_i;
    }
}

inline void compute_lte_pops_flat(
    const CompAtom<fp_t>& atom,
    const Atmosphere& atmos,
    const Fp2d& pops
) {
    const auto& temperature = atmos.temperature.collapse();
    const auto& ne = atmos.ne.collapse();
    const auto& nhtot = atmos.nh_tot.collapse();
    parallel_for(
        "LTE Pops",
        SimpleBounds<1>(pops.extent(1)),
        YAKL_LAMBDA (int64_t k) {
            lte_pops(
                atom.energy,
                atom.g,
                atom.stage,
                temperature(k),
                ne(k),
                atom.abundance * nhtot(k),
                pops,
                k
            );
        }
    );
}

/**
 * Computes the LTE populations in state. Assumes state->pops is already allocated.
*/
inline void compute_lte_pops(State* state) {
    auto pops_dims = state->pops.get_dimensions();
    const auto& pops = state->pops.reshape<2>(Dims(pops_dims(0), pops_dims(1) * pops_dims(2)));
    compute_lte_pops_flat(state->atom, state->atmos, pops);
}

template <typename T=fp_t>
inline fp_t stat_eq(State* state) {
    auto Gamma_host = state->Gamma.createHostCopy();
    const auto& Gamma = state->Gamma.reshape<3>(Dims(
        state->Gamma.extent(0),
        state->Gamma.extent(1),
        state->Gamma.extent(2) * state->Gamma.extent(3)
    ));
    yakl::Array<T, 3, yakl::memDevice> GammaT("GammaT", Gamma.extent(2), Gamma.extent(0), Gamma.extent(1));
    yakl::Array<T*, 1, yakl::memDevice> GammaT_ptrs("GammaT_ptrs", GammaT.extent(0));
    const auto& pops = state->pops.reshape<2>(Dims(
        state->pops.extent(0),
        state->pops.extent(1) * state->pops.extent(2)
    ));
    yakl::Array<T, 2, yakl::memDevice> new_pops("new_pops", GammaT.extent(0), GammaT.extent(1));
    yakl::Array<T, 1, yakl::memDevice> n_total("new_pops", GammaT.extent(0));
    yakl::Array<T*, 1, yakl::memDevice> new_pops_ptrs("new_pops_ptrs", GammaT.extent(0));
    yakl::Array<i32, 1, yakl::memDevice> i_elim("i_elim", GammaT.extent(0));
    yakl::Array<i32, 2, yakl::memDevice> ipivs("ipivs", new_pops.extent(0), new_pops.extent(1));
    yakl::Array<i32*, 1, yakl::memDevice> ipiv_ptrs("ipiv_ptrs", new_pops.extent(0));
    yakl::Array<i32, 1, yakl::memDevice> info("info", new_pops.extent(0));
    parallel_for(
        "Max Pops",
        SimpleBounds<1>(pops.extent(1)),
        YAKL_LAMBDA (int64_t k) {
            fp_t n_max = pops(0, k);
            i_elim(k) = 0;
            n_total(k) = FP(0.0);
            for (int i = 0; i < pops.extent(0); ++i) {
                fp_t n = pops(i, k);
                n_total(k) += n;
                if (n > n_max) {
                    i_elim(k) = i;
                }
            }
        }
    );
    yakl::fence();

    parallel_for(
        "Transpose Gamma",
        SimpleBounds<3>(Gamma.extent(2), Gamma.extent(1), Gamma.extent(0)),
        YAKL_LAMBDA (int k, int i, int j) {
            // if (i_elim(k) == i) {
            //     GammaT(k, j, i) = FP(1.0);
            // } else {
                GammaT(k, j, i) = Gamma(i, j, k);
            // }
        }
    );
    yakl::fence();

    parallel_for(
        "Gamma fixup",
        SimpleBounds<1>(GammaT.extent(0)),
        YAKL_LAMBDA (i64 k) {
            for (int i = 0; i < GammaT.extent(1); ++i) {
                T diag = FP(0.0);
                GammaT(k, i, i) = FP(0.0);
                for (int j = 0; j < GammaT.extent(2); ++j) {
                    diag += GammaT(k, i, j);
                }
                GammaT(k, i, i) = -diag;
            }
        }
    );
    parallel_for(
        "Transpose Pops",
        SimpleBounds<2>(pops.extent(0), pops.extent(1)),
        YAKL_LAMBDA (int i, int64_t k) {
            if (i_elim(k) == i) {
                // T n_total = FP(0.0);
                // for (int ii = 0; ii < pops.extent(0); ++ii) {
                //     n_total += pops(ii, k);
                // }
                // new_pops(k, i) = n_total;
                // new_pops(k, i) = FP(1.0);
                new_pops(k, i) = n_total(k);
            } else {
                new_pops(k, i) = FP(0.0);
            }
        }
    );
    parallel_for(
        "Setup pointers",
        SimpleBounds<1>(GammaT_ptrs.extent(0)),
        YAKL_LAMBDA (int64_t k) {
            GammaT_ptrs(k) = &GammaT(k, 0, 0);
            new_pops_ptrs(k) = &new_pops(k, 0);
            ipiv_ptrs(k) = &ipivs(k, 0);
        }
    );
    yakl::fence();

    auto GammaT_host = GammaT.createHostCopy();
    auto pops_host = pops.createHostCopy();
    // const int print_idx = 452 * state->atmos.temperature.extent(1) + 519;
    const int print_idx = std::min(
        int(128 * state->atmos.temperature.extent(1) + 128),
        int(state->atmos.temperature.extent(0) * state->atmos.temperature.extent(1) - 1)
    );
    for (int i = 0; i < GammaT_host.extent(2); ++i) {
        for (int j = 0; j < GammaT_host.extent(1); ++j) {
            fmt::print("{:e}, ", GammaT_host(print_idx, j, i));
        }
        fmt::print("\n");
    }
    fmt::print("pops pre ");
    for (int i = 0; i < pops_host.extent(0); ++i) {
        fmt::print("{:e}, ", pops_host(i, print_idx));
    }
    fmt::print("\n");


    parallel_for(
        "Conservation eqn",
        SimpleBounds<3>(GammaT.extent(0), GammaT.extent(1), GammaT.extent(2)),
        YAKL_LAMBDA (i64 k, int i, int j) {
            if (i_elim(k) == i) {
                GammaT(k, j, i) = FP(1.0);
            }
        }
    );

    yakl::fence();

    static_assert(
        std::is_same_v<T, f32> || std::is_same_v<T, f64>,
        "What type are you asking the poor stat_eq function to use internally?"
    );
    if constexpr (std::is_same_v<T, f32>) {
        magma_sgesv_batched_small(
            GammaT.extent(1),
            1,
            GammaT_ptrs.get_data(),
            GammaT.extent(1),
            ipiv_ptrs.get_data(),
            new_pops_ptrs.get_data(),
            new_pops.extent(1),
            info.get_data(),
            GammaT.extent(0),
            state->magma_queue
        );
    } else if constexpr (std::is_same_v<T, f64>) {
        magma_dgesv_batched_small(
            GammaT.extent(1),
            1,
            GammaT_ptrs.get_data(),
            GammaT.extent(1),
            ipiv_ptrs.get_data(),
            new_pops_ptrs.get_data(),
            new_pops.extent(1),
            info.get_data(),
            GammaT.extent(0),
            state->magma_queue
        );
    }

    magma_queue_sync(state->magma_queue);
    yakl::fence();
    parallel_for(
        "info check",
        SimpleBounds<1>(info.extent(0)),
        YAKL_LAMBDA (int k) {
            if (info(k) != 0) {
                printf("%d: %d\n", k, info(k));
            }
        }
    );

    // parallel_for(
    //     "Normalise new pops vec",
    //     SimpleBounds<1>(new_pops.extent(0)),
    //     YAKL_LAMBDA (i64 k) {
    //         T sum = FP(0.0);
    //         for (int i = 0; i < new_pops.extent(1); ++i) {
    //             sum += new_pops(k, i);
    //         }
    //         for (int i = 0; i < new_pops.extent(1); ++i) {
    //             new_pops(k, i) /= sum;
    //         }
    //     }
    // );

    yakl::fence();

    Fp2d max_rel_change("max rel change", new_pops.extent(0), new_pops.extent(1));
    const auto& flat_temp = state->atmos.temperature.collapse();
    parallel_for(
        "Compute max change",
        SimpleBounds<2>(new_pops.extent(0), new_pops.extent(1)),
        YAKL_LAMBDA (int64_t k, int i) {
            fp_t change = FP(0.0);
            // if (flat_temp(k) < FP(5.0e4)) {
                // change = std::abs(FP(1.0) - pops(i, k) / (new_pops(k, i) * n_total(k)));
                change = std::abs(FP(1.0) - pops(i, k) / new_pops(k, i));
            // }
            max_rel_change(k, i) = change;
        }
    );
    yakl::fence();
    parallel_for(
        "Copy & transpose pops",
        SimpleBounds<2>(pops.extent(0), pops.extent(1)),
        YAKL_LAMBDA (int i, int64_t k) {
            // pops(i, k) = new_pops(k, i) * n_total(k);
            pops(i, k) = new_pops(k, i);
        }
    );
    pops_host = pops.createHostCopy();
    fmt::print("pops post ");
    for (int i = 0; i < pops_host.extent(0); ++i) {
        fmt::print("{:e}, ", pops_host(i, print_idx));
    }
    fmt::print("\n");
    fp_t max_change = yakl::intrinsics::maxval(max_rel_change);
    int max_change_loc = yakl::intrinsics::maxloc(max_rel_change.collapse());
    auto temp_h = state->atmos.temperature.createHostCopy();

    yakl::fence();
    int max_change_level = max_change_loc % state->pops.extent(0);
    max_change_loc /= state->pops.extent(0);
    int max_change_x = max_change_loc % state->pops.extent(2);
    max_change_loc /= state->pops.extent(2);
    int max_change_z = max_change_loc;
    fmt::println(
        "Max Change: {} (@ l={}, ({}, {})) [T={}]",
        max_change,
        max_change_level,
        max_change_z,
        max_change_x,
        temp_h(max_change_z, max_change_x)
    );
    return max_change;
}

#else
#endif