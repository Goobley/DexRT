#if !defined(DEXRT_CRTAF_PARSER_HPP)
#define DEXRT_CRTAF_PARSER_HPP

#include <cstdio>
#include <string>
#include <yaml-cpp/yaml.h>
#include <fmt/core.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "Types.hpp"

template <typename T=fp_t>
inline ModelAtom<T> parse_crtaf_model(const std::string& path) {
    YAML::Node file = YAML::LoadFile(path);

    if (!file["crtaf_meta"]) {
        throw std::runtime_error(fmt::format("Did not find `crtaf_meta` mapping in {}, is this a model atom?", path));
    }

    if (file["crtaf_meta"]["level"].as<std::string>() != "simplified") {
        throw std::runtime_error("Can only parse \"simplified\" models");
    }

    ModelAtom<T> model;
    auto elem = file["element"];
    model.element.symbol = elem["symbol"].as<std::string>();
    model.element.mass = elem["atomic_mass"].as<T>();
    model.element.abundance = elem["abundance"].as<T>();
    model.element.Z = elem["Z"].as<int>();

    // NOTE(cmo): Parse levels
    assert(file["levels"].IsMap());
    int num_levels = file["levels"].size();
    model.levels.reserve(num_levels);
    for (const auto& l : file["levels"]) {
        YAML::Node key = l.first;
        YAML::Node value = l.second;

        AtomicLevel<T> new_level;
        new_level.energy = value["energy_eV"]["value"].as<T>();
        new_level.g = value["g"].as<int>();
        new_level.stage = value["stage"].as<int>();
        if (value["label"]) {
            new_level.label = value["label"].as<std::string>();
        }
        new_level.key = key.as<std::string>();
        model.levels.emplace_back(new_level);
    }
    // NOTE(cmo): Sort levels by increasing energy
    std::sort(model.levels.begin(), model.levels.end(), [](const auto& a, const auto& b) {
        return a.energy < b.energy;
    });
    // NOTE(cmo): Mapping from key to level idx
    std::unordered_map<std::string, int> level_idx_mapping;
    for (int i = 0; i < model.levels.size(); ++i) {
        const auto& level = model.levels[i];
        level_idx_mapping[level.key] = i;
    }

    for (const auto& l : file["lines"]) {
        AtomicLine<T> new_line;
        std::string type = l["type"].as<std::string>();
        if (type == "Voigt") {
            new_line.type = LineProfileType::Voigt;
        } else if (type == "PRD-Voigt") {
            new_line.type = LineProfileType::PrdVoigt;
        } else {
            throw std::runtime_error(fmt::format("Unexpected line type: {}\n", type));
        }

        std::string upper = l["transition"][0].as<std::string>();
        std::string lower = l["transition"][1].as<std::string>();
        new_line.j = level_idx_mapping.at(upper);
        new_line.i = level_idx_mapping.at(lower);
        new_line.f = l["f_value"].as<T>();
        new_line.Aji = l["Aji"]["value"].as<T>();
        new_line.Bji = l["Bji"]["value"].as<T>();
        new_line.Bji_wavelength = l["Bji_wavelength"]["value"].as<T>();
        new_line.Bij = l["Bij"]["value"].as<T>();
        new_line.Bij_wavelength = l["Bij_wavelength"]["value"].as<T>();
        new_line.lambda0 = l["lambda0"]["value"].as<T>();

        bool got_natural = false;
        for (const auto& b : l["broadening"]) {
            std::string type = b["type"].as<std::string>();
            if (type == "Natural") {
                if (got_natural) {
                    throw std::runtime_error("Got more than one source of natural broadening on a line!");
                }

                new_line.g_natural = b["value"]["value"].as<T>();
                got_natural = true;
            } else if (type == "Scaled_Exponents") {
                ScaledExponentsBroadening<T> new_b;
                new_b.scaling = b["scaling"].as<T>();
                new_b.temperature_exponent = b["temperature_exponent"].as<T>();
                new_b.hydrogen_exponent = b["hydrogen_exponent"].as<T>();
                new_b.electron_exponent = b["electron_exponent"].as<T>();
                new_line.broadening.emplace_back(new_b);
            } else {
                throw std::runtime_error(fmt::format("Got unexpected broadening type {}\n", type));
            }
        }
        
        const auto q = l["wavelength_grid"];
        std::string grid_type = q["type"].as<std::string>();
        if (grid_type == "Linear") {
            T half_width = q["delta_lambda"]["value"].as<T>();
            int n_lambda = q["n_lambda"].as<int>();
            T step_size = (half_width + half_width) / T(n_lambda - 1);
            new_line.wavelength.reserve(n_lambda);
            const T start = new_line.lambda0 - half_width;
            for (int i = 0; i < n_lambda; ++i) {
                new_line.wavelength.push_back(start + i * step_size);
            }
        } else if (grid_type == "Tabulated") {
            int n_lambda = q["wavelengths"]["value"].size();
            new_line.wavelength.reserve(n_lambda);
            for (const auto& entry: q["wavelengths"]["value"]) {
                new_line.wavelength.push_back(new_line.lambda0 + entry.as<T>());
            }
        } else {
            throw std::runtime_error(fmt::format("Got unexpected wavelength grid type {}\n", grid_type));
        }

        model.lines.emplace_back(new_line);
    }

    for (const auto& c : file["continua"]) {
        std::string upper = c["transition"][0].as<std::string>();
        std::string lower = c["transition"][1].as<std::string>();
        int j = level_idx_mapping.at(upper);
        int i = level_idx_mapping.at(lower);
        T lambda_edge = model.transition_wavelength(j, i);
        std::string type = c["type"].as<std::string>();

        if (type != "Tabulated") {
            throw std::runtime_error(fmt::format("Can only parse Tabulated continua, got {}\n", type));
        } 

        int n_lambda = c["value"].size();
        AtomicContinuum<T> new_cont;
        new_cont.j = j;
        new_cont.i = i;
        new_cont.wavelength.reserve(n_lambda);
        new_cont.sigma.reserve(n_lambda);
        for (int i = 0; i < n_lambda; ++i) {
            new_cont.wavelength.push_back(c["value"][i][0].as<T>());
            new_cont.sigma.push_back(c["value"][i][1].as<T>());
        }
        model.continua.emplace_back(new_cont);
    }

    for (const auto& c : file["collisions"]) {
        std::string upper = c["transition"][0].as<std::string>();
        std::string lower = c["transition"][1].as<std::string>();
        int j = level_idx_mapping.at(upper);
        int i = level_idx_mapping.at(lower);

        for (const auto& coll : c["data"]) {
            InterpCollRate<T> new_coll;
            new_coll.j = j;
            new_coll.i = i;

            std::string type = coll["type"].as<std::string>();
            CollRateType coll_type;
            if (type == "Omega") {
                coll_type = CollRateType::Omega;
            } else if (type == "CI") {
                coll_type = CollRateType::CI;
            } else if (type == "CE") {
                coll_type = CollRateType::CE;
            } else if (type == "CP") {
                coll_type = CollRateType::CP;
            } else if (type == "CH") {
                coll_type = CollRateType::CH;
            } else if (type == "ChargeExcH") {
                coll_type = CollRateType::ChargeExcH;
            } else if (type == "ChargeExcP") {
                coll_type = CollRateType::ChargeExcP;
            } else {
                throw std::runtime_error(fmt::format("Unexpected collisional rate type {}\n", type));
            }
            new_coll.type = coll_type;
            int n_temperature = coll["temperature"]["value"].size();
            new_coll.temperature.reserve(n_temperature);
            new_coll.data.reserve(n_temperature);

            for (const auto& t : coll["temperature"]["value"]) {
                new_coll.temperature.push_back(t.as<T>());
            }
            for (const auto& d : coll["data"]["value"]) {
                new_coll.data.push_back(d.as<T>());
            }

            model.coll_rates.emplace_back(new_coll);
        }
    }

    return model;
}


#else
#endif