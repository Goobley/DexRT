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
            new_line.type = LineProfileType.Voigt;
        } else if (type == "PRD-Voigt") {
            new_line.type == LineProfileType.PrdVoigt;
        } else {
            throw std::runtime_error(fmt::format("Unexpected line type: {}\n", type).c_str());
        }

        std::string upper = l["transition"][0].as<std::string>();
        std::string lower = l["transition"][1].as<std::string>();
        new_line.j = level_idx_mapping.at(upper);
        new_line.i = level_idx_mapping.at(lower);
        new_line.f = l["f"].as<T>();
        new_line.Aji = l["Aji"].as<T>();
        new_line.Bji = l["Bji"].as<T>();
        new_line.Bji_wavelength = l["Bji_wavelength"].as<T>();
        new_line.Bij = l["Bij"].as<T>();
        new_line.Bij_wavelength = l["Bij_wavelength"].as<T>();
        new_line.lambda0 = l["lambda0"].as<T>();

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
                throw std::runtime_error(fmt::format("Got unexpected broadening type {}\n", type).c_str());
            }
        }
    }

    return model;
}


#else
#endif