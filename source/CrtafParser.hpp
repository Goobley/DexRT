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

inline ModelAtom parse_crtaf_model(const std::string& path) {
    YAML::Node file = YAML::LoadFile(path);

    if (!file["crtaf_meta"]) {
        throw std::runtime_error(fmt::format("Did not find `crtaf_meta` mapping in {}, is this a model atom?", path));
    }

    if (file["crtaf_meta"]["level"].as<std::string>() != "simplified") {
        throw std::runtime_error("Can only parse \"simplified\" models");
    }

    ModelAtom model;
    auto elem = file["element"];
    model.element.symbol = elem["symbol"].as<std::string>();
    model.element.mass = elem["atomic_mass"].as<fp_t>();
    model.element.abundance = elem["abundance"].as<fp_t>();
    model.element.Z = elem["Z"].as<int>();

    // NOTE(cmo): Parse levels
    assert(file["levels"].IsMap());
    int num_levels = file["levels"].size();
    model.levels.reserve(num_levels);
    for (const auto& l : file["levels"]) {
        YAML::Node key = l.first;
        YAML::Node value = l.second;

        AtomicLevel new_level;
        new_level.energy = value["energy_eV"]["value"].as<fp_t>();
        new_level.g = value["g"].as<int>();
        new_level.stage = value["stage"].as<int>();
        if (value["label"]) {
            new_level.label = value["label"].as<std::string>();
        }
        new_level.key = key.as<std::string>();
        model.levels.emplace_back(new_level);
    }
    // NOTE(cmo): Sort levels by increasing energy
    std::sort(model.levels.begin(), model.levels.end(), [](AtomicLevel a, AtomicLevel b) {
        return a.energy < b.energy;
    });
    // NOTE(cmo): Mapping from key to level idx
    std::unordered_map<std::string, int> level_idx_mapping;
    for (int i = 0; i < model.levels.size(); ++i) {
        const AtomicLevel& level = model.levels[i];
        level_idx_mapping[level.key] = i;
    }

    return model;
}


#else
#endif