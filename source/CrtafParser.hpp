#if !defined(DEXRT_CRTAF_PARSER_HPP)
#define DEXRT_CRTAF_PARSER_HPP

#include <cstdio>
#include <string>
#include <yaml-cpp/yaml.h>
#include <vector>
#include "Types.hpp"

inline YAML::Node parse_crtaf_model(const std::string& path) {
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

    assert(file["levels"].IsMap());
    int num_levels = file["levels"].size();
    model.levels.reserve(num_levels);
    for (const auto l : file["levels"]) {
        AtomicLevel new_level;
        new_level.energy = l["energy_eV"].as<fp_t>();
        new_level.g = l["g"].as<int>();
        new_level.stage = l["stage"].as<int>();
        if (l["label"]) {
            new_level.label = l["label"].as<std::string>();
        }
        // TODO(cmo): Need to get the key here.
    }

    



    return file;
}


#else
#endif