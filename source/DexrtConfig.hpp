#if !defined(DEXRT_DEXRT_CONFIG_HPP)
#define DEXRT_DEXRT_CONFIG_HPP
#include <string>
#include <vector>
#include "Config.hpp"
#include "Types.hpp"
#include "BoundaryType.hpp"
#include <yaml-cpp/yaml.h>

struct DexrtConfig {
    DexrtMode mode = DexrtMode::NonLte;
    fp_t mem_pool_initial_gb = FP(2.0);
    fp_t mem_pool_grow_gb = FP(1.0);
    std::string atmos_path;
    std::string output_path;
    std::vector<std::string> atom_paths;
    std::vector<AtomicTreatment> atom_modes;
    BoundaryType boundary = BoundaryType::Zero;
    bool sparse_calculation = false;
    fp_t threshold_temperature = FP(0.0);
    int max_iter = 200;
    fp_t pop_tol = FP(1e-3);
    bool conserve_charge = false;
    bool conserve_pressure = false;
    int snapshot_frequency = 0;
    int initial_lambda_iterations = 2;
};

void parse_extra_givenfs(DexrtConfig* cfg, const YAML::Node& file) {
    DexrtConfig& config = *cfg;
    config.boundary = BoundaryType::Zero;
    config.sparse_calculation = false;
    config.threshold_temperature = FP(0.0);
    config.max_iter = 0;
    config.pop_tol = FP(1.0);
    config.conserve_charge = false;
    config.conserve_pressure = false;
}

void parse_extra_lte(DexrtConfig* cfg, const YAML::Node& file) {
    DexrtConfig& config = *cfg;

    if (file["sparse_calculation"]) {
        config.sparse_calculation = file["sparse_calculation"].as<bool>();
    }
    if (file["threshold_temperature"]) {
        config.threshold_temperature = file["threshold_temperature"].as<fp_t>();
    }
    if (!file["atoms"] || !file["atoms"].IsMap()) {
        throw std::runtime_error("No atoms map found in config file.");
    } else {
        for (const auto& v : file["atoms"]) {
            std::string name = v.first.as<std::string>();
            auto table = v.second;
            if (!table["path"]) {
                throw std::runtime_error(fmt::format("No path found for atom with key {}", name));
            } else {
                config.atom_paths.push_back(table["path"].as<std::string>());
            }

            if (table["treatment"]) {
                std::string mode = table["treatment"].as<std::string>();
                if (mode == "Detailed") {
                    config.atom_modes.push_back(AtomicTreatment::Detailed);
                } else if (mode == "Golding") {
                    config.atom_modes.push_back(AtomicTreatment::Golding);
                } else if (mode == "Active") {
                    config.atom_modes.push_back(AtomicTreatment::Active);
                } else {
                    throw std::runtime_error(fmt::format("Invalid atomic treatment for atom with key {}: {}", name, mode));
                }
            } else {
                config.atom_modes.push_back(AtomicTreatment::Active);
            }
        }
    }

    if (file["boundary_type"]) {
        std::string mode = file["boundary_type"].as<std::string>();
        if (mode == "Zero") {
            config.boundary = BoundaryType::Zero;
        } else if (mode == "Promweaver") {
            config.boundary = BoundaryType::Promweaver;
        } else {
            throw std::runtime_error(fmt::format("Unexpected boundary condition type: {}", mode));
        }
    }
}

void parse_extra_nonlte(DexrtConfig* cfg, const YAML::Node& file) {
    // NOTE(cmo): Everything needed in Lte is also needed in NonLte.
    parse_extra_lte(cfg, file);
    DexrtConfig& config = *cfg;

    if (file["max_iter"]) {
        config.max_iter = file["max_iter"].as<int>();
    }
    if (file["pop_tol"]) {
        config.pop_tol = file["pop_tol"].as<fp_t>();
    }
    if (file["conserve_charge"]) {
        config.conserve_charge = file["conserve_charge"].as<bool>();
    }
    if (file["conserve_pressure"]) {
        config.conserve_pressure = file["conserve_pressure"].as<bool>();
    }
    if (file["snapshot_frequency"]) {
        config.snapshot_frequency = file["snapshot_frequency"].as<int>();
    }
    if (file["initial_lambda_iterations"]) {
        config.initial_lambda_iterations = file["initial_lambda_iterations"].as<int>();
    }
}

DexrtConfig parse_dexrt_config(const std::string& path) {
    DexrtConfig config;
    config.atmos_path = "dexrt_atmos.nc";
    config.output_path = "dexrt.nc";

    YAML::Node file = YAML::LoadFile(path);
    if (file["system"]) {
        auto system = file["system"];
        if (system["mem_pool_initial_gb"]) {
            config.mem_pool_initial_gb = system["mem_pool_initial_gb"].as<fp_t>();
        }
        if (system["mem_pool_grow_gb"]) {
            config.mem_pool_grow_gb = system["mem_pool_grow_gb"].as<fp_t>();
        }
    }
    if (file["atmos_path"]) {
        config.atmos_path = file["atmos_path"].as<std::string>();
    }
    if (file["output_path"]) {
        config.output_path = file["output_path"].as<std::string>();
    }
    if (file["mode"]) {
        std::string mode = file["mode"].as<std::string>();
        if (mode == "Lte") {
            config.mode = DexrtMode::Lte;
        } else if (mode == "NonLte") {
            config.mode = DexrtMode::NonLte;
        } else if (mode == "GivenFs") {
            config.mode = DexrtMode::GivenFs;
        } else {
            throw std::runtime_error(fmt::format("Unexpected program mode: {}", mode));
        }
    }

    switch (config.mode) {
        case DexrtMode::Lte: {
            parse_extra_lte(&config, file);
        } break;
        case DexrtMode::NonLte: {
            parse_extra_nonlte(&config, file);
        } break;
        case DexrtMode::GivenFs: {
            parse_extra_givenfs(&config, file);
        } break;
    }

    return config;
}

#else
#endif