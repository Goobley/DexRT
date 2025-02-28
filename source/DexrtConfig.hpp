#if !defined(DEXRT_DEXRT_CONFIG_HPP)
#define DEXRT_DEXRT_CONFIG_HPP
#include <string>
#include <vector>
#include "Config.hpp"
#include "Types.hpp"
#include "BoundaryType.hpp"
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

struct DexrtOutputConfig {
    bool sparse = false;
    bool wavelength = true;
    bool J = true;
    bool pops = true;
    bool lte_pops = true;
    bool ne = true;
    bool nh_tot = true;
    bool max_mip_level = true;
    bool alo = false;
    bool active = true;
    std::vector<int> cascades;
};

struct DexrtMipConfig {
    fp_t opacity_threshold = FP(0.25);
    fp_t log_chi_mip_variance = FP(1.0);
    fp_t log_eta_mip_variance = FP(1.0);
    std::vector<int> mip_levels;
};

struct DexrtNgConfig {
    bool enable = true;
    fp_t threshold = FP(5e-2);
    fp_t lower_threshold = FP(2e-4);
};

struct DexrtConfig {
    DexrtMode mode = DexrtMode::NonLte;
    fp_t mem_pool_gb = FP(4.0);
    std::string own_path;
    std::string atmos_path;
    std::string output_path;
    std::string initial_pops_path;
    DexrtOutputConfig output;
    bool store_J_on_cpu = true;
    std::vector<std::string> atom_paths;
    std::vector<AtomicTreatment> atom_modes;
    BoundaryType boundary = BoundaryType::Zero;
    bool sparse_calculation = false;
    bool final_dense_fs = true;
    fp_t threshold_temperature = FP(0.0);
    int max_iter = 200;
    fp_t pop_tol = FP(1e-3);
    bool conserve_charge = false;
    bool conserve_pressure = false;
    int snapshot_frequency = 0;
    int initial_lambda_iterations = 2;
    int max_cascade = 5;
    DexrtMipConfig mip_config;
    DexrtNgConfig ng;
};

inline void parse_extra_givenfs(DexrtConfig* cfg, const YAML::Node& file) {
    DexrtConfig& config = *cfg;
    config.boundary = BoundaryType::Zero;
    config.sparse_calculation = false;
    config.threshold_temperature = FP(0.0);
    config.max_iter = 0;
    config.pop_tol = FP(1.0);
    config.conserve_charge = false;
    config.conserve_pressure = false;
}

inline void parse_extra_lte(DexrtConfig* cfg, const YAML::Node& file) {
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

inline void parse_extra_nonlte(DexrtConfig* cfg, const YAML::Node& file) {
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
    if (file["final_dense_fs"]) {
        config.final_dense_fs = file["final_dense_fs"].as<bool>();
    }
    if (file["ng_config"]) {
        auto ng_config = file["ng_config"];
        if (ng_config["enable"]) {
            config.ng.enable = ng_config["enable"].as<bool>();
        }
        if (ng_config["threshold"]) {
            config.ng.threshold = ng_config["threshold"].as<fp_t>();
        }
        if (ng_config["lower_threshold"]) {
            config.ng.lower_threshold = ng_config["lower_threshold"].as<fp_t>();
        }
    }
}

inline void parse_and_update_dexrt_output_config(DexrtConfig* cfg, const YAML::Node& file) {
    DexrtConfig& config = *cfg;
    DexrtOutputConfig out;

    auto output = file["output"];
    if (output["sparse"]) {
        out.sparse = output["sparse"].as<bool>();
    }
    if (output["wavelength"]) {
        out.wavelength = output["wavelength"].as<bool>();
    }
    if (output["J"]) {
        out.J = output["J"].as<bool>();
    }
    if (output["pops"]) {
        out.pops = output["pops"].as<bool>();
    }
    if (output["lte_pops"]) {
        out.lte_pops = output["lte_pops"].as<bool>();
    }
    if (output["ne"]) {
        out.ne = output["ne"].as<bool>();
    }
    if (output["nh_tot"]) {
        out.nh_tot = output["nh_tot"].as<bool>();
    }
    if (output["alo"]) {
        out.alo = output["alo"].as<bool>();
    }
    if (output["cascades"]) {
        for (const auto& c : output["cascades"]) {
            out.cascades.push_back(c.as<int>());
        }
    }

    // NOTE(cmo): Setup applicable based on mode.
    if (config.mode == DexrtMode::GivenFs) {
        out.ne = false;
        out.nh_tot = false;
        out.pops = false;
        out.lte_pops = false;
        out.active = false;
        out.alo = false;
    } else if (config.mode == DexrtMode::Lte) {
        out.ne = false;
        out.nh_tot = false;
        out.lte_pops = false;
        out.alo = false;
        out.active = true;
    } else if (config.mode == DexrtMode::NonLte) {
        out.active = true;
        if (!(config.conserve_charge || config.conserve_pressure)) {
            out.ne = false;
            out.nh_tot = false;
        }
    }
    if (out.cascades.size() > 0) {
        if constexpr (DIR_BY_DIR) {
            fmt::println(stderr, "Cascade output requested, but DIR_BY_DIR is enabled, only the entries corresponding to the final direction of C0 will be output.");
        }
        for (int casc_to_output : out.cascades) {
            if (casc_to_output > config.max_cascade) {
                throw std::runtime_error(fmt::format("Output of cascade {} requested, greater than max cascade {}.", casc_to_output, config.max_cascade));
            }
            if constexpr (PINGPONG_BUFFERS) {
                if (casc_to_output > 1) {
                    throw std::runtime_error(fmt::format("Output of cascade {} requested, which is > 1, and PINGPONG_BUFFERS is enabled, overwriting the other cascades to save memory", casc_to_output));
                }
            }
        }
    }

    config.output = out;
}

inline void parse_mip_config(DexrtConfig* cfg, const YAML::Node& file) {
    DexrtConfig& config(*cfg);
    config.mip_config.mip_levels.resize(config.max_cascade + 1);

    // NOTE(cmo): Parse the mip level: extend or truncate sequence as necessary
    if (!file["mip_config"] || !file["mip_config"]["mip_levels"]) {
        for (int i = 0; i <= config.max_cascade; ++i) {
            config.mip_config.mip_levels[i] = 0;
        }
    } else {
        auto mip_level = file["mip_config"]["mip_levels"];
        if (mip_level.IsSequence()) {
            int len = std::min(i32(config.max_cascade + 1), i32(mip_level.size()));
            for (int i = 0; i < len; ++i) {
                config.mip_config.mip_levels[i] = mip_level[i].as<i32>();
            }
            for (int i = len; i <= config.max_cascade; ++i) {
                config.mip_config.mip_levels[i] = config.mip_config.mip_levels[len-1];
            }
        } else {
            i32 level = mip_level.as<i32>();
            for (int i = 0; i <= config.max_cascade; ++i) {
                config.mip_config.mip_levels[i] = level;
            }
        }
    }

    if (file["mip_config"]) {
        auto& mip_config = file["mip_config"];
        if (mip_config["log_chi_mip_variance"]) {
            config.mip_config.log_chi_mip_variance = mip_config["log_chi_mip_variance"].as<fp_t>();
        }
        if (mip_config["log_eta_mip_variance"]) {
            config.mip_config.log_eta_mip_variance = mip_config["log_eta_mip_variance"].as<fp_t>();
        }
        if (mip_config["opacity_threshold"]) {
            config.mip_config.opacity_threshold = mip_config["opacity_threshold"].as<fp_t>();
        }
    }
}


inline DexrtConfig parse_dexrt_config(const std::string& path) {
    DexrtConfig config{};
    config.own_path = path;
    config.atmos_path = "dexrt_atmos.nc";
    config.output_path = "dexrt.nc";

    YAML::Node file = YAML::LoadFile(path);
    if (file["system"]) {
        auto system = file["system"];
        if (system["mem_pool_gb"]) {
            config.mem_pool_gb = system["mem_pool_gb"].as<fp_t>();
        } else if (system["mem_pool_initial_gb"]) {
            fmt::println("Found deprecated \"mem_pool_initial_gb\", using that value. The pool no longer grows and should be set with key \"mem_pool_gb\".");
            config.mem_pool_gb = system["mem_pool_initial_gb"].as<fp_t>();
        }
    }
    if (file["atmos_path"]) {
        config.atmos_path = file["atmos_path"].as<std::string>();
    }
    if (file["output_path"]) {
        config.output_path = file["output_path"].as<std::string>();
    }
    if (file["initial_pops_path"]) {
        config.initial_pops_path = file["initial_pops_path"].as<std::string>();
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
    if (file["store_J_on_cpu"]) {
        config.store_J_on_cpu = file["store_J_on_cpu"].as<bool>();
    }

    if (file["max_cascade"]) {
        config.max_cascade = file["max_cascade"].as<int>();
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

    if (file["output"]) {
        parse_and_update_dexrt_output_config(&config, file);
    }

    parse_mip_config(&config, file);

    return config;
}

#else
#endif