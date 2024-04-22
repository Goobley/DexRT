#include "Types.hpp"
#include <argparse/argparse.hpp>
#include <string>
#include <vector>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <YAKL_netcdf.h>
#include "Utils.hpp"

struct RayConfig {
    std::string atmos_path;
    std::string atom_path;
    std::string dex_output_path;
    std::string ray_output_path;
    std::vector<fp_t> muz;
    std::vector<fp_t> mux;
    std::vector<fp_t> wavelength;
    bool rotate_aabb = true;
};

struct RaySet {
    Fp2dHost start_coord;
    vec3 mu;
};

RayConfig parse_config(const std::string& path) {
    RayConfig config;
    config.atmos_path = "atmos.nc";
    config.atom_path = "../tests/test_CaII.yaml";
    config.dex_output_path = "output.nc";
    config.ray_output_path = "ray_output.nc";

    YAML::Node file = YAML::LoadFile(path);
    if (file["atmos_path"]) {
        config.atmos_path = file["atmos_path"].as<std::string>();
    }
    if (file["atom_path"]) {
        config.atom_path = file["atom_path"].as<std::string>();
    }
    if (file["dex_output_path"]) {
        config.dex_output_path = file["dex_output_path"].as<std::string>();
    }
    if (file["ray_output_path"]) {
        config.ray_output_path = file["ray_output_path"].as<std::string>();
    }

    if (file["rotate_aabb"]) {
        config.rotate_aabb = file["rotate_aabb"].as<bool>();
    }

    auto require_key = [&] (const std::string& key) {
        if (!file[key]) {
            throw std::runtime_error(fmt::format("{} key must be present in config file.", key));
        }
    };
    require_key("muz");
    require_key("mux");
    require_key("num_rays");

    auto parse_one_or_more_float_to_vector = [&] (const std::string& key) {
        std::vector<fp_t> result;
        if (!file["key"]) {
            return result;
        }

        if (file[key].IsSequence()) {
            result.reserve(file[key].size());
            for (const auto& v : file[key]) {
                result.push_back(v.as<fp_t>());
            }
        } else {
            result.push_back(file[key].as<fp_t>());
        }
        return result;
    };
    config.muz = parse_one_or_more_float_to_vector("muz");
    config.mux = parse_one_or_more_float_to_vector("mux");
    config.wavelength = parse_one_or_more_float_to_vector("wavelength");
    if ((config.muz.size() != config.mux.size()) || config.muz.size() == 0) {
        throw std::runtime_error("muz and mux must be provided and have the same number of entries.");
    }

    if (config.wavelength.size() == 0) {
        yakl::Array<f32, 1, yakl::memHost> wavelengths;
        yakl::SimpleNetCDF nc;
        nc.open(path, yakl::NETCDF_MODE_READ);
        nc.read(wavelengths, "wavelength");
        config.wavelength.reserve(wavelengths.extent(0));
        for (int i = 0; i < wavelengths.extent(0); ++i) {
            config.wavelength.push_back(wavelengths(i));
        }
    }
    return config;
}

RaySet compute_ray_set(const RayConfig& cfg, const Atmosphere& atmos, int mu_idx) {
    RaySet result;
    if (cfg.rotate_aabb) {

    } else {
        Fp2dHost ray_pos("ray_starts", atmos.temperature.extent(1), 2);
        fp_t z_max = fp_t(atmos.temperature.extent(0));
        for (int i = 0; i < atmos.temperature.extent(1); ++i) {
            ray_pos(i, 0) = (i + FP(0.5));
            ray_pos(i, 1) = z_max;
        }
        vec3 mu;
        mu(0) = cfg.mux[mu_idx];
        mu(2) = cfg.muz[mu_idx];
        mu(1) = std::sqrt(FP(1.0) - square(mu(0)) - square(mu(2)));

        result.start_coord = ray_pos;
        result.mu = mu;
    }
    return result;
}


int main(int argc, const char* argv[]) {
    argparse::ArgumentParser program("DexRT Ray");
    program.add_argument("--config")
        .default_value(std::string("dexrt_ray.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_epilog("Single-pass formal solver for post-processing Dex models.");

    program.parse_args(argc, argv);

    auto config = parse_config(program.get<std::string>("--config"));


    return 0;
}