#include "Types.hpp"
#include <argparse/argparse.hpp>
#include <string>
#include <vector>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <YAKL_netcdf.h>
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include "CrtafParser.hpp"
#include "Populations.hpp"
#include "PromweaverBoundary.hpp"
#include "DexrtConfig.hpp"
#include "JasPP.hpp"
#include "MiscSparse.hpp"
#include "PostProcessingCore.hpp"
#include <sstream>
#include "tqdm.hpp"

int get_dexrt_dimensionality() {
    return 2;
}

struct RayConfig {
    fp_t mem_pool_gb = FP(2.0);
    std::string own_path;
    std::string dexrt_config_path;
    std::string ray_output_path;
    std::vector<fp_t> muz;
    std::vector<fp_t> mux;
    std::vector<fp_t> wavelength;
    bool rotate_aabb = true;
    bool output_cfn = false;
    bool output_eta_chi = false;
    DexrtConfig dexrt;
};

RayConfig parse_ray_config(const std::string& path) {
    RayConfig config;
    config.own_path = path;
    config.dexrt_config_path = "dexrt.yaml";
    config.ray_output_path = "ray_output.nc";

    YAML::Node file = YAML::LoadFile(path);
    if (file["dexrt_config_path"]) {
        config.dexrt_config_path = file["dexrt_config_path"].as<std::string>();
    }
    if (file["ray_output_path"]) {
        config.ray_output_path = file["ray_output_path"].as<std::string>();
    }

    if (file["rotate_aabb"]) {
        config.rotate_aabb = file["rotate_aabb"].as<bool>();
    }
    if (file["output_cfn"]) {
        config.output_cfn = file["output_cfn"].as<bool>();
    }
    if (file["output_eta_chi"]) {
        config.output_eta_chi = file["output_eta_chi"].as<bool>();
    }

    if (file["system"]) {
        auto system = file["system"];
        if (system["mem_pool_gb"]) {
            config.mem_pool_gb = system["mem_pool_gb"].as<fp_t>();
        } else if (system["mem_pool_initial_gb"]) {
            fmt::println("Found deprecated \"mem_pool_initial_gb\", using that value. The pool no longer grows and should be set with key \"mem_pool_gb\".");
            config.mem_pool_gb = system["mem_pool_initial_gb"].as<fp_t>();
        }
    }

    config.dexrt = load_and_parse_dexrt_config(config.dexrt_config_path);

    auto require_key = [&] (const std::string& key) {
        if (!file[key]) {
            throw std::runtime_error(fmt::format("{} key must be present in config file.", key));
        }
    };
    require_key("muz");
    require_key("mux");

    auto parse_one_or_more_float_to_vector = [&] (const std::string& key) {
        std::vector<fp_t> result;
        if (!file[key]) {
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
        throw std::runtime_error("muz and mux must be provided and have the same number of entries (non-zero).");
    }

    // NOTE(cmo): Can't do this before yakl init, but we need to know pool params. Defer this. It's a bit messy.
    // if (config.wavelength.size() == 0) {
    //     yakl::Array<f32, 1, yakl::memHost> wavelengths;
    //     yakl::SimpleNetCDF nc;
    //     nc.open(config.dexrt.output_path, yakl::NETCDF_MODE_READ);
    //     nc.read(wavelengths, "wavelength");
    //     config.wavelength.reserve(wavelengths.extent(0));
    //     for (int i = 0; i < wavelengths.extent(0); ++i) {
    //         config.wavelength.push_back(wavelengths(i));
    //     }
    //     nc.close();
    // }
    return config;
}

void load_wavelength_if_missing(RayConfig* cfg) {
    RayConfig& config = *cfg;
    if (config.wavelength.size() == 0) {
        yakl::Array<f32, 1, yakl::memHost> wavelengths;
        yakl::SimpleNetCDF nc;
        nc.open(config.dexrt.output_path, yakl::NETCDF_MODE_READ);
        nc.read(wavelengths, "wavelength");
        config.wavelength.reserve(wavelengths.extent(0));
        for (int i = 0; i < wavelengths.extent(0); ++i) {
            config.wavelength.push_back(wavelengths(i));
        }
        nc.close();
    }
}

bool dex_data_is_sparse(const yakl::SimpleNetCDF& nc) {
    int ncid = nc.file.ncid;
    size_t len = 0;
    auto check_error = [](int ierr) {
        if (ierr != NC_NOERR) {
            throw std::runtime_error(fmt::format("Error determining sparsity: {}", nc_strerror(ierr)));
        }
    };
    int ierr = nc_inq_att(ncid, NC_GLOBAL, "output_format", nullptr, &len);
    if (ierr == NC_ENOTATT) {
        // NOTE(cmo): No "output_format" attribute found, i.e. old file -> dense
        return false;
    } else if (ierr != NC_NOERR) {
        check_error(ierr);
    }
    std::string format(len, 'x');
    ierr = nc_get_att_text(ncid, NC_GLOBAL, "output_format", format.data());
    check_error(ierr);
    bool is_sparse = (format == "sparse");
    return is_sparse;
}

void load_dex_output(const DexrtConfig& config, DexRayState* state) {
    yakl::SimpleNetCDF nc;
    nc.open(config.output_path, yakl::NETCDF_MODE_READ);
    const bool is_sparse = dex_data_is_sparse(nc);

    if (is_sparse) {
        nc.read(state->pops, "pops");
    } else {
        // NOTE(cmo): Need to sparsify pops
        Fp3d temp_pops;
        nc.read(temp_pops, "pops");
        const i32 num_level = temp_pops.extent(0);
        state->pops = Fp2d("pops", num_level, state->atmos.temperature.extent(0));

        JasUnpack((*state), mr_block_map, pops);
        dex_parallel_for(
            mr_block_map.block_map.loop_bounds(),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                for (int i = 0; i < num_level; ++i) {
                    pops(i, ks) = temp_pops(i, coord.z, coord.x);
                }
            }
        );
        Kokkos::fence();
    }
}

void update_atmosphere(const DexrtConfig& config, DexRayState* state) {
    yakl::SimpleNetCDF nc;
    nc.open(config.output_path, yakl::NETCDF_MODE_READ);
    if (!(nc.varExists("ne") || nc.varExists("nh_tot"))) {
        return;
    }

    JasUnpack((*state), atmos);
    const bool is_sparse = dex_data_is_sparse(nc);
    if (is_sparse) {
        if (nc.varExists("ne")) {
            nc.read(atmos.ne, "ne");
        }
        if (nc.varExists("nh_tot")) {
            nc.read(atmos.nh_tot, "nh_tot");
        }
    } else {
        auto dehydrate_2d_arr = [state](Fp1d& dst, const Fp2d& src) {
            JasUnpack((*state), mr_block_map);

            dex_parallel_for(
                FlatLoop<2>(mr_block_map.block_map.loop_bounds()),
                KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                    IdxGen idx_gen(mr_block_map);
                    i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                    Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                    dst(ks) = src(coord.z, coord.x);
                }
            );
        };
        if (nc.varExists("ne")) {
            Fp2d temp_ne;
            nc.read(temp_ne, "ne");
            dehydrate_2d_arr(atmos.ne, temp_ne);
        }
        if (nc.varExists("nh_tot")) {
            Fp2d temp_nh_tot;
            nc.read(temp_nh_tot, "nh_tot");
            dehydrate_2d_arr(atmos.nh_tot, temp_nh_tot);
        }
        Kokkos::fence();
    }
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("DexRT Ray");
    program.add_argument("--config")
        .default_value(std::string("dexrt_ray.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_argument("--quiet")
        .default_value(false)
        .implicit_value(true)
        .help("Whether to print progress");
    program.add_epilog("Single-pass formal solver for post-processing Dex models.");

    program.parse_known_args(argc, argv);

    RayConfig config = parse_ray_config(program.get<std::string>("--config"));
    bool quiet = program.get<bool>("--quiet");
    Kokkos::initialize(argc, argv);
    yakl::init(
        yakl::InitConfig()
            .set_pool_size_mb(config.mem_pool_gb * 1024)
    );
    {
        load_wavelength_if_missing(&config);
        if (config.dexrt.mode == DexrtMode::GivenFs) {
            throw std::runtime_error(fmt::format("Models run in GivenFs mode not supported by {}", argv[0]));
        }
        if (config.dexrt.boundary != BoundaryType::Promweaver) {
            throw std::runtime_error(fmt::format("Only promweaver boundaries are supported by {}", argv[0]));
        }
        std::vector<ModelAtom<f64>> crtaf_models;
        // TODO(cmo): Override atoms in ray config
        crtaf_models.reserve(config.dexrt.atom_paths.size());
        for (int i = 0; i < config.dexrt.atom_paths.size(); ++i) {
            const auto& p = config.dexrt.atom_paths[i];
            const auto& model_config = config.dexrt.atom_configs[i];
            crtaf_models.emplace_back(parse_crtaf_model<f64>(p, model_config));
        }
        AtomicDataHostDevice<fp_t> atomic_data = to_atomic_data<fp_t, f64>(
            crtaf_models,
            ToAtomicDataOptions{
                .limit_line_edge_bins=false
            }
        );

        DexRayState state{
            .adata = atomic_data.device,
            .phi = VoigtProfile<fp_t>(
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.15), 1024},
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(1.5e3), 64 * 1024}
            ),
            .nh_lte = HPartFn(),
        };

        const i32 max_mip_level = 0;
        state.atmos = state.mr_block_map.init_and_sparsify_atmos(
            config.dexrt.atmos_path,
            config.dexrt.threshold_temperature,
            max_mip_level
        );
        configure_mr_block_map(state.mr_block_map);
        update_atmosphere(config.dexrt, &state);
        load_dex_output(config.dexrt, &state);

        state.eta = Fp1d(
            "eta",
            state.atmos.temperature.extent(0)
        );
        state.chi = Fp1d(
            "chi",
            state.atmos.temperature.extent(0)
        );

        auto out = setup_output(config.ray_output_path, config, state.atmos);

        auto mu_iterator = tq::trange(config.muz.size());
        std::ostringstream ostream_redirect;
        if (quiet) {
            mu_iterator.set_ostream(ostream_redirect);
        }
        for (int mu : mu_iterator) {
            state.ray_set = compute_ray_set<yakl::memDevice>(config, state.atmos, mu);
            // TODO(cmo): Hoist this if possible
            PwBc<> pw_bc = load_bc(
                config.dexrt.atmos_path,
                state.ray_set.wavelength,
                config.dexrt.boundary,
                PromweaverResampleType::Interpolation
            );

            if (
                !state.ray_I.initialized()
                || (state.ray_I.extent(0) != state.ray_set.wavelength.extent(0))
                || (state.ray_I.extent(1) != state.ray_set.start_coord.extent(0))
            ) {
                state.ray_I = Fp2d(
                    "I",
                    state.ray_set.wavelength.extent(0),
                    state.ray_set.start_coord.extent(0)
                );
                state.ray_tau = Fp2d(
                    "tau",
                    state.ray_set.wavelength.extent(0),
                    state.ray_set.start_coord.extent(0)
                );
            }
            DexRayStateAndBc<PwBc<>> ray_state{
                .state = state,
                .bc = pw_bc
            };
            compute_ray_intensity(&ray_state, config);
            // NOTE(cmo): state isn't captured by reference (the arrays are), so if the depth data arrays are modified, this won't propagate back to the original state, so we pass ray_state.state.
            write_output_plane(out, ray_state.state, config, mu);
        }
    }
    yakl::finalize();
    Kokkos::finalize();

    return 0;
}