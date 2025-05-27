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
#include "EmisOpac.hpp"
#include "RayMarching.hpp"
#include "DexrtConfig.hpp"
#include "JasPP.hpp"
#include "MiscSparse.hpp"
#include "GitVersion.hpp"

int get_dexrt_dimensionality() {
    return 3;
}

struct RayConfig {
    fp_t mem_pool_gb = FP(4.0);
    std::string own_path;
    std::string dexrt_config_path;
    std::string ray_output_path;
    vec2 image_size;
    int supersample = 1;
    std::vector<vec3> view_ray;
    std::vector<vec3> up_ray;
    std::vector<vec3> right_ray;
    std::vector<vec3> image_corner;
    std::vector<fp_t> wavelength;
    DexrtConfig dexrt;
};

template <int mem_space=yakl::memDevice>
struct RaySet {
    yakl::Array<fp_t, 3, mem_space> start_coord; // [y, x, pos]
    vec3 view_ray;
    yakl::Array<fp_t, 1, mem_space> wavelength;
};

struct DexRayState {
    SparseAtmosphere atmos;
    AtomicData<fp_t> adata;
    MultiResBlockMap<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3> mr_block_map;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;

    RaySet<yakl::memDevice> ray_set;
    Fp2d pops;
    Fp3d ray_I; // [wavelength, pos_y, pos_x]
    Fp3d ray_tau; // [wavelength, pos_y, pos_x]
};

template <typename Bc>
struct DexRayStateAndBc {
    DexRayState state;
    Bc bc;
};

struct DexOutput {
    // TODO(cmo): When we have scattering, add J
    Fp2d pops;
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

    if (file["system"]) {
        auto system = file["system"];
        if (system["mem_pool_gb"]) {
            config.mem_pool_gb = system["mem_pool_gb"].as<fp_t>();
        } else if (system["mem_pool_initial_gb"]) {
            fmt::println("Found deprecated \"mem_pool_initial_gb\", using that value. The pool no longer grows and should be set with key \"mem_pool_gb\".");
            config.mem_pool_gb = system["mem_pool_initial_gb"].as<fp_t>();
        }
    }

    config.dexrt = parse_dexrt_config(config.dexrt_config_path);

    auto require_key = [&] (const std::string& key) {
        if (!file[key]) {
            throw std::runtime_error(fmt::format("{} key must be present in config file.", key));
        }
    };
    require_key("view_ray");
    require_key("up_ray");
    require_key("right_ray");
    require_key("image_corner");
    require_key("image_size");

    auto parse_one_or_more_vec3_to_vector = [&] (const std::string& key) {
        std::vector<vec3> result;
        if (!file[key]) {
            return result;
        }

        auto parse_vec3 = [] (const YAML::Node& seq) -> vec3 {
            if (seq.size() != 3) {
                throw std::runtime_error(fmt::format("Asked to parse vec3 but got {} entries", seq.size()));
            }
            vec3 result;
            for (int i = 0; i < 3; ++i) {
                result(i) = seq[i].as<fp_t>();
            }
            return result;
        };

        if (!(file[key].IsSequence() && file[key].size() > 0)) {
            throw std::runtime_error(fmt::format("Error parsing {}", key));
        }

        if (file[key][0].IsSequence()) {
            // multiple
            result.reserve(file[key].size());
            for (const auto& subseq : file[key]) {
                result.push_back(parse_vec3(subseq));
            }
        } else {
            // one
            result.push_back(parse_vec3(file[key]));
        }
        return result;
    };
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

    config.view_ray = parse_one_or_more_vec3_to_vector("view_ray");
    config.up_ray = parse_one_or_more_vec3_to_vector("up_ray");
    config.right_ray = parse_one_or_more_vec3_to_vector("right_ray");
    config.image_corner = parse_one_or_more_vec3_to_vector("image_corner");
    config.wavelength = parse_one_or_more_float_to_vector("wavelength");
    if ((config.view_ray.size() != config.up_ray.size()) || (config.up_ray.size() != config.right_ray.size()) || config.view_ray.size() == 0) {
        throw std::runtime_error("view_ray, up_ray, and right_ray must be provided and have the same number of entries (non-zero).");
    }

    if (!(file["image_size"].IsSequence() && file["image_size"].size() == 2)) {
        throw std::runtime_error("Image size should be a vec2 of voxel-space image plane size");
    }
    config.image_size(0) = file["image_size"][0].as<fp_t>();
    config.image_size(1) = file["image_size"][1].as<fp_t>();

    if (file["supersample"]) {
        config.supersample = file["supersample"].as<int>();
    }

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
    return true;
}

BlockMap<BLOCK_SIZE> dex_block_map(const DexrtConfig& config, const Atmosphere& atmos, yakl::SimpleNetCDF& nc) {
    int block_size_file = 0;
    int ncid = nc.file.ncid;
    int ierr = nc_get_att_int(ncid, NC_GLOBAL, "block_size", &block_size_file);
    if (ierr != NC_NOERR) {
        throw std::runtime_error(fmt::format("Unable to load block_size from dex output: {}", nc_strerror(ierr)));
    }

    if (BLOCK_SIZE != block_size_file) {
        throw std::runtime_error(
            fmt::format(
                "Compiled BLOCK_SIZE ({}) != block_size in dex output ({}), please recompile so these are the same",
                BLOCK_SIZE,
                block_size_file
            )
        );
    }

    BlockMap<BLOCK_SIZE> block_map;
    block_map.init(atmos, config.threshold_temperature);
    return block_map;
}

DexOutput load_dex_output(const DexrtConfig& config) {
    yakl::SimpleNetCDF nc;
    nc.open(config.output_path, yakl::NETCDF_MODE_READ);

    Fp2d pops;
    nc.read(pops, "pops");
    DexOutput result {
        .pops = pops
    };
    return result;
}

void update_atmosphere(const DexrtConfig& config, SparseAtmosphere* atmos) {
    yakl::SimpleNetCDF nc;
    nc.open(config.output_path, yakl::NETCDF_MODE_READ);
    if (!(nc.varExists("ne") || nc.varExists("nh_tot"))) {
        return;
    }

    if (nc.varExists("ne")) {
        nc.read(atmos->ne, "ne");
    }
    if (nc.varExists("nh_tot")) {
        nc.read(atmos->nh_tot, "nh_tot");
    }
}

void configure_mr_block_map(const MultiResBlockMap<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>& mr_block_map) {
    const auto& block_map = mr_block_map.block_map;
    yakl::Array<i32, 1, yakl::memDevice> max_mip_level("max mip entries", block_map.num_z_tiles() * block_map.num_y_tiles() * block_map.num_x_tiles());
    max_mip_level = 0;
    Kokkos::fence();
    dex_parallel_for(
        "Set active blocks in mr_block_map",
        FlatLoop<1>(block_map.loop_bounds().dim(0)),
        YAKL_LAMBDA (i64 tile_idx) {
            MRIdxGen3d idx_gen(mr_block_map);
            Coord3 tile_coord = idx_gen.compute_tile_coord(tile_idx);
            i64 flat_entry = mr_block_map.lookup.flat_tile_index(tile_coord);
            max_mip_level(flat_entry) = 1;
        }
    );
    Kokkos::fence();
    mr_block_map.lookup.pack_entries(max_mip_level);
}

template <int mem_space=yakl::memDevice>
RaySet<mem_space> compute_ray_set(const RayConfig& cfg, const SparseAtmosphere& atmos, int mu_idx) {
    RaySet<mem_space> result;

    i32 supersample = cfg.supersample;
    int num_x = int(cfg.image_size(0) * supersample);
    int num_y = int(cfg.image_size(1) * supersample);
    vec3 corner = cfg.image_corner[mu_idx];
    vec3 up = cfg.up_ray[mu_idx];
    vec3 right = cfg.right_ray[mu_idx];
    result.view_ray = cfg.view_ray[mu_idx];
    fp_t mu_scale = std::sqrt(square(result.view_ray(0)) + square(result.view_ray(1)) + square(result.view_ray(2)));
    for (int m = 0; m < 3; ++m) {
        result.view_ray(m) /= mu_scale;
    }
    result.start_coord = Fp3d("start_pos", num_y, num_x, 3);
    dex_parallel_for(
        "Compute ray dirs",
        FlatLoop<2>(num_y, num_x),
        KOKKOS_LAMBDA (int y, int x) {
            for (int i = 0; i < 3; ++i) {
                result.start_coord(y, x, i) = corner(i) + up(i) * (y + FP(0.5)) / fp_t(supersample) + right(i) * (x + FP(0.5)) / fp_t(supersample);
            }
        }
    );
    Kokkos::fence();
    Fp1dHost wave("wavelength", cfg.wavelength.size());
    for (int la = 0; la < cfg.wavelength.size(); ++la) {
        wave(la) = cfg.wavelength[la];
    }
    if constexpr (mem_space == yakl::memHost) {
        result.wavelength = wave;
    } else {
        result.wavelength = wave.createDeviceCopy();
    }

    return result;
}

template <typename Bc>
void compute_ray_intensity(DexRayStateAndBc<Bc>* st, const RayConfig& config) {
    DexRayStateAndBc<Bc>& state = *st;
    JasUnpack(state.state, atmos, pops, ray_set, ray_I, ray_tau, adata, phi, nh_lte, mr_block_map);
    const auto& bc(state.bc);

    Fp2d n_star("n_star", pops.extent(0), pops.extent(1));
    Fp1d eta("eta", pops.extent(1));
    Fp1d chi("chi", pops.extent(1));

    for (int la = 0; la < ray_set.wavelength.extent(0); ++la) {
        dex_parallel_for(
            "Compute emis/opac",
            mr_block_map.block_map.loop_bounds(),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen3d idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                const fp_t lambda = ray_set.wavelength(la);
                vec3 ray_mu;
                ray_mu(0) = -ray_set.view_ray(0);
                ray_mu(1) = -ray_set.view_ray(1);
                ray_mu(2) = -ray_set.view_ray(2);

                const fp_t vel = (
                    atmos.vx(ks) * ray_mu(0)
                    + atmos.vy(ks) * ray_mu(1)
                    + atmos.vz(ks) * ray_mu(2)
                );
                const bool have_h = (adata.Z(0) == 1);
                fp_t nh0;
                if (have_h) {
                    nh0 = pops(0, ks);
                } else {
                    nh0 = nh_lte(atmos.temperature(ks), atmos.ne(ks), atmos.nh_tot(ks));
                }
                AtmosPointParams local_atmos{
                    .temperature = atmos.temperature(ks),
                    .ne = atmos.ne(ks),
                    .vturb = atmos.vturb(ks),
                    .nhtot = atmos.nh_tot(ks),
                    .vel = vel,
                    .nh0 = nh0
                };
                auto eta_chi = emis_opac(
                    EmisOpacSpecState<>{
                        .adata = adata,
                        .profile = phi,
                        .lambda = lambda,
                        .n = pops,
                        .n_star_scratch = n_star,
                        .k = ks,
                        .atmos = local_atmos
                    }
                );
                eta(ks) = eta_chi.eta;
                chi(ks) = eta_chi.chi;
            }
        );
        Kokkos::fence();
        dex_parallel_for(
            "Trace Rays (front-to-back)",
            FlatLoop<2>(ray_set.start_coord.extent(0), ray_set.start_coord.extent(1)),
            YAKL_LAMBDA (int yi, int xi) {
                vec3 start_pos;
                start_pos(0) = ray_set.start_coord(yi, xi, 0);
                start_pos(1) = ray_set.start_coord(yi, xi, 1);
                start_pos(2) = ray_set.start_coord(yi, xi, 2);

                const fp_t vox_scale = atmos.voxel_scale;
                RaySegment<3> ray_seg(start_pos, ray_set.view_ray, FP(0.0), FP(8192.0));

                MRIdxGen3d idx_gen(mr_block_map);
                auto s = MultiLevelDDA<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>(idx_gen);
                const bool have_marcher = s.init(ray_seg, 0, nullptr);

                fp_t boundary_I = FP(0.0);
                if (ray_seg.d(2) < FP(0.0)) {
                    vec3 pos;
                    pos(0) = ray_seg.o(0) * vox_scale + atmos.offset_x;
                    pos(1) = ray_seg.o(1) * vox_scale + atmos.offset_y;
                    pos(2) = ray_seg.o(2) * vox_scale + atmos.offset_z;
                    vec3 ray_mu;
                    ray_mu(0) = -ray_set.view_ray(0);
                    ray_mu(1) = -ray_set.view_ray(1);
                    ray_mu(2) = -ray_set.view_ray(2);

                    fp_t I_sample = sample_boundary(bc, la, pos, ray_mu);
                    boundary_I = I_sample;
                }
                if (!have_marcher) {
                    ray_I(la, yi, xi) = boundary_I;
                    ray_tau(la, yi, xi) = FP(0.0);
                    return;
                }

                fp_t I = FP(0.0);
                fp_t cumulative_tau = FP(0.0);

                do {
                    if (s.can_sample()) {
                        i64 ks = idx_gen.idx(
                            0,
                            Coord3{.x = s.curr_coord(0), .y = s.curr_coord(1), .z = s.curr_coord(2)}
                        );

                        fp_t eta_s = eta(ks);
                        fp_t chi_s = chi(ks) + FP(1e-20);
                        fp_t tau = chi_s * s.dt * vox_scale;
                        fp_t source_fn = eta_s / chi_s;
                        fp_t one_m_edt = -std::expm1(-tau);
                        fp_t cumulative_trans = std::exp(-cumulative_tau);

                        fp_t local_I = one_m_edt * source_fn;
                        I += cumulative_trans * local_I;
                        cumulative_tau += tau;
                    }

                } while (s.step_through_grid());

                I += std::exp(-cumulative_tau) * boundary_I;

                ray_I(la, yi, xi) = I;
                ray_tau(la, yi, xi) = cumulative_tau;
            }
        );
        Kokkos::fence();
    }
}

yakl::SimpleNetCDF setup_output(const std::string& path, const RayConfig& cfg, const SparseAtmosphere& atmos) {
    yakl::SimpleNetCDF nc;
    nc.create(path, yakl::NETCDF_MODE_REPLACE);

    nc.createDim("wavelength", cfg.wavelength.size());

    FpConst1dHost wavelength("wavelength", cfg.wavelength.data(), cfg.wavelength.size());
    nc.write(wavelength, "wavelength", {"wavelength"});

    const auto ncwrap = [] (int ierr, int line) {
        if (ierr != NC_NOERR) {
            printf("NetCDF Error writing attributes at dexrt_ray.cpp:%d\n", line);
            printf("%s\n",nc_strerror(ierr));
            yakl::yakl_throw(nc_strerror(ierr));
        }
    };
    int ncid = nc.file.ncid;
    std::string name = "dexrt_ray (3d)";
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "program", name.size(), name.c_str()),
        __LINE__
    );

    std::string git_hash(GIT_HASH);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "git_hash", git_hash.size(), git_hash.c_str()),
        __LINE__
    );

    f64 voxel_scale = atmos.voxel_scale;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "voxel_scale", NC_DOUBLE, 1, &voxel_scale),
        __LINE__
    );
    f64 pixel_scale = atmos.voxel_scale / f64(cfg.supersample);
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "pixel_scale", NC_DOUBLE, 1, &pixel_scale),
        __LINE__
    );

    const auto& config_path(cfg.own_path);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "config_path", config_path.size(), config_path.c_str()),
        __LINE__
    );

    return nc;
}

void write_output_plane(
    yakl::SimpleNetCDF& nc,
    const DexRayState& state,
    const RayConfig& config,
    int mu_idx
) {
    nc.write(state.ray_I, fmt::format("I_{}", mu_idx), {"wavelength", "im_y", "im_x"});
    nc.write(state.ray_tau, fmt::format("tau_{}", mu_idx), {"wavelength", "im_y", "im_x"});
}


int main(int argc, char** argv) {
    argparse::ArgumentParser program("DexRT Ray");
    program.add_argument("--config")
        .default_value(std::string("dexrt_ray.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_epilog("Single-pass formal solver for post-processing Dex models.");

    program.parse_known_args(argc, argv);

    RayConfig config = parse_ray_config(program.get<std::string>("--config"));
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
        DexRayState state;
        state.phi = VoigtProfile<fp_t>();
        state.nh_lte = HPartFn();

        AtmosphereNd<3, yakl::memHost> atmos = load_atmos_3d_host(config.dexrt.atmos_path);
        std::vector<ModelAtom<f64>> crtaf_models;
        // TODO(cmo): Override atoms in ray config
        crtaf_models.reserve(config.dexrt.atom_paths.size());
        for (int i = 0; i < config.dexrt.atom_paths.size(); ++i) {
            const auto& p = config.dexrt.atom_paths[i];
            const auto& model_config = config.dexrt.atom_configs[i];
            crtaf_models.emplace_back(parse_crtaf_model<f64>(p, model_config));
        }
        AtomicDataHostDevice<fp_t> atomic_data = to_atomic_data<fp_t, f64>(crtaf_models);
        state.adata = atomic_data.device;

        BlockMap<BLOCK_SIZE_3D, 3> block_map;
        block_map.init(atmos, config.dexrt.threshold_temperature);
        i32 max_mip_level = 0;
        state.mr_block_map.init(block_map, max_mip_level);
        configure_mr_block_map(state.mr_block_map);
        state.atmos = sparsify_atmosphere(atmos, block_map);

        update_atmosphere(config.dexrt, &state.atmos);
        DexOutput model_output = load_dex_output(config.dexrt);
        state.pops = model_output.pops;

        auto out = setup_output(config.ray_output_path, config, state.atmos);

        for (int mu = 0; mu < config.view_ray.size(); ++mu) {
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
                || (state.ray_I.extent(2) != state.ray_set.start_coord.extent(1))
            ) {
                state.ray_I = Fp3d(
                    "I",
                    state.ray_set.wavelength.extent(0),
                    state.ray_set.start_coord.extent(0),
                    state.ray_set.start_coord.extent(1)
                );
                state.ray_tau = Fp3d(
                    "tau",
                    state.ray_set.wavelength.extent(0),
                    state.ray_set.start_coord.extent(0),
                    state.ray_set.start_coord.extent(1)
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