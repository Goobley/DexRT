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

template <int mem_space=yakl::memDevice>
struct RaySet {
    yakl::Array<fp_t, 2, mem_space> start_coord;
    vec3 mu;
    yakl::Array<fp_t, 1, mem_space> wavelength;
};

struct DexRayState {
    SparseAtmosphere atmos;
    MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE, 2> mr_block_map;
    AtomicData<fp_t> adata;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    Fp2d pops;
    RaySet<> ray_set;
    Fp2d ray_I; // [wavelength, pos]
    Fp2d ray_tau; // [wavelength, pos]
    Fp1d eta;
    Fp1d chi;

    // NOTE(cmo): Depth-data output, big, and only allocated if needed
    yakl::Array<i64, 1, yakl::memDevice> num_steps;
    Fp2d pos; // Depth, locations after end of ray are padded with nan [num_steps, ray_idx]
    Fp3d cont_fn; // [wavelength, num_steps, ray_idx]
    Fp3d chi_tau;
    Fp3d source_fn_depth;
    Fp3d tau_depth;
    Fp3d eta_depth;
    Fp3d chi_depth;
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

struct AabbPoints {
    vec2 br;
    vec2 bl;
    vec2 tr;
    vec2 tl;
};

struct LineSeg {
    vec2 start;
    vec2 end;
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

    config.dexrt = parse_dexrt_config(config.dexrt_config_path);

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

void configure_mr_block_map(const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE, 2>& mr_block_map) {
    const auto& block_map = mr_block_map.block_map;
    yakl::Array<i32, 1, yakl::memDevice> max_mip_level("max mip entries", block_map.num_z_tiles() * block_map.num_x_tiles());
    max_mip_level = 0;
    Kokkos::fence();
    dex_parallel_for(
        "Set active blocks in mr_block_map",
        FlatLoop<1>(block_map.loop_bounds().dim(0)),
        YAKL_LAMBDA (i64 tile_idx) {
            MRIdxGen idx_gen(mr_block_map);
            Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);
            i64 flat_entry = mr_block_map.lookup.flat_tile_index(tile_coord);
            max_mip_level(flat_entry) = 1;
        }
    );
    Kokkos::fence();
    mr_block_map.lookup.pack_entries(max_mip_level);
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

template <int mem_space=yakl::memDevice>
RaySet<mem_space> compute_ray_set(const RayConfig& cfg, const SparseAtmosphere& atmos, int mu_idx) {
    RaySet<mem_space> result;
    if (cfg.rotate_aabb) {
        auto matvec = [] (const mat2x2& mat, const vec2& vec) {
            vec2 result;
            result(0) = mat(0, 0) * vec(0) + mat(0, 1) * vec(1);
            result(1) = mat(1, 0) * vec(0) + mat(1, 1) * vec(1);
            return result;
        };
        // NOTE(cmo): Do AABB shenanigans in x-z plane only
        fp_t muz = cfg.muz[mu_idx];
        fp_t mux = cfg.mux[mu_idx];
        fp_t mu_norm = std::sqrt(square(muz) + square(mux));
        muz /= mu_norm;
        mux /= mu_norm;
        vec2 mu_2d;
        mu_2d(0) = mux;
        mu_2d(1) = muz;

        mat2x2 reverse_rot;
        fp_t reverse_angle = std::acos(muz) * std::copysign(FP(1.0), mux);
        fp_t rs = std::sin(reverse_angle);
        fp_t rc = std::cos(reverse_angle);
        reverse_rot(0, 0) = rc;
        reverse_rot(0, 1) = -rs;
        reverse_rot(1, 0) = rs;
        reverse_rot(1, 1) = rc;

        mat2x2 rot;
        fp_t angle = -reverse_angle;
        fp_t s = std::sin(angle);
        fp_t c = std::cos(angle);
        rot(0, 0) = c;
        rot(0, 1) = -s;
        rot(1, 0) = s;
        rot(1, 1) = c;

        AabbPoints aabb;
        int nz = atmos.num_z;
        int nx = atmos.num_x;
        aabb.bl(0) = FP(0.0);
        aabb.bl(1) = FP(0.0);
        aabb.br(0) = fp_t(nx);
        aabb.br(1) = FP(0.0);
        aabb.tl(0) = FP(0.0);
        aabb.tl(1) = fp_t(nz);
        aabb.tr(0) = fp_t(nx);
        aabb.tr(1) = fp_t(nz);

        vec2 centre;
        centre(0) = FP(0.5) * (FP(0.0) + fp_t(nx));
        centre(1) = FP(0.5) * (FP(0.0) + fp_t(nz));

        AabbPoints rot_box;
        rot_box.bl = matvec(
            reverse_rot,
            (aabb.bl - centre)
        ) + centre;
        rot_box.br = matvec(
            reverse_rot,
            (aabb.br - centre)
        ) + centre;
        rot_box.tl = matvec(
            reverse_rot,
            (aabb.tl - centre)
        ) + centre;
        rot_box.tr = matvec(
            reverse_rot,
            (aabb.tr - centre)
        ) + centre;

        auto reduce_and_clip = [](
            fp_t (*red_op)(fp_t, fp_t),
            fp_t (*clip_op)(fp_t),
            const AabbPoints& aabb,
            int idx
        ) -> fp_t {
            return clip_op(
                red_op(
                    red_op(
                        red_op(
                            aabb.bl(idx),
                            aabb.br(idx)
                        ),
                        aabb.tl(idx)
                    ),
                    aabb.tr(idx)
                )
            );
        };
        // NOTE(cmo): The compiler can't directly take a reference to min here
        auto clip_noop = [](fp_t a) { return a; };
        auto min_wrap = [](fp_t a, fp_t b) { return std::min(a, b); };
        auto max_wrap = [](fp_t a, fp_t b) { return std::max(a, b); };
        auto floor_wrap = [](fp_t a) { return std::floor(a); };
        auto ceil_wrap = [](fp_t a) { return std::ceil(a); };
        fp_t min_x = reduce_and_clip(min_wrap, clip_noop, rot_box, 0) - FP(1.0);
        fp_t max_x = reduce_and_clip(max_wrap, clip_noop, rot_box, 0)  + FP(1.0);
        fp_t min_z = reduce_and_clip(min_wrap, clip_noop, rot_box, 1) - FP(1.0);
        fp_t max_z = reduce_and_clip(max_wrap, clip_noop, rot_box, 1)  + FP(1.0);

        // NOTE(cmo): Top surface of the AABB of the reverse rotated AABB of the domain
        LineSeg rot_image_plane;
        rot_image_plane.start(0) = min_x;
        rot_image_plane.start(1) = max_z;
        rot_image_plane.end(0) = max_x;
        rot_image_plane.end(1) = max_z;

        LineSeg image_plane;
        image_plane.start = matvec(
            rot,
            (rot_image_plane.start - centre)
        ) + centre;
        image_plane.end = matvec(
            rot,
            (rot_image_plane.end - centre)
        ) + centre;

        fp_t seg_length = std::sqrt(yakl::intrinsics::sum(square(image_plane.end - image_plane.start)));
        vec2 image_plane_dir = (image_plane.end - image_plane.start) / seg_length;

        int num_rays = int(std::ceil(seg_length));
        Fp2dHost ray_pos("ray_starts", num_rays, 2);
        for (int i = 0; i < num_rays; ++i) {
            ray_pos(i, 0) = image_plane.start(0) + (FP(0.5) + i) * image_plane_dir(0);
            ray_pos(i, 1) = image_plane.start(1) + (FP(0.5) + i) * image_plane_dir(1);
        }
        vec3 mu;
        mu(0) = -cfg.mux[mu_idx];
        mu(2) = -cfg.muz[mu_idx];
        mu_norm = square(mu(0)) + square(mu(2));
        if (mu_norm >= FP(1.0)) {
            mu(0) /= mu_norm;
            mu(1) = FP(0.0);
            mu(2) /= mu_norm;
        } else {
            mu(1) = std::sqrt(FP(1.0) - square(mu(0)) - square(mu(2)));
        }

        result.mu = mu;
        if constexpr (mem_space == yakl::memHost) {
            result.start_coord = ray_pos;
        } else {
            result.start_coord = ray_pos.createDeviceCopy();
        }
    } else {
        Fp2dHost ray_pos("ray_starts", atmos.num_x, 2);
        fp_t z_max = fp_t(atmos.num_z);
        for (int i = 0; i < atmos.num_x; ++i) {
            ray_pos(i, 0) = (i + FP(0.5));
            ray_pos(i, 1) = z_max;
        }
        vec3 mu;
        mu(0) = -cfg.mux[mu_idx];
        mu(2) = -cfg.muz[mu_idx];
        fp_t mu_norm = square(mu(0)) + square(mu(2));
        if (mu_norm >= FP(1.0)) {
            mu(0) /= mu_norm;
            mu(1) = FP(0.0);
            mu(2) /= mu_norm;
        } else {
            mu(1) = std::sqrt(FP(1.0) - square(mu(0)) - square(mu(2)));
        }

        result.mu = mu;
        if constexpr (mem_space == yakl::memHost) {
            result.start_coord = ray_pos;
        } else {
            result.start_coord = ray_pos.createDeviceCopy();
        }
    }

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
YAKL_INLINE fp_t sample_bc(const Bc& bc, int la, vec2 pos, vec2 mu) {
    vec3 pos3;
    pos3(0) = pos(0);
    pos3(1) = FP(0.0);
    pos3(2) = pos(1);
    vec3 mu3;
    mu3(0) = mu(0);
    mu3(1) = FP(0.0);
    mu3(2) = mu(1);
    return sample_boundary(bc, la, pos3, mu3);
}

template <typename Bc>
void compute_ray_intensity(DexRayStateAndBc<Bc>* st, const RayConfig& config) {
    DexRayStateAndBc<Bc>& state = *st;
    JasUnpack(state.state, atmos, pops, ray_set, ray_I, ray_tau, eta, chi, adata, phi, nh_lte);
    JasUnpack(state.state, num_steps, pos, cont_fn, source_fn_depth, tau_depth, eta_depth, chi_depth);
    JasUnpack(state.state, chi_tau, mr_block_map);
    auto& bc(state.bc);

    Fp2d n_star = Fp2d("flat_n_star", pops.extent(0), pops.extent(1));

    for (int wave = 0; wave < ray_set.wavelength.extent(0); ++wave) {
        dex_parallel_for(
            "Compute eta, chi",
            mr_block_map.block_map.loop_bounds(),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                fp_t lambda = ray_set.wavelength(wave);
                // NOTE(cmo): The projection vector is inverted here as we are
                // tracing front-to-back.
                fp_t v_proj = (
                    atmos.vx(ks) * -ray_set.mu(0)
                    + atmos.vy(ks) * -ray_set.mu(1)
                    + atmos.vz(ks) * -ray_set.mu(2)
                );
                const bool have_h = (adata.Z(0) == 1);
                fp_t nh0;
                if (have_h) {
                    nh0 = pops(0, ks);
                } else {
                    nh0 = nh_lte(atmos.temperature(ks), atmos.ne(ks), atmos.nh_tot(ks));
                }
                AtmosPointParams local_atmos {
                    .temperature = atmos.temperature(ks),
                    .ne = atmos.ne(ks),
                    .vturb = atmos.vturb(ks),
                    .nhtot = atmos.nh_tot(ks),
                    .vel = v_proj,
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
        yakl::fence();

        if (wave == 0 && (config.output_cfn || config.output_eta_chi)) {
            num_steps = std::remove_reference_t<decltype(num_steps)>(
                "num steps",
                ray_set.start_coord.extent(0)
            );
            i64 num_rays = ray_set.start_coord.extent(0);
            dex_parallel_for(
                "Compute max steps",
                FlatLoop<1>(num_rays),
                YAKL_LAMBDA (int ray_idx) {
                    vec2 start_pos;
                    start_pos(0) = ray_set.start_coord(ray_idx, 0);
                    start_pos(1) = ray_set.start_coord(ray_idx, 1);

                    vec2 flatland_mu;
                    fp_t flatland_mu_norm = std::sqrt(square(ray_set.mu(0)) + square(ray_set.mu(2)));
                    flatland_mu(0) = ray_set.mu(0) / flatland_mu_norm;
                    flatland_mu(1) = ray_set.mu(2) / flatland_mu_norm;

                    fp_t max_dim = std::max(fp_t(atmos.num_z), fp_t(atmos.num_x));

                    RaySegment<2> ray_seg(start_pos, flatland_mu, FP(0.0), max_dim);
                    MRIdxGen idx_gen(mr_block_map);
                    auto s = MultiLevelDDA<BLOCK_SIZE, ENTRY_SIZE>(idx_gen);
                    const bool have_marcher = s.init(ray_seg, 0, nullptr);

                    if (!have_marcher) {
                        num_steps(ray_idx) = 0;
                        return;
                    }

                    i64 step_count = 0;
                    do {
                        if (s.can_sample()) {
                            step_count += 1;
                        }
                    } while (s.step_through_grid());

                    num_steps(ray_idx) = step_count;
                }
            );
            yakl::fence();

            i64 max_steps = yakl::intrinsics::maxval(num_steps);
            fmt::println(
                "Num rays: {}, max steps: {}, product: {}k",
                num_rays,
                max_steps,
                fp_t(num_rays * max_steps) / FP(1024.0)
            );
            pos = Fp2d("ray coord start", max_steps, num_rays);
            pos = std::nanf("");
            if (config.output_cfn) {
                cont_fn = Fp3d(
                    "cont fn",
                    ray_set.wavelength.extent(0),
                    max_steps,
                    num_rays
                );
                chi_tau = Fp3d(
                    "chi/tau",
                    ray_set.wavelength.extent(0),
                    max_steps,
                    num_rays
                );
                source_fn_depth = Fp3d(
                    "source fn",
                    ray_set.wavelength.extent(0),
                    max_steps,
                    num_rays
                );
                tau_depth = Fp3d(
                    "tau depth",
                    ray_set.wavelength.extent(0),
                    max_steps,
                    num_rays
                );
                cont_fn = std::nanf("");
                chi_tau = std::nanf("");
                source_fn_depth = std::nanf("");
                tau_depth = std::nanf("");
            }
            if (config.output_eta_chi) {
                eta_depth = Fp3d(
                    "eta depth",
                    ray_set.wavelength.extent(0),
                    max_steps,
                    num_rays
                );
                chi_depth = Fp3d(
                    "chi depth",
                    ray_set.wavelength.extent(0),
                    max_steps,
                    num_rays
                );
                eta_depth = std::nanf("");
                chi_depth = std::nanf("");
            }
            yakl::fence();
        }

        const bool output_cfn = config.output_cfn;
        const bool output_eta_chi = config.output_eta_chi;

        dex_parallel_for(
            "Trace Rays (front-to-back)",
            FlatLoop<1>(ray_set.start_coord.extent(0)),
            YAKL_LAMBDA (int ray_idx) {
                vec2 start_pos;
                start_pos(0) = ray_set.start_coord(ray_idx, 0);
                start_pos(1) = ray_set.start_coord(ray_idx, 1);

                vec2 flatland_mu;
                fp_t flatland_mu_norm = std::sqrt(square(ray_set.mu(0)) + square(ray_set.mu(2)));
                flatland_mu(0) = ray_set.mu(0) / flatland_mu_norm;
                flatland_mu(1) = ray_set.mu(2) / flatland_mu_norm;

                fp_t max_dim = std::max(fp_t(atmos.num_z), fp_t(atmos.num_x));

                RaySegment<2> ray_seg(start_pos, flatland_mu, FP(0.0), max_dim);
                MRIdxGen idx_gen(mr_block_map);
                auto s = MultiLevelDDA<BLOCK_SIZE, ENTRY_SIZE>(idx_gen);
                const bool have_marcher = s.init(ray_seg, 0, nullptr);

                fp_t boundary_I = FP(0.0);
                if (ray_seg.d(1) < FP(0.0)) {
                    vec2 sample;
                    sample(0) = start_pos(0) * atmos.voxel_scale + atmos.offset_x;
                    sample(1) = start_pos(1) * atmos.voxel_scale + atmos.offset_z;
                    boundary_I = sample_bc(bc, wave, sample, flatland_mu);
                }
                if (!have_marcher) {
                    ray_I(wave, ray_idx) = boundary_I;
                    ray_tau(wave, ray_idx) = FP(0.0);
                    return;
                }

                fp_t I = FP(0.0);
                fp_t cumulative_tau = FP(0.0);

                const fp_t distance_factor = atmos.voxel_scale / std::sqrt(FP(1.0) - square(ray_set.mu(1)));

                i64 step_idx = 0;
                do {
                    if (s.can_sample()) {
                        i64 ks = idx_gen.idx(
                            0,
                            Coord2{.x = s.curr_coord(0), .z = s.curr_coord(1)}
                        );
                        fp_t eta_s = eta(ks);
                        fp_t chi_s = chi(ks) + FP(1e-20);
                        fp_t tau = chi_s * s.dt * distance_factor;
                        fp_t source_fn = eta_s / chi_s;
                        fp_t one_m_edt = -std::expm1(-tau);
                        fp_t cumulative_trans = std::exp(-cumulative_tau);

                        fp_t local_I = one_m_edt * source_fn;
                        I += cumulative_trans * local_I;
                        cumulative_tau += tau;
                        if (output_cfn || output_eta_chi) {
                            if (wave == 0) {
                                pos(step_idx, ray_idx) = s.t;
                            }

                            if (output_cfn) {
                                source_fn_depth(wave, step_idx, ray_idx) = source_fn;
                                tau_depth(wave, step_idx, ray_idx) = cumulative_tau;
                                cont_fn(wave, step_idx, ray_idx) = eta_s * cumulative_trans;
                                chi_tau(wave, step_idx, ray_idx) = chi_s / cumulative_tau;
                            }
                            if (output_eta_chi) {
                                eta_depth(wave, step_idx, ray_idx) = eta_s;
                                chi_depth(wave, step_idx, ray_idx) = chi_s;
                            }
                        }
                        step_idx += 1;
                    }
                } while (s.step_through_grid());

                I += std::exp(-cumulative_tau) * boundary_I;

                ray_I(wave, ray_idx) = I;
                ray_tau(wave, ray_idx) = cumulative_tau;
            }
        );
    }
    Kokkos::fence();
}

yakl::SimpleNetCDF setup_output(const std::string& path, const RayConfig& cfg, const SparseAtmosphere& atmos) {
    yakl::SimpleNetCDF nc;
    nc.create(path, yakl::NETCDF_MODE_REPLACE);

    nc.createDim("mu", cfg.muz.size());
    nc.createDim("3d-space", 3);
    nc.createDim("2d-space", 2);
    nc.createDim("wavelength", cfg.wavelength.size());

    FpConst1dHost wavelength("wavelength", cfg.wavelength.data(), cfg.wavelength.size());
    nc.write(wavelength, "wavelength", {"wavelength"});
    Fp2dHost mu("mu", cfg.muz.size(), 3);
    for (int i = 0; i < cfg.muz.size(); ++i) {
        mu(i, 0) = cfg.mux[i];
        mu(i, 1) = std::sqrt(std::max(FP(0.0), FP(1.0) - square(cfg.mux[i]) - square(cfg.muz[i])));
        mu(i, 2) = cfg.muz[i];
    }
    nc.write(mu, "mu", {"mu", "3d-space"});

    const auto ncwrap = [] (int ierr, int line) {
        if (ierr != NC_NOERR) {
            printf("NetCDF Error writing attributes at dexrt_ray.cpp:%d\n", line);
            printf("%s\n",nc_strerror(ierr));
            yakl::yakl_throw(nc_strerror(ierr));
        }
    };
    int ncid = nc.file.ncid;
    std::string name = "dexrt_ray (2d)";
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
    std::string ray_idx = fmt::format("rays_{}", mu_idx);
    nc.write(state.ray_I, fmt::format("I_{}", mu_idx), {"wavelength", ray_idx});
    nc.write(state.ray_tau, fmt::format("tau_{}", mu_idx), {"wavelength", ray_idx});
    nc.write(state.ray_set.start_coord, fmt::format("ray_start_{}", mu_idx), {ray_idx, "2d-space"});

    if (config.output_cfn || config.output_eta_chi) {
        std::string num_steps = fmt::format("num_steps_{}", mu_idx);
        nc.write(state.num_steps, fmt::format("rays_{}_num_steps", mu_idx), {ray_idx});
        nc.write(state.pos, fmt::format("rays_{}_pos", mu_idx), {num_steps, ray_idx});
        if (config.output_cfn) {
            nc.write(state.cont_fn, fmt::format("rays_{}_cont_fn", mu_idx), {"wavelength", num_steps, ray_idx});
            nc.write(state.chi_tau, fmt::format("rays_{}_chi_tau", mu_idx), {"wavelength", num_steps, ray_idx});
            nc.write(state.source_fn_depth, fmt::format("rays_{}_source_fn", mu_idx), {"wavelength", num_steps, ray_idx});
            nc.write(state.tau_depth, fmt::format("rays_{}_tau", mu_idx), {"wavelength", num_steps, ray_idx});
        }
        if (config.output_eta_chi) {
            nc.write(state.eta_depth, fmt::format("rays_{}_eta", mu_idx), {"wavelength", num_steps, ray_idx});
            nc.write(state.chi_depth, fmt::format("rays_{}_chi", mu_idx), {"wavelength", num_steps, ray_idx});
        }
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
        AtomicDataHostDevice<fp_t> atomic_data = to_atomic_data<fp_t, f64>(crtaf_models);

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