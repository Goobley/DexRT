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

struct RayConfig {
    fp_t mem_pool_initial_gb = FP(2.0);
    fp_t mem_pool_grow_gb = FP(1.4);
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
    Atmosphere atmos;
    AtomicData<fp_t> adata;
    yakl::Array<bool, 2, yakl::memDevice> active;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    Fp3d pops;
    RaySet<> ray_set;
    Fp2d ray_I; // [wavelength, pos]
    Fp2d ray_tau; // [wavelength, pos]
    Fp2d eta;
    Fp2d chi;

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
    Fp3d pops;
    yakl::Array<bool, 2, yakl::memDevice> active;
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
        if (system["mem_pool_initial_gb"]) {
            config.mem_pool_initial_gb = system["mem_pool_initial_gb"].as<fp_t>();
        }
        if (system["mem_pool_grow_gb"]) {
            config.mem_pool_grow_gb = system["mem_pool_grow_gb"].as<fp_t>();
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

DexOutput load_dex_output(const DexrtConfig& config, const Atmosphere& atmos) {
    yakl::SimpleNetCDF nc;
    nc.open(config.output_path, yakl::NETCDF_MODE_READ);
    const bool is_sparse = dex_data_is_sparse(nc);
    DexOutput result;

    if (is_sparse) {
        BlockMap<BLOCK_SIZE> block_map = dex_block_map(config, atmos, nc);
        Fp2d temp_pops;
        nc.read(temp_pops, "pops");
        Fp3dHost full_pops = rehydrate_sparse_quantity(block_map, temp_pops);
        result.pops = full_pops.createDeviceCopy();

        // NOTE(cmo): If we're sparse -- we need active, so rely on it being there.
        // currently active is always written dense -- but we reconstruct it from the block_map here anyway
        auto active = reify_active_c0(block_map);
        result.active = decltype(result.active)("active", active.extent(0), active.extent(1));
        parallel_for(
            SimpleBounds<2>(active.extent(0), active.extent(1)),
            YAKL_LAMBDA (int i, int j) {
                result.active(i, j) = active(i, j);
            }
        );
        yakl::fence();
    } else {
        nc.read(result.pops, "pops");

        if (nc.varExists("active")) {
            // NOTE(cmo): This is dumb, but isn't going to cause an issue, and netCDF doesn't let us save bool arrays.
            yakl::Array<unsigned char, 2, yakl::memHost> active_char;
            nc.read(active_char, "active");
            yakl::Array<bool, 2, yakl::memHost> active_host("active", active_char.extent(0), active_char.extent(1));
            for (int z = 0; z < active_char.extent(0); ++z) {
                for (int x = 0; x < active_char.extent(1); ++x) {
                    active_host(z, x) = active_char(z, x);
                }
            }
            result.active = active_host.createDeviceCopy();
        } else {
            // NOTE(cmo): If the active mask isn't in the file, then set everything to true
            result.active = decltype(result.active)("active", result.pops.extent(1), result.pops.extent(2));
            result.active = true;
            yakl::fence();
        }
    }

    return result;
}

void update_atmosphere(const DexrtConfig& config, Atmosphere* atmos) {
    yakl::SimpleNetCDF nc;
    nc.open(config.output_path, yakl::NETCDF_MODE_READ);
    if (!(nc.varExists("ne") || nc.varExists("nh_tot"))) {
        return;
    }

    const bool is_sparse = dex_data_is_sparse(nc);
    if (is_sparse) {
        BlockMap<BLOCK_SIZE> block_map = dex_block_map(config, *atmos, nc);

        auto load_and_rehydrate_if_present = [&](const std::string& name) -> std::optional<Fp2d> {
            if (!nc.varExists(name)) {
                return std::nullopt;
            }
            Fp1d temp;
            nc.read(temp, name);
            // NOTE(cmo): This is a little inefficient, but eh.
            Fp2dHost hydrated = rehydrate_sparse_quantity(block_map, temp);
            return hydrated.createDeviceCopy();
        };
        auto ne = load_and_rehydrate_if_present("ne");
        if (ne) {
            atmos->ne = *ne;
        }
        auto nh_tot = load_and_rehydrate_if_present("nh_tot");
        if (nh_tot) {
            atmos->nh_tot = *nh_tot;
        }
    } else {
        if (nc.varExists("ne")) {
            nc.read(atmos->ne, "ne");
        }
        if (nc.varExists("nh_tot")) {
            nc.read(atmos->nh_tot, "nh_tot");
        }
    }
}

template <int mem_space=yakl::memDevice>
RaySet<mem_space> compute_ray_set(const RayConfig& cfg, const Atmosphere& atmos, int mu_idx) {
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
        int nz = atmos.temperature.extent(0);
        int nx = atmos.temperature.extent(1);
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
        Fp2dHost ray_pos("ray_starts", atmos.temperature.extent(1), 2);
        fp_t z_max = fp_t(atmos.temperature.extent(0));
        for (int i = 0; i < atmos.temperature.extent(1); ++i) {
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
    JasUnpack(state.state, chi_tau, active);
    auto& bc(state.bc);

    FlatAtmosphere<fp_t> flatmos = flatten(atmos);
    Fp2d flat_pops = pops.reshape(pops.extent(0), pops.extent(1) * pops.extent(2));
    // Fp2d flat_n_star = flat_pops.createDeviceObject();
    Fp2d flat_n_star = Fp2d("flat_n_star", flat_pops.extent(0), flat_pops.extent(1));
    Fp1d flat_eta = eta.collapse();
    Fp1d flat_chi = chi.collapse();
    auto flat_active = active.collapse();

    for (int wave = 0; wave < ray_set.wavelength.extent(0); ++wave) {
        parallel_for(
            "Compute eta, chi",
            SimpleBounds<1>(flatmos.temperature.extent(0)),
            YAKL_LAMBDA (i64 k) {
                if (!flat_active(k)) {
                    flat_eta(k) = FP(0.0);
                    flat_chi(k) = FP(0.0);
                    return;
                }
                fp_t lambda = ray_set.wavelength(wave);
                // NOTE(cmo): The projection vector is inverted here as we are
                // tracing front-to-back.
                fp_t v_proj = (
                    flatmos.vx(k) * -ray_set.mu(0)
                    + flatmos.vy(k) * -ray_set.mu(1)
                    + flatmos.vz(k) * -ray_set.mu(2)
                );
                // TODO(cmo): Tidy this up.
                const bool have_h = (adata.Z(0) == 1);
                fp_t nh0;
                if (have_h) {
                    nh0 = flat_pops(0, k);
                } else {
                    nh0 = nh_lte(flatmos.temperature(k), flatmos.ne(k), flatmos.nh_tot(k));
                }
                AtmosPointParams local_atmos {
                    .temperature = flatmos.temperature(k),
                    .ne = flatmos.ne(k),
                    .vturb = flatmos.vturb(k),
                    .nhtot = flatmos.nh_tot(k),
                    .vel = v_proj,
                    .nh0 = nh0
                };

                auto eta_chi = emis_opac(
                    EmisOpacSpecState<>{
                        .adata = adata,
                        .profile = phi,
                        .lambda = lambda,
                        .n = flat_pops,
                        .n_star_scratch = flat_n_star,
                        .k = k,
                        .atmos = local_atmos
                    }
                );

                flat_eta(k) = eta_chi.eta;
                flat_chi(k) = eta_chi.chi;
            }
        );
        yakl::fence();

        if (wave == 0 && (config.output_cfn || config.output_eta_chi)) {
            num_steps = std::remove_reference_t<decltype(num_steps)>(
                "num steps",
                ray_set.start_coord.extent(0)
            );
            i64 num_rays = ray_set.start_coord.extent(0);
            parallel_for(
                "Compute max steps",
                SimpleBounds<1>(num_rays),
                YAKL_LAMBDA (int ray_idx) {
                    vec2 start_pos;
                    start_pos(0) = ray_set.start_coord(ray_idx, 0);
                    start_pos(1) = ray_set.start_coord(ray_idx, 1);

                    vec2 flatland_mu;
                    fp_t flatland_mu_norm = std::sqrt(square(ray_set.mu(0)) + square(ray_set.mu(2)));
                    flatland_mu(0) = ray_set.mu(0) / flatland_mu_norm;
                    flatland_mu(1) = ray_set.mu(2) / flatland_mu_norm;

                    fp_t max_dim = std::max(fp_t(atmos.temperature.extent(0)), fp_t(atmos.temperature.extent(1)));
                    vec2 end_pos;
                    end_pos(0) = start_pos(0) + flatland_mu(0) * FP(10.0) * max_dim;
                    end_pos(1) = start_pos(1) + flatland_mu(1) * FP(10.0) * max_dim;

                    vec2 box_x;
                    box_x(0) = FP(0.0);
                    box_x(1) = fp_t(atmos.temperature.extent(1));
                    vec2 box_z;
                    box_z(0) = FP(0.0);
                    box_z(1) = fp_t(atmos.temperature.extent(0));

                    ivec2 domain_size;
                    domain_size(0) = atmos.temperature.extent(1);
                    domain_size(1) = atmos.temperature.extent(0);
                    RayMarchState2d s;
                    bool have_marcher = s.init(
                        start_pos,
                        end_pos,
                        domain_size
                    );

                    if (!have_marcher) {
                        num_steps(ray_idx) = 0;
                        return;
                    }

                    i64 step_count = 0;
                    do {
                        const auto& sample_coord(s.curr_coord);
                        if (sample_coord(0) < 0 || sample_coord(0) >= domain_size(0)) {
                            break;
                        }
                        if (sample_coord(1) < 0 || sample_coord(1) >= domain_size(1)) {
                            break;
                        }
                        step_count += 1;
                    } while (next_intersection(&s));

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

        parallel_for(
            "Trace Rays (front-to-back)",
            SimpleBounds<1>(ray_set.start_coord.extent(0)),
            YAKL_LAMBDA (int ray_idx) {
                vec2 start_pos;
                start_pos(0) = ray_set.start_coord(ray_idx, 0);
                start_pos(1) = ray_set.start_coord(ray_idx, 1);

                vec2 flatland_mu;
                fp_t flatland_mu_norm = std::sqrt(square(ray_set.mu(0)) + square(ray_set.mu(2)));
                flatland_mu(0) = ray_set.mu(0) / flatland_mu_norm;
                flatland_mu(1) = ray_set.mu(2) / flatland_mu_norm;

                fp_t max_dim = std::max(fp_t(atmos.temperature.extent(0)), fp_t(atmos.temperature.extent(1)));
                vec2 end_pos;
                end_pos(0) = start_pos(0) + flatland_mu(0) * FP(10.0) * max_dim;
                end_pos(1) = start_pos(1) + flatland_mu(1) * FP(10.0) * max_dim;

                vec2 box_x;
                box_x(0) = FP(0.0);
                box_x(1) = fp_t(atmos.temperature.extent(1));
                vec2 box_z;
                box_z(0) = FP(0.0);
                box_z(1) = fp_t(atmos.temperature.extent(0));

                ivec2 domain_size;
                domain_size(0) = atmos.temperature.extent(1);
                domain_size(1) = atmos.temperature.extent(0);
                RayMarchState2d s;
                bool have_marcher = s.init(
                    start_pos,
                    end_pos,
                    domain_size
                );

                if (!have_marcher) {
                    if (end_pos(1) < start_pos(1)) {
                        vec2 sample;
                        sample(0) = start_pos(0) * atmos.voxel_scale + atmos.offset_x;
                        sample(1) = start_pos(1) * atmos.voxel_scale + atmos.offset_z;
                        fp_t bc_I = sample_bc(bc, wave, sample, flatland_mu);
                        ray_I(wave, ray_idx) = bc_I;
                    } else {
                        ray_I(wave, ray_idx) = FP(0.0);
                    }
                    ray_tau(wave, ray_idx) = FP(0.0);
                    return;
                }

                fp_t I = FP(0.0);
                fp_t cumulative_tau = FP(0.0);

                const fp_t distance_factor = atmos.voxel_scale / std::sqrt(FP(1.0) - square(ray_set.mu(1)));

                i64 step_idx = 0;
                do {
                    const auto& sample_coord(s.curr_coord);
                    if (sample_coord(0) < 0 || sample_coord(0) >= domain_size(0)) {
                        break;
                    }
                    if (sample_coord(1) < 0 || sample_coord(1) >= domain_size(1)) {
                        break;
                    }
                    fp_t eta_s = eta(sample_coord(1), sample_coord(0));
                    fp_t chi_s = chi(sample_coord(1), sample_coord(0)) + FP(1e-20);
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
                } while (next_intersection(&s));

                // NOTE(cmo): Check BC
                if (s.p1(1) < s.p0(1)) {
                    vec2 sample;
                    sample(0) = s.p1(0) * atmos.voxel_scale + atmos.offset_x;
                    sample(1) = s.p1(1) * atmos.voxel_scale + atmos.offset_z;
                    fp_t bc_I = sample_bc(bc, wave, sample, flatland_mu);
                    I += std::exp(-cumulative_tau) * bc_I;
                }

                ray_I(wave, ray_idx) = I;
                ray_tau(wave, ray_idx) = cumulative_tau;
            }
        );
    }
}

yakl::SimpleNetCDF setup_output(const std::string& path, const RayConfig& cfg, const Atmosphere& atmos) {
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


int main(int argc, const char* argv[]) {
    argparse::ArgumentParser program("DexRT Ray");
    program.add_argument("--config")
        .default_value(std::string("dexrt_ray.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_epilog("Single-pass formal solver for post-processing Dex models.");

    program.parse_args(argc, argv);

    RayConfig config = parse_ray_config(program.get<std::string>("--config"));
    yakl::init(
        yakl::InitConfig()
            .set_pool_initial_mb(config.mem_pool_initial_gb * 1024)
            .set_pool_grow_mb(config.mem_pool_grow_gb * 1024)
    );
    {
        load_wavelength_if_missing(&config);
        if (config.dexrt.mode == DexrtMode::GivenFs) {
            throw std::runtime_error(fmt::format("Models run in GivenFs mode not supported by {}", argv[0]));
        }
        if (config.dexrt.boundary != BoundaryType::Promweaver) {
            throw std::runtime_error(fmt::format("Only promweaver boundaries are supported by {}", argv[0]));
        }
        Atmosphere atmos = load_atmos(config.dexrt.atmos_path);
        update_atmosphere(config.dexrt, &atmos);
        std::vector<ModelAtom<f64>> crtaf_models;
        // TODO(cmo): Override atoms in ray config
        crtaf_models.reserve(config.dexrt.atom_paths.size());
        for (auto p : config.dexrt.atom_paths) {
            crtaf_models.emplace_back(parse_crtaf_model<f64>(p));
        }
        AtomicDataHostDevice<fp_t> atomic_data = to_atomic_data<fp_t, f64>(crtaf_models);
        DexOutput model_output = load_dex_output(config.dexrt, atmos);

        DexRayState state{
            .atmos = atmos,
            .adata = atomic_data.device,
            .active = model_output.active,
            .phi = VoigtProfile<fp_t>(
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.15), 1024},
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(1.5e3), 64 * 1024}
            ),
            .nh_lte = HPartFn(),
            .pops = model_output.pops
        };


        state.eta = Fp2d(
            "eta",
            atmos.temperature.extent(0),
            atmos.temperature.extent(1)
        );
        state.chi = Fp2d(
            "chi",
            atmos.temperature.extent(0),
            atmos.temperature.extent(1)
        );

        auto out = setup_output(config.ray_output_path, config, atmos);

        for (int mu = 0; mu < config.muz.size(); ++mu) {
            state.ray_set = compute_ray_set<yakl::memDevice>(config, atmos, mu);
            // TODO(cmo): Hoist this if possible
            PwBc<> pw_bc = load_bc(
                config.dexrt.atmos_path,
                state.ray_set.wavelength,
                config.dexrt.boundary
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

    return 0;
}