#include "State3d.hpp"
#include "CascadeState3d.hpp"
#include "RcUtilsModes3d.hpp"
#include "StaticFormalSolution3d.hpp"
#include "DynamicFormalSolution3d.hpp"
#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include "YAKL_netcdf.h"
#include "Atmosphere.hpp"
#include "Populations.hpp"
#include "CrtafParser.hpp"
#include "MiscSparse.hpp"
#include "Collisions.hpp"
#include "ProfileNormalisation.hpp"
#include "ChargeConservation.hpp"
#include "PressureConservation.hpp"
#include "NgAcceleration.hpp"
#include "GitVersion.hpp"

void allocate_J(State3d* state) {
    JasUnpack((*state), config, mr_block_map, adata);
    const auto& block_map = mr_block_map.block_map;
    const bool sparse = config.sparse_calculation;
    i64 num_cells = mr_block_map.block_map.get_num_active_cells();
    i32 wave_dim = adata.wavelength.extent(0);
    if (config.mode == DexrtMode::GivenFs) {
        wave_dim = state->given_state.emis.extent(0);
    }

    if (!sparse) {
        num_cells = block_map.num_x_tiles() * block_map.num_y_tiles() * block_map.num_z_tiles() * cube(BLOCK_SIZE_3D);
    }

    if (config.store_J_on_cpu && config.mode != DexrtMode::GivenFs) {
        state->J = Fp2d("J", 1, num_cells);
        state->J_cpu = Fp2dHost("JHost", wave_dim, num_cells);
    } else {
        state->J = Fp2d("J", wave_dim, num_cells);
    }
    state->J = FP(0.0);
}

CascadeRays3d init_given_emis_opac(State3d* st, const DexrtConfig& config) {
    if (config.mode != DexrtMode::GivenFs) {
        return CascadeRays3d{};
    }
    yakl::SimpleNetCDF nc;
    nc.open(config.atmos_path, yakl::NETCDF_MODE_READ);
    int x_dim = nc.getDimSize("x");
    int y_dim = nc.getDimSize("y");
    int z_dim = nc.getDimSize("z");
    int wave_dim = nc.getDimSize("wavelength");
    fmt::println("{} {} {} {}", wave_dim, z_dim, y_dim, x_dim);

    Fp4d eta("eta", wave_dim, z_dim, y_dim, x_dim);
    Fp4d chi("chi", wave_dim, z_dim, y_dim, x_dim);
    nc.read(eta, "eta");
    nc.read(chi, "chi");

    f32 voxel_scale = FP(1.0);
    if (nc.varExists("voxel_scale")) {
        nc.read(voxel_scale, "voxel_scale");
    }
    st->given_state.voxel_scale = voxel_scale;
    st->atmos.voxel_scale = voxel_scale;
    fmt::println("Scale: {} m", st->given_state.voxel_scale);

    BlockMap<BLOCK_SIZE_3D, 3> block_map;
    block_map.init(Dims<3>{.x = x_dim, .y = y_dim, .z = z_dim});

    i32 max_mip_level = 0;
    for (int i = 0; i <= config.max_cascade; ++i) {
        max_mip_level = std::max(max_mip_level, config.mip_config.mip_levels[i]);
    }
    st->mr_block_map.init(block_map, max_mip_level);

    st->given_state.emis = eta;
    st->given_state.opac = chi;

    CascadeRays3d c0_rays;
    c0_rays.num_probes(0) = x_dim;
    c0_rays.num_probes(1) = y_dim;
    c0_rays.num_probes(2) = z_dim;
    c0_rays.num_az_rays = C0_AZ_RAYS_3D;
    c0_rays.num_polar_rays = C0_POLAR_RAYS_3D;

    return c0_rays;
}

CascadeRays3d init_atmos_atoms (State3d* st, const DexrtConfig& config) {
    if (!(config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte)) {
        return CascadeRays3d{};
    }

    State3d& state = *st;

    AtmosphereNd<3, yakl::memHost> atmos = load_atmos_3d_host(config.atmos_path);
    std::vector<ModelAtom<f64>> crtaf_models;
    crtaf_models.reserve(config.atom_paths.size());
    for (auto p : config.atom_paths) {
        crtaf_models.emplace_back(parse_crtaf_model<f64>(p));
    }
    AtomicDataHostDevice<fp_t> atomic_data = to_atomic_data<fp_t, f64>(crtaf_models);
    state.adata = atomic_data.device;
    state.adata_host = atomic_data.host;
    state.have_h = atomic_data.have_h_model;
    state.atoms = extract_atoms(atomic_data.device, atomic_data.host);
    GammaAtomsAndMapping gamma_atoms = extract_atoms_with_gamma_and_mapping(atomic_data.device, atomic_data.host);
    state.atoms_with_gamma = gamma_atoms.atoms;
    state.atoms_with_gamma_mapping = gamma_atoms.mapping;

    BlockMap<BLOCK_SIZE_3D, 3> block_map;
    block_map.init(atmos, config.threshold_temperature);
    i32 max_mip_level = 0;
    for (int i = 0; i <= config.max_cascade; ++i) {
        max_mip_level = std::max(max_mip_level, config.mip_config.mip_levels[i]);
    }
    if (state.config.mode != DexrtMode::GivenFs && LINE_SCHEME == LineCoeffCalc::Classic) {
        max_mip_level = 0;
        state.println("Mips not supported with LineCoeffCalc::Classic");
    }
    state.mr_block_map.init(block_map, max_mip_level);

    state.atmos = sparsify_atmosphere(atmos, block_map);

    state.phi = VoigtProfile<fp_t>(
        VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.4), 1024},
        VoigtProfile<fp_t>::Linspace{FP(0.0), FP(3e3), 64 * 1024}
    );
    state.nh_lte = HPartFn();
    state.println("Scale: {} m", state.atmos.voxel_scale);

    i64 num_active_cells = block_map.get_num_active_cells();

    const int n_level_total = state.adata.energy.extent(0);
    state.pops = Fp2d("pops", n_level_total, num_active_cells);

    // if (config.mode == DexrtMode::NonLte) {
        for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
            const int n_level = state.adata_host.num_level(ia);
            state.Gamma.emplace_back(
                decltype(state.Gamma)::value_type("Gamma", n_level, n_level, num_active_cells)
            );
        }
        state.wphi = Fp2d("wphi", state.adata.lines.extent(0), num_active_cells);
    // }

    // NOTE(cmo): We just have one of these chained for each boundary type -- they don't do anything if this configuration doesn't need them to.
    state.pw_bc = load_bc(config.atmos_path, state.adata.wavelength, config.boundary);
    state.boundary = config.boundary;

    // NOTE(cmo): This doesn't actually know that things will be allocated sparse
    CascadeRays3d c0_rays;
    const auto space_dims = atmos.temperature.get_dimensions();
    c0_rays.num_probes(0) = space_dims(2);
    c0_rays.num_probes(1) = space_dims(1);
    c0_rays.num_probes(2) = space_dims(0);
    c0_rays.num_az_rays = C0_AZ_RAYS_3D;
    c0_rays.num_polar_rays = C0_POLAR_RAYS_3D;

    // state.max_block_mip = decltype(state.max_block_mip)(
    //     "max_block_mip",
    //     (state.adata.wavelength.extent(0) + c0_rays.wave_batch - 1) / c0_rays.wave_batch,
    //     block_map.num_z_tiles(),
    //     block_map.num_x_tiles()
    // );
    return c0_rays;
}

void init_state(State3d* state, const DexrtConfig& config) {
    state->config = config;
    CascadeRays3d c0_rays;
    if (config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte) {
        c0_rays = init_atmos_atoms(state, config);
    } else {
        c0_rays = init_given_emis_opac(state, config);
    }

    constexpr int RcMode = RC_flags_storage_3d();
    state->c0_size = cascade_rays_to_storage<RcMode>(c0_rays);

    allocate_J(state);
}

/// Add Dex's metadata to the file using attributes. The netcdf layer needs extending to do this, so I'm just throwing it in manually.
void add_netcdf_attributes(const State3d& state, const yakl::SimpleNetCDF& file, i32 num_iter) {
    const auto ncwrap = [&] (int ierr, int line) {
        if (ierr != NC_NOERR) {
            state.println("NetCDF Error writing attributes at main.cpp:{}", line);
            state.println("{}",nc_strerror(ierr));
            yakl::yakl_throw(nc_strerror(ierr));
        }
    };
    int ncid = file.file.ncid;
    if (ncid == -999) {
        throw std::runtime_error("File appears to have been closed before writing attributes!");
    }

    std::string name = "dexrt (3d)";
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "program", name.size(), name.c_str()),
        __LINE__
    );

    std::string precision = "f64";
#ifdef DEXRT_SINGLE_PREC
    precision = "f32";
#endif
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "precision", precision.size(), precision.c_str()),
        __LINE__
    );

    f64 probe0_length = C0_LENGTH_3D;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "probe0_length", NC_DOUBLE, 1, &probe0_length),
        __LINE__
    );
    i32 probe0_az_rays = C0_AZ_RAYS_3D;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "probe0_az_rays", NC_INT, 1, &probe0_az_rays),
        __LINE__
    );
    i32 probe0_polar_rays = C0_POLAR_RAYS_3D;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "probe0_polar_rays", NC_INT, 1, &probe0_polar_rays),
        __LINE__
    );
    i32 probe0_spacing = 1;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "probe0_spacing", NC_INT, 1, &probe0_spacing),
        __LINE__
    );
    i32 az_cascade_branching = USE_BRANCHING_FACTOR_3D ? AZ_BRANCHING_FACTOR_3D : (1 << AZ_BRANCHING_EXP_3D);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "az_branching_factor", NC_INT, 1, &az_cascade_branching),
        __LINE__
    );
    i32 polar_cascade_branching = USE_BRANCHING_FACTOR_3D ? POLAR_BRANCHING_FACTOR_3D : (1 << POLAR_BRANCHING_EXP_3D);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "polar_branching_factor", NC_INT, 1, &polar_cascade_branching),
        __LINE__
    );
    i32 interval_scale_factor = USE_SCALE_FACTOR_3D ? SPATIAL_SCALE_FACTOR_3D : (1 << SPATIAL_SCALE_EXP_3D);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "interval_scale_factor", NC_INT, 1, &interval_scale_factor),
        __LINE__
    );
    std::string ang_quad_type(AngularQuadratureNames[int(ANGULAR_QUADRATURE_TYPE)]);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "angular_quadrature", ang_quad_type.size(), ang_quad_type.c_str()),
        __LINE__
    );
    if (ANGULAR_QUADRATURE_TYPE == AngularQuadratureType::Healpix) {
        i32 healpix_order = HEALPIX_ORDER;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "healpix_order", NC_INT, 1, &healpix_order),
            __LINE__
        );
    }
    i32 max_cascade = state.config.max_cascade;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "max_cascade", NC_INT, 1, &max_cascade),
        __LINE__
    );
    i32 last_casc_to_inf = LAST_CASCADE_TO_INFTY_3D;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "last_casc_to_infty", NC_INT, 1, &last_casc_to_inf),
        __LINE__
    );
    f64 last_casc_dist = LAST_CASCADE_MAX_DIST_3D;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "last_cascade_max_distance", NC_DOUBLE, 1, &last_casc_dist),
        __LINE__
    );
    i32 dir_by_dir = DIR_BY_DIR_3D;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "dir_by_dir", NC_INT, 1, &dir_by_dir),
        __LINE__
    );
    i32 pingpong = PINGPONG_BUFFERS;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "pingpong_buffers", NC_INT, 1, &pingpong),
        __LINE__
    );
    i32 store_tau_cascades = STORE_TAU_CASCADES;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "store_tau_cascades", NC_INT, 1, &store_tau_cascades),
        __LINE__
    );
    f64 thermal_vel_frac = ANGLE_INVARIANT_THERMAL_VEL_FRAC;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "angle_invariant_thermal_vel_frac", NC_DOUBLE, 1, &thermal_vel_frac),
        __LINE__
    );
    i32 conserve_pressure_nr = true;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "conserve_pressure_nr", NC_INT, 1, &conserve_pressure_nr),
        __LINE__
    );

    i32 num_atom = state.adata_host.num_level.extent(0);
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_atom", NC_INT, 1, &num_atom),
        __LINE__
    );
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_level", NC_INT, num_atom, state.adata_host.num_level.get_data()),
        __LINE__
    );
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_line", NC_INT, state.adata_host.num_line.extent(0), state.adata_host.num_line.get_data()),
        __LINE__
    );
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "line_start", NC_INT, state.adata_host.line_start.extent(0), state.adata_host.line_start.get_data()),
        __LINE__
    );
    yakl::Array<f64, 1, yakl::memHost> lambda0("lambda0", state.adata_host.lines.extent(0));
    for (int i = 0; i < lambda0.extent(0); ++i) {
        lambda0(i) = state.adata_host.lines(i).lambda0;
    }
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "lambda0", NC_DOUBLE, lambda0.extent(0), lambda0.get_data()),
        __LINE__
    );

    // NOTE(cmo): Hack to save timing data. These functions only print to stdout -- want to redirect that.
    auto cout_buf = std::cout.rdbuf();
    std::ostringstream timer_buffer;
    std::cout.rdbuf(timer_buffer.rdbuf());
    yakl::timer_finalize();
    std::cout.rdbuf(cout_buf);
    std::string timer_data = timer_buffer.str();
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "timing", timer_data.size(), timer_data.c_str()),
        __LINE__
    );
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_iter", NC_INT, 1, &num_iter),
        __LINE__
    );

    std::string output_format = state.config.output.sparse ? "sparse" : "full";
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "output_format", output_format.size(), output_format.c_str()),
        __LINE__
    );
    i32 final_dense_fs = state.config.final_dense_fs;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "final_dense_fs", NC_INT, 1, &final_dense_fs),
        __LINE__
    );

    std::string line_scheme_name(LineCoeffCalcNames[int(LINE_SCHEME_3D)]);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "line_calculation_scheme", line_scheme_name.size(), line_scheme_name.c_str()),
        __LINE__
    );

    i32 block_size = BLOCK_SIZE_3D;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "block_size", NC_INT, 1, &block_size),
        __LINE__
    );
    i32 nx_blocks = state.mr_block_map.block_map.num_x_tiles();
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_x_blocks", NC_INT, 1, &nx_blocks),
        __LINE__
    );
    i32 ny_blocks = state.mr_block_map.block_map.num_y_tiles();
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_y_blocks", NC_INT, 1, &nx_blocks),
        __LINE__
    );
    i32 nz_blocks = state.mr_block_map.block_map.num_z_tiles();
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_z_blocks", NC_INT, 1, &nz_blocks),
        __LINE__
    );

    // if constexpr (LINE_SCHEME_3D == LineCoeffCalc::VelocityInterp) {
    //     i32 interp_bins = INTERPOLATE_DIRECTIONAL_BINS;
    //     ncwrap(
    //         nc_put_att_int(ncid, NC_GLOBAL, "interpolate_directional_bins", NC_INT, 1, &interp_bins),
    //         __LINE__
    //     );

    //     f64 interp_max_width = INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH;
    //     ncwrap(
    //         nc_put_att_double(ncid, NC_GLOBAL, "interpolate_direction_max_thermal_width", NC_DOUBLE, 1, &interp_max_width),
    //         __LINE__
    //     );
    // }

    ncwrap(
        nc_put_att_int(
            ncid, NC_GLOBAL, "mip_levels", NC_INT,
            state.config.max_cascade+1,
            state.config.mip_config.mip_levels.data()
        ),
        __LINE__
    );

    std::string git_hash(GIT_HASH);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "git_hash", git_hash.size(), git_hash.c_str()),
        __LINE__
    );

    f64 voxel_scale = state.atmos.voxel_scale;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "voxel_scale", NC_DOUBLE, 1, &voxel_scale),
        __LINE__
    );

    const auto& config_path(state.config.own_path);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "config_path", config_path.size(), config_path.c_str()),
        __LINE__
    );
}

void save_results(const State3d& state, const CascadeState3d& casc_state, i32 num_iter) {
    const auto& config = state.config;
    const auto& out_cfg = config.output;

    yakl::SimpleNetCDF nc;
    nc.create(config.output_path, yakl::NETCDF_MODE_REPLACE);
    fmt::println("Saving output to {}...", config.output_path);

    add_netcdf_attributes(state, nc, num_iter);
    const auto& block_map = state.mr_block_map.block_map;
    bool sparse_J = (state.J.extent(1) == state.atmos.temperature.extent(0));

    auto maybe_rehydrate_and_write = [&](
        auto arr,
        const std::string& name,
        std::vector<std::string> leading_dim_names
    ) {
        auto& dim_names = leading_dim_names;
        if (out_cfg.sparse) {
            dim_names.insert(dim_names.end(), {"ks"});
            nc.write(arr, name, dim_names);
        } else {
            auto hydrated = rehydrate_sparse_quantity(block_map, arr);
            dim_names.insert(dim_names.end(), {"z", "y", "x"});
            nc.write(hydrated, name, dim_names);
        }
    };

    if (out_cfg.J) {
        if (config.store_J_on_cpu) {
            if (sparse_J) {
                maybe_rehydrate_and_write(state.J_cpu, "J", {"wavelength"});
            } else {
                nc.write(
                    state.J_cpu.reshape(
                        state.J_cpu.extent(0),
                        state.atmos.num_z,
                        state.atmos.num_y,
                        state.atmos.num_x
                    ),
                    "J",
                    {"wavelength", "z", "y", "x"}
                );
            }
        } else {
            if (sparse_J) {
                maybe_rehydrate_and_write(state.J, "J", {"wavelength"});
            } else {
                nc.write(
                    state.J.reshape(
                        state.J.extent(0),
                        state.atmos.num_z,
                        state.atmos.num_y,
                        state.atmos.num_x
                    ),
                    "J",
                    {"wavelength", "z", "y", "x"}
                );
            }
        }
    }
    if (out_cfg.wavelength && state.adata.wavelength.initialized()) {
        nc.write(state.adata.wavelength, "wavelength", {"wavelength"});
    }
    if (out_cfg.pops && state.pops.initialized()) {
        maybe_rehydrate_and_write(state.pops, "pops", {"level"});
    }
    if (out_cfg.lte_pops) {
        Fp2d lte_pops = state.pops.createDeviceObject();
        compute_lte_pops(&state, lte_pops);
        yakl::fence();
        maybe_rehydrate_and_write(lte_pops, "lte_pops", {"level"});
    }
    if (out_cfg.ne && state.atmos.ne.initialized()) {
        maybe_rehydrate_and_write(state.atmos.ne, "ne", {});
    }
    if (out_cfg.nh_tot && state.atmos.nh_tot.initialized()) {
        maybe_rehydrate_and_write(state.atmos.nh_tot, "nh_tot", {});
    }
    if (out_cfg.alo && casc_state.alo.initialized()) {
        nc.write(casc_state.alo, "alo", {"casc_shape"});
    }
    for (int casc : out_cfg.cascades) {
        // NOTE(cmo): The validity of these + necessary warning were checked/output in the config parsing step
        std::string name = fmt::format("I_C{}", casc);
        std::string shape = fmt::format("casc_shape_{}", casc);
        nc.write(casc_state.i_cascades[casc], name, {shape});
        if constexpr (STORE_TAU_CASCADES) {
            name = fmt::format("tau_C{}", casc);
            nc.write(casc_state.tau_cascades[casc], name, {shape});
        }
    }
    if (out_cfg.sparse) {
        nc.write(block_map.active_tiles, "morton_tiles", {"num_active_tiles"});
    }
    nc.close();
}

void copy_J_plane_to_host(const State3d& state, int la) {
    const Fp2dHost J_copy = state.J.createHostCopy();
    // TODO(cmo): Replace with a memcpy?
    for (i64 ks = 0; ks < J_copy.extent(1); ++ks) {
        state.J_cpu(la, ks) = J_copy(0, ks);
    }
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("DexRT 3D");
    program.add_argument("--config")
        .default_value(std::string("dexrt.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_argument("--restart-from")
        .nargs(1)
        .help("Path to snapshot file")
        .metavar("FILE");
    program.add_epilog("DexRT 3D Radiance Cascade based non-LTE solver.");

    program.parse_args(argc, argv);
    const DexrtConfig config = parse_dexrt_config(program.get<std::string>("--config"));

    Kokkos::initialize();
    yakl::init(
        yakl::InitConfig()
            .set_pool_size_mb(config.mem_pool_gb * 1024)
    );
    {
        yakl::timer_start("DexRT");
        State3d state;
        init_state(&state, config);

        CascadeState3d casc_state;
        casc_state.init(state, config.max_cascade);

        int num_iter = 1;
        if (config.mode == DexrtMode::GivenFs) {
            static_formal_sol_rc_given_3d(state, casc_state);
        } else {
            compute_lte_pops(&state);
            const bool non_lte = config.mode == DexrtMode::NonLte;
            const fp_t non_lte_tol = config.pop_tol;
            fp_t max_change = FP(1.0);

            const bool do_restart = false;
            const bool conserve_charge = config.conserve_charge;
            const bool actually_conserve_charge = state.have_h && conserve_charge;
            if (!actually_conserve_charge && conserve_charge) {
                throw std::runtime_error("Charge conservation enabled without a model H!");
            }
            const bool conserve_pressure = config.conserve_pressure;
            if (conserve_pressure && !conserve_charge) {
                throw std::runtime_error("Cannot enable pressure conservation without charge conservation.");
            }
            const bool actually_conserve_pressure = actually_conserve_charge && conserve_pressure;
            const int initial_lambda_iterations = config.initial_lambda_iterations;
            const int max_iters = config.max_iter;

            int i = 0;

            if (actually_conserve_charge && !do_restart) {
                // TODO(cmo): Make all of these parameters configurable
                state.println("-- Iterating LTE n_e/pressure --");
                fp_t lte_max_change = FP(1.0);
                int lte_i = 0;
                while ((lte_max_change > FP(1e-5) || lte_i < 6) && lte_i < max_iters) {
                    lte_i += 1;
                    compute_nh0(state);
                    compute_collisions_to_gamma(&state);
                    lte_max_change = stat_eq(&state, StatEqOptions{
                        .ignore_change_below_ntot_frac=FP(1e-7)
                    });
                    if (lte_i < 2) {
                        continue;
                    }
                    // NOTE(cmo): Ignore what the lte_change actually is
                    // from stat eq... it will "converge" essentially
                    // instantly due to linearity, so whilst the error may
                    // be above a threshold, it's unlikely to get
                    // meaningfully better after the second iteration
                    fp_t nr_update = nr_post_update(&state, NrPostUpdateOptions{
                        .ignore_change_below_ntot_frac = FP(1e-7),
                        .conserve_pressure = false
                    });
                    lte_max_change = nr_update;
                    if (actually_conserve_pressure) {
                        fp_t nh_tot_update = simple_conserve_pressure(&state);
                        lte_max_change = std::max(nh_tot_update, lte_max_change);
                    }
                }
                state.println("Ran for {} iterations", lte_i);
            }

            // if (do_restart) {
            //     i = handle_restart(&state, *restart_path);
            // }

            NgAccelerator ng;
            if (config.ng.enable) {
                ng.init(
                    NgAccelArgs{
                        .num_level=(i64)state.pops.extent(0),
                        .num_space=(i64)state.pops.extent(1),
                        .accel_tol=config.ng.threshold,
                        .lower_tol=config.ng.lower_threshold
                    }
                );
                ng.accelerate(state, FP(1.0));
            }
            bool accelerated = false;

            state.println("-- Non-LTE Iterations --");
            while (((max_change > non_lte_tol || i < (initial_lambda_iterations+1)) && i < max_iters) || accelerated) {
                state.println("==== FS {} ====", i);
                compute_nh0(state);
                compute_collisions_to_gamma(&state);
                compute_profile_normalisation(state, casc_state, true);
                state.J = FP(0.0);
                if (config.store_J_on_cpu) {
                    state.J_cpu = FP(0.0);
                }
                Kokkos::fence();
                for (int la = 0; la < state.adata_host.wavelength.extent(0); ++la) {
                    if (config.store_J_on_cpu) {
                        state.J = FP(0.0);
                        Kokkos::fence();
                    }
                    bool lambda_iterate = i < initial_lambda_iterations;
                    dynamic_formal_sol_rc_3d(state, casc_state, lambda_iterate, la);
                    if (config.store_J_on_cpu) {
                        copy_J_plane_to_host(state, la);
                    }
                }
                if (!non_lte) {
                    break;
                }
                state.println("  == Statistical equilibrium ==");
                max_change = stat_eq(
                    &state,
                    StatEqOptions{
                        .ignore_change_below_ntot_frac=std::min(FP(1e-6), non_lte_tol)
                    }
                );
                if (i > 0 && actually_conserve_charge) {
                    fp_t nr_update = nr_post_update(
                        &state,
                        NrPostUpdateOptions{
                            .ignore_change_below_ntot_frac = std::min(FP(1e-6), non_lte_tol),
                            .conserve_pressure = actually_conserve_pressure
                        }
                    );
                    // wave_dist.update_ne(&state);
                    max_change = std::max(nr_update, max_change);
                    // if (actually_conserve_pressure) {
                    //     wave_dist.update_nh_tot(&state);
                    // }
                }
                if (config.ng.enable) {
                    accelerated = ng.accelerate(state, max_change);
                    if (accelerated) {
                        state.println("  ~~ Ng Acceleration! (📉 or 💣) ~~");
                    }
                }
                i += 1;
            }
            if (state.config.mode == DexrtMode::NonLte && state.config.sparse_calculation && state.config.final_dense_fs) {
                state.config.sparse_calculation = false;
                allocate_J(&state);
                casc_state.probes_to_compute.init(state, casc_state.num_cascades);

                state.println("Final FS (dense)");
                compute_nh0(state);
                state.J = FP(0.0);
                if (config.store_J_on_cpu) {
                    state.J_cpu = FP(0.0);
                }
                Kokkos::fence();
                for (int la = 0; la < state.adata_host.wavelength.extent(0); ++la) {
                    if (config.store_J_on_cpu) {
                        state.J = FP(0.0);
                        Kokkos::fence();
                    }
                    bool lambda_iterate = true;
                    dynamic_formal_sol_rc_3d(state, casc_state, lambda_iterate, la);
                    if (config.store_J_on_cpu) {
                        copy_J_plane_to_host(state, la);
                    }
                }
                state.config.sparse_calculation = true;
            }
            num_iter = i;
        }
        save_results(state, casc_state, num_iter);
        yakl::timer_stop("DexRT");
    }
    yakl::finalize();
    Kokkos::finalize();
    return 0;
}