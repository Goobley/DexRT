#include <math.h>
#include <limits>
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include "Populations.hpp"
#include "CrtafParser.hpp"
#include "Collisions.hpp"
#include "Voigt.hpp"
#include "NgAcceleration.hpp"
#include "StaticFormalSolution.hpp"
#include "DynamicFormalSolution.hpp"
#include "ChargeConservation.hpp"
#include "PressureConservation.hpp"
#include "ProfileNormalisation.hpp"
#include "PromweaverBoundary.hpp"
#include "DexrtConfig.hpp"
#include "BlockMap.hpp"
#include "MiscSparse.hpp"
#include "YAKL_netcdf.h"
#include <vector>
#include <string>
#include <optional>
#include <fmt/core.h>
#include <argparse/argparse.hpp>
#include <sstream>
#include "GitVersion.hpp"
#include "WavelengthParallelisation.hpp"

#include <random>

void allocate_J(State* state) {
    JasUnpack((*state), config, mr_block_map, c0_size, adata);
    const auto& block_map = mr_block_map.block_map;
    const bool sparse = config.sparse_calculation;
    i64 num_cells = mr_block_map.block_map.get_num_active_cells();
    i32 wave_dim = adata.wavelength.extent(0);
    if (config.mode == DexrtMode::GivenFs) {
        wave_dim = state->given_state.emis.extent(2);
    }

    if (!sparse) {
        num_cells = block_map.num_x_tiles() * block_map.num_z_tiles() * square(BLOCK_SIZE);
    }

    if (config.store_J_on_cpu) {
        state->J = Fp2d("J", c0_size.wave_batch, num_cells);
        state->J_cpu = Fp2dHost("JHost", wave_dim, num_cells);
    } else {
        state->J = Fp2d("J", wave_dim, num_cells);
    }
    state->J = FP(0.0);
    // TODO(cmo): If we have scattering terms and are updating J, the old
    // contents should probably be moved first, but we don't have these terms yet.
}

CascadeRays init_atmos_atoms (State* st, const DexrtConfig& config) {
    if (!(config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte)) {
        return CascadeRays{};
    }

    State& state = *st;

    Atmosphere atmos = load_atmos(config.atmos_path);
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

    BlockMap<BLOCK_SIZE> block_map;
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

    if (config.mode == DexrtMode::NonLte) {
        for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
            const int n_level = state.adata_host.num_level(ia);
            state.Gamma.emplace_back(
                decltype(state.Gamma)::value_type("Gamma", n_level, n_level, num_active_cells)
            );
        }
        state.wphi = Fp2d("wphi", state.adata.lines.extent(0), num_active_cells);
    }

    // NOTE(cmo): We just have one of these chained for each boundary type -- they don't do anything if this configuration doesn't need them to.
    state.pw_bc = load_bc(config.atmos_path, state.adata.wavelength, config.boundary);
    state.boundary = config.boundary;

    // NOTE(cmo): This doesn't actually know that things will be allocated sparse
    CascadeRays c0_rays;
    const auto space_dims = atmos.temperature.get_dimensions();
    c0_rays.num_probes(0) = space_dims(1);
    c0_rays.num_probes(1) = space_dims(0);
    c0_rays.num_flat_dirs = PROBE0_NUM_RAYS;
    c0_rays.num_incl = NUM_INCL;
    c0_rays.wave_batch = WAVE_BATCH;

    state.max_block_mip = decltype(state.max_block_mip)(
        "max_block_mip",
        (state.adata.wavelength.extent(0) + c0_rays.wave_batch - 1) / c0_rays.wave_batch,
        block_map.num_z_tiles(),
        block_map.num_x_tiles()
    );
    return c0_rays;
}

CascadeRays init_given_emis_opac(State* st, const DexrtConfig& config) {
    if (config.mode != DexrtMode::GivenFs) {
        return CascadeRays{};
    }
    yakl::SimpleNetCDF nc;
    nc.open(config.atmos_path, yakl::NETCDF_MODE_READ);
    int x_dim = nc.getDimSize("x");
    int z_dim = nc.getDimSize("z");
    int wave_dim = nc.getDimSize("wavelength");

    typedef yakl::Array<f32, 3, yakl::memHost> Fp3dLoad;

    Fp3dLoad eta("eta", z_dim, x_dim, wave_dim);
    Fp3dLoad chi("chi", z_dim, x_dim, wave_dim);
    nc.read(eta, "eta");
    nc.read(chi, "chi");

    f32 voxel_scale = FP(1.0);
    if (nc.varExists("voxel_scale")) {
        nc.read(voxel_scale, "voxel_scale");
    }
    st->atmos.voxel_scale = voxel_scale;
    st->given_state.voxel_scale = voxel_scale;
    st->println("Scale: {} m", st->atmos.voxel_scale);
    BlockMap<BLOCK_SIZE> block_map;
    block_map.init(Dims<2>{.x = x_dim, .z = z_dim});
    i32 max_mip_level = 0;
    for (int i = 0; i <= config.max_cascade; ++i) {
        max_mip_level = std::max(max_mip_level, config.mip_config.mip_levels[i]);
    }
    // NOTE(cmo): Everything is assumed active
    st->mr_block_map.init(block_map, max_mip_level);

#ifdef DEXRT_SINGLE_PREC
    st->given_state.emis = eta.createDeviceCopy();
    st->given_state.opac = chi.createDeviceCopy();
#else
    auto etad = eta.createDeviceCopy();
    auto chid = chi.createDeviceCopy();
    Fp3d emis("eta", z_dim, x_dim, wave_dim);
    Fp3d opac("chi", z_dim, x_dim, wave_dim);
    yakl::fence();
    dex_parallel_for(
        "Convert f32->f64",
        FlatLoop<3>(z_dim, x_dim, wave_dim),
        YAKL_LAMBDA (int z, int x, int la) {
            emis(z, x, la) = etad(z, x, la);
        }
    );
    dex_parallel_for(
        "Convert f32->f64",
        FlatLoop<3>(z_dim, x_dim, wave_dim),
        YAKL_LAMBDA (int z, int x, int la) {
            opac(z, x, la) = chid(z, x, la);
        }
    );
    yakl::fence();
    st->given_state.emis = emis;
    st->given_state.opac = opac;
#endif

    // NOTE(cmo): Only zero boundaries are supported here.
    st->boundary = BoundaryType::Zero;
    CascadeRays c0_rays;
    c0_rays.num_probes(0) = x_dim;
    c0_rays.num_probes(1) = z_dim;
    c0_rays.num_flat_dirs = PROBE0_NUM_RAYS;
    c0_rays.num_incl = NUM_INCL;
    c0_rays.wave_batch = WAVE_BATCH;

    st->max_block_mip = decltype(st->max_block_mip)(
        "max_block_mip",
        (wave_dim + c0_rays.wave_batch - 1) / c0_rays.wave_batch,
        block_map.num_z_tiles(),
        block_map.num_x_tiles()
    );
    yakl::fence();
    return c0_rays;
}

void init_cascade_sized_arrays(State* state, const DexrtConfig& config) {
}

void init_state (State* state, const DexrtConfig& config) {
    state->config = config;
    setup_comm(state);

    CascadeRays c0_rays;
    if (config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte) {
        c0_rays = init_atmos_atoms(state, config);
    } else {
        c0_rays = init_given_emis_opac(state, config);
    }

    constexpr int RcMode = RC_flags_storage_2d();
    state->c0_size = cascade_rays_to_storage<RcMode>(c0_rays);

    allocate_J(state);
    init_cascade_sized_arrays(state, config);

    Fp1dHost muy("muy", NUM_INCL);
    Fp1dHost wmuy("wmuy", NUM_INCL);
    for (int i = 0; i < NUM_INCL; ++i) {
        muy(i) = INCL_RAYS[i];
        wmuy(i) = INCL_WEIGHTS[i];
    }
    state->incl_quad.muy = muy.createDeviceCopy();
    state->incl_quad.wmuy = wmuy.createDeviceCopy();

#ifdef DEXRT_USE_MAGMA
    magma_init();
    magma_queue_create(0, &state->magma_queue);
#endif
}

void finalize_state(State* state) {
#ifdef DEXRT_USE_MAGMA
    magma_queue_destroy(state->magma_queue);
    magma_finalize();
#endif
}

/// Loads the populations, ne, and nh_tot from restart_path. Returns the current
/// iteration number.  Will crash/throw exception if file can't be found or
/// dimensions are wrong etc.
int handle_restart(State* st, const std::string& restart_path) {
    State& state(*st);
    yakl::SimpleNetCDF nc;
    nc.open(restart_path, yakl::NETCDF_MODE_READ);

    i64 num_active = nc.getDimSize("ks");
    i64 num_level = nc.getDimSize("level");
    if (num_active != state.atmos.temperature.extent(0)) {
        throw std::runtime_error("Restart atmosphere size does not match loaded size, have you changed a parameter (e.g. threshold_temperature)?");
    }
    if (num_level != state.pops.extent(0)) {
        throw std::runtime_error("Restart number of levels does not match size set in config. Have the models changed?");
    }

    nc.read(state.pops, "pops");
    if (nc.varExists("ne")) {
        nc.read(state.atmos.ne, "ne");
    }
    if (nc.varExists("nh_tot")) {
        nc.read(state.atmos.nh_tot, "nh_tot");
    }

    int ncid = nc.file.ncid;
    int num_iter = 0;
    int ierr = nc_get_att_int(ncid, NC_GLOBAL, "num_iter", &num_iter);
    if (ierr != NC_NOERR) {
        throw std::runtime_error(fmt::format("Unable to get num_iter from restart file: {}", nc_strerror(ierr)));
    }

    return num_iter;
}

/// Dump a snapshot. File name determined automatically (e.g. main output
/// dexrt_output.nc -> dexrt_output_snapshot.nc)
void save_snapshot(const State& state, int num_iter) {
    yakl::SimpleNetCDF nc;
    std::string name(state.config.output_path);
    std::string ext(".nc");
    std::string new_ext("_snapshot.nc");
    auto nc_pos = name.rfind(ext);
    if (nc_pos == std::string::npos) {
        // NOTE(cmo): Didn't find it, so just append
        name += new_ext;
    } else {
        name.replace(nc_pos, ext.size(), new_ext);
    }

    nc.create(name, yakl::NETCDF_MODE_REPLACE);
    state.println("Saving snapshot to {}...", name);

    int ncid = nc.file.ncid;
    int ierr = nc_put_att_int(ncid, NC_GLOBAL, "num_iter", NC_INT, 1, &num_iter);
    if (ierr != NC_NOERR) {
        throw std::runtime_error(fmt::format("Unable to write num_iter to snapshot file: {}", nc_strerror(ierr)));
    }

    nc.write(state.pops, "pops", {"level", "ks"});
    if (state.config.conserve_charge) {
        nc.write(state.atmos.ne, "ne", {"ks"});
    }
    if (state.config.conserve_pressure) {
        nc.write(state.atmos.nh_tot, "nh_tot", {"ks"});
    }
}

/// Load populations from the specified path (variable name "pops"). Will be
/// assumed to be sparse if the ks dimension exists (e.g. from a previous run),
/// or dense otherwise.
void load_initial_pops(State* st, const std::string& initial_pops_path) {
    State& state(*st);
    yakl::SimpleNetCDF nc;
    nc.open(initial_pops_path, yakl::NETCDF_MODE_READ);

    bool load_sparse = nc.dimExists("ks");
    state.println(
        "Loading populations from {}, appear to be {}.",
        initial_pops_path,
        load_sparse ? "sparse" : "dense"
    );

    i64 num_level = nc.getDimSize("level");
    if (num_level != state.pops.extent(0)) {
        throw std::runtime_error("Restart number of levels does not match size set in config. Have the models changed?");
    }

    if (load_sparse) {
        i64 num_active = nc.getDimSize("ks");
        if (num_active != state.pops.extent(1)) {
            throw std::runtime_error("Initial pops spatial dimension (ks) does not match loaded size, have you changed a parameter (e.g. threshold_temperature)?");
        }
        nc.read(state.pops, "pops");
    } else {
        i64 nx = nc.getDimSize("x");
        i64 nz = nc.getDimSize("z");

        if (nx != state.atmos.num_x || nz != state.atmos.num_z) {
            throw std::runtime_error(
                fmt::format(
                    "Initial pops file dimensions [x: {}, z: {}] do not match atmos: [{}, {}]",
                    nx,
                    nz,
                    state.atmos.num_x,
                    state.atmos.num_z
                )
            );
        }
        Fp3d temp_pops("temp_pops", num_level, nz, nx);
        nc.read(temp_pops, "pops");

        JasUnpack(state, mr_block_map, pops);
        auto bounds = mr_block_map.block_map.loop_bounds();
        dex_parallel_for(
            "Sparsify new pops",
            FlatLoop<3>(
                num_level,
                bounds.dim(0),
                bounds.dim(1)
            ),
            YAKL_LAMBDA (i32 level, i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);

                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
                pops(level, ks) = temp_pops(level, coord.z, coord.x);
            }
        );
        yakl::fence();
    }
}

/// Called to copy J from GPU to plane of host array if config.store_J_on_cpu
void copy_J_plane_to_host(const State& state, int la_start, int la_end) {
    int wave_batch = la_end - la_start;
    const Fp2dHost J_copy = state.J.createHostCopy();
    // TODO(cmo): Replace with a memcpy?
    for (int wave = 0; wave < wave_batch; ++wave) {
        for (i64 ks = 0; ks < J_copy.extent(1); ++ks) {
            state.J_cpu(la_start + wave, ks) = J_copy(wave, ks);
        }
    }
}

void setup_wavelength_batch(const State& state, int la_start, int la_end) {
    if (state.config.store_J_on_cpu) {
        state.J = FP(0.0);
        yakl::fence();
    }
}

void finalise_wavelength_batch(const State& state, int la_start, int la_end) {
    if (state.config.store_J_on_cpu) {
        copy_J_plane_to_host(state, la_start, la_end);
    }

    const i32 wave_batch_idx = la_start / state.c0_size.wave_batch;
    JasUnpack(state, max_block_mip, mr_block_map);
    dex_parallel_for(
        "Copy max mip",
        FlatLoop<1>(state.mr_block_map.block_map.loop_bounds().dim(0)),
        YAKL_LAMBDA (i64 tile_idx) {
            MRIdxGen idx_gen(mr_block_map);
            Coord2 coord = idx_gen.loop_coord(0, tile_idx, 0);
            Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);
            i32 mip_level = idx_gen.get_sample_level(coord);
            max_block_mip(wave_batch_idx, tile_coord.z, tile_coord.x) = mip_level;
        }
    );
    yakl::fence();
}

/// Add Dex's metadata to the file using attributes. The netcdf layer needs extending to do this, so I'm just throwing it in manually.
void add_netcdf_attributes(const State& state, const yakl::SimpleNetCDF& file, i32 num_iter) {
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

    std::string name = "dexrt (2d)";
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
    std::string method(RcConfigurationNames[int(RC_CONFIG)]);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "RC_method", method.size(), method.c_str()),
        __LINE__
    );

    if (RC_CONFIG == RcConfiguration::ParallaxFixInner) {
        i32 inner_parallax_merge_lim = INNER_PARALLAX_MERGE_ABOVE_CASCADE;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "inner_parallax_merge_above_cascade", NC_INT, 1, &inner_parallax_merge_lim),
            __LINE__
        );
    }
    if (RC_CONFIG == RcConfiguration::ParallaxFix) {
        i32 parallax_merge_lim = PARALLAX_MERGE_ABOVE_CASCADE;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "parallax_merge_above_cascade", NC_INT, 1, &parallax_merge_lim),
            __LINE__
        );
    }

    std::string raymarch_type(RaymarchTypeNames[int(RAYMARCH_TYPE)]);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "raymarch_type", raymarch_type.size(), raymarch_type.c_str()),
        __LINE__
    );
    if (RAYMARCH_TYPE == RaymarchType::LineSweep) {
        i32 line_sweep_on_and_above = LINE_SWEEP_START_CASCADE;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "line_sweep_start_cascade", NC_INT, 1, &line_sweep_on_and_above),
            __LINE__
        );
    }

    f64 probe0_length = PROBE0_LENGTH;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "probe0_length", NC_DOUBLE, 1, &probe0_length),
        __LINE__
    );
    i32 probe0_num_rays = PROBE0_NUM_RAYS;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "probe0_num_rays", NC_INT, 1, &probe0_num_rays),
        __LINE__
    );
    i32 probe0_spacing = PROBE0_SPACING;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "probe0_spacing", NC_INT, 1, &probe0_spacing),
        __LINE__
    );
    i32 cascade_branching = CASCADE_BRANCHING_FACTOR;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "cascade_branching_factor", NC_INT, 1, &cascade_branching),
        __LINE__
    );
    i32 multiple_branching_factors = VARY_BRANCHING_FACTOR;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "multiple_branching_factors", NC_INT, 1, &multiple_branching_factors),
        __LINE__
    );
    if (VARY_BRANCHING_FACTOR) {
        i32 upper_branching = UPPER_BRANCHING_FACTOR;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "upper_branching_factor", NC_INT, 1, &upper_branching),
            __LINE__
        );
        i32 branch_switch = BRANCHING_FACTOR_SWITCH;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "branching_factor_switch", NC_INT, 1, &branch_switch),
            __LINE__
        );
    }
    i32 max_cascade = state.config.max_cascade;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "max_cascade", NC_INT, 1, &max_cascade),
        __LINE__
    );
    i32 last_casc_to_inf = LAST_CASCADE_TO_INFTY;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "last_casc_to_infty", NC_INT, 1, &last_casc_to_inf),
        __LINE__
    );
    f64 last_casc_dist = LAST_CASCADE_MAX_DIST;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "last_cascade_max_distance", NC_DOUBLE, 1, &last_casc_dist),
        __LINE__
    );
    i32 preaverage = PREAVERAGE;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "preaverage", NC_INT, 1, &preaverage),
        __LINE__
    );
    i32 dir_by_dir = DIR_BY_DIR;
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
    i32 conserve_pressure_nr = CONSERVE_PRESSURE_NR;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "conserve_pressure_nr", NC_INT, 1, &conserve_pressure_nr),
        __LINE__
    );

    i32 warp_size = DEXRT_WARP_SIZE;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "warp_size", NC_INT, 1, &warp_size),
        __LINE__
    );
    i32 wave_batch = WAVE_BATCH;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "wave_batch", NC_INT, 1, &wave_batch),
        __LINE__
    );
    i32 num_incl = NUM_INCL;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_incl", NC_INT, 1, &num_incl),
        __LINE__
    );
    f64 incl_rays[NUM_INCL];
    f64 incl_weights[NUM_INCL];
    for (int i = 0; i < NUM_INCL; ++i) {
        incl_rays[i] = INCL_RAYS[i];
        incl_weights[i] = INCL_WEIGHTS[i];
    }
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "incl_rays", NC_DOUBLE, num_incl, incl_rays),
        __LINE__
    );
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "incl_weights", NC_DOUBLE, num_incl, incl_weights),
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

    std::string line_scheme_name(LineCoeffCalcNames[int(LINE_SCHEME)]);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "line_calculation_scheme", line_scheme_name.size(), line_scheme_name.c_str()),
        __LINE__
    );

    i32 block_size = BLOCK_SIZE;
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "block_size", NC_INT, 1, &block_size),
        __LINE__
    );
    i32 nx_blocks = state.mr_block_map.block_map.num_x_tiles();
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_x_blocks", NC_INT, 1, &nx_blocks),
        __LINE__
    );
    i32 nz_blocks = state.mr_block_map.block_map.num_z_tiles();
    ncwrap(
        nc_put_att_int(ncid, NC_GLOBAL, "num_z_blocks", NC_INT, 1, &nz_blocks),
        __LINE__
    );

    if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
        i32 interp_bins = INTERPOLATE_DIRECTIONAL_BINS;
        ncwrap(
            nc_put_att_int(ncid, NC_GLOBAL, "interpolate_directional_bins", NC_INT, 1, &interp_bins),
            __LINE__
        );

        f64 interp_max_width = INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH;
        ncwrap(
            nc_put_att_double(ncid, NC_GLOBAL, "interpolate_direction_max_thermal_width", NC_DOUBLE, 1, &interp_max_width),
            __LINE__
        );
    }

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

void save_results(const State& state, const CascadeState& casc_state, i32 num_iter) {
    const auto& config = state.config;
    const auto& out_cfg = config.output;
    if (state.mpi_state.rank != 0) {
        return;
    }

    yakl::SimpleNetCDF nc;
    nc.create(config.output_path, yakl::NETCDF_MODE_REPLACE);
    state.println("Saving output to {}...", config.output_path);
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
            dim_names.insert(dim_names.end(), {"z", "x"});
            nc.write(hydrated, name, dim_names);
        }
    };

    if (out_cfg.J) {
        if (config.store_J_on_cpu) {
            if (sparse_J) {
                maybe_rehydrate_and_write(state.J_cpu, "J", {"wavelength"});
            } else {
                Fp3dHost J_full = state.J_cpu.reshape(state.J_cpu.extent(0), block_map.num_z_tiles() * BLOCK_SIZE, block_map.num_x_tiles() * BLOCK_SIZE);
                nc.write(J_full, "J", {"wavelength", "z", "x"});
            }
        } else {
            if (sparse_J) {
                maybe_rehydrate_and_write(state.J, "J", {"wavelength"});
            } else {
                Fp3d J_full = state.J.reshape(state.J.extent(0), block_map.num_z_tiles() * BLOCK_SIZE, block_map.num_x_tiles() * BLOCK_SIZE);
                nc.write(J_full, "J", {"wavelength", "z", "x"});
            }
        }
        nc.write(state.max_block_mip, "max_mip_block", {"wavelength_batch", "tile_z", "tile_x"});
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
    if (out_cfg.active) {
        // NOTE(cmo): Currently active is always written dense
        const auto& active_char = reify_active_c0(block_map);
        nc.write(active_char, "active", {"z", "x"});
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

int main(int argc, char** argv) {
    init_mpi(argc, argv);

    argparse::ArgumentParser program("DexRT");
    program.add_argument("--config")
        .default_value(std::string("dexrt.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_argument("--restart-from")
        .nargs(1)
        .help("Path to snapshot file")
        .metavar("FILE");
    program.add_epilog("DexRT Radiance Cascade based non-LTE solver.");

    program.parse_args(argc, argv);

    std::optional<std::string> restart_path = program.present("--restart-from");

    const DexrtConfig config = parse_dexrt_config(program.get<std::string>("--config"));

    Kokkos::initialize();
    yakl::init(
        yakl::InitConfig()
            .set_pool_size_mb(config.mem_pool_gb * 1024)
    );

    {
        State state;
        // NOTE(cmo): Allocate the arrays in state, and fill emission/opacity if
        // not using an atmosphere
        init_state(&state, config);
        CascadeState casc_state;
        casc_state.init(state, config.max_cascade);
        yakl::timer_start("DexRT");
        state.max_block_mip = -1;
        yakl::fence();

        // NOTE(cmo): Provided emissivity and opacity in file: static solution.
        if (config.mode == DexrtMode::GivenFs) {
            int num_waves = state.given_state.emis.extent(2);
            for (
                int la_start = 0;
                la_start < num_waves;
                la_start += state.c0_size.wave_batch
            ) {
                const int la_end = std::min(la_start + state.c0_size.wave_batch, num_waves);
                setup_wavelength_batch(state, la_start, la_end);
                static_formal_sol_given_rc(
                    state,
                    casc_state,
                    true,
                    la_start,
                    la_end
                );
                finalise_wavelength_batch(state, la_start, la_end);
            }

            yakl::timer_stop("DexRT");
            save_results(state, casc_state, 1);
        } else {
            compute_lte_pops(&state);
            if (state.config.initial_pops_path.size() > 0) {
                load_initial_pops(&state, state.config.initial_pops_path);
            }
            const bool non_lte = config.mode == DexrtMode::NonLte;
            const bool do_restart = bool(restart_path);
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

            const fp_t non_lte_tol = config.pop_tol;
            auto& waves = state.adata_host.wavelength;
            fp_t max_change = FP(1.0);
            int num_iter = 1;

            WavelengthDistributor wave_dist;
            wave_dist.init(state.mpi_state, waves.extent(0), state.c0_size.wave_batch);

            if (non_lte) {
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

                if (do_restart) {
                    i = handle_restart(&state, *restart_path);
                }

                state.println("-- Non-LTE Iterations --");
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
                while (((max_change > non_lte_tol || i < (initial_lambda_iterations+1)) && i < max_iters) || accelerated) {
                    state.println("==== FS {} ====", i);
                    compute_nh0(state);

                    if (state.mpi_state.rank == 0) {
                        compute_collisions_to_gamma(&state);
                    } else {
                        for (int ia = 0; ia < state.Gamma.size(); ++ia) {
                            state.Gamma[ia] = FP(0.0);
                        }
                        yakl::fence();
                    }

                    compute_profile_normalisation(state, casc_state);
                    state.J = FP(0.0);
                    if (config.store_J_on_cpu) {
                        state.J_cpu = FP(0.0);
                    }
                    yakl::fence();
                    WavelengthBatch wave_batch;
                    wave_dist.wait_for_all(state.mpi_state);
                    wave_dist.reset();
                    while (wave_dist.next_batch(state.mpi_state, &wave_batch)) {
                        setup_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
                        bool lambda_iterate = i < initial_lambda_iterations;
                        dynamic_formal_sol_rc(
                            state,
                            casc_state,
                            lambda_iterate,
                            wave_batch.la_start,
                            wave_batch.la_end
                        );
                        finalise_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
                    }
                    yakl::fence();
                    wave_dist.wait_for_all(state.mpi_state);

                    state.println("  == Statistical equilibrium ==");
                    wave_dist.reduce_Gamma(&state);
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
                                .conserve_pressure = CONSERVE_PRESSURE_NR && actually_conserve_pressure
                            }
                        );
                        wave_dist.update_ne(&state);
                        max_change = std::max(nr_update, max_change);
                        if (!CONSERVE_PRESSURE_NR && actually_conserve_pressure) {
                            fp_t nh_tot_update = simple_conserve_pressure(&state);
                            max_change = std::max(nh_tot_update, max_change);
                        }
                        if (actually_conserve_pressure) {
                            wave_dist.update_nh_tot(&state);
                        }
                    }
                    if (config.ng.enable) {
                        accelerated = ng.accelerate(state, max_change);
                        if (accelerated) {
                            state.println("  ~~ Ng Acceleration! (📉 or 💣) ~~");
                        }
                    }
                    wave_dist.update_pops(&state);
                    i += 1;
                    if (
                        (state.config.snapshot_frequency != 0) &&
                        (i % state.config.snapshot_frequency == 0)
                    ) {
                        save_snapshot(state, i);
                    }
                }
                if (state.config.sparse_calculation && state.config.final_dense_fs) {
                    state.config.sparse_calculation = false;
                    allocate_J(&state);
                    casc_state.probes_to_compute.init(state, casc_state.num_cascades);

                    state.println("Final FS (dense)");
                    compute_nh0(state);
                    state.J = FP(0.0);
                    if (config.store_J_on_cpu) {
                        state.J_cpu = FP(0.0);
                    }
                    yakl::fence();
                    wave_dist.reset();
                    WavelengthBatch wave_batch;
                    while (wave_dist.next_batch(state.mpi_state, &wave_batch)) {
                        setup_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
                        bool lambda_iterate = i < initial_lambda_iterations;
                        dynamic_formal_sol_rc(
                            state,
                            casc_state,
                            lambda_iterate,
                            wave_batch.la_start,
                            wave_batch.la_end
                        );
                        finalise_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
                    }
                    yakl::fence();
                    wave_dist.wait_for_all(state.mpi_state);
                    state.config.sparse_calculation = true;
                }
                num_iter = i;
            } else {
                state.J = FP(0.0);
                compute_nh0(state);
                if (config.store_J_on_cpu) {
                    state.J_cpu = FP(0.0);
                }
                yakl::fence();
                wave_dist.reset();
                WavelengthBatch wave_batch;
                while (wave_dist.next_batch(state.mpi_state, &wave_batch)) {
                    setup_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
                    fmt::println(
                        "Computing wavelengths [{}, {}] ({}, {})",
                        wave_batch.la_start,
                        wave_batch.la_end,
                        waves(wave_batch.la_start),
                        waves(wave_batch.la_end-1)
                    );
                    bool lambda_iterate = true;
                    dynamic_formal_sol_rc(state, casc_state, lambda_iterate, wave_batch.la_start, wave_batch.la_end);
                    finalise_wavelength_batch(state, wave_batch.la_start, wave_batch.la_end);
                }
                wave_dist.wait_for_all(state.mpi_state);
            }
            yakl::timer_stop("DexRT");
            wave_dist.reduce_J(&state);
            save_results(state, casc_state, num_iter);
        }
        finalize_state(&state);
    }
    yakl::finalize();
    finalise_mpi();
    Kokkos::finalize();
    return 0;

}
