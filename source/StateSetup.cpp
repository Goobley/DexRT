#include "StateSetup.hpp"
#include "ScState.hpp"
#include "GammaMatrix.hpp"
#include "CascadeState.hpp"
#include "Populations.hpp"
#include "CrtafParser.hpp"
#include "PromweaverBoundary.hpp"
#include "MiscSparse.hpp"
#include "RcUtilsModes.hpp"
#include "YAKL_netcdf.h"
#include "Atmosphere.hpp"
#include <fmt/core.h>
#include <array>
#include <type_traits>

void allocate_J(State* state) {
    JasUnpack((*state), config, mr_block_map, c0_size, adata);
    const auto& block_map = mr_block_map.block_map;
    const bool sparse = config.sparse_calculation;
    i64 num_cells = mr_block_map.get_num_active_cells();
    i32 wave_dim = adata.wavelength.extent(0);
    if (config.mode == DexrtMode::GivenFs) {
        wave_dim = state->given_state.emis.extent(2);
    }

    if (!sparse) {
        num_cells = i64(block_map.num_x_tiles()) * block_map.num_z_tiles() * square(BLOCK_SIZE);
    }

    state->J = Fp2d("J", yakl::DimsT<i64>(c0_size.wave_batch, num_cells));
    state->J_cpu = Rad16ArrayHost("JHost", wave_dim, num_cells);
    if constexpr (HALF_RADIATION_STORAGE) {
        state->J_stage = Rad16Array("JStage", c0_size.wave_batch, num_cells);
    }
    state->J = FP(0.0);
    state->J_cpu.fill(FP(0.0));
    // TODO(cmo): If we have scattering terms and are updating J, the old
    // contents should probably be moved first, but we don't have these terms yet.
}

void allocate_rad_loss(State* state) {
    JasUnpack((*state), config, mr_block_map, c0_size, adata);
    if (config.rad_loss == RadLossType::None) {
        return;
    }
    i64 num_cells = mr_block_map.get_num_active_cells();
    i32 wave_dim = adata.wavelength.extent(0);
    if (config.mode == DexrtMode::GivenFs) {
        wave_dim = state->given_state.emis.extent(2);
    }

    int rad_loss_cpu_leading_dim = wave_dim;
    if (config.rad_loss == RadLossType::Integrated) {
        rad_loss_cpu_leading_dim = 1;
    }

    state->rad_loss = decltype(state->rad_loss)("L", yakl::DimsT<i64>(c0_size.wave_batch, num_cells));
    state->rad_loss_integrated = decltype(state->rad_loss_integrated)("LIntegrated", yakl::DimsT<i64>(num_cells));
    state->rad_loss_cpu = decltype(state->rad_loss_cpu)("LHost", yakl::DimsT<i64>(rad_loss_cpu_leading_dim, num_cells));
    zero_rad_loss(*state);
}

/// Zero all of the rad_loss arrays, e.g. at the start of an iteration
void zero_rad_loss(const State& state) {
    if (!state.rad_loss.initialized()) {
        return;
    }
    state.rad_loss = FP(0.0);
    state.rad_loss_integrated = FP(0.0);
    state.rad_loss_cpu = FP(0.0);
    yakl::fence();
}

/// The Promweaver boundary. dexrt_sc: read by rank 0 and broadcast.
template <typename StateT>
PwBc<> load_bc_replicated(const StateT& state, const DexrtConfig& config) {
    if (!std::is_same_v<StateT, ScState> || state.mpi_state.num_ranks == 1) {
        return load_bc(config.atmos_path, state.adata.wavelength, config.boundary, PromweaverResampleType::FluxConserving);
    }
    PwBc<> bc{};
#ifdef HAVE_MPI
    const MPI_Comm comm = state.mpi_state.comm;
    fp_t mu[3] = {FP(0.0), FP(0.0), FP(0.0)};
    i64 dims[2] = {0, 0};
    Fp2dHost I;
    if (state.mpi_state.rank == 0) {
        bc = load_bc(config.atmos_path, state.adata.wavelength, config.boundary, PromweaverResampleType::FluxConserving);
        mu[0] = bc.mu_min;
        mu[1] = bc.mu_max;
        mu[2] = bc.mu_step;
        if (bc.I.initialized()) {
            I = bc.I.createHostCopy();
            dims[0] = I.extent(0);
            dims[1] = I.extent(1);
        }
    }
    MPI_Bcast(mu, 3, get_FpMpi(), 0, comm);
    MPI_Bcast(dims, 2, MPI_INT64_T, 0, comm);
    if (state.mpi_state.rank != 0) {
        bc.mu_min = mu[0];
        bc.mu_max = mu[1];
        bc.mu_step = mu[2];
        if (dims[0] > 0) {
            I = Fp2dHost("pw_bc_I", dims[0], dims[1]);
        }
    }
    if (dims[0] > 0) {
        MPI_Bcast(I.data(), int(dims[0] * dims[1]), get_FpMpi(), 0, comm);
        if (state.mpi_state.rank != 0) {
            bc.I = I.createDeviceCopy();
        }
    }
#endif
    return bc;
}

template <typename StateT>
void init_atomic_data(StateT* st, const DexrtConfig& config) {
    StateT& state = *st;
    std::vector<ModelAtom<f64>> crtaf_models;
    crtaf_models.reserve(config.atom_paths.size());
    for (int i = 0; i < config.atom_paths.size(); ++i) {
        const auto& p = config.atom_paths[i];
        const auto& model_config = config.atom_configs[i];
        crtaf_models.emplace_back(parse_crtaf_model<f64>(p, model_config));
    }
    AtomicDataHostDevice<fp_t> atomic_data = to_atomic_data<fp_t, f64>(crtaf_models);
    state.adata = atomic_data.device;
    state.adata_host = atomic_data.host;
    state.have_h = atomic_data.have_h_model;
    state.atoms = extract_atoms(atomic_data.device, atomic_data.host);
    GammaAtomsAndMapping gamma_atoms = extract_atoms_with_gamma_and_mapping(atomic_data.device, atomic_data.host);
    state.atoms_with_gamma = gamma_atoms.atoms;
    state.atoms_with_gamma_mapping = gamma_atoms.mapping;
    state.rate_diag.init_stage_index(state.adata_host);
}

template <typename StateT>
void init_atom_storage(StateT* st, const DexrtConfig& config) {
    StateT& state = *st;
    state.phi = VoigtProfile<fp_t>::make();
    state.nh_lte = HPartFn();
    state.println("Scale: {} m", state.atmos.voxel_scale);

    i64 num_active_cells = state.atmos.temperature.extent(0);

    const int n_level_total = state.adata.energy.extent(0);
    state.pops = Fp2d("pops", n_level_total, num_active_cells);

    if (config.mode == DexrtMode::NonLte) {
        state.precon_R = PreconRMat("precon_R", num_transitions(state.adata_host), 2, num_active_cells);
        state.precon_R = GammaFp(0.0);
        state.rate_diag.allocate(state.adata_host, config.output, num_active_cells);
        state.wphi = Rad16Array("wphi", state.adata.lines.extent(0), num_active_cells);
    }

    // NOTE(cmo): We just have one of these chained for each boundary type -- they don't do anything if this configuration doesn't need them to.
    state.pw_bc = load_bc_replicated(state, config);
    state.boundary = config.boundary;

    if (!config.background_path.empty()) {
        load_background(&state, config.background_path);
    }
}

template void init_atomic_data<State>(State* st, const DexrtConfig& config);
template void init_atomic_data<ScState>(ScState* st, const DexrtConfig& config);
template void init_atom_storage<State>(State* st, const DexrtConfig& config);
template void init_atom_storage<ScState>(ScState* st, const DexrtConfig& config);

CascadeRays init_atmos_atoms(State* st, const DexrtConfig& config) {
    if (!(config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte)) {
        return CascadeRays{};
    }

    State& state = *st;
    init_atomic_data(st, config);

    i32 max_mip_level = 0;
    for (int i = 0; i <= config.max_cascade; ++i) {
        max_mip_level = std::max(max_mip_level, config.mip_config.mip_levels[i]);
    }
    if (state.config.mode != DexrtMode::GivenFs && LINE_SCHEME == LineCoeffCalc::Classic) {
        max_mip_level = 0;
        state.println("Mips not supported with LineCoeffCalc::Classic");
    }
    state.atmos = state.mr_block_map.init_and_sparsify_atmos(config.atmos_path, config.threshold_temperature, max_mip_level);

    init_atom_storage(st, config);

    // NOTE(cmo): This doesn't actually know that things will be allocated sparse
    CascadeRays c0_rays;
    c0_rays.num_probes(0) = state.atmos.num_x;
    c0_rays.num_probes(1) = state.atmos.num_z;
    c0_rays.num_flat_dirs = PROBE0_NUM_RAYS;
    c0_rays.num_incl = NUM_INCL;
    c0_rays.wave_batch = WAVE_BATCH;

    const auto& block_map = state.mr_block_map.block_map;
    state.max_block_mip = decltype(state.max_block_mip)("max_block_mip",
        (state.adata.wavelength.extent(0) + c0_rays.wave_batch - 1) / c0_rays.wave_batch,
        block_map.num_z_tiles(),
        block_map.num_x_tiles());
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
    st->given_state.num_y = 1;
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

    st->max_block_mip = decltype(st->max_block_mip)("max_block_mip",
        (wave_dim + c0_rays.wave_batch - 1) / c0_rays.wave_batch,
        block_map.num_z_tiles(),
        block_map.num_x_tiles());
    yakl::fence();
    return c0_rays;
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
    if (state.mpi_state.rank != 0) {
        return;
    }

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
            FlatLoop<3>(num_level, bounds.dim(0), bounds.dim(1)),
            YAKL_LAMBDA(i32 level, i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);

                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
                pops(level, ks) = temp_pops(level, coord.z, coord.x);
            });
        yakl::fence();
    }
}

/// Copy J from GPU to plane of host array
void copy_J_plane_to_host(const State& state, int la_start, int la_end) {
    page_rows_to_host(state.J_cpu, la_start, state.J, la_end - la_start, state.J_stage);
}

/// Add the batch's rad_loss into rad_loss_integrated, and copy its planes to
/// the host if we're storing them per wavelength
void finalise_rad_loss_batch(const State& state, int la_start, int la_end) {
    if (!state.rad_loss.initialized()) {
        return;
    }

    const int wave_batch = la_end - la_start;
    JasUnpack(state, rad_loss, rad_loss_integrated);
    dex_parallel_for(
        "Integrate rad_loss batch",
        FlatLoop<1>(rad_loss.extent(1)),
        YAKL_LAMBDA (i64 ks) {
            RadLossFp batch_loss = RadLossFp(0.0);
            for (int wave = 0; wave < wave_batch; ++wave) {
                batch_loss += rad_loss(wave, ks);
            }
            rad_loss_integrated(ks) += batch_loss;
        }
    );
    yakl::fence();

    if (state.config.rad_loss == RadLossType::PerWavelength) {
        const auto rad_loss_copy = state.rad_loss.createHostCopy();
        // TODO(cmo): Replace with a memcpy?
        for (int wave = 0; wave < wave_batch; ++wave) {
            for (i64 ks = 0; ks < rad_loss_copy.extent(1); ++ks) {
                state.rad_loss_cpu(la_start + wave, ks) = rad_loss_copy(wave, ks);
            }
        }
    }
}

/// Copy rad_loss_integrated to the host array if we're storing the integrated
/// loss (the per wavelength planes are copied in finalise_rad_loss_batch)
void copy_rad_loss_to_host(const State& state) {
    if (!state.rad_loss.initialized() || state.config.rad_loss != RadLossType::Integrated) {
        return;
    }
    const auto integrated = state.rad_loss_integrated.createHostCopy();
    for (i64 ks = 0; ks < integrated.extent(0); ++ks) {
        state.rad_loss_cpu(0, ks) = integrated(ks);
    }
}
