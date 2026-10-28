#include "StateSetup.hpp"
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

// NOTE(cmo): Defined in BlockMap.cpp. Calling it rather than the inline
// load_atmos keeps its CUDA lambdas instantiated in a single translation unit.
template <int NumDim>
std::conditional_t<NumDim < 3, AtmosphereNd<NumDim>, AtmosphereNd<NumDim, yakl::memHost>>
load_dense_atmos(const std::string& path);
template <>
AtmosphereNd<2> load_dense_atmos<2>(const std::string& path);

namespace {

/// Rows [z_start, z_end) of a dense atmosphere.
Atmosphere crop_atmos_z(const Atmosphere& atmos, i32 z_start, i32 z_end) {
    Atmosphere result = atmos;
    const i32 nz = z_end - z_start;
    const i32 nx = atmos.temperature.extent(1);
    auto crop = [&](const Fp2d& a) -> Fp2d {
        if (!a.initialized()) {
            return a;
        }
        Fp2d out(a.label(), nz, nx);
        dex_parallel_for(
            "Crop atmosphere z",
            FlatLoop<2>(nz, nx),
            YAKL_LAMBDA (i32 z, i32 x) {
                out(z, x) = a(z + z_start, x);
            }
        );
        return out;
    };
    result.temperature = crop(atmos.temperature);
    result.pressure = crop(atmos.pressure);
    result.ne = crop(atmos.ne);
    result.nh_tot = crop(atmos.nh_tot);
    result.nh0 = crop(atmos.nh0);
    result.vturb = crop(atmos.vturb);
    result.vx = crop(atmos.vx);
    result.vy = crop(atmos.vy);
    result.vz = crop(atmos.vz);
    result.e_int = crop(atmos.e_int);
    yakl::fence();
    return result;
}

} // namespace

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
    i64 num_cells = mr_block_map.block_map.get_num_active_cells();
    i32 wave_dim = adata.wavelength.extent(0);
    if (config.mode == DexrtMode::GivenFs) {
        wave_dim = state->given_state.emis.extent(2);
    }

    int rad_loss_leading_dim = c0_size.wave_batch;
    int rad_loss_cpu_leading_dim = wave_dim;
    if (config.rad_loss == RadLossType::Integrated) {
        rad_loss_leading_dim = 1;
        rad_loss_cpu_leading_dim = 1;
    }

    state->rad_loss = decltype(state->rad_loss)("L", yakl::DimsT<i64>(rad_loss_leading_dim, num_cells));
    state->rad_loss_cpu = decltype(state->rad_loss_cpu)("LHost", yakl::DimsT<i64>(rad_loss_cpu_leading_dim, num_cells));
    state->rad_loss_cpu = FP(0.0);
    state->rad_loss = FP(0.0);
}

CascadeRays init_atmos_atoms (State* st, const DexrtConfig& config, const AtmosZWindow* window) {
    if (!(config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte)) {
        return CascadeRays{};
    }

    State& state = *st;

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

    i32 max_mip_level = 0;
    for (int i = 0; i <= config.max_cascade; ++i) {
        max_mip_level = std::max(max_mip_level, config.mip_config.mip_levels[i]);
    }
    if (state.config.mode != DexrtMode::GivenFs && LINE_SCHEME == LineCoeffCalc::Classic) {
        max_mip_level = 0;
        state.println("Mips not supported with LineCoeffCalc::Classic");
    }
    if (window) {
        if (atmosphere_file_is_sparse(config.atmos_path)) {
            throw std::runtime_error("Loading a z window of an atmosphere requires a dense atmosphere file.");
        }
        const Atmosphere atmos = crop_atmos_z(load_dense_atmos<2>(config.atmos_path), window->z_start, window->z_end);
        BlockMap<BLOCK_SIZE> block_map;
        block_map.init(atmos, config.threshold_temperature);
        state.mr_block_map.init(block_map, max_mip_level);
        state.atmos = sparsify_atmosphere(atmos, block_map);
    } else {
        state.atmos = state.mr_block_map.init_and_sparsify_atmos(config.atmos_path, config.threshold_temperature, max_mip_level);
    }

    state.phi = VoigtProfile<fp_t>(
        VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.4), 1024},
        VoigtProfile<fp_t>::Linspace{FP(0.0), FP(3e3), 64 * 1024}
    );
    state.nh_lte = HPartFn();
    state.println("Scale: {} m", state.atmos.voxel_scale);

    i64 num_active_cells = state.mr_block_map.get_num_active_cells();

    const int n_level_total = state.adata.energy.extent(0);
    state.pops = Fp2d("pops", n_level_total, num_active_cells);

    if (config.mode == DexrtMode::NonLte) {
        for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
            state.precon_R.emplace_back(
                PreconRMat("precon_R", num_transitions(state.adata_host, ia), 2, num_active_cells)
            );
            state.precon_R.back() = GammaFp(0.0);
        }
        state.rate_diag.allocate(state.adata_host, config.output, num_active_cells);
        state.wphi = Rad16Array("wphi", state.adata.lines.extent(0), num_active_cells);
    }

    // NOTE(cmo): We just have one of these chained for each boundary type -- they don't do anything if this configuration doesn't need them to.
    state.pw_bc = load_bc(config.atmos_path, state.adata.wavelength, config.boundary, PromweaverResampleType::FluxConserving);
    state.boundary = config.boundary;

    // NOTE(cmo): This doesn't actually know that things will be allocated sparse
    CascadeRays c0_rays;
    c0_rays.num_probes(0) = state.atmos.num_x;
    c0_rays.num_probes(1) = state.atmos.num_z;
    c0_rays.num_flat_dirs = PROBE0_NUM_RAYS;
    c0_rays.num_incl = NUM_INCL;
    c0_rays.wave_batch = WAVE_BATCH;

    const auto& block_map = state.mr_block_map.block_map;
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

/// Copy J from GPU to plane of host array
void copy_J_plane_to_host(const State& state, int la_start, int la_end) {
    page_rows_to_host(state.J_cpu, la_start, state.J, la_end - la_start, state.J_stage);
}

/// Copy rad_loss from GPU to plane of host array
void copy_rad_loss_plane_to_host(const State& state, int la_start, int la_end) {
    if (!state.rad_loss.initialized()) {
        return;
    }

    int wave_batch = la_end - la_start;
    const auto rad_loss_copy = state.rad_loss.createHostCopy();
    if (state.config.rad_loss == RadLossType::PerWavelength) {
        // TODO(cmo): Replace with a memcpy?
        for (int wave = 0; wave < wave_batch; ++wave) {
            for (i64 ks = 0; ks < rad_loss_copy.extent(1); ++ks) {
                state.rad_loss_cpu(la_start + wave, ks) = rad_loss_copy(wave, ks);
            }
        }
    } else {
        KOKKOS_ASSERT(rad_loss_copy.extent(0) == 1);
        for (i64 ks = 0; ks < rad_loss_copy.extent(1); ++ks) {
            state.rad_loss_cpu(0, ks) += rad_loss_copy(0, ks);
        }
    }
}
