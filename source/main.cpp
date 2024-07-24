#include <math.h>
#include <limits>
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include "Populations.hpp"
#include "RadianceCascades.hpp"
#include "CrtafParser.hpp"
#include "Collisions.hpp"
#include "Voigt.hpp"
#include "StaticFormalSolution.hpp"
#include "DynamicFormalSolution.hpp"
#include "GammaMatrix.hpp"
#include "ChargeConservation.hpp"
#include "PressureConservation.hpp"
#include "ProfileNormalisation.hpp"
#include "PromweaverBoundary.hpp"
#include "DexrtConfig.hpp"
#ifdef HAVE_MPI
    #include "YAKL_pnetcdf.h"
#else
    #include "YAKL_netcdf.h"
#endif
#include <vector>
#include <string>
#include <optional>
#include <fmt/core.h>
#include <argparse/argparse.hpp>
#include <sstream>

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
    state.atmos = atmos;
    state.phi = VoigtProfile<fp_t>(
        VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.4), 1024},
        VoigtProfile<fp_t>::Linspace{FP(0.0), FP(3e3), 64 * 1024}
    );
    // TODO(cmo): Check range of Voigt terms against model
    state.nh_lte = HPartFn();
    fmt::println("Scale: {} m", state.atmos.voxel_scale);

    const auto space_dims = atmos.temperature.get_dimensions();
    int space_x = space_dims(0);
    int space_y = space_dims(1);
    int cascade_0_x_probes = space_dims(1) / PROBE0_SPACING;
    int cascade_0_z_probes = space_dims(0) / PROBE0_SPACING;
    // TODO(cmo): Need to decide whether to allow non-probe 1 spacing... it
    // probably doesn't make sense for us to have any interest in the level
    // populations not on a probe - we don't have any way to compute this outside LTE.
    const int n_level_total = state.adata.energy.extent(0);
    state.pops = Fp3d("pops", n_level_total, space_x, space_y);

    if (config.mode == DexrtMode::NonLte) {
        for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
            const int n_level = state.adata_host.num_level(ia);
            state.Gamma.emplace_back(
                Fp4d("Gamma", n_level, n_level, space_x, space_y)
            );
        }
        state.wphi = Fp3d("wphi", state.adata.lines.extent(0), space_x, space_y);
    }
    state.active = decltype(state.active)("active", space_x, space_y);
    const auto& temperature = state.atmos.temperature;
    const auto& active = state.active;
    const fp_t threshold = config.threshold_temperature;
    const bool sparse_calculation = config.sparse_calculation;
    parallel_for(
        "Active bits",
        SimpleBounds<2>(temperature.extent(0), temperature.extent(1)),
        YAKL_LAMBDA (int z, int x) {
            if (!sparse_calculation || threshold == FP(0.0)) {
                active(z, x) = true;
            } else {
                active(z, x) = (temperature(z, x) <= threshold);
            }
        }
    );
    yakl::fence();
    const auto& cpu_active = active.createHostCopy();
    yakl::fence();
    i64 active_count = 0;
    for (int z = 0; z < cpu_active.extent(0); ++z) {
        for (int x = 0; x < cpu_active.extent(1); ++x) {
            active_count += cpu_active(z, x);
        }
    }
    fmt::println("Active cells: {}/{}", active_count, active.extent(0) * active.extent(1));

    // NOTE(cmo): We just have one of these chained for each boundary type -- they don't do anything if this configuration doesn't need them to.
    state.pw_bc = load_bc(config.atmos_path, state.adata.wavelength, config.boundary);
    state.boundary = config.boundary;

    CascadeRays c0_rays;
    c0_rays.num_probes(0) = cascade_0_x_probes;
    c0_rays.num_probes(1) = cascade_0_z_probes;
    c0_rays.num_flat_dirs = PROBE0_NUM_RAYS;
    c0_rays.num_incl = NUM_INCL;
    c0_rays.wave_batch = WAVE_BATCH;

    if (config.store_J_on_cpu) {
        state.J = Fp3d("J", c0_rays.wave_batch, space_x, space_y);
        state.J_cpu = Fp3dHost("JHost", state.adata.wavelength.extent(0), space_x, space_y);
    } else {
        state.J = Fp3d("J", state.adata.wavelength.extent(0), space_x, space_y);
    }
    state.J = FP(0.0);
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
    fmt::println("Scale: {} m", st->atmos.voxel_scale);

#ifdef DEXRT_SINGLE_PREC
    st->given_state.emis = eta.createDeviceCopy();
    st->given_state.opac = chi.createDeviceCopy();
#else
    auto etad = eta.createDeviceCopy();
    auto chid = chi.createDeviceCopy();
    Fp3d emis("eta", z_dim, x_dim, wave_dim);
    Fp3d opac("chi", z_dim, x_dim, wave_dim);
    yakl::fence();
    parallel_for(
        "Convert f32->f64",
        SimpleBounds<3>(z_dim, x_dim, wave_dim),
        YAKL_LAMBDA (int z, int x, int la) {
            emis(z, x, la) = etad(z, x, la);
        }
    );
    parallel_for(
        "Convert f32->f64",
        SimpleBounds<3>(z_dim, x_dim, wave_dim),
        YAKL_LAMBDA (int z, int x, int la) {
            opac(z, x, la) = chid(z, x, la);
        }
    );
    yakl::fence();
    st->given_state.emis = emis;
    st->given_state.opac = opac;
#endif

    // NOTE(cmo): The raymarcher checks whether cells are active, we could remove, but this is a mode used infrequently, so set everything active.
    st->active = yakl::Array<bool, 2, yakl::memDevice>("active cells", eta.extent(0), eta.extent(1));
    st->active = true;

    // NOTE(cmo): Only zero boundaries are supported here.
    st->boundary = BoundaryType::Zero;
    CascadeRays c0_rays;
    c0_rays.num_probes(0) = x_dim;
    c0_rays.num_probes(1) = z_dim;
    c0_rays.num_flat_dirs = PROBE0_NUM_RAYS;
    c0_rays.num_incl = NUM_INCL;
    c0_rays.wave_batch = WAVE_BATCH;

    if (config.store_J_on_cpu) {
        st->J = Fp3d("J", c0_rays.wave_batch, z_dim, x_dim);
        st->J_cpu = Fp3dHost("JHost", wave_dim, z_dim, x_dim);
    } else {
        st->J = Fp3d("J", wave_dim, z_dim, x_dim);
    }
    st->J = FP(0.0);
    yakl::fence();
    return c0_rays;
}

void init_cascade_sized_arrays(State* state, const DexrtConfig& config) {
    if (config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte) {

        if (config.mode == DexrtMode::NonLte) {
            i64 alo_size = cascade_entries(state->c0_size);
            state->alo = Fp1d(
                "ALO",
                alo_size
            );
        }
        state->dynamic_opac = decltype(state->dynamic_opac)(
            "Dynamic Emis/Opac",
            state->c0_size.num_probes(1),
            state->c0_size.num_probes(0),
            state->c0_size.wave_batch
        );
    }
}

void init_state (State* state, const DexrtConfig& config) {
    state->config = config;
    CascadeRays c0_rays;
    if (config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte) {
        c0_rays = init_atmos_atoms(state, config);
    } else {
        c0_rays = init_given_emis_opac(state, config);
    }

    constexpr int RcMode = RC_flags_storage();
    state->c0_size = cascade_rays_to_storage<RcMode>(c0_rays);

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

void init_active_probes(const State& state, CascadeState* casc) {
    // TODO(cmo): This is a poor strategy for 3D, but simple for now. To be done properly in parallel we need to do some stream compaction. e.g. thrust::copy_if
    // Really this function is backwards. We can loop over each probe of i+1 and
    // check the dependents in i, in parallel. The merge process remains the
    // same.
    // NOTE(cmo): Active probes in c_0
    CascadeState& casc_state = *casc;
    auto& active_bool = state.active;
    yakl::Array<u64, 2, yakl::memDevice> prev_active(
        "casc_active",
        state.active.extent(0),
        state.active.extent(1)
    );
    parallel_for(
        SimpleBounds<2>(prev_active.extent(0), prev_active.extent(1)),
        YAKL_LAMBDA (int z, int x) {
            if (active_bool(z, x)) {
                prev_active(z, x) = 1;
            } else {
                prev_active(z, x) = 0;
            }
        }
    );
    yakl::fence();
    u64 num_active = yakl::intrinsics::sum(prev_active);
    auto prev_active_h = prev_active.createHostCopy();
    yakl::fence();
    yakl::Array<i32, 2, yakl::memHost> probes_to_compute_h("c0 to compute", num_active, 2);
    i32 idx = 0;
    for (int z = 0; z < prev_active_h.extent(0); ++z) {
        for (int x = 0; x < prev_active_h.extent(1); ++x) {
            if (prev_active_h(z, x)) {
                probes_to_compute_h(idx, 0) = x;
                probes_to_compute_h(idx, 1) = z;
                idx += 1;
            }
        }
    }
    auto probes_to_compute = probes_to_compute_h.createDeviceCopy();
    casc_state.probes_to_compute.emplace_back(probes_to_compute);
    fmt::println(
        "C0 Active Probes {}/{} ({}%)",
        num_active,
        prev_active.extent(0)*prev_active.extent(1),
        fp_t(num_active) / fp_t(prev_active.extent(0)*prev_active.extent(1)) * FP(100.0)
    );


    for (int cascade_idx = 1; cascade_idx <= casc_state.num_cascades; ++cascade_idx) {
        CascadeStorage dims = cascade_size(state.c0_size, cascade_idx);
        yakl::Array<u64, 2, yakl::memDevice> curr_active(
            "casc_active",
            dims.num_probes(1),
            dims.num_probes(0)
        );
        curr_active = 0;
        yakl::fence();
        auto my_atomic_max = YAKL_LAMBDA (u64& ref, unsigned long long int val) {
            yakl::atomicMax(
                *reinterpret_cast<unsigned long long int*>(&ref),
                val
            );
        };
        parallel_for(
            SimpleBounds<2>(prev_active.extent(0), prev_active.extent(1)),
            YAKL_LAMBDA (int z, int x) {
                int z_bc = std::max(int((z - 1) / 2), 0);
                int x_bc = std::max(int((x - 1) / 2), 0);
                const bool z_clamp = (z_bc == 0) || (z_bc == (curr_active.extent(0) - 1));
                const bool x_clamp = (x_bc == 0) || (x_bc == (curr_active.extent(1) - 1));

                if (!prev_active(z, x)) {
                    return;
                }

                // NOTE(cmo): Atomically set the (up-to) 4 valid
                // probes for this active probe of cascade_idx-1
                my_atomic_max(curr_active(z_bc, x_bc), 1);
                if (!x_clamp) {
                    my_atomic_max(curr_active(z_bc, x_bc+1), 1);
                }
                if (!z_clamp) {
                    my_atomic_max(curr_active(z_bc+1, x_bc), 1);
                }
                if (!(z_clamp || x_clamp)) {
                    my_atomic_max(curr_active(z_bc+1, x_bc+1), 1);
                }
            }
        );
        yakl::fence();
        i64 num_active = yakl::intrinsics::sum(curr_active);
        auto curr_active_h = curr_active.createHostCopy();
        yakl::fence();
        yakl::Array<i32, 2, yakl::memHost> probes_to_compute_h("probes to compute", num_active, 2);
        i32 idx = 0;
        for (int z = 0; z < curr_active_h.extent(0); ++z) {
            for (int x = 0; x < curr_active_h.extent(1); ++x) {
                if (curr_active_h(z, x)) {
                    probes_to_compute_h(idx, 0) = x;
                    probes_to_compute_h(idx, 1) = z;
                    idx += 1;
                }
            }
        }
        auto probes_to_compute = probes_to_compute_h.createDeviceCopy();
        casc_state.probes_to_compute.emplace_back(probes_to_compute);
        prev_active = curr_active;
        fmt::println(
            "C{} Active Probes {}/{} ({}%)",
            cascade_idx,
            num_active,
            prev_active.extent(0)*prev_active.extent(1),
            fp_t(num_active) / fp_t(prev_active.extent(0)*prev_active.extent(1)) * FP(100.0)
        );
    }
}

/// Called to copy J from GPU to plane of host array if config.store_J_on_cpu
void copy_J_plane_to_host(const State& state, int la_start, int la_end) {
    int wave_batch = la_end - la_start;
    const Fp3dHost J_copy = state.J.createHostCopy();
    // TODO(cmo): Replace with a memcpy?
    for (int wave = 0; wave < wave_batch; ++wave) {
        for (int z = 0; z < J_copy.extent(1); ++z) {
            for (int x = 0; x < J_copy.extent(2); ++x) {
                state.J_cpu(la_start + wave, z, x) = J_copy(wave, z, x);
            }
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
}

/// Add Dex's metadata to the file using attributes. The netcdf layer needs extending to do this, so I'm just throwing it in manually.
void add_netcdf_attributes(const State& state, const yakl::SimpleNetCDF& file, i32 num_iter) {
    const auto ncwrap = [] (int ierr, int line) {
        if (ierr != NC_NOERR) {
            printf("NetCDF Error writing attributes at main.cpp:%d\n", line);
            printf("%s\n",nc_strerror(ierr));
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
    i32 max_cascade = MAX_CASCADE;
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
    f64 thermal_vel_frac = ANGLE_INVARIANT_THERMAL_VEL_FRAC;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "angle_invariant_thermal_vel_frac", NC_DOUBLE, 1, &thermal_vel_frac),
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
}

void save_results(const State& state, const CascadeState& casc_state, i32 num_iter) {
    const auto& config = state.config;
    const auto& out_cfg = config.output;

    yakl::SimpleNetCDF nc;
    nc.create(config.output_path, yakl::NETCDF_MODE_REPLACE);
    fmt::println("Saving output to {}...", config.output_path);
    add_netcdf_attributes(state, nc, num_iter);

    if (out_cfg.J) {
        if (config.store_J_on_cpu) {
            nc.write(state.J_cpu, "J", {"wavelength", "z", "x"});
        } else {
            nc.write(state.J, "J", {"wavelength", "z", "x"});
        }
    }

    if (out_cfg.wavelength && state.adata.wavelength.initialized()) {
        nc.write(state.adata.wavelength, "wavelength", {"wavelength"});
    }
    if (out_cfg.pops && state.pops.initialized()) {
        nc.write(state.pops, "pops", {"level", "z", "x"});
    }
    if (out_cfg.lte_pops) {
        Fp3d lte_pops = state.pops.createDeviceObject();
        compute_lte_pops(&state, lte_pops);
        yakl::fence();
        nc.write(lte_pops, "lte_pops", {"level", "z", "x"});
    }
    if (out_cfg.ne && state.atmos.ne.initialized()) {
        nc.write(state.atmos.ne, "ne", {"z", "x"});
    }
    if (out_cfg.nh_tot && state.atmos.nh_tot.initialized()) {
        nc.write(state.atmos.nh_tot, "nh_tot", {"z", "x"});
    }
    if (out_cfg.alo && state.alo.initialized()) {
        nc.write(state.alo, "alo", {"casc_shape"});
    }
    if (out_cfg.active && state.active.initialized()) {
        const auto& active_copy = state.active.createHostCopy();
        yakl::Array<unsigned char, 2, yakl::memHost> active_char(
            "active_char",
            (unsigned char*)active_copy.get_data(),
            active_copy.extent(0),
            active_copy.extent(1)
        );
        nc.write(active_char, "active", {"z", "x"});
    }
    for (int casc : out_cfg.cascades) {
        // NOTE(cmo): The validity of these + necessary warning were checked/output in the config parsing step
        std::string name = fmt::format("I_C{}", casc);
        std::string shape = fmt::format("casc_shape_{}", casc);
        nc.write(casc_state.i_cascades[casc], name, {shape});
        name = fmt::format("tau_C{}", casc);
        nc.write(casc_state.tau_cascades[casc], name, {shape});
    }
    nc.close();
}

int main(int argc, char** argv) {
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    argparse::ArgumentParser program("DexRT");
    program.add_argument("--config")
        .default_value(std::string("dexrt.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_epilog("DexRT Radiance Cascade based non-LTE solver.");

    program.parse_args(argc, argv);

    const DexrtConfig config = parse_dexrt_config(program.get<std::string>("--config"));

    yakl::init(
        yakl::InitConfig()
            .set_pool_initial_mb(config.mem_pool_initial_gb * 1024)
            .set_pool_grow_mb(config.mem_pool_grow_gb * 1024)
    );

    {
        State state;
        // NOTE(cmo): Allocate the arrays in state, and fill emission/opacity if
        // not using an atmosphere
        init_state(&state, config);
        CascadeState casc_state = CascadeState_new(state.c0_size, MAX_CASCADE);
        yakl::timer_start("DexRT");

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

            Fp1d dummy_wavelengths;
            Fp3d dummy_pops;
            yakl::timer_stop("DexRT");
            save_results(state, casc_state, 1);
        } else {
            if (config.sparse_calculation) {
                init_active_probes(state, &casc_state);
            }
            compute_lte_pops(&state);
            const bool non_lte = config.mode == DexrtMode::NonLte;
            const bool conserve_charge = config.conserve_charge;
            const bool actually_conserve_charge = state.have_h && conserve_charge;
            const bool conserve_pressure = config.conserve_pressure;
            const bool actually_conserve_pressure = actually_conserve_charge && conserve_pressure;
            const int initial_lambda_iterations = config.initial_lambda_iterations;
            const int max_iters = config.max_iter;

            constexpr fp_t non_lte_tol = FP(1e-3);
            auto& waves = state.adata_host.wavelength;
            auto fs_fn = dynamic_formal_sol_rc;
            // if (static_soln) {
            //     fs_fn = static_formal_sol_rc;
            // }
            fp_t max_change = FP(1.0);
            int num_iter = 1;
            if (non_lte) {
                int i = 0;
                if (actually_conserve_charge) {
                    fmt::println("-- Iterating LTE n_e/pressure --");
                    fp_t lte_max_change = FP(1.0);
                    int lte_i = 0;
                    while ((lte_max_change > FP(1e-3) || lte_i < 6) && lte_i < max_iters) {
                        lte_i += 1;
                        compute_nh0(state);
                        compute_collisions_to_gamma(&state);
                        // NOTE(cmo): When dealing with just the collisional
                        // rates, numerical precision can be a bit of a pain,
                        // and get stuck in loops. Here we ignore changes to
                        // populations below 1e-12 of the total species fraction
                        // -- they are very unlikely overly important for charge
                        // conservation stuff. This is just to get a starting
                        // guess anyway.
                        lte_max_change = stat_eq<f64>(&state, StatEqOptions{
                            .ignore_change_below_ntot_frac=FP(1e-12)
                        });
                        if (lte_i < 2) {
                            continue;
                        }
                        fp_t nr_update = nr_post_update<f64>(&state, NrPostUpdateOptions{
                            .ignore_change_below_ntot_frac=FP(1e-12)
                        });
                        lte_max_change = std::max(nr_update, lte_max_change);
                        if (actually_conserve_pressure) {
                            fp_t nh_tot_update = simple_conserve_pressure(&state);
                            lte_max_change = std::max(nh_tot_update, lte_max_change);
                        }
                    }
                    fmt::println("Ran for {} iterations", lte_i);
                }
                fmt::println("-- Non-LTE Iterations --");
                while ((max_change > non_lte_tol || i < (initial_lambda_iterations+1)) && i < max_iters) {
                    fmt::println("FS {}", i);
                    compute_nh0(state);
                    compute_collisions_to_gamma(&state);
                    state.J = FP(0.0);
                    compute_profile_normalisation(state, casc_state);
                    if (config.store_J_on_cpu) {
                        state.J_cpu = FP(0.0);
                    }
                    yakl::fence();
                    for (
                        int la_start = 0;
                        la_start < waves.extent(0);
                        la_start += state.c0_size.wave_batch
                    ) {
                        int la_end = std::min(la_start + state.c0_size.wave_batch, int(waves.extent(0)));
                        setup_wavelength_batch(state, la_start, la_end);
                        bool lambda_iterate = i < initial_lambda_iterations;
                        fs_fn(
                            state,
                            casc_state,
                            lambda_iterate,
                            la_start,
                            la_end
                        );
                        finalise_wavelength_batch(state, la_start, la_end);
                    }
                    yakl::fence();
                    fmt::println("Stat eq");
                    max_change = stat_eq<f64>(&state);
                    if (i > 0 && actually_conserve_charge) {
                        fp_t nr_update = nr_post_update<f64>(&state);
                        max_change = std::max(nr_update, max_change);
                        if (actually_conserve_pressure) {
                            fp_t nh_tot_update = simple_conserve_pressure(&state);
                            max_change = std::max(nh_tot_update, max_change);
                        }
                    }
                    i += 1;
                }
                if (state.config.sparse_calculation) {
                    state.config.sparse_calculation = false;
                    fmt::println("Final FS (dense)");
                    compute_nh0(state);
                    state.J = FP(0.0);
                    if (config.store_J_on_cpu) {
                        state.J_cpu = FP(0.0);
                    }
                    yakl::fence();
                    for (
                        int la_start = 0;
                        la_start < waves.extent(0);
                        la_start += state.c0_size.wave_batch
                    ) {
                        int la_end = std::min(la_start + state.c0_size.wave_batch, int(waves.extent(0)));
                        setup_wavelength_batch(state, la_start, la_end);
                        bool lambda_iterate = i < initial_lambda_iterations;
                        fs_fn(
                            state,
                            casc_state,
                            lambda_iterate,
                            la_start,
                            la_end
                        );
                        finalise_wavelength_batch(state, la_start, la_end);
                    }
                    yakl::fence();
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
                for (int la_start = 0; la_start < waves.extent(0); la_start += state.c0_size.wave_batch) {
                    int la_end = std::min(la_start + state.c0_size.wave_batch, int(waves.extent(0)));
                    setup_wavelength_batch(state, la_start, la_end);
                    fmt::println(
                        "Computing wavelengths [{}, {}] ({}, {})",
                        la_start,
                        la_end,
                        waves(la_start),
                        waves(la_end-1)
                    );
                    bool lambda_iterate = true;
                    fs_fn(state, casc_state, lambda_iterate, la_start, la_end);
                    finalise_wavelength_batch(state, la_start, la_end);
                }
            }
            yakl::timer_stop("DexRT");
            save_results(state, casc_state, num_iter);
        }
        finalize_state(&state);
    }
    yakl::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
