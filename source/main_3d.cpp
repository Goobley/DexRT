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


void save_results(const State3d& state, const CascadeState3d& casc_state) {
    const auto& config = state.config;

    yakl::SimpleNetCDF nc;
    nc.create(config.output_path, yakl::NETCDF_MODE_REPLACE);
    fmt::println("Saving output to {}...", config.output_path);
    if (state.config.mode == DexrtMode::GivenFs) {
        Fp4d J4d = state.J.reshape(
            state.given_state.emis.extent(0),
            state.given_state.emis.extent(1),
            state.given_state.emis.extent(2),
            state.given_state.emis.extent(3)
        );
        nc.write(J4d, "J", {"wavelength", "z", "y", "x"});
        return;
    }

    if (state.config.sparse_calculation) {
        // nc.write(rehydrate_sparse_quantity(state.mr_block_map.block_map, state.J_cpu), "J", {"wavelength", "z", "y", "x"});
        nc.write(state.J_cpu, "J", {"wavelength", "ks"});
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
    nc.write(state.adata.wavelength, "wavelength", {"wavelength"});
    nc.write(state.mr_block_map.block_map.active_tiles, "morton_tiles", {"num_active_tiles"});
    nc.write(state.pops, "pops", {"level", "ks"});
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

        if (config.mode == DexrtMode::GivenFs) {
            static_formal_sol_rc_given_3d(state, casc_state);
        } else {
            compute_lte_pops(&state);
            const bool non_lte = config.mode == DexrtMode::NonLte;
            const fp_t non_lte_tol = config.pop_tol;
            fp_t max_change = FP(1.0);
            int num_iter = 1;
            const int initial_lambda_iterations = config.initial_lambda_iterations;
            const int max_iters = config.max_iter;

            // if (non_lte) {
                int i = 0;
                state.println("-- Non-LTE Iterations --");
                while (((max_change > non_lte_tol || i < (initial_lambda_iterations+1)) && i < max_iters)) {
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
                        dynamic_formal_sol_rc_3d(state, casc_state, la);
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
                    i += 1;
                }
            // }
        }
        save_results(state, casc_state);
        yakl::timer_stop("DexRT");
    }
    yakl::finalize();
    Kokkos::finalize();
    return 0;
}