#include "State3d.hpp"
#include "CascadeState3d.hpp"
#include "RcUtilsModes3d.hpp"
#include "StaticFormalSolution3d.hpp"
#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include "YAKL_netcdf.h"

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

void save_results(const State3d& state, const CascadeState3d& casc_state) {
    const auto& config = state.config;

    yakl::SimpleNetCDF nc;
    nc.create(config.output_path, yakl::NETCDF_MODE_REPLACE);
    fmt::println("Saving output to {}...", config.output_path);
    Fp4d J4d = state.J.reshape(
        state.given_state.emis.extent(0),
        state.given_state.emis.extent(1),
        state.given_state.emis.extent(2),
        state.given_state.emis.extent(3)
    );
    nc.write(J4d, "J", {"wavelength", "z", "y", "x"});
    // nc.write(casc_state.i_cascades[0], "C0", {"C0_dim"});
    // nc.write(casc_state.i_cascades[1], "C1", {"C1_dim"});
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
        state.config = config;
        CascadeRays3d c0_rays = init_given_emis_opac(&state, config);
        constexpr int RcMode = RC_flags_storage_3d();
        state.c0_size = cascade_rays_to_storage<RcMode>(c0_rays);
        state.J = Fp2d("J", state.given_state.emis.extent(0), state.mr_block_map.get_num_active_cells());
        CascadeState3d casc_state;
        casc_state.init(state, config.max_cascade);

        static_formal_sol_rc_given_3d(state, casc_state);
        save_results(state, casc_state);
        yakl::timer_stop("DexRT");
    }
    yakl::finalize();
    Kokkos::finalize();
    return 0;
}