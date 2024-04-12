#if !defined(DEXRT_ATMOSPHERE_HPP)
#define DEXRT_ATMOSPHERE_HPP

#include "Config.hpp"
#include "Types.hpp"
#include <string>
#ifdef HAVE_MPI
    #include "YAKL_pnetcdf.h"
#else
    #include "YAKL_netcdf.h"
#endif

inline Atmosphere load_atmos(const std::string& path) {
#ifdef HAVE_MPI
    static_assert(false, "Only normal netcdf supported for atmosphere loading currently.");
#endif


    yakl::SimpleNetCDF nc;
    nc.open(path, yakl::NETCDF_MODE_READ);
    int x_dim = nc.getDimSize("x");
    int z_dim = nc.getDimSize("z");

    Atmosphere result;
    result.temperature = Fp2d("temperature", z_dim, x_dim);
    result.pressure = Fp2d("pressure", z_dim, x_dim);
    result.ne = Fp2d("ne", z_dim, x_dim);
    result.nh_tot = Fp2d("nh_tot", z_dim, x_dim);
    result.vturb = Fp2d("vturb", z_dim, x_dim);
    result.vx = Fp2d("vx", z_dim, x_dim);
    result.vy = Fp2d("vy", z_dim, x_dim);
    result.vz = Fp2d("vz", z_dim, x_dim);

    nc.read(result.voxel_scale, "voxel_scale");
    nc.read(result.temperature, "temperature");
    nc.read(result.pressure, "pressure");
    nc.read(result.ne, "ne");
    nc.read(result.nh_tot, "nh_tot");
    nc.read(result.vturb, "vturb");
    nc.read(result.vx, "vx");
    nc.read(result.vy, "vy");
    nc.read(result.vz, "vz");

    return result;
}

#else
#endif