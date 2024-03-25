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
    nc.open("atmos.nc", yakl::NETCDF_MODE_READ);
    int x_dim = nc.getDimSize("x");
    int z_dim = nc.getDimSize("z");

    Atmosphere result;
    result.temperature = Fp2d("temperature", x_dim, z_dim);
    result.pressure = Fp2d("pressure", x_dim, z_dim);
    result.ne = Fp2d("ne", x_dim, z_dim);
    result.nh_tot = Fp2d("nh_tot", x_dim, z_dim);
    result.vturb = Fp2d("vturb", x_dim, z_dim);

    nc.read(result.voxel_scale, "voxel_scale");
    nc.read(result.temperature, "temperature");
    nc.read(result.pressure, "pressure");
    nc.read(result.ne, "ne");
    nc.read(result.nh_tot, "nh_tot");
    nc.read(result.vturb, "vturb");

    return result;
}

#else
#endif