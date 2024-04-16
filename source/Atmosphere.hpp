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

    if (nc.varExists("altitude")) {
        nc.read(result.altitude, "altitude");
    }

    return result;
}

template <typename T=fp_t>
FlatAtmosphere<T> flatten(const Atmosphere& atmos) {
    FlatAtmosphere<T> result;
    result.voxel_scale = atmos.voxel_scale;
    result.temperature = atmos.temperature.collapse();
    result.pressure = atmos.pressure.collapse();
    result.ne = atmos.ne.collapse();
    result.nh_tot = atmos.nh_tot.collapse();
    result.vturb = atmos.vturb.collapse();
    result.vx = atmos.vx.collapse();
    result.vy = atmos.vy.collapse();
    result.vz = atmos.vz.collapse();
    return result;
}

template <typename T>
YAKL_INLINE fp_t compute_vnorm(const FlatAtmosphere<T>& atmos, i64 k) {
    return std::sqrt(
        square(atmos.vx(k))
        + square(atmos.vy(k))
        + square(atmos.vz(k))
    );
}

#else
#endif