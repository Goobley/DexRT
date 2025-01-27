#if !defined(DEXRT_ATMOSPHERE_HPP)
#define DEXRT_ATMOSPHERE_HPP

#include "Config.hpp"
#include "Types.hpp"
#include <string>
#include "ExYakl/netcdf.hpp"
#include "JasPP.hpp"

inline Atmosphere load_atmos(const std::string& path) {
    typedef KView<f32*, Kokkos::HostSpace> Fp1dLoad;
    typedef KView<f32**, Kokkos::HostSpace> Fp2dLoad;

    ExYakl::SimpleNetCDF nc;
    nc.open(path, ExYakl::NETCDF_MODE_READ);
    int x_dim = nc.getDimSize("x");
    int z_dim = nc.getDimSize("z");

    Fp2dLoad temperature("temperature", z_dim, x_dim);
    Fp2dLoad pressure("pressure", z_dim, x_dim);
    Fp2dLoad ne("ne", z_dim, x_dim);
    Fp2dLoad nh_tot("nh_tot", z_dim, x_dim);
    Fp2dLoad vturb("vturb", z_dim, x_dim);
    Fp2dLoad vx("vx", z_dim, x_dim);
    Fp2dLoad vy("vy", z_dim, x_dim);
    Fp2dLoad vz("vz", z_dim, x_dim);

    f32 voxel_scale;
    nc.read(voxel_scale, "voxel_scale");
    nc.read(temperature, "temperature");
    nc.read(pressure, "pressure");
    nc.read(ne, "ne");
    nc.read(nh_tot, "nh_tot");
    nc.read(vturb, "vturb");
    nc.read(vx, "vx");
    nc.read(vy, "vy");
    nc.read(vz, "vz");

    f32 offset_x = FP(0.0);
    f32 offset_y = FP(0.0);
    f32 offset_z = FP(0.0);

    if (nc.varExists("offset_x")) {
        nc.read(offset_x, "offset_x");
    }
    if (nc.varExists("offset_y")) {
        nc.read(offset_y, "offset_y");
    }
    if (nc.varExists("offset_z")) {
        nc.read(offset_z, "offset_z");
    }
    Atmosphere result{
        .voxel_scale = voxel_scale,
        .offset_x = offset_x,
        .offset_y = offset_y,
        .offset_z = offset_z
    };
#ifdef DEXRT_SINGLE_PREC
    result.temperature = create_device_copy(temperature);
    result.pressure = create_device_copy(pressure);
    result.ne = create_device_copy(ne);
    result.nh_tot = create_device_copy(nh_tot);
    result.vturb = create_device_copy(vturb);
    result.vx = create_device_copy(vx);
    result.vy = create_device_copy(vy);
    result.vz = create_device_copy(vz);
#else
    static_assert(false, "Has not been updated for Kokkos yet")
    result.temperature = Fp2d("temperature", z_dim, x_dim);
    result.pressure = Fp2d("pressure", z_dim, x_dim);
    result.ne = Fp2d("ne", z_dim, x_dim);
    result.nh_tot = Fp2d("nh_tot", z_dim, x_dim);
    result.vturb = Fp2d("vturb", z_dim, x_dim);
    result.vx = Fp2d("vx", z_dim, x_dim);
    result.vy = Fp2d("vy", z_dim, x_dim);
    result.vz = Fp2d("vz", z_dim, x_dim);

    #define DEX_DEV_COPY(X) auto JasConcat(X, dev) = X.createDeviceCopy()
    #define DEX_FLOAT_CONVERT(X) parallel_for( \
        "convert", \
        SimpleBounds<2>(z_dim, x_dim), \
        KOKKOS_LAMBDA (int z, int x) { \
            result.X(z, x) = JasConcat(X, dev)(z, x); \
        } \
    )

    DEX_DEV_COPY(temperature);
    DEX_DEV_COPY(pressure);
    DEX_DEV_COPY(ne);
    DEX_DEV_COPY(nh_tot);
    DEX_DEV_COPY(vturb);
    DEX_DEV_COPY(vx);
    DEX_DEV_COPY(vy);
    DEX_DEV_COPY(vz);
    yakl::fence();
    DEX_FLOAT_CONVERT(temperature);
    DEX_FLOAT_CONVERT(pressure);
    DEX_FLOAT_CONVERT(ne);
    DEX_FLOAT_CONVERT(nh_tot);
    DEX_FLOAT_CONVERT(vturb);
    DEX_FLOAT_CONVERT(vx);
    DEX_FLOAT_CONVERT(vy);
    DEX_FLOAT_CONVERT(vz);
    yakl::fence();

    #undef DEX_DEV_COPY
    #undef DEX_FLOAT_CONVERT

#endif

    result.nh0 = Fp2d("nh0", z_dim, x_dim);
    Kokkos::deep_copy(result.nh0, FP(0.0));

    fp_t max_vel_2;
    dex_parallel_reduce(
        "Atmosphere Max Vel",
        FlatLoop<2>(vx.extent(0), vx.extent(1)),
        KOKKOS_LAMBDA (int z, int x, fp_t& running_max) {
            fp_t vel2 = square(result.vz(z, x)) + square(result.vy(z, x)) + square(result.vx(z, x));
            if (vel2 > running_max) {
                running_max = vel2;
            }
        },
        Kokkos::Max<fp_t>(max_vel_2)
    );

    result.moving = max_vel_2 > FP(10.0);

    return result;
}

template <typename T=fp_t, class Atmosphere>
FlatAtmosphere<T> flatten(const Atmosphere& atmos) {
    FlatAtmosphere<T> result;
    result.voxel_scale = atmos.voxel_scale;
    result.offset_x = atmos.offset_x;
    result.offset_y = atmos.offset_y;
    result.offset_z = atmos.offset_z;
    result.moving = atmos.moving;
    result.temperature = collapse(atmos.temperature);
    result.pressure = collapse(atmos.pressure);
    result.ne = collapse(atmos.ne);
    result.nh_tot = collapse(atmos.nh_tot);
    result.nh0 = collapse(atmos.nh0);
    result.vturb = collapse(atmos.vturb);
    result.vx = collapse(atmos.vx);
    result.vy = collapse(atmos.vy);
    result.vz = collapse(atmos.vz);
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

template <class Atmosphere>
YAKL_INLINE vec3 get_offsets(const Atmosphere& atmos) {
    vec3 result;
    result(0) = atmos.offset_x;
    result(1) = atmos.offset_y;
    result(2) = atmos.offset_z;
    return result;
}

#else
#endif