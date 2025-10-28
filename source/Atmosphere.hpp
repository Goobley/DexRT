#if !defined(DEXRT_ATMOSPHERE_HPP)
#define DEXRT_ATMOSPHERE_HPP

#include "Config.hpp"
#include "Types.hpp"
#include <string>
#include "YAKL_netcdf.h"
#include "JasPP.hpp"

inline Atmosphere load_atmos(const std::string& path) {
    typedef yakl::Array<f32, 1, yakl::memHost> Fp1dLoad;
    typedef yakl::Array<f32, 2, yakl::memHost> Fp2dLoad;

    yakl::SimpleNetCDF nc;
    nc.open(path, yakl::NETCDF_MODE_READ);
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
    result.temperature = temperature.createDeviceCopy();
    result.pressure = pressure.createDeviceCopy();
    result.ne = ne.createDeviceCopy();
    result.nh_tot = nh_tot.createDeviceCopy();
    result.vturb = vturb.createDeviceCopy();
    result.vx = vx.createDeviceCopy();
    result.vy = vy.createDeviceCopy();
    result.vz = vz.createDeviceCopy();
#else
    result.temperature = Fp2d("temperature", z_dim, x_dim);
    result.pressure = Fp2d("pressure", z_dim, x_dim);
    result.ne = Fp2d("ne", z_dim, x_dim);
    result.nh_tot = Fp2d("nh_tot", z_dim, x_dim);
    result.vturb = Fp2d("vturb", z_dim, x_dim);
    result.vx = Fp2d("vx", z_dim, x_dim);
    result.vy = Fp2d("vy", z_dim, x_dim);
    result.vz = Fp2d("vz", z_dim, x_dim);

    #define DEX_DEV_COPY(X) auto JasConcat(X, dev) = X.createDeviceCopy()
    #define DEX_FLOAT_CONVERT(X) dex_parallel_for( \
        "convert", \
        FlatLoop<2>(z_dim, x_dim), \
        YAKL_LAMBDA (int z, int x) { \
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
    result.nh0 = FP(0.0);

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
    result.moving = std::sqrt(max_vel_2) > FP(10.0);

    return result;
}

inline AtmosphereNd<3, yakl::memHost> load_atmos_3d_host(const std::string& path) {
    typedef yakl::Array<f32, 1, yakl::memHost> Fp1dLoad;
    typedef yakl::Array<f32, 2, yakl::memHost> Fp2dLoad;
    typedef yakl::Array<f32, 3, yakl::memHost> Fp3dLoad;

    yakl::SimpleNetCDF nc;
    nc.open(path, yakl::NETCDF_MODE_READ);
    int x_dim = nc.getDimSize("x");
    int y_dim = nc.getDimSize("y");
    int z_dim = nc.getDimSize("z");

    Fp3dLoad temperature("temperature", z_dim, y_dim, x_dim);
    Fp3dLoad pressure("pressure", z_dim, y_dim, x_dim);
    Fp3dLoad ne("ne", z_dim, y_dim, x_dim);
    Fp3dLoad nh_tot("nh_tot", z_dim, y_dim, x_dim);
    Fp3dLoad vturb("vturb", z_dim, y_dim, x_dim);
    Fp3dLoad vx("vx", z_dim, y_dim, x_dim);
    Fp3dLoad vy("vy", z_dim, y_dim, x_dim);
    Fp3dLoad vz("vz", z_dim, y_dim, x_dim);

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
    AtmosphereNd<3, yakl::memHost> result{
        .voxel_scale = voxel_scale,
        .offset_x = offset_x,
        .offset_y = offset_y,
        .offset_z = offset_z
    };
#ifdef DEXRT_SINGLE_PREC
    result.temperature = temperature;
    result.pressure = pressure;
    result.ne = ne;
    result.nh_tot = nh_tot;
    result.vturb = vturb;
    result.vx = vx;
    result.vy = vy;
    result.vz = vz;
#else
    result.temperature = Fp3dHost("temperature", z_dim, y_dim, x_dim);
    result.pressure = Fp3dHost("pressure", z_dim, y_dim, x_dim);
    result.ne = Fp3dHost("ne", z_dim, y_dim, x_dim);
    result.nh_tot = Fp3dHost("nh_tot", z_dim, y_dim, x_dim);
    result.vturb = Fp3dHost("vturb", z_dim, y_dim, x_dim);
    result.vx = Fp3dHost("vx", z_dim, y_dim, x_dim);
    result.vy = Fp3dHost("vy", z_dim, y_dim, x_dim);
    result.vz = Fp3dHost("vz", z_dim, y_dim, x_dim);

    FlatLoop<3> convert_loop(z_dim, y_dim, x_dim);
    #define DEX_FLOAT_CONVERT(X) Kokkos::parallel_for( \
        "convert", \
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, convert_loop.num_iter), \
        KOKKOS_LAMBDA (i64 i) { \
            auto idxs = convert_loop.unpack(i); \
            result.X(idxs[0], idxs[1], idxs[2]) = JAS_EXPAND(X)(idxs[0], idxs[1], idxs[2]); \
        } \
    )

    DEX_FLOAT_CONVERT(temperature);
    DEX_FLOAT_CONVERT(pressure);
    DEX_FLOAT_CONVERT(ne);
    DEX_FLOAT_CONVERT(nh_tot);
    DEX_FLOAT_CONVERT(vturb);
    DEX_FLOAT_CONVERT(vx);
    DEX_FLOAT_CONVERT(vy);
    DEX_FLOAT_CONVERT(vz);
    Kokkos::fence();

    #undef DEX_FLOAT_CONVERT
#endif

    result.nh0 = Fp3dHost("nh0", z_dim, y_dim, x_dim);
    result.nh0 = FP(0.0);

    fp_t max_vel_2;
    dex_parallel_reduce<Kokkos::DefaultHostExecutionSpace>(
        "Atmosphere Max Vel",
        FlatLoop<3>(vx.extent(0), vx.extent(1), vx.extent(2)),
        KOKKOS_LAMBDA (int z, int y, int x, fp_t& running_max) {
            fp_t vel2 = square(result.vz(z, y, x)) + square(result.vy(z, y, x)) + square(result.vx(z, y, x));
            if (vel2 > running_max) {
                running_max = vel2;
            }
        },
        Kokkos::Max<fp_t>(max_vel_2)
    );
    result.moving = std::sqrt(max_vel_2) > FP(10.0);

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
    result.temperature = atmos.temperature.collapse();
    result.pressure = atmos.pressure.collapse();
    result.ne = atmos.ne.collapse();
    result.nh_tot = atmos.nh_tot.collapse();
    result.nh0 = atmos.nh0.collapse();
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

template <class Atmosphere>
YAKL_INLINE vec3 get_offsets(const Atmosphere& atmos) {
    vec3 result;
    result(0) = atmos.offset_x;
    result(1) = atmos.offset_y;
    result(2) = atmos.offset_z;
    return result;
}

inline bool atmosphere_file_is_sparse(const std::string& path) {
    yakl::SimpleNetCDF nc;
    nc.open(path, yakl::NETCDF_MODE_READ);
    int ncid = nc.file.ncid;

    constexpr const char* sparse = "sparse";
    nc_type dtype;
    size_t att_len;
    if (nc_inq_att(ncid, NC_GLOBAL, sparse, &dtype, &att_len) != NC_NOERR) {
        // NOTE(cmo): If not specified, it's dense.
        return false;
    }
    if (att_len != 1) {
        throw std::runtime_error("Expected sparse attribute to be 1x int");
    }

    i32 is_sparse;
    if (nc_get_att_int(ncid, NC_GLOBAL, sparse, &is_sparse) != NC_NOERR) {
        throw std::runtime_error("NetCDF error");
    }
    return is_sparse;
}

template <int NumDim>
inline SparseAtmosphere load_sparse_atmosphere(yakl::SimpleNetCDF& nc) {
    typedef yakl::Array<f32, 1, yakl::memDevice> FpLoad;

    auto ncwrap = [] (int ierr , int line) {
        if (ierr != NC_NOERR) {
            printf("NetCDF Error at line: %d\n", line);
            printf("%s\n",nc_strerror(ierr));
            Kokkos::abort(nc_strerror(ierr));
        }
    };

    const int ncid = nc.file.ncid;
    int x_dim, y_dim, z_dim, block_size;
    ncwrap(nc_get_att_int(ncid, NC_GLOBAL, "num_x", &x_dim), __LINE__);
    ncwrap(nc_get_att_int(ncid, NC_GLOBAL, "num_z", &z_dim), __LINE__);
    y_dim = 1;
    if constexpr (NumDim > 2) {
        ncwrap(nc_get_att_int(ncid, NC_GLOBAL, "num_y", &y_dim), __LINE__);
    }
    ncwrap(nc_get_att_int(ncid, NC_GLOBAL, "block_size", &block_size), __LINE__);

    if constexpr (NumDim == 2) {
        if (block_size != BLOCK_SIZE) {
            throw std::runtime_error(fmt::format("block_size attribute in atmosphere file ({}) does not match compiled BLOCK_SIZE ({}).", block_size, BLOCK_SIZE));
        }
    } else {
        if (block_size != BLOCK_SIZE_3D) {
            throw std::runtime_error(fmt::format("block_size attribute in atmosphere file ({}) does not match compiled BLOCK_SIZE_3D ({}).", block_size, BLOCK_SIZE_3D));
        }
    }

    i64 num_active_cells = nc.getDimSize("cells");
    FpLoad temperature("temperature", num_active_cells);
    FpLoad pressure("pressure", num_active_cells);
    FpLoad ne("ne", num_active_cells);
    FpLoad nh_tot("nh_tot", num_active_cells);
    FpLoad vturb("vturb", num_active_cells);
    FpLoad vx("vx", num_active_cells);
    FpLoad vy("vy", num_active_cells);
    FpLoad vz("vz", num_active_cells);

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
    SparseAtmosphere result{
        .voxel_scale = voxel_scale,
        .offset_x = offset_x,
        .offset_y = offset_y,
        .offset_z = offset_z,
        .num_x = x_dim,
        .num_y = y_dim,
        .num_z = z_dim
    };

#ifdef DEXRT_SINGLE_PREC
    result.temperature = temperature;
    result.pressure = pressure;
    result.ne = ne;
    result.nh_tot = nh_tot;
    result.vturb = vturb;
    result.vx = vx;
    result.vy = vy;
    result.vz = vz;
#else
    result.temperature = Fp1d("temperature", num_active_cells);
    result.pressure = Fp1d("pressure", num_active_cells);
    result.ne = Fp1d("ne", num_active_cells);
    result.nh_tot = Fp1d("nh_tot", num_active_cells);
    result.vturb = Fp1d("vturb", num_active_cells);
    result.vx = Fp1d("vx", num_active_cells);
    result.vy = Fp1d("vy", num_active_cells);
    result.vz = Fp1d("vz", num_active_cells);

    #define DEX_FLOAT_CONVERT(X) dex_parallel_for( \
        "convert", \
        FlatLoop<1>(num_active_cells), \
        YAKL_LAMBDA (i64 x) { \
            result.X(x) = X(x); \
        } \
    )

    DEX_FLOAT_CONVERT(temperature);
    DEX_FLOAT_CONVERT(pressure);
    DEX_FLOAT_CONVERT(ne);
    DEX_FLOAT_CONVERT(nh_tot);
    DEX_FLOAT_CONVERT(vturb);
    DEX_FLOAT_CONVERT(vx);
    DEX_FLOAT_CONVERT(vy);
    DEX_FLOAT_CONVERT(vz);
    yakl::fence();

    #undef DEX_FLOAT_CONVERT

#endif

    result.nh0 = Fp1d("nh0", num_active_cells);
    result.nh0 = FP(0.0);

    fp_t max_vel_2;
    dex_parallel_reduce(
        "Atmosphere Max Vel",
        FlatLoop<1>(vx.extent(0)),
        KOKKOS_LAMBDA (i64 x, fp_t& running_max) {
            fp_t vel2 = square(result.vz(x)) + square(result.vy(x)) + square(result.vx(x));
            if (vel2 > running_max) {
                running_max = vel2;
            }
        },
        Kokkos::Max<fp_t>(max_vel_2)
    );
    result.moving = std::sqrt(max_vel_2) > FP(10.0);

    return result;
}

#else
#endif