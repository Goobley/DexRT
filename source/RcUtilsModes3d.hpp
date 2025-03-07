#if !defined(DEXRT_UTILS_MODES_3D_HPP)
#define DEXRT_UTILS_MODES_3D_HPP
#include "Types.hpp"
#include "RcConstants.hpp"
#include "RcUtilsModes.hpp"
#include "Utils.hpp"

/// Returns the packed RC flags that affect cascade storage (i.e. known at compile time)
YAKL_INLINE constexpr int RC_flags_storage_3d() {
    return RC_flags_pack(RcFlags{
        .preaverage=false,
        .dir_by_dir=DIR_BY_DIR_3D
    });
}

namespace DexImpl {
    /// Simple x^exponent calculation for both being integers whilst
    /// remaining relatively efficient
    KOKKOS_FORCEINLINE_FUNCTION constexpr int powi(int x, int exponent) {
        // We right-shift the exponent each iteration until there's no bits
        // left. If odd, we have to self-multiply twice (as (3 >> 1) == 1).
        int result = 1;
        while (exponent) {
            if (exponent & 1) {
                result *= x;
            }
            x *= x;
            exponent = exponent >> 1;
        }
        return result;
    }
}

KOKKOS_FORCEINLINE_FUNCTION constexpr int az_rays_factor(int n) {
    if constexpr (USE_BRANCHING_FACTOR_3D) {
        return DexImpl::powi(AZ_BRANCHING_FACTOR_3D, n);
    }
    return 1 << (AZ_BRANCHING_EXP_3D * n);
}

KOKKOS_FORCEINLINE_FUNCTION constexpr int polar_rays_factor(int n) {
    if constexpr (USE_BRANCHING_FACTOR_3D) {
        return DexImpl::powi(POLAR_BRANCHING_FACTOR_3D, n);
    }
    return 1 << (POLAR_BRANCHING_EXP_3D * n);
}

KOKKOS_FORCEINLINE_FUNCTION constexpr int num_az_rays(int n) {
    return C0_AZ_RAYS_3D * az_rays_factor(n);
}

KOKKOS_FORCEINLINE_FUNCTION constexpr int num_polar_rays(int n) {
    return C0_POLAR_RAYS_3D * polar_rays_factor(n);
}

YAKL_INLINE CascadeStorage3d cascade_size(const CascadeStorage3d& c0, int n) {
    CascadeStorage3d c;
    c.num_probes(0) = std::max(1, (c0.num_probes(0) >> n));
    c.num_probes(1) = std::max(1, (c0.num_probes(1) >> n));
    c.num_probes(2) = std::max(1, (c0.num_probes(2) >> n));
    const int az_factor = az_rays_factor(n);
    c.num_az_rays = c0.num_az_rays * az_factor;
    const int polar_factor = polar_rays_factor(n);
    c.num_polar_rays = c0.num_polar_rays * polar_factor;
    return c;
}

YAKL_INLINE CascadeRays3d cascade_compute_size(const CascadeRays3d& c0, int n) {
    CascadeRays3d c;
    c.num_probes(0) = std::max(1, (c0.num_probes(0) >> n));
    c.num_probes(1) = std::max(1, (c0.num_probes(1) >> n));
    c.num_probes(2) = std::max(1, (c0.num_probes(2) >> n));
    const int az_factor = az_rays_factor(n);
    c.num_az_rays = c0.num_az_rays * az_factor;
    const int polar_factor = polar_rays_factor(n);
    c.num_polar_rays = c0.num_polar_rays * polar_factor;
    return c;
}

template <int RcMode>
YAKL_INLINE CascadeRays3d cascade_compute_size(const CascadeStorage3d& c0, int n) {
    CascadeRays3d c;
    c.num_probes(0) = std::max(1, (c0.num_probes(0) >> n));
    c.num_probes(1) = std::max(1, (c0.num_probes(1) >> n));
    c.num_probes(2) = std::max(1, (c0.num_probes(2) >> n));
    if constexpr  (RcMode & RC_DIR_BY_DIR) {
        c.num_az_rays = (c0.num_az_rays * C0_AZ_RAYS_3D) * az_rays_factor(n);
        c.num_polar_rays = (c0.num_polar_rays * C0_POLAR_RAYS_3D) * polar_rays_factor(n);
    } else {
        c.num_az_rays = c0.num_az_rays * az_rays_factor(n);
        c.num_polar_rays = c0.num_polar_rays * polar_rays_factor(n);
    }
    return c;
}

template <int RcMode>
YAKL_INLINE CascadeStorage3d cascade_rays_to_storage(const CascadeRays3d& r) {
    CascadeStorage3d c;
    c.num_probes(0) = r.num_probes(0);
    c.num_probes(1) = r.num_probes(1);
    c.num_probes(2) = r.num_probes(2);
    if constexpr (RcMode & RC_DIR_BY_DIR) {
        c.num_az_rays = r.num_az_rays / C0_AZ_RAYS_3D;
        c.num_polar_rays = r.num_polar_rays / C0_POLAR_RAYS_3D;
    } else {
        c.num_az_rays = r.num_az_rays;
        c.num_polar_rays = r.num_polar_rays;
    }
    return c;
}

template <int RcMode>
YAKL_INLINE CascadeRays3d cascade_storage_to_rays(const CascadeStorage3d& c) {
    CascadeRays3d r;
    r.num_probes(0) = c.num_probes(0);
    r.num_probes(1) = c.num_probes(1);
    r.num_probes(2) = c.num_probes(2);
    if constexpr (RcMode & RC_DIR_BY_DIR) {
        r.num_az_rays = c.num_az_rays * C0_AZ_RAYS_3D;
        r.num_polar_rays = c.num_polar_rays * C0_POLAR_RAYS_3D;
    } else {
        r.num_az_rays = c.num_az_rays;
        r.num_polar_rays = c.num_polar_rays;
    }
    return r;
}

struct TexelsPerRay3d {
    int az;
    int polar;
};

template <int RcMode>
YAKL_INLINE TexelsPerRay3d upper_texels_per_ray_3d(int n) {
    TexelsPerRay3d t;
    if constexpr (USE_BRANCHING_FACTOR_3D) {
        t.az = AZ_BRANCHING_FACTOR_3D;
        t.polar = POLAR_BRANCHING_FACTOR_3D;
    } else {
        t.az = (1 << AZ_BRANCHING_EXP_3D);
        t.polar = (1 << POLAR_BRANCHING_EXP_3D);
    }
    return t;
}

/// The number of sub-tasks that need to be computed per cascade. Normally 1,
/// but the number of c0 rays in the case of DIR_BY_DIR.
template <int RcMode>
YAKL_INLINE constexpr int subset_tasks_per_cascade_3d() {
    if constexpr (RcMode & RC_DIR_BY_DIR) {
        return C0_AZ_RAYS_3D * C0_POLAR_RAYS_3D;
    } else {
        return 1;
    }
}

template <int RcMode>
YAKL_INLINE constexpr CascadeRaysSubset3d nth_rays_subset(const CascadeRays3d& rays, int n) {
    if constexpr (RcMode & RC_DIR_BY_DIR) {
        constexpr int num_az_sets = C0_AZ_RAYS_3D;
        constexpr int num_polar_sets = C0_POLAR_RAYS_3D;
        const int az_per_subset = rays.num_az_rays / num_az_sets;
        const int polar_per_subset = rays.num_polar_rays / num_polar_sets;

        // NOTE(cmo): We run sets fast over az. i.e. nested [polar, az]
        const int polar_index = n / num_az_sets;
        const int az_index = n - polar_index * num_az_sets;

        return CascadeRaysSubset3d{
            .start_az_rays = az_index * az_per_subset,
            .num_az_rays = az_per_subset,
            .start_polar_rays = polar_index * polar_per_subset,
            .num_polar_rays = polar_per_subset
        };
    } else {
        return CascadeRaysSubset3d{
            .start_az_rays = 0,
            .num_az_rays = rays.num_az_rays,
            .start_polar_rays = 0,
            .num_polar_rays = rays.num_polar_rays
        };
    }
}

/// Number of entries to store cascade
YAKL_INLINE i64 cascade_entries(const CascadeStorage3d& c) {
    i64 result = c.num_probes(0);
    result *= c.num_probes(1);
    result *= c.num_probes(2);
    result *= c.num_az_rays;
    result *= c.num_polar_rays;
    return result;
}

YAKL_INLINE vec3 probe_pos(ivec3 probe_coord, int n) {
    fp_t probe_spacing = PROBE0_SPACING * (1 << n);
    vec3 pos;
    pos(0) = (fp_t(probe_coord(0)) + FP(0.5)) * probe_spacing;
    pos(1) = (fp_t(probe_coord(1)) + FP(0.5)) * probe_spacing;
    pos(2) = (fp_t(probe_coord(2)) + FP(0.5)) * probe_spacing;
    return pos;
}

struct TrilinearCorner {
    ivec3 corner;
    vec3 frac;
};

YAKL_INLINE TrilinearCorner trilinear_corner(ivec3 probe_coord) {
    TrilinearCorner result;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        result.corner(i) = std::max(int((probe_coord(i) - 1) / 2), 0);
        // NOTE(cmo): Weights for this corner
        if (probe_coord(i) == 0) {
            // NOTE(cmo): Clamp first row/col/layer
            result.frac(i) = FP(1.0);
        } else {
            result.frac(i) = FP(0.25) + FP(0.5) * (probe_coord(i) & 1);
        }
    }
    return result;
}

YAKL_INLINE vec<8> trilinear_weights(const TrilinearCorner& tri) {
    vec<8> result;
    result(0) = tri.frac(0) * tri.frac(1) * tri.frac(2); // u_bc, v_bc, w_bc
    result(1) = (FP(1.0) - tri.frac(0)) * tri.frac(1) * tri.frac(2); // u_uc, v_bc, w_bc
    result(2) = tri.frac(0) * (FP(1.0) - tri.frac(1)) * tri.frac(2); // u_bc, v_uc, w_bc
    result(3) = tri.frac(0) * tri.frac(1) * (FP(1.0) - tri.frac(2)); // u_bc, v_bc, w_uc
    result(4) = (FP(1.0) - tri.frac(0)) * (FP(1.0) - tri.frac(1)) * tri.frac(2); // u_uc, v_uc, w_bc
    result(5) = (FP(1.0) - tri.frac(0)) * tri.frac(1) * (FP(1.0) - tri.frac(2)); // u_uc, v_bc, w_uc
    result(6) = tri.frac(0) * (FP(1.0) - tri.frac(1)) * (FP(1.0) - tri.frac(2)); // u_bc, v_uc, w_uc
    result(7) = (FP(1.0) - tri.frac(0)) * (FP(1.0) - tri.frac(1)) * (FP(1.0) - tri.frac(2)); // u_uc, v_uc, w_uc

    return result;
}

YAKL_INLINE ivec3 trilinear_coord(const TrilinearCorner& trilin, const ivec3& num_probes, int sample) {
    ivec3 coord;

    switch (sample) {
        case 0: {
            coord(0) = trilin.corner(0);
            coord(1) = trilin.corner(1);
            coord(2) = trilin.corner(2);
        } break;
        case 1: {
            coord(0) = trilin.corner(0) + 1;
            coord(1) = trilin.corner(1);
            coord(2) = trilin.corner(2);
        } break;
        case 2: {
            coord(0) = trilin.corner(0);
            coord(1) = trilin.corner(1) + 1;
            coord(2) = trilin.corner(2);

        } break;
        case 3: {
            coord(0) = trilin.corner(0);
            coord(1) = trilin.corner(1);
            coord(2) = trilin.corner(2) + 1;
        } break;
        case 4: {
            coord(0) = trilin.corner(0) + 1;
            coord(1) = trilin.corner(1) + 1;
            coord(2) = trilin.corner(2);

        } break;
        case 5: {
            coord(0) = trilin.corner(0) + 1;
            coord(1) = trilin.corner(1);
            coord(2) = trilin.corner(2) + 1;
        } break;
        case 6: {
            coord(0) = trilin.corner(0);
            coord(1) = trilin.corner(1) + 1;
            coord(2) = trilin.corner(2) + 1;
        } break;
        case 7: {
            coord(0) = trilin.corner(0) + 1;
            coord(1) = trilin.corner(1) + 1;
            coord(2) = trilin.corner(2) + 1;
        } break;
        default: {
            assert(false);
        }
    }
    coord(0) = std::min(coord(0), num_probes(0) - 1);
    coord(1) = std::min(coord(1), num_probes(1) - 1);
    coord(2) = std::min(coord(2), num_probes(2) - 1);
    return coord;
}

/// The index of the desired _ray_ within the cascade, irrespective of storage layout.
struct ProbeIndex3d {
    ivec3 coord;
    int polar;
    int az;
};

/// The index of the desired _texel_ within the cascade, affected by storage layout.
struct ProbeStorageIndex3d {
    ivec3 coord;
    int polar;
    int az;
};

template <int RcMode>
YAKL_INLINE i64 probe_linear_index(const CascadeStorage3d& dims, const ProbeIndex3d& probe) {
    // NOTE(cmo): probe_coord is stored as [u, v, w], but these are stored in the buffer as [w, v, u]
    // Current cascade storage is [w, v, u, polar, az]. If we default to dir_by_dir, we might be best at [polar, az, w, v, u]
    i64 idx;
    i64 stride;
    i32 polar, az;
    if constexpr (RcMode & RC_DIR_BY_DIR) {
        i32 factor = probe.az / dims.num_az_rays;
        az = probe.az - factor * dims.num_az_rays;
        idx = az;
        stride = dims.num_az_rays;
        factor = probe.polar / dims.num_polar_rays;
        polar = probe.polar - factor * dims.num_polar_rays;
        idx += stride * polar;
        stride *= dims.num_polar_rays;
    } else {
        az = probe.az;
        idx = probe.az;
        stride = dims.num_az_rays;
        polar = probe.polar;
        idx += stride * probe.polar;
        stride *= dims.num_polar_rays;
    }

    idx += stride * probe.coord(0);
    stride *= dims.num_probes(0);
    idx += stride * probe.coord(1);
    stride *= dims.num_probes(1);
    idx += stride * probe.coord(2);
#ifdef DEXRT_DEBUG
    if (az >= dims.num_az_rays) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [az] ({} >= {}).", az, dims.num_az_rays);
        );
        printf("%d > %d\n", az, dims.num_az_rays);
        yakl::yakl_throw("DexRT Error: Cascade index [az] out of bounds.");
    }
    if (polar >= dims.num_polar_rays) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [polar] ({} >= {}).", polar, dims.num_polar_rays);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [polar] out of bounds.");
    }
    if (probe.coord(0) >= dims.num_probes(0)) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [coord(0)] ({} >= {}).", probe.coord(0), dims.num_probes(0));
        );
        yakl::yakl_throw("DexRT Error: Cascade index [coord(0)] out of bounds.");
    }
    if (probe.coord(1) >= dims.num_probes(1)) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [coord(1)] ({} >= {}).", probe.coord(1), dims.num_probes(1));
        );
        yakl::yakl_throw("DexRT Error: Cascade index [coord(1)] out of bounds.");
    }
    if (probe.coord(2) >= dims.num_probes(2)) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [coord(2)] ({} >= {}).", probe.coord(2), dims.num_probes(2));
        );
        yakl::yakl_throw("DexRT Error: Cascade index [coord(2)] out of bounds.");
    }
#endif
    return idx;
}

template <int RcMode>
YAKL_INLINE i64 probe_linear_index(const CascadeStorage3d& dims, const ProbeStorageIndex3d& probe) {
    // NOTE(cmo): probe_coord is stored as [u, v, w], but these are stored in the buffer as [w, v, u]
    // Current cascade storage is [w, v, u, polar, az]. If we default to dir_by_dir, we might be best at [polar, az, w, v, u]
    i64 idx = probe.az;
    i64 stride = dims.num_az_rays;
    idx += stride * probe.polar;
    stride *= dims.num_polar_rays;
    idx += stride * probe.coord(0);
    stride *= dims.num_probes(0);
    idx += stride * probe.coord(1);
    stride *= dims.num_probes(1);
    idx += stride * probe.coord(2);
#ifdef DEXRT_DEBUG
    if (probe.az >= dims.num_az_rays) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [az] ({} >= {}).", probe.az, dims.num_az_rays);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [az] out of bounds.");
    }
    if (probe.polar >= dims.num_polar_rays) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [polar] ({} >= {}).", probe.polar, dims.num_polar_rays);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [polar] out of bounds.");
    }
    if (probe.coord(0) >= dims.num_probes(0)) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [coord(0)] ({} >= {}).", probe.coord(0), dims.num_probes(0));
        );
        yakl::yakl_throw("DexRT Error: Cascade index [coord(0)] out of bounds.");
    }
    if (probe.coord(1) >= dims.num_probes(1)) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [coord(1)] ({} >= {}).", probe.coord(1), dims.num_probes(1));
        );
        yakl::yakl_throw("DexRT Error: Cascade index [coord(1)] out of bounds.");
    }
    if (probe.coord(2) >= dims.num_probes(2)) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [coord(2)] ({} >= {}).", probe.coord(2), dims.num_probes(2));
        );
        yakl::yakl_throw("DexRT Error: Cascade index [coord(2)] out of bounds.");
    }
#endif
    return idx;
}

template <int RcMode>
YAKL_INLINE i64 probe_linear_index(const CascadeRays3d& dims, const ProbeIndex3d& probe) {
    CascadeStorage storage = cascade_rays_to_storage<RcMode>(dims);
    return probe_linear_index<RcMode>(storage, probe);
}

template <int RcMode>
YAKL_INLINE fp_t probe_fetch(const FpConst1d& casc, const CascadeStorage3d& dims, const ProbeStorageIndex3d& index) {
    i64 lin_idx = probe_linear_index<RcMode>(dims, index);
    return casc(lin_idx);
}

template <int RcMode>
YAKL_INLINE fp_t probe_fetch(const FpConst1d& casc, const CascadeRays3d& dims, const ProbeIndex3d& index) {
    i64 lin_idx = probe_linear_index<RcMode>(dims, index);
    return casc(lin_idx);
}

YAKL_INLINE IntervalLength cascade_interval_length_3d(int num_cascades, int n) {
    IntervalLength length;

    if constexpr (USE_SCALE_FACTOR_3D) {
        length = {
            .from = C0_LENGTH_3D * ((n == 0) ? FP(0.0) : DexImpl::powi(SPATIAL_SCALE_FACTOR_3D, n-1)),
            .to = C0_LENGTH_3D * DexImpl::powi(SPATIAL_SCALE_FACTOR_3D, n)
        };
    } else {
        length = {
            .from = C0_LENGTH_3D * ((n == 0) ? FP(0.0) : (1 << ((n-1) * SPATIAL_SCALE_EXP_3D))),
            .to = C0_LENGTH_3D * (1 << (n * SPATIAL_SCALE_EXP_3D))
        };
    }
    if (LAST_CASCADE_TO_INFTY && n == num_cascades) {
        length.to = LAST_CASCADE_MAX_DIST;
    }
    return length;
}

/// The direction of propagation along the ray (i.e. towards the probe), rather
/// than in the classical RC sense. Use with t0/t1 as defined in `probe_ray`
/// below.
YAKL_INLINE vec3 ray_dir(const CascadeRays3d& dims, int phi_idx, int theta_idx) {
    namespace Const = ConstantsFP;
    fp_t phi = FP(2.0) * Const::pi / fp_t(dims.num_az_rays) * (phi_idx + FP(0.5)); // (0, 2pi)
    fp_t cos_theta = FP(2.0) / fp_t(dims.num_polar_rays) * (theta_idx + FP(0.5)) - FP(1.0); // (-1, 1)
    fp_t sin_theta = std::sqrt(FP(1.0) - square(cos_theta));

    vec3 dir;
    dir(0) = std::cos(phi) * sin_theta;
    dir(1) = std::sin(phi) * sin_theta;
    dir(2) = cos_theta;
    return dir;
}

/// Set up a RaySegment for a particular ray of a probe. No modifications to
/// this have to be made before computing the RTE along it, i.e. the direction
/// points towards the probe centre.
YAKL_INLINE RaySegment<3> probe_ray(const CascadeRays3d& dims, int num_cascades, int n, const ProbeIndex3d& probe) {
    // NOTE(cmo): In 3D we invert directly so that we're tracing towards the
    // probe. In this case we keep the direction the same but set t0 = -t1 and
    // t1 = -t0 to offset the ray to the other side of the probe so it's tracing
    // towards the probe.
    vec3 o = probe_pos(probe.coord, n);
    vec3 dir = ray_dir(dims, probe.az, probe.polar);
    IntervalLength length = cascade_interval_length_3d(num_cascades, n);
    const fp_t t0 = -length.to;
    const fp_t t1 = -length.from;
    RaySegment<3> ray(o, dir, t0, t1);
    return ray;
}

YAKL_INLINE RaySegment<3> trilinear_probe_ray(
    const CascadeRays3d& dims,
    const CascadeRays3d& upper_dims,
    int num_cascades,
    int n,
    const ProbeIndex3d& probe,
    const ivec3& upper_probe_coord
) {
    // NOTE(cmo): In 3D we invert directly so that we're tracing towards the
    // probe. In this case we keep the direction the same but set t0 = -t1 and
    // t1 = -t0 to offset the ray to the other side of the probe so it's tracing
    // towards the probe.
    vec3 o = probe_pos(probe.coord, n);
    vec3 upper_o = probe_pos(upper_probe_coord, n+1);
    vec3 dir = ray_dir(dims, probe.az, probe.polar);
    IntervalLength length = cascade_interval_length_3d(num_cascades, n);

    vec3 end_pos = o - dir * length.from;
    // NOTE(cmo): The effective origin
    vec3 start_pos = upper_o - dir * length.to;
    const fp_t t0 = FP(0.0);
    vec3 effective_dir = end_pos - start_pos;
    const fp_t t1 = std::sqrt(square(effective_dir(0)) + square(effective_dir(1)) + square(effective_dir(2)));
    effective_dir = effective_dir * (FP(1.0) / t1);

    RaySegment<3> ray(start_pos, effective_dir, t0, t1);
    return ray;
}


#else
#endif