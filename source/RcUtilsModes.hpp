#if !defined(DEXRT_UTILS_MODES_HPP)
#define DEXRT_UTILS_MODES_HPP
#include "Types.hpp"
#include "State.hpp"
#include "RcConstants.hpp"

/// Returns the packed RC flags that affect cascade storage (i.e. known at compile time)
YAKL_INLINE constexpr int RC_flags_storage_2d() {
    return RC_flags_pack(RcFlags{
        .preaverage=PREAVERAGE,
        .dir_by_dir=DIR_BY_DIR
    });
}

YAKL_INLINE constexpr int num_flat_rays_exponent(int n) {
    if constexpr (VARY_BRANCHING_FACTOR) {
        if (n < BRANCHING_FACTOR_SWITCH) {
            return CASCADE_BRANCHING_FACTOR * n;
        } else {
            return CASCADE_BRANCHING_FACTOR * (BRANCHING_FACTOR_SWITCH - 1)
                + UPPER_BRANCHING_FACTOR * (n - (BRANCHING_FACTOR_SWITCH - 1));
        }
    } else {
        return CASCADE_BRANCHING_FACTOR * n;
    }
}

YAKL_INLINE constexpr int num_flat_rays_storage(const CascadeStorage& c0, int n) {
    int exponent = num_flat_rays_exponent(n);
    return c0.num_flat_dirs * (1 << exponent);
}

YAKL_INLINE constexpr int num_flat_rays_compute(const CascadeRays& c0, int n) {
    int exponent = num_flat_rays_exponent(n);
    return c0.num_flat_dirs * (1 << exponent);
}

YAKL_INLINE CascadeStorage cascade_size(const CascadeStorage& c0, int n) {
    CascadeStorage c;
    c.num_probes(0) = std::max(1, (c0.num_probes(0) >> n));
    c.num_probes(1) = std::max(1, (c0.num_probes(1) >> n));
    c.num_flat_dirs = num_flat_rays_storage(c0, n);
    c.num_incl = c0.num_incl;
    c.wave_batch = c0.wave_batch;
    return c;
}

YAKL_INLINE CascadeRays cascade_compute_size(const CascadeRays& c0, int n) {
    CascadeRays c;
    c.num_probes(0) = std::max(1, (c0.num_probes(0) >> n));
    c.num_probes(1) = std::max(1, (c0.num_probes(1) >> n));
    c.num_flat_dirs = num_flat_rays_compute(c0, n);
    c.num_incl = c0.num_incl;
    c.wave_batch = c0.wave_batch;
    return c;
}

template <int RcMode>
YAKL_INLINE CascadeRays cascade_compute_size(const CascadeStorage& c0, int n) {
    CascadeRays c;
    c.num_probes(0) = std::max(1, (c0.num_probes(0) >> n));
    c.num_probes(1) = std::max(1, (c0.num_probes(1) >> n));
    if constexpr (VARY_BRANCHING_FACTOR) {
        c.num_flat_dirs = num_flat_rays_storage(c0, n);
        if constexpr (RcMode & RC_DIR_BY_DIR) {
            c.num_flat_dirs *= PROBE0_NUM_RAYS;
        }
    } else {
        if constexpr (RcMode & RC_PREAVERAGE) {
            c.num_flat_dirs = c0.num_flat_dirs * (1 << (CASCADE_BRANCHING_FACTOR * (n + 1)));
        } else if constexpr  (RcMode & RC_DIR_BY_DIR) {
            c.num_flat_dirs = (c0.num_flat_dirs * PROBE0_NUM_RAYS) * (1 << (CASCADE_BRANCHING_FACTOR * n));
        } else {
            c.num_flat_dirs = c0.num_flat_dirs * (1 << (CASCADE_BRANCHING_FACTOR * n));
        }
    }
    c.num_incl = c0.num_incl;
    c.wave_batch = c0.wave_batch;
    return c;
}

template <int RcMode>
YAKL_INLINE CascadeStorage cascade_rays_to_storage(const CascadeRays& r) {
    CascadeStorage c;
    c.num_probes(0) = r.num_probes(0);
    c.num_probes(1) = r.num_probes(1);
    if constexpr (RcMode & RC_PREAVERAGE) {
        c.num_flat_dirs = r.num_flat_dirs / (1 << CASCADE_BRANCHING_FACTOR);
    } else if constexpr (RcMode & RC_DIR_BY_DIR) {
        c.num_flat_dirs = r.num_flat_dirs / PROBE0_NUM_RAYS;
    } else {
        c.num_flat_dirs = r.num_flat_dirs;
    }
    c.num_incl = r.num_incl;
    c.wave_batch = r.wave_batch;
    return c;
}

template <int RcMode>
YAKL_INLINE CascadeRays cascade_storage_to_rays(const CascadeStorage& c) {
    CascadeRays r;
    r.num_probes(0) = c.num_probes(0);
    r.num_probes(1) = c.num_probes(1);
    if constexpr (RcMode & RC_PREAVERAGE) {
        r.num_flat_dirs = c.num_flat_dirs * (1 << CASCADE_BRANCHING_FACTOR);
    } else if constexpr (RcMode & RC_DIR_BY_DIR) {
        r.num_flat_dirs = c.num_flat_dirs * PROBE0_NUM_RAYS;
    } else {
        r.num_flat_dirs = c.num_flat_dirs;
    }
    r.num_incl = c.num_incl;
    r.wave_batch = c.wave_batch;
    return r;
}

/// The number of rays to be computed for each texel in the cascade arrays.
/// These are essentially used for preaveraging, where we store the 4 rays in
/// one write to global (since they are always read as an averaged group).
template <int RcMode>
YAKL_INLINE constexpr int rays_per_stored_texel() {
    if constexpr (RcMode & RC_PREAVERAGE) {
        return (1 << CASCADE_BRANCHING_FACTOR);
    } else {
        return 1;
    }
}

/// The number of upper ray directions to be loaded/averaged for each ray in the
/// current cascade. n refers to the current cascade, and returns the number of
/// ray directions to average in n+1
template <int RcMode>
YAKL_INLINE int upper_texels_per_ray(int n) {
    if constexpr (RcMode & RC_PREAVERAGE) {
        return 1;
    } else {
        if constexpr (VARY_BRANCHING_FACTOR) {
            if ((n + 1) < BRANCHING_FACTOR_SWITCH) {
                return (1 << CASCADE_BRANCHING_FACTOR);
            } else {
                return (1 << UPPER_BRANCHING_FACTOR);
            }
        } else {
            return (1 << CASCADE_BRANCHING_FACTOR);
        }
    }
}

/// The associated ray index in cascade n+1 from the direction in n
YAKL_INLINE int upper_ray_idx(int this_dir, int n) {
    if constexpr (VARY_BRANCHING_FACTOR) {
        if ((n + 1) < BRANCHING_FACTOR_SWITCH) {
            return this_dir * (1 << CASCADE_BRANCHING_FACTOR);
        } else {
            return this_dir * (1 << UPPER_BRANCHING_FACTOR);
        }
    } else {
        return this_dir * (1 << CASCADE_BRANCHING_FACTOR);
    }
}

/// Number of flat directions to average in C0 merge to J.
template <int RcMode>
YAKL_INLINE constexpr int c0_dirs_to_average() {
    if constexpr (RcMode & RC_PREAVERAGE) {
        return PROBE0_NUM_RAYS / (1 << CASCADE_BRANCHING_FACTOR);
    } else {
        return PROBE0_NUM_RAYS;
    }
}

/// The number of sub-tasks that need to be computed per cascade. Normally 1,
/// but the number of c0 rays in the case of DIR_BY_DIR.
template <int RcMode>
YAKL_INLINE constexpr int subset_tasks_per_cascade() {
    if constexpr (RcMode & RC_DIR_BY_DIR) {
        return PROBE0_NUM_RAYS;
    } else {
        return 1;
    }
}

template <int RcMode>
YAKL_INLINE constexpr CascadeRaysSubset nth_rays_subset(const CascadeRays& rays, int n) {
    if constexpr (RcMode & RC_DIR_BY_DIR) {
        const int dirs_per_subset = rays.num_flat_dirs / PROBE0_NUM_RAYS;
        return CascadeRaysSubset{
            .start_probes=ivec2(0),
            .num_probes=rays.num_probes,
            .start_flat_dirs=n * dirs_per_subset,
            .num_flat_dirs=dirs_per_subset,
            .start_wave_batch=0,
            .wave_batch=rays.wave_batch,
            .start_incl=0,
            .num_incl=rays.num_incl
        };
    } else {
        return CascadeRaysSubset{
            .start_probes=ivec2(0),
            .num_probes=rays.num_probes,
            .start_flat_dirs=0,
            .num_flat_dirs=rays.num_flat_dirs,
            .start_wave_batch=0,
            .wave_batch=rays.wave_batch,
            .start_incl=0,
            .num_incl=rays.num_incl
        };
    }
}


/// Number of entries to store cascade
YAKL_INLINE i64 cascade_entries(const CascadeStorage& c) {
    i64 result = c.num_probes(0);
    result *= c.num_probes(1);
    result *= c.num_flat_dirs;
    result *= c.num_incl;
    result *= c.wave_batch;
    return result;
}

/// Number of entries per probe of this level
YAKL_INLINE i64 single_probe_storage(const CascadeStorage& c) {
    i64 result = 1;
    result *= c.num_flat_dirs;
    result *= c.num_incl;
    result *= c.wave_batch;
    return result;
}

YAKL_INLINE constexpr fp_t probe_spacing(int n) {
    fp_t probe_spacing = PROBE0_SPACING * (1 << n);
    return probe_spacing;
}

YAKL_INLINE vec2 probe_pos(ivec2 probe_coord, int n) {
    fp_t probe_spacing = PROBE0_SPACING * (1 << n);
    vec2 pos;
    pos(0) = (fp_t(probe_coord(0)) + FP(0.5)) * probe_spacing;
    pos(1) = (fp_t(probe_coord(1)) + FP(0.5)) * probe_spacing;
    return pos;
}

struct BilinearCorner {
    ivec2 corner;
    vec2 frac;
};

YAKL_INLINE BilinearCorner bilinear_corner(ivec2 probe_coord) {
    BilinearCorner result;
    result.corner(0) = std::max(int((probe_coord(0) - 1) / 2), 0);
    result.corner(1) = std::max(int((probe_coord(1) - 1) / 2), 0);
    // NOTE(cmo): Weights for this corner
    if (probe_coord(0) == 0) {
        // NOTE(cmo): Clamp first row
        result.frac(0) = FP(1.0);
    } else {
        result.frac(0) = FP(0.25) + FP(0.5) * (probe_coord(0) % 2);
    }
    if (probe_coord(1) == 0) {
        // NOTE(cmo): Clamp first col
        result.frac(1) = FP(1.0);
    } else {
        result.frac(1) = FP(0.25) + FP(0.5) * (probe_coord(1) % 2);
    }
    return result;
}

YAKL_INLINE vec4 bilinear_weights(const BilinearCorner& bilin) {
    vec4 result;
    result(0) = bilin.frac(0) * bilin.frac(1); // u_bc, v_bc
    result(1) = (FP(1.0) - bilin.frac(0)) * bilin.frac(1); // u_uc, v_bc
    result(2) = bilin.frac(0) * (FP(1.0) - bilin.frac(1)); // u_bc, v_uc
    result(3) = (FP(1.0) - bilin.frac(0)) * (FP(1.0) - bilin.frac(1)); // u_uc, v_uc
    return result;
}

template <int u, int v>
YAKL_INLINE ivec2 bilinear_offset() {
    ivec2 result;
    result(0) = u;
    result(1) = v;
    return result;
};

YAKL_INLINE ivec2 bilinear_offset(const BilinearCorner& bilin, const ivec2& num_probes, int sample) {
    // const bool u0 = bilin.corner(0) == 0;
    const bool u0 = false; // NOTE(cmo): Handled by initial weight
    const bool u_max = bilin.corner(0) == (num_probes(0) - 1);
    const bool u_clamp = (u0 || u_max);

    // const bool v0 = bilin.corner(1) == 0;
    const bool v0 = false; // NOTE(cmo): Handled by initial weight
    const bool v_max = bilin.corner(1) == (num_probes(1) - 1);
    const bool v_clamp = (v0 || v_max);
    switch (sample) {
        case 0: {
            return bilinear_offset<0, 0>();
        } break;
        case 1: {
            if (u_clamp) {
                return bilinear_offset<0, 0>();
            } else {
                return bilinear_offset<1, 0>();
            }
        } break;
        case 2: {
            if (v_clamp) {
                return bilinear_offset<0, 0>();
            } else {
                return bilinear_offset<0, 1>();
            }
        } break;
        case 3: {
            if (u_clamp && v_clamp) {
                return bilinear_offset<0, 0>();
            } else if (u_clamp) {
                return bilinear_offset<0, 1>();
            } else if (v_clamp) {
                return bilinear_offset<1, 0>();
            } else {
                return bilinear_offset<1, 1>();
            }
        } break;
    }
    assert(false);
    return bilinear_offset<0, 0>();
}

/// The index of the desired _ray_ within the cascade, irrespective of storage layout.
struct ProbeIndex {
    ivec2 coord;
    int dir;
    int incl;
    int wave;
};

/// The index of the desired _texel_ within the cascade, affected by storage layout.
struct ProbeStorageIndex {
    ivec2 coord;
    int dir;
    int incl;
    int wave;
};

template <int RcMode>
YAKL_INLINE i64 probe_linear_index(const CascadeStorage& dims, const ProbeIndex& probe) {
    // NOTE(cmo): probe_coord is stored as [u, v], but these are stored in the buffer as [v, u]
    // Current cascade storage is [v, u, ray, wave, incl] to give coalesced warp access
    i64 idx = probe.incl;
    i64 dim_mul = dims.num_incl;
    idx += dim_mul * probe.wave;
    dim_mul *= dims.wave_batch;

    int dir = probe.dir;
    if constexpr (RcMode & RC_PREAVERAGE) {
        dir /= (1 << CASCADE_BRANCHING_FACTOR);
    } else if constexpr (RcMode & RC_DIR_BY_DIR) {
        dir = dir % dims.num_flat_dirs;
    }
    idx += dim_mul * dir;
    dim_mul *= dims.num_flat_dirs;
    idx += dim_mul * probe.coord(0);
    dim_mul *= dims.num_probes(0);
    idx += dim_mul * probe.coord(1);
#ifdef DEXRT_DEBUG
    if (probe.incl >= dims.num_incl) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [incl] ({} >= {}).", probe.incl, dims.num_incl);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [incl] out of bounds.");
    }
    if (probe.wave >= dims.wave_batch) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [wave] ({} >= {}).", probe.wave, dims.wave_batch);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [wave] out of bounds.");
    }
    if (dir >= dims.num_flat_dirs) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [dir] ({} >= {}).", dir, dims.num_flat_dirs);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [dir] out of bounds.");
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
#endif
    return idx;
}

template <int RcMode>
YAKL_INLINE i64 probe_linear_index(const CascadeStorage& dims, const ProbeStorageIndex& probe) {
    // NOTE(cmo): probe_coord is stored as [u, v], but these are stored in the buffer as [v, u]
    // Current cascade storage is [v, u, ray, wave, incl] to give coalesced warp access
    i64 idx = probe.incl;
    i64 dim_mul = dims.num_incl;
    idx += dim_mul * probe.wave;
    dim_mul *= dims.wave_batch;
    idx += dim_mul * probe.dir;
    dim_mul *= dims.num_flat_dirs;
    idx += dim_mul * probe.coord(0);
    dim_mul *= dims.num_probes(0);
    idx += dim_mul * probe.coord(1);
#ifdef DEXRT_DEBUG
    if (probe.incl >= dims.num_incl) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [incl] ({} >= {}).", probe.incl, dims.num_incl);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [incl] out of bounds.");
    }
    if (probe.wave >= dims.wave_batch) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [wave] ({} >= {}).", probe.wave, dims.wave_batch);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [wave] out of bounds.");
    }
    if (probe.dir >= dims.num_flat_dirs) {
        YAKL_EXECUTE_ON_HOST_ONLY(
            fmt::println(stderr, "DexRT Error: Cascade index [dir] ({} >= {}).", probe.dir, dims.num_flat_dirs);
        );
        yakl::yakl_throw("DexRT Error: Cascade index [dir] out of bounds.");
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
#endif
    return idx;
}

template <int RcMode>
YAKL_INLINE i64 probe_linear_index(const CascadeRays& dims, const ProbeIndex& probe) {
    CascadeStorage storage = cascade_rays_to_storage<RcMode>(dims);
    return probe_linear_index<RcMode>(storage, probe);
}

template <int RcMode>
YAKL_INLINE fp_t probe_fetch(const FpConst1d& casc, const CascadeStorage& dims, const ProbeStorageIndex& index) {
    i64 lin_idx = probe_linear_index<RcMode>(dims, index);
    return casc(lin_idx);
}

template <int RcMode>
YAKL_INLINE fp_t probe_fetch(const FpConst1d& casc, const CascadeRays& dims, const ProbeIndex& index) {
    i64 lin_idx = probe_linear_index<RcMode>(dims, index);
    return casc(lin_idx);
}

struct IntervalLength {
    fp_t from;
    fp_t to;
};

YAKL_INLINE IntervalLength cascade_interval_length(int num_cascades, int n) {
    int near_exp = 0;
    if (n > 0) {
        near_exp = num_flat_rays_exponent(n-1);
    }
    int far_exp = num_flat_rays_exponent(n);

    IntervalLength length = {
        .from = PROBE0_LENGTH * ((n == 0) ? FP(0.0) : (1 << near_exp)),
        .to = PROBE0_LENGTH * (1 << far_exp)
    };
    if (LAST_CASCADE_TO_INFTY && n == num_cascades) {
        length.to = LAST_CASCADE_MAX_DIST;
    }
    return length;
}

YAKL_INLINE vec2 ray_dir(const CascadeRays& dims, int phi_idx) {
    namespace Const = ConstantsFP;
    // NOTE(cmo): If this angle-generation code is adjusted, also adjust probe_frac_dir_idx
    fp_t phi = FP(2.0) * Const::pi / fp_t(dims.num_flat_dirs) * (phi_idx + FP(0.5));
    vec2 dir;
    dir(0) = std::cos(phi);
    dir(1) = std::sin(phi);
    return dir;
}

YAKL_INLINE RayProps ray_props(const CascadeRays& dims, int num_cascades, int n, const ProbeIndex& probe) {
    RayProps ray;
    ray.centre = probe_pos(probe.coord, n);

    ray.dir = ray_dir(dims, probe.dir);

    IntervalLength length = cascade_interval_length(num_cascades, n);
    ray.start(0) = ray.centre(0) + ray.dir(0) * length.from;
    ray.start(1) = ray.centre(1) + ray.dir(1) * length.from;
    ray.end(0) = ray.centre(0) + ray.dir(0) * length.to;
    ray.end(1) = ray.centre(1) + ray.dir(1) * length.to;
    return ray;
}

YAKL_INLINE RayProps invert_direction(RayProps props) {
    RayProps invert(props);
    invert.end = props.start;
    invert.start = props.end;
    invert.dir(0) = -props.dir(0);
    invert.dir(1) = -props.dir(1);
    return invert;
}

/// Compute mu for traversing this ray from end to start
YAKL_INLINE vec3 inverted_mu(const RayProps& ray, fp_t incl) {
    const fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
    vec3 mu;
    mu(0) = -ray.dir(0) * sin_theta;
    mu(1) = -incl;
    mu(2) = -ray.dir(1) * sin_theta;
    return mu;
}

/// Returns the fractional index associated with a direction. Note that this is
/// in the range [-num_flat_dirs/2, num_flat_dirs/2] and needs to be remapped
/// accordingly.
YAKL_INLINE fp_t probe_frac_dir_idx(const CascadeRays& dims, vec2 dir) {
    // NOTE(cmo): If this index-generation code is adjusted, also adjust ray_props
    using ConstantsFP::pi;
    fp_t angle = std::atan2(dir(1), dir(0));
    // if (angle < FP(0.0)) {
    //     angle += FP(2.0) * pi;
    // }
    fp_t angle_ratio = angle / (FP(2.0) * pi);
    fp_t frac_idx = angle_ratio * fp_t(dims.num_flat_dirs) - FP(0.5);
    // if (frac_idx < FP(0.0)) {
    //     frac_idx += dims.num_flat_dirs;
    // }
    return frac_idx;
}

struct SphericalAngularRange {
    /// polar (y) angle radians [min, max]
    vec2 theta_range;
    /// x-z angle radians [min, max]
    vec2 phi_range;
};

/// Returns the theta and phi ranges associated with a c0 ray (currently assumes all inclinations)
YAKL_INLINE
SphericalAngularRange c0_flat_dir_angular_range(
    const CascadeRays& dims,
    const InclQuadrature& incl_quad,
    int phi_idx
) {
    using ConstantsFP::pi;

    SphericalAngularRange result;
    result.theta_range(0) = std::acos(incl_quad.muy(dims.num_incl-1));
    result.theta_range(1) = std::acos(incl_quad.muy(0));

    const fp_t phi_step = FP(2.0) * pi / fp_t(dims.num_flat_dirs);
    result.phi_range(0) = phi_step * phi_idx;
    result.phi_range(1) = phi_step * (phi_idx+1);
    return result;
}


#else
#endif