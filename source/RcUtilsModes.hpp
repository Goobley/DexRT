#if !defined(DEXRT_UTILS_MODES_HPP)
#define DEXRT_UTILS_MODES_HPP
#include "Types.hpp"

constexpr int RC_DYNAMIC = 0x1;

struct CascadeDims {
    ivec2 num_probes;
    int num_flat_dirs;
    int wave_batch;
    int num_incl;
};

YAKL_INLINE CascadeDims cascade_size(const CascadeDims& c0, int n) {
    CascadeDims c;
    c.num_probes(0) = std::max(1, (c0.num_probes(0) >> n));
    c.num_probes(1) = std::max(1, (c0.num_probes(1) >> n));
    c.num_flat_dirs = c0.num_flat_dirs * (1 << (CASCADE_BRANCHING_FACTOR * n));
    c.num_incl = c0.num_incl;
    c.wave_batch = c0.wave_batch;
    return c;
}

YAKL_INLINE vec2 probe_pos(ivec2 probe_coord, const CascadeDims& c0, int n) {
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
    result.frac(0) = FP(0.25) + FP(0.5) * (probe_coord(0) % 2);
    result.frac(1) = FP(0.25) + FP(0.5) * (probe_coord(1) % 2);
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

YAKL_INLINE ivec2 bilinear_offset(const BilinearCorner& bilin, const CascadeDims& dims, int sample) {
    const bool u0 = bilin.corner(0) == 0;
    const bool u_max = bilin.corner(0) == (dims.num_probes(0) - 1);
    const bool u_clamp = (u0 || u_max);

    const bool v0 = bilin.corner(1) == 0;
    const bool v_max = bilin.corner(1) == (dims.num_probes(1) - 1);
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
}

struct ProbeIndex {
    ivec2 coord;
    int dir;
    int incl;
    int wave;
};

YAKL_INLINE i64 probe_linear_index(const CascadeDims& dims, const ProbeIndex& probe) {
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
    return idx;
}


#else
#endif