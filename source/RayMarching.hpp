#if !defined(DEXRT_RAY_MARCHING_2_HPP)
#define DEXRT_RAY_MARCHING_2_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "JasPP.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "EmisOpac.hpp"
#include "DirectionalEmisOpacInterp.hpp"
#include <optional>

struct RayMarchState2d {
    /// Start pos
    vec2 p0;
    /// end pos
    vec2 p1;

    /// Current cell coordinate
    ivec2 curr_coord;
    /// Next cell coordinate
    ivec2 next_coord;
    /// Final cell coordinate -- for intersections with the outer edge of the
    /// box, this isn't floor(p1), but inside it is.
    ivec2 final_coord;
    /// Integer step dir
    ivec2 step;

    /// t to next hit per axis
    vec2 next_hit;
    /// t increment per step per axis
    vec2 delta;
    /// t to stop at
    fp_t max_t;

    /// axis increment
    vec2 direction;
    /// value of t at current intersection (far side of curr_coord, just before entering next_coord)
    fp_t t = FP(0.0);
    /// length of step
    fp_t dt = FP(0.0);
};

struct RayStartEnd {
    yakl::SArray<fp_t, 1, NUM_DIM> start;
    yakl::SArray<fp_t, 1, NUM_DIM> end;
};

struct Box {
    vec2 dims[NUM_DIM];
};

template <int NumAz=NUM_INCL>
struct Raymarch2dStaticArgs {
    FpConst3d eta = Fp3d();
    FpConst3d chi = Fp3d();
    vec2 ray_start;
    vec2 ray_end;
    vec2 centre;
    yakl::SArray<fp_t, 1, NumAz> az_rays;
    yakl::SArray<fp_t, 1, NumAz> az_weights;
    yakl::SArray<fp_t, 1, 2> direction;
    int la = -1; /// sentinel value of -1 to ignore bc
    const PwBc<>& bc;
    Fp3d alo = Fp3d();
    fp_t distance_scale = FP(1.0);
    vec3 offset = {};
};


/** Clips a ray to the specified box
 * \param ray the start/end points of the ray
 * \param box the box dimensions
 * \param start_clipped an out param specifying whether the start point was
 * clipped (i.e. may need to sample BC). Can be nullptr if not interest
 * \returns An optional ray start/end, nullopt if the ray is entirely outside.
*/
template <int NumDim=NUM_DIM>
YAKL_INLINE std::optional<RayStartEnd> clip_ray_to_box(RayStartEnd ray, Box box, bool* start_clipped=nullptr) {
    RayStartEnd result(ray);
    yakl::SArray<fp_t, 1, NumDim> length;
    fp_t clip_t_start = FP(0.0);
    fp_t clip_t_end = FP(0.0);
    if (start_clipped) {
        *start_clipped = false;
    }
    for (int d = 0; d < NumDim; ++d) {
        length(d) = ray.end(d) - ray.start(d);
        int clip_idx = -1;
        if (ray.start(d) < box.dims[d](0)) {
            clip_idx = 0;
        } else if (ray.start(d) > box.dims[d](1)) {
            clip_idx = 1;
        }
        if (clip_idx != -1) {
            fp_t clip_t = (box.dims[d](clip_idx) - ray.start(d)) / length(d);
            if (clip_t > clip_t_start) {
                clip_t_start = clip_t;
            }
        }

        clip_idx = -1;
        if (ray.end(d) < box.dims[d](0)) {
            clip_idx = 0;
        } else if (ray.end(d) > box.dims[d](1)) {
            clip_idx = 1;
        }
        if (clip_idx != -1) {
            fp_t clip_t = (box.dims[d](clip_idx) - ray.end(d)) / -length(d);
            if (clip_t > clip_t_end) {
                clip_t_end = clip_t;
            }
        }
    }

    if (
        clip_t_start < 0 ||
        clip_t_end < 0 ||
        clip_t_start + clip_t_end >= FP(1.0)
    ) {
        // NOTE(cmo): We've moved forwards from start enough, and back from end
        // enough that there's none of the original ray actually intersecting
        // the clip planes! Or numerical precision issues accidentally forced a
        // fake intersection.
        return std::nullopt;
    }

    if (clip_t_start > FP(0.0)) {
        for (int d = 0; d < NumDim; ++d) {
            result.start(d) += clip_t_start * length(d);
        }
        if (start_clipped) {
            *start_clipped = true;
        }
    }
    if (clip_t_end > FP(0.0)) {
        for (int d = 0; d < NumDim; ++d) {
            result.end(d) -= clip_t_end * length(d);
        }
    }
    // NOTE(cmo): Catch precision errors with a clamp -- without this we will
    // stop the ray at the edge of the box to floating point precision, but it's
    // better for these to line up perfectly.
    for (int d = 0; d < NumDim; ++d) {
        if (result.start(d) < box.dims[d](0)) {
            result.start(d) = box.dims[d](0);
        } else if (result.start(d) > box.dims[d](1)) {
            result.start(d) = box.dims[d](1);
        }
        if (result.end(d) < box.dims[d](0)) {
            result.end(d) = box.dims[d](0);
        } else if (result.end(d) > box.dims[d](1)) {
            result.end(d) = box.dims[d](1);
        }
    }

    return result;
}

// NOTE(cmo): Based on Nanovdb templated implementation
template <int axis>
YAKL_INLINE fp_t step_marcher(RayMarchState2d* state) {
    auto& s = *state;
    fp_t new_t = s.next_hit(axis);
    s.next_hit(axis) += s.delta(axis);
    s.next_coord(axis) += s.step(axis);
    return new_t;
}

YAKL_INLINE fp_t step_marcher(RayMarchState2d* state) {
    auto& s = *state;
    int axis = 0;
    if (s.next_hit(1) < s.next_hit(0)) {
        axis = 1;
    }
    switch (axis) {
        case 0: {
            return step_marcher<0>(state);
        } break;
        case 1: {
            return step_marcher<1>(state);
        } break;
    }
}

YAKL_INLINE bool next_intersection(RayMarchState2d* state) {
    using namespace yakl::componentwise;
    using yakl::intrinsics::sum;

    auto& s = *state;
    const fp_t prev_t = s.t;
    for (int d = 0; d < NUM_DIM; ++d) {
        s.curr_coord(d) = s.next_coord(d);
    }

    fp_t new_t = step_marcher(state);

    if (new_t > s.max_t && prev_t < s.max_t) {
        // NOTE(cmo): The end point is in the box we have just stepped through
        decltype(s.p1) prev_hit = s.p0 +  prev_t * s.direction;
        s.dt = std::sqrt(sum(square(s.p1 - prev_hit)));
        new_t = s.max_t;
        // NOTE(cmo): Set curr_coord to a value we know we clamped inside the
        // grid: minimise accumulated error
        for (int d = 0; d < NUM_DIM; ++d) {
            s.curr_coord(d) = s.final_coord(d);
        }
    } else {
        // NOTE(cmo): Progress as normal
        s.dt = new_t - prev_t;
    }

    s.t = new_t;
    return new_t <= s.max_t;
}

/**
 * Create a new state for grid traversal using DDA. The ray is first clipped to
 * the grid, and if it is outside, nullopt is returned.
 * \param start_pos The start position of the ray
 * \param end_pos The start position of the ray
 * \param domain_size The domain size
 * \param start_clipped whether the start position was clipped; i.e. sample the BC.
*/
template <int NumDim=NUM_DIM>
YAKL_INLINE std::optional<RayMarchState2d> RayMarch2d_new(
    vec2 start_pos,
    vec2 end_pos,
    ivec2 domain_size,
    bool* start_clipped=nullptr
) {
    Box box;
    for (int d = 0; d < NumDim; ++d) {
        box.dims[d](0) = FP(0.0);
        box.dims[d](1) = domain_size(d);
    }
    auto clipped = clip_ray_to_box({start_pos, end_pos}, box, start_clipped);
    if (!clipped) {
        return std::nullopt;
    }

    start_pos = clipped->start;
    end_pos = clipped->end;

    RayMarchState2d r{};
    r.p0 = start_pos;
    r.p1 = end_pos;

    fp_t length = FP(0.0);
    for (int d = 0; d < NumDim; ++d) {
        r.curr_coord(d) = std::min(int(std::floor(start_pos(d))), domain_size(d)-1);
        r.direction(d) = end_pos(d) - start_pos(d);
        length += square(end_pos(d) - start_pos(d));
        r.final_coord(d) = std::min(int(std::floor(end_pos(d))), domain_size(d)-1);
    }
    r.next_coord = r.curr_coord;
    length = std::sqrt(length);
    r.max_t = length;

    fp_t inv_length = FP(1.0) / length;
    for (int d = 0; d < NumDim; ++d) {
        r.direction(d) *= inv_length;
        if (r.direction(d) > FP(0.0)) {
            r.next_hit(d) = fp_t(r.curr_coord(d) + 1 - r.p0(d)) / r.direction(d);
            r.step(d) = 1;
        } else if (r.direction(d) == FP(0.0)) {
            r.step(d) = 0;
            r.next_hit(d) = FP(1e24);
        } else {
            r.step(d) = -1;
            r.next_hit(d) = (r.curr_coord(d) - r.p0(d)) / r.direction(d);
        }
        r.delta(d) = fp_t(r.step(d)) / r.direction(d);
    }

    r.t = FP(0.0);
    r.dt = FP(0.0);

    // NOTE(cmo): Initialise to the first intersection so dt != 0
    next_intersection(&r);

    return r;
}

template <typename Alo>
YAKL_INLINE RadianceInterval<Alo> merge_intervals(
    RadianceInterval<Alo> closer,
    RadianceInterval<Alo> further
) {
    fp_t transmission = std::exp(-closer.tau);
    closer.I += transmission * further.I;
    closer.tau += further.tau;
    return closer;
}

struct Raymarch2dDynamicState {
    vec3 mu;
    const yakl::Array<const u16, 1, yakl::memDevice> active_set;
    const yakl::Array<bool, 3, yakl::memDevice>& dynamic_opac;
    const Atmosphere& atmos;
    const AtomicData<fp_t>& adata;
    const VoigtProfile<fp_t, false>& profile;
    const Fp2d& nh0;
    const Fp2d& n; // flattened
};

struct Raymarch2dDynamicInterpState {
    vec3 mu;
    const FpConst2d& vx;
    const FpConst2d& vy;
    const FpConst2d& vz;
    const yakl::Array<bool, 3, yakl::memDevice>& dynamic_opac;
    const yakl::Array<i64, 2, yakl::memDevice>& active_map;
    const DirectionalEmisOpacInterp& dir_interp;
};

template <typename Bc, class DynamicState=DexEmpty>
struct Raymarch2dArgs {
    const CascadeStateAndBc<Bc>& casc_state_bc;
    RayProps ray;
    fp_t distance_scale = FP(1.0);
    fp_t incl;
    fp_t incl_weight;
    int wave;
    int la;
    vec3 offset;
    const yakl::Array<bool, 2, yakl::memDevice>& active;
    const DynamicState& dyn_state;
};

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
YAKL_INLINE RadianceInterval<Alo> dda_raymarch_2d(
    const Raymarch2dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, casc_state_bc, ray, distance_scale, incl, incl_weight, wave, la, offset, dyn_state);
    JasUnpack(casc_state_bc, state, bc);
    constexpr bool dynamic = RcMode & RC_DYNAMIC;
    constexpr bool dynamic_interp = (RcMode & RC_DYNAMIC) && (RcMode & RC_DYNAMIC_INTERP);
    static_assert(
        !dynamic || (dynamic && (
            std::is_same_v<DynamicState, Raymarch2dDynamicState>
            || (dynamic_interp && std::is_same_v<DynamicState, Raymarch2dDynamicInterpState>)
            )
        ),
        "If dynamic must provide dynamic state"
    );

    auto domain_dims = state.eta.get_dimensions();
    ivec2 domain_size;
    // NOTE(cmo): This is swapped as the coord is still x,y,z, but the array is indexed (z,y,x)
    domain_size(0) = domain_dims(1);
    domain_size(1) = domain_dims(0);
    bool start_clipped;
    auto marcher = RayMarch2d_new(ray.start, ray.end, domain_size, &start_clipped);
    RadianceInterval<Alo> result{};
    if ((RcMode & RC_SAMPLE_BC) && (!marcher || start_clipped)) {
        // NOTE(cmo): Check the ray is going up along z.
        if ((ray.dir(1) > FP(0.0)) && la != -1) {
            const fp_t cos_theta = incl;
            const fp_t sin_theta = std::sqrt(FP(1.0) - square(cos_theta));
            vec3 mu;
            mu(0) = -ray.dir(0) * sin_theta;
            mu(1) = cos_theta;
            mu(2) = -ray.dir(1) * sin_theta;
            vec3 pos;
            pos(0) = ray.centre(0) * distance_scale + offset(0);
            pos(1) = offset(1);
            pos(2) = ray.centre(1) * distance_scale + offset(2);

            fp_t I_sample = sample_boundary(bc, la, pos, mu);
            result.I = I_sample;
        }
    }
    if (!marcher) {
        return result;
    }

    RayMarchState2d s = *marcher;

    // NOTE(cmo): one_m_edt is also the ALO
    fp_t eta_s = FP(0.0), chi_s = FP(1e-20), one_m_edt = FP(0.0);
    do {
        const auto& sample_coord(s.curr_coord);
        const int u = sample_coord(0);
        const int v = sample_coord(1);
        one_m_edt = FP(0.0);

        if (u < 0 || u >= domain_size(0)) {
            break;
        }
        if (v < 0 || v >= domain_size(1)) {
            break;
        }
        if (!args.active(v, u)) {
            continue;
        }

        eta_s = state.eta(v, u, wave);
        chi_s = state.chi(v, u, wave) + FP(1e-15);
        if constexpr (dynamic && !dynamic_interp) {
            // NOTE(cmo): dyn_state is Raymarch2dDynamicState
            const Atmosphere& atmos = dyn_state.atmos;
            const i64 k = v * atmos.temperature.extent(1) + u;
            if (
                dyn_state.dynamic_opac(v, u, wave)
                && dyn_state.active_set.extent(0) > 0
            ) {
                const auto& mu = dyn_state.mu;
                fp_t vel = (
                    atmos.vx.get_data()[k] * mu(0)
                    + atmos.vy.get_data()[k] * mu(1)
                    + atmos.vz.get_data()[k] * mu(2)
                );
                AtmosPointParams local_atmos{
                    .temperature = atmos.temperature.get_data()[k],
                    .ne = atmos.ne.get_data()[k],
                    .vturb = atmos.vturb.get_data()[k],
                    .nhtot = atmos.nh_tot.get_data()[k],
                    .vel = vel,
                    .nh0 = dyn_state.nh0.get_data()[k]
                };
                auto lines = emis_opac(
                    EmisOpacState<fp_t>{
                        .adata = dyn_state.adata,
                        .profile = dyn_state.profile,
                        .la = la,
                        .n = dyn_state.n,
                        .k = k,
                        .atmos = local_atmos,
                        .active_set = dyn_state.active_set,
                        .mode = EmisOpacMode::DynamicOnly
                    }
                );

                eta_s += lines.eta;
                chi_s += lines.chi;
            }
        } else if constexpr (dynamic && dynamic_interp) {
            // NOTE(cmo): dyn_state is Raymarch2dDynamicInterpState
            if (dyn_state.dynamic_opac(v, u, wave)) {
                const int ks = dyn_state.active_map(v, u);
                if (ks != -1) {
                    const auto& mu = dyn_state.mu;
                    const fp_t vel = (
                        dyn_state.vx(v, u) * mu(0)
                        + dyn_state.vy(v, u) * mu(1)
                        + dyn_state.vz(v, u) * mu(2)
                    );
                    auto contrib = dyn_state.dir_interp.sample(ks, wave, vel);

                    // NOTE(cmo): We are overwriting here, not adding.
                    eta_s = contrib.eta;
                    chi_s = contrib.chi + FP(1e-15);
                }
            }
        }

        fp_t tau = chi_s * s.dt * distance_scale;
        fp_t source_fn = eta_s / chi_s;

        // NOTE(cmo): implicit assumption muy != 1.0
        fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
        fp_t tau_mu = tau / sin_theta;
        fp_t edt;
        if (tau_mu < FP(1e-2)) {
            edt = FP(1.0) + (-tau_mu) + FP(0.5) * square(tau_mu);
            one_m_edt = -std::expm1(-tau_mu);
        } else {
            edt = std::exp(-tau_mu);
            one_m_edt = -std::expm1(-tau_mu);
        }
        result.tau += tau_mu;
        result.I = result.I * edt + source_fn * one_m_edt;
    } while (next_intersection(&s));

    if constexpr ((RcMode & RC_COMPUTE_ALO) && !std::is_same_v<Alo, DexEmpty>) {
        result.alo = std::max(one_m_edt, FP(0.0));
    }

    return result;
}


#else
#endif