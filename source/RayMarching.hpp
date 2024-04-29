#if !defined(DEXRT_RAYMARCHING_HPP)
#define DEXRT_RAYMARCHING_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "JasPP.hpp"
#include "PromweaverBoundary.hpp"
#include <optional>

using yakl::intrinsics::minloc;

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
            r.next_hit(d) = (r.curr_coord(d) + 1 - r.p0(d)) / r.direction(d);
            r.step(d) = 1;
        } else if (r.direction(d) == FP(0.0)) {
            r.step(d) = 0;
            r.next_hit(d) = FP(1e24);
        } else {
            r.step(d) = -1;
            r.next_hit(d) = (r.curr_coord(d) - r.p0(d)) / r.direction(d);
        }
        r.delta(d) = r.step(d) / r.direction(d);
    }

    r.t = FP(0.0);
    r.dt = FP(0.0);

    // NOTE(cmo): Initialise to the first intersection so dt != 0
    next_intersection(&r);

    return r;
}

template <int NumAz=NUM_INCL, int NumComponents=NUM_COMPONENTS>
YAKL_INLINE yakl::SArray<fp_t, 2, NumComponents, NumAz> empty_hit() {
    yakl::SArray<fp_t, 2, NumComponents, NumAz> result;
    result = FP(0.0);
    return result;
}

template <bool SampleBoundary=false, int NumWavelengths=NUM_WAVELENGTHS, int NumAz=NUM_INCL, int NumComponents=NUM_COMPONENTS>
YAKL_INLINE yakl::SArray<fp_t, 2, NumComponents, NumAz> dda_raymarch_2d(
    const Raymarch2dStaticArgs<NumAz>& args
) {
    JasUnpack(args, eta, chi, ray_start, ray_end, az_rays, az_weights);
    JasUnpack(args, alo, distance_scale, la, direction, bc);

    auto domain_dims = eta.get_dimensions();
    ivec2 domain_size;
    // NOTE(cmo): This is swapped as the coord is still x,y,z, but the array is indexed (z,y,x)
    domain_size(0) = domain_dims(1);
    domain_size(1) = domain_dims(0);
    bool start_clipped;
    auto marcher = RayMarch2d_new(ray_start, ray_end, domain_size, &start_clipped);
    yakl::SArray<fp_t, 2, NumComponents, NumAz> result(FP(0.0));
    if (SampleBoundary && USE_BC && (!marcher || start_clipped)) {
        // NOTE(cmo): Sample BC
        static_assert(!(USE_BC && USE_MIPMAPS), "BCs not currently supported with mipmaps");
        static_assert(!(USE_BC && !USE_ATMOSPHERE), "BCs not supported outside of atmosphere mode");

        // NOTE(cmo): Check the ray is going up along z.
        if ((direction(1) > FP(0.0)) && la != -1) {
            vec3 mu;
            mu(0) = -direction(0);
            mu(1) = FP(0.0);
            mu(2) = -direction(1);
            vec3 pos;
            pos(0) = args.centre(0) * distance_scale + args.offset(0);
            pos(1) = args.offset(1);
            pos(2) = args.centre(1) * distance_scale + args.offset(2);

            fp_t I_sample = sample_boundary(bc, la, pos, mu);
            if constexpr (PWBC_SAMPLE_CONE) {
                constexpr fp_t cone_half_angle = FP(2.0) * FP(M_PI) / FP(2048.0);
                constexpr fp_t edge_weight = FP(2.5) / FP(9.0);
                constexpr fp_t centre_weight = FP(4.0)/ FP(9.0);
                constexpr fp_t gl_sample =  FP(0.7745966692414834);
                const fp_t cos_cone = std::cos(cone_half_angle * gl_sample);
                const fp_t sin_cone = std::sin(cone_half_angle * gl_sample);
                I_sample *= centre_weight;

                // cos(x+y) = cosx cosy - sinx siny
                // cos(x-y) = cosx cosy + sinx siny
                // sin(x+y) = sinx cosy + cosx siny
                // sin(x-y) = sinx cosy - cosx siny
                vec3 mu_cone;
                mu_cone(0) = mu(0) * cos_cone - mu(2) * sin_cone;
                mu_cone(1) = FP(0.0);
                mu_cone(2) = mu(2) * cos_cone + mu(0) * sin_cone;
                I_sample += edge_weight * sample_boundary(bc, la, pos, mu_cone);

                mu_cone(0) = mu(0) * cos_cone + mu(2) * sin_cone;
                mu_cone(1) = FP(0.0);
                mu_cone(2) = mu(2) * cos_cone - mu(0) * sin_cone;
                I_sample += edge_weight * sample_boundary(bc, la, pos, mu_cone);
            }
            // NOTE(cmo): The extra terms are correcting for solid angle so J is correct
            // const fp_t start_I = I_sample * std::abs(mu(0)) * FP(0.5) * FP(M_PI);
            const fp_t start_I = I_sample * FP(0.5) * FP(M_PI);
            for (int r = 0; r < NumAz; ++r) {
                result(0, r) = start_I * std::sqrt(FP(1.0) - square(az_rays(r)));
            }
        }
    }
    if (!marcher) {
        return result;
    }

    RayMarchState2d s = *marcher;

    yakl::SArray<fp_t, 1, NumWavelengths> sample;
    yakl::SArray<fp_t, 1, NumWavelengths> chi_sample;

    do {
        const auto& sample_coord(s.curr_coord);

        if (sample_coord(0) < 0 || sample_coord(0) >= domain_size(0)) {
            auto hit = s.p0 + s.t * s.direction;
            if (false) {
                printf("out x <%d, %d>, (%f, %f), [%f,%f] -> [%f,%f]\n",
                sample_coord(0), sample_coord(1), hit(0), hit(1),
                s.p0(0), s.p0(1), s.p1(0), s.p1(1)
                );
            }
            break;
        }
        if (sample_coord(1) < 0 || sample_coord(1) >= domain_size(1)) {
            auto hit = s.p0 + s.t * s.direction;
            if (false) {
                printf("out y <%d, %d>, (%f, %g), [%f,%f] -> [%f,%f]\n",
                sample_coord(0), sample_coord(1), hit(0), hit(1),
                s.p0(0), s.p0(1), s.p1(0), s.p1(1)
                );
            }
            break;
        }

        for (int i = 0; i < NumWavelengths; ++i) {
            sample(i) = eta(sample_coord(1), sample_coord(0), i);
        }
        for (int i = 0; i < NumWavelengths; ++i) {
            chi_sample(i) = chi(sample_coord(1), sample_coord(0), i) + FP(1e-20);
        }

        const bool final_step = (s.t == s.max_t);
        const bool accumulate_alo = (final_step && alo.initialized());

        for (int i = 0; i < NumWavelengths; ++i) {
            fp_t tau = chi_sample(i) * s.dt * distance_scale;
            fp_t source_fn = sample(i) / chi_sample(i);

            for (int r = 0; r < NumAz; ++r) {
                const fp_t weight = FP(1.0) / PROBE0_NUM_RAYS;
                if (az_rays(r) == FP(0.0)) {
                    result(2*i, r) = source_fn;
                    if (accumulate_alo) {
                        // NOTE(cmo): We add the local weight since tau is infinite, i.e. one_m_edt == 1.0
                        yakl::atomicAdd(alo(sample_coord(1), sample_coord(0), r), weight);
                    }
                } else {
                    fp_t mu = std::sqrt(FP(1.0) - square(az_rays(r)));
                    fp_t tau_mu = tau / mu;
                    fp_t edt, one_m_edt;
                    if (tau_mu < FP(1e-2)) {
                        edt = FP(1.0) + (-tau_mu) + FP(0.5) * square(tau_mu);
                        one_m_edt = -std::expm1(-tau_mu);
                    } else {
                        edt = std::exp(-tau_mu);
                        one_m_edt = -std::expm1(-tau_mu);
                    }
                    result(2*i+1, r) += tau_mu;
                    result(2*i, r) = result(2*i, r) * edt + source_fn * one_m_edt;
                    if (accumulate_alo) {
                        yakl::atomicAdd(alo(sample_coord(1), sample_coord(0), r), weight * one_m_edt);
                    }
                }
            }
        }
    } while (next_intersection(&s));

    return result;
}

template <bool SampleBoundary=false, bool UseMipmaps=USE_MIPMAPS, int NumWavelengths=NUM_WAVELENGTHS, int NumAz=NUM_INCL, int NumComponents=NUM_COMPONENTS>
YAKL_INLINE yakl::SArray<fp_t, 2, NumComponents, NumAz> raymarch_2d(
    const CascadeRTState& state,
    const Raymarch2dStaticArgs<NumAz>& args
) {
    // NOTE(cmo): Swap start/end to facilitate solution to RTE. Could reframe
    // and go the other way, dropping out of the march early if we have
    // traversed sufficient optical depth.
    fp_t factor = args.distance_scale;
    if constexpr (UseMipmaps) {
        fp_t mip_factor = (1 << state.mipmap_factor);
        JasUnpack(args, ray_start, ray_end);
        ray_start(0) = ray_start(0) / mip_factor;
        ray_start(1) = ray_start(1) / mip_factor;
        ray_end(0) = ray_end(0) / mip_factor;
        ray_end(1) = ray_end(1) / mip_factor;
        factor *= mip_factor;
    }
    const FpConst3d& eta = state.eta;
    const FpConst3d& chi = state.chi;

    return dda_raymarch_2d<SampleBoundary, NumWavelengths, NumAz, NumComponents>(
        Raymarch2dStaticArgs<NumAz>{
            .eta = eta,
            .chi = chi,
            .ray_start = args.ray_end,
            .ray_end = args.ray_start,
            .centre = args.centre,
            .az_rays = args.az_rays,
            .az_weights = args.az_weights,
            .direction = args.direction,
            .la = args.la,
            .bc = args.bc,
            .alo = args.alo,
            .distance_scale = factor,
            .offset = args.offset,
        }
    );
}

#else
#endif