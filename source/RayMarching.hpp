#if !defined(DEXRT_RAYMARCHING_HPP)
#define DEXRT_RAYMARCHING_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include <optional>

using yakl::intrinsics::minloc;

template <int NumDim=NUM_DIM>
YAKL_INLINE std::optional<RayStartEnd> clip_ray_to_box(RayStartEnd ray, Box box) {
    RayStartEnd result(ray);
    yakl::SArray<fp_t, 1, NumDim> length;
    fp_t clip_t_start = FP(0.0);
    fp_t clip_t_end = FP(0.0);
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

YAKL_INLINE bool next_intersection(RayMarchState2d* state) {
    using namespace yakl::componentwise;
    using yakl::intrinsics::sum;

    auto& s = *state;
    const fp_t prev_t = s.t;
    s.curr_coord = s.next_coord;

    int axis = minloc(s.next_hit);
    fp_t new_t = s.next_hit(axis);
    s.next_hit(axis) += s.delta(axis);
    s.next_coord(axis) += s.step(axis);

    if (new_t > s.max_t && prev_t < s.max_t) {
        // NOTE(cmo): The end point is in the box we have just stepped through
        decltype(s.p1) prev_hit = s.p0 +  prev_t * s.direction;
        s.dt = std::sqrt(sum(square(s.p1 - prev_hit)));
        new_t = s.max_t;
        // NOTE(cmo): Set curr_coord to a value we know we clamped inside the
        // grid: minimise accumulated error
        for (int d = 0; d < NUM_DIM; ++d) {
            s.curr_coord(d) = int(std::floor(s.p1(d)));
        }
    } else {
        // NOTE(cmo): Progress as normal
        s.dt = new_t - prev_t;
    }

    s.t = new_t;
    return new_t <= s.max_t;
}

template <int NumDim=2>
YAKL_INLINE std::optional<RayMarchState2d> RayMarch2d_new(vec2 start_pos, vec2 end_pos, ivec2 domain_size) {
    Box box;
    for (int d = 0; d < NumDim; ++d) {
        box.dims[d](0) = FP(0.0);
        box.dims[d](1) = domain_size(d) - 1;
    }
    auto clipped = clip_ray_to_box({start_pos, end_pos}, box);
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
        r.curr_coord(d) = int(std::floor(start_pos(d)));
        r.direction(d) = end_pos(d) - start_pos(d);
        length += square(end_pos(d) - start_pos(d));
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
            r.next_hit(d) = FP(1e8);
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

template <int NumAz=NUM_AZ, int NumComponents=NUM_COMPONENTS>
YAKL_INLINE yakl::SArray<fp_t, 2, NumComponents, NumAz> empty_hit() {
    yakl::SArray<fp_t, 2, NumComponents, NumAz> result;
    result = FP(0.0);
    return result;
}

template <int NumWavelengths=NUM_WAVELENGTHS, int NumAz=NUM_AZ, int NumComponents=NUM_COMPONENTS>
YAKL_INLINE yakl::SArray<fp_t, 2, NumComponents, NumAz> dda_raymarch_2d(
    // eta in volumetric
    const FpConst3d& domain,
    const FpConst3d& chi,
    vec2 ray_start,
    vec2 ray_end,
    yakl::SArray<fp_t, 1, NumAz> az_rays,
    fp_t distance_scale = FP(1.0)
) {
    auto domain_dims = domain.get_dimensions();
    ivec2 domain_size;
    domain_size(0) = domain_dims(0);
    domain_size(1) = domain_dims(1);
    auto marcher = RayMarch2d_new(ray_start, ray_end, domain_size);
    if (!marcher) {
        return empty_hit<NumAz, NumComponents>();
    }

    RayMarchState2d s = *marcher;

    yakl::SArray<fp_t, 2, NumComponents, NumAz> result(FP(0.0));
    yakl::SArray<fp_t, 1, NumWavelengths> sample;
    yakl::SArray<fp_t, 1, NumWavelengths> chi_sample;

    do {
        auto sample_coord = s.curr_coord;

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
            sample(i) = domain(sample_coord(0), sample_coord(1), i);
        }
        for (int i = 0; i < NumWavelengths; ++i) {
            chi_sample(i) = chi(sample_coord(0), sample_coord(1), i);
        }

        for (int i = 0; i < NumWavelengths; ++i) {
            fp_t tau = chi_sample(i) * s.dt * distance_scale;
            fp_t source_fn = sample(i) / chi_sample(i);

            for (int r = 0; r < NumAz; ++r) {
                if (az_rays(r) == FP(0.0)) {
                    result(2*i, r) = source_fn;
                } else {
                    fp_t mu = az_rays(r);
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
                }
            }
        }
    } while (next_intersection(&s));

    return result;
}

template <bool UseMipmaps=USE_MIPMAPS, int NumWavelengths=NUM_WAVELENGTHS, int NumAz=NUM_AZ, int NumComponents=NUM_COMPONENTS>
YAKL_INLINE yakl::SArray<fp_t, 2, NumComponents, NumAz> raymarch_2d(
    const CascadeRTState& state,
    vec2 ray_start,
    vec2 ray_end,
    const yakl::SArray<fp_t, 1, NumAz>& az_rays
) {
    // NOTE(cmo): Swap start/end to facilitate solution to RTE. Could reframe
    // and go the other way, dropping out of the march early if we have
    // traversed sufficient optical depth.
    if constexpr (UseMipmaps) {
        fp_t factor = (1 << state.mipmap_factor);
        ray_start(0) = ray_start(0) / factor;
        ray_start(1) = ray_start(1) / factor;
        ray_end(0) = ray_end(0) / factor;
        ray_end(1) = ray_end(1) / factor;

        const FpConst3d& eta = state.eta;
        const FpConst3d& chi = state.chi;
        return dda_raymarch_2d<NumWavelengths, NumAz, NumComponents>(eta, chi, ray_end, ray_start, az_rays, factor);

    } else {
        const FpConst3d& domain = state.eta;
        const FpConst3d& chi = state.chi;
        return dda_raymarch_2d<NumWavelengths, NumAz, NumComponents>(domain, chi, ray_end, ray_start, az_rays);
    }
}

#else
#endif