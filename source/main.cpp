#include <math.h>
#include <limits>
#include "Config.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include <stdio.h>
#ifdef HAVE_MPI
    #include "YAKL_pnetcdf.h"
#endif
#include <vector>
#include <optional>

using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;

YAKL_INLINE void model_original(Fp3d emission, int x, int y) {
    constexpr fp_t intensity_scale = FP(3.0);
    if (x >= 25 && x < 35) {
        if (y >= 235 && y < 275) {
            emission(x, y, 0) = intensity_scale;
        }
    }

    if (x >= 295 && x < 311) {
        if (y >= 145 && y < 365) {
            emission(x, y, 0) = FP(1e-6);
            emission(x, y, 1) = FP(1e-6);
            emission(x, y, 2) = FP(1e-6);
        }
    }

    if (x >= 253 && x < 260) {
        if (y >= 253 && y < 260) {
            emission(x, y, 1) = FP(4.0) * intensity_scale;
        }
    }

    if (x >= 220 && x < 222) {
        if (y >= 220 && y < 222) {
            emission(x, y, 0) = FP(1.414) * intensity_scale;
            emission(x, y, 2) = FP(1.414) * intensity_scale;
        }
    }

    if (x >= 405 && x < 410) {
        if (y >= 252 && y < 262) {
            emission(x, y, 2) = intensity_scale;
        }
    }
}

YAKL_INLINE void draw_disk(
    Fp3d emission,
    vec2 centre, 
    fp_t radius, 
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color,
    int x,
    int y
) {
    fp_t dx = fp_t(x) - centre(0);
    fp_t dy = fp_t(y) - centre(1);

    if ((dx * dx + dy * dy) <= (radius * radius)) {
        for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
            emission(x, y, i) = color(i);
        }
    }
}

YAKL_INLINE void model_A(Fp3d emission, int x, int y) {
    vec2 centre;
    yakl::SArray<fp_t, 1, 3> color;

    centre(0) = 30;
    centre(1) = 30;
    color(0) = FP(10.0);
    color(1) = FP(10.0);
    color(2) = FP(10.0);
    draw_disk(emission, centre, 30, color, x, y);

    centre(0) = 50;
    centre(1) = 180;
    color(0) = FP(1e-6);
    color(1) = FP(1e-6);
    color(2) = FP(1e-6);
    draw_disk(emission, centre, 6, color, x, y);

    centre(0) = CANVAS_X / 2;
    centre(1) = CANVAS_Y / 2;
    color(0) = FP(1.0);
    color(1) = FP(1.0);
    color(2) = FP(1.0);
    draw_disk(emission, centre, 2, color, x, y);
}

YAKL_INLINE void model_B(Fp3d emission, int x, int y) {
    int centre = int(CANVAS_X / 2);
    vec2 c;
    yakl::SArray<fp_t, 1, 3> color;
    color(0) = FP(0.0);
    color(1) = FP(1.0);
    color(2) = FP(0.0);
    for (int cx = centre - 256; cx < centre + 256; ++cx) {
        c(0) = centre;
        c(1) = cx;

        draw_disk(emission, c, 6, color, x, y);
    }

    color(0) = FP(1e-6);
    color(1) = FP(1e-6);
    color(2) = FP(1e-6);
    for (int cx = centre - 50; cx < centre + 50; ++cx) {
        c(0) = centre + 100;
        c(1) = cx;

        draw_disk(emission, c, 6, color, x, y);
    }

    color(0) = FP(10.0);
    color(1) = FP(0.0);
    color(2) = FP(0.0);
    for (int cx = centre - 50; cx < centre + 50; ++cx) {
        c(0) = 100;
        c(1) = cx;

        draw_disk(emission, c, 6, color, x, y);
    }

    c(0) = centre + 400;
    c(1) = centre;
    color(0) = FP(0.0);
    color(1) = FP(0.0);
    color(2) = FP(0.5);
    draw_disk(emission, c, 40, color, x, y);
}

YAKL_INLINE void model_D_emission(Fp3d emission, int x, int y) {
    int centre = int(CANVAS_X / 2);
    vec2 c;
    yakl::SArray<fp_t, 1, 3> color;
    c(0) = centre + 400;
    c(1) = centre;
    color(0) = FP(0.0);
    color(1) = FP(0.0);
    color(2) = FP(2.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = centre - 400;
    c(1) = centre;
    color(0) = FP(2.0);
    color(1) = FP(0.0);
    color(2) = FP(0.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = centre;
    c(1) = centre - 400;
    color(0) = FP(0.0);
    color(1) = FP(0.0);
    color(2) = FP(0.5);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = centre;
    c(1) = centre + 400;
    color(0) = FP(0.5);
    color(1) = FP(0.0);
    color(2) = FP(0.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = 200;
    c(1) = 200;
    color(0) = FP(0.0);
    color(1) = FP(3.0);
    color(2) = FP(0.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = CANVAS_X - 200;
    c(1) = 200;
    color(0) = FP(0.0);
    color(1) = FP(0.0);
    color(2) = FP(3.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = 400;
    c(1) = 700;
    color(0) = FP(3.0);
    color(1) = FP(0.0);
    color(2) = FP(3.0);
    draw_disk(emission, c, 40, color, x, y);
}

YAKL_INLINE void model_D_absorption(Fp3d chi, int x, int y) {
    int centre = int(CANVAS_X / 2);
    vec2 c;
    yakl::SArray<fp_t, 1, 3> color;
    fp_t bg = FP(1e-10);
    for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
        chi(x, y, i) = bg;
    }
    c(0) = centre + 400;
    c(1) = centre;
    color(0) = bg;
    color(1) = bg;
    color(2) = FP(0.5);
    draw_disk(chi, c, 40, color, x, y);

    c(0) = centre - 400;
    c(1) = centre;
    color(0) = FP(0.5);
    color(1) = bg;
    color(2) = bg;
    draw_disk(chi, c, 40, color, x, y);

    c(0) = centre;
    c(1) = centre - 400;
    color(0) = bg;
    color(1) = bg;
    color(2) = FP(0.5);
    draw_disk(chi, c, 40, color, x, y);

    c(0) = centre;
    c(1) = centre + 400;
    color(0) = FP(0.5);
    color(1) = bg;
    color(2) = bg;
    draw_disk(chi, c, 40, color, x, y);

    c(0) = centre + 340;
    c(1) = centre;
    color(0) = FP(0.2);
    color(1) = FP(0.2);
    color(2) = FP(0.2);
    draw_disk(chi, c, 6, color, x, y);
    c(0) = centre - 340;
    c(1) = centre;
    draw_disk(chi, c, 6, color, x, y);

    int box_size = 250;
    if (x >= centre - box_size && x < centre + box_size && y >= centre - box_size && y < centre + box_size) {
        fp_t chi_r = FP(1e-4);
        chi(x, y, 0) = chi_r;
        chi(x, y, 1) = FP(1e2) * chi_r;
        chi(x, y, 2) = FP(1e2) * chi_r;
    }

    c(0) = 200;
    c(1) = 200;
    color(0) = bg;
    color(1) = FP(1.0);
    color(2) = bg;
    draw_disk(chi, c, 40, color, x, y);

    c(0) = CANVAS_X - 200;
    c(1) = 200;
    color(0) = bg;
    color(1) = bg;
    color(2) = FP(1.0);
    draw_disk(chi, c, 40, color, x, y);

    c(0) = 400;
    c(1) = 700;
    color(0) = FP(1.0);
    color(1) = bg;
    color(2) = FP(1.0);
    draw_disk(chi, c, 40, color, x, y);
}

template <typename T>
YAKL_INLINE T sign(T t) {
    return std::copysign(T(1.0), t);
}

template <typename T>
YAKL_INLINE constexpr T square(T t) {
    return t * t;
}

template <typename T>
YAKL_INLINE constexpr T cube(T t) {
    return t * t * t;
}

yakl::SArray<fp_t, 1, NUM_AZ> get_az_rays() {
    yakl::SArray<fp_t, 1, NUM_AZ> az_rays;
    for (int r = 0; r < NUM_AZ; ++r) {
        az_rays(r) = AZ_RAYS[r];
    }
    return az_rays;
}

yakl::SArray<fp_t, 1, NUM_AZ> get_az_weights() {
    yakl::SArray<fp_t, 1, NUM_AZ> az_weights;
    for (int r = 0; r < NUM_AZ; ++r) {
        az_weights(r) = AZ_WEIGHTS[r];
    }
    return az_weights;
}

YAKL_INLINE std::optional<RayStartEnd> clip_ray_to_box(RayStartEnd ray, Box box) {
    RayStartEnd result(ray);
    yakl::SArray<fp_t, 1, NUM_DIM> length;
    fp_t clip_t_start = FP(-1.0);
    fp_t clip_t_end = FP(-1.0);
    for (int d = 0; d < NUM_DIM; ++d) {
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

    if (clip_t_start + clip_t_end >= 1) {
        // NOTE(cmo): We've moved forwards from start enough, and back from end
        // enough that there's none of the original ray actually intersecting
        // the clip planes!
        return std::nullopt;
    }

    if (clip_t_start >= FP(0.0)) {
        for (int d = 0; d < NUM_DIM; ++d) {
            result.start(d) += clip_t_start * length(d); 
        }
    }
    if (clip_t_end >= FP(0.0)) {
        for (int d = 0; d < NUM_DIM; ++d) {
            result.end(d) -= clip_t_end * length(d); 
        }
    }
    // NOTE(cmo): Catch precision errors with a clamp -- without this we will
    // stop the ray at the edge of the box to floating point precision, but it's
    // better for these to line up perfectly.
    for (int d = 0; d < NUM_DIM; ++d) {
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

YAKL_INLINE std::optional<RayMarchState> RayMarch_new(vec2 start_pos, vec2 end_pos, ivec2 domain_size) {
    Box box;
    for (int d = 0; d < NUM_DIM; ++d) {
        box.dims[d](0) = FP(0.0);
        box.dims[d](1) = domain_size(d) - 1;
    }
    auto clipped = clip_ray_to_box({start_pos, end_pos}, box);
    if (!clipped) {
        return std::nullopt;
    }

    start_pos = clipped->start;
    end_pos = clipped->end;
    
    RayMarchState r{};
    r.p0 = start_pos;
    r.p1 = end_pos;
    r.p(0) = int(std::floor(start_pos(0)));
    r.p(1) = int(std::floor(start_pos(1)));

    r.step(0) = sign(end_pos(0) - start_pos(0));
    r.step(1) = sign(end_pos(1) - start_pos(1));

    r.end(0) = int(std::floor(end_pos(0))) + r.step(0);
    r.end(1) = int(std::floor(end_pos(1))) + r.step(1);

    fp_t dx = end_pos(0) - start_pos(0);
    fp_t dy = end_pos(1) - start_pos(1);
    fp_t length = FP(1.0) / std::sqrt(dx*dx + dy*dy);
    dx *= length;
    dy *= length;
    r.d(0) = dx;
    r.d(1) = dy;

    r.tm(0) = (start_pos(0) - r.p(0)) / dx * r.step(0);
    r.td(0) = r.step(0) / dx;
    if (dx == FP(0.0)) {
        r.tm(0) = FP(1e10);
        r.td(0) = FP(0.0);
    }

    r.tm(1) = (start_pos(1) - r.p(1)) / dy * r.step(1);
    r.td(1) = r.step(1) / dy;
    if (dy == FP(0.0)) {
        r.tm(1) = FP(1e10);
        r.td(1) = FP(0.0);
    }

    return r;
}

YAKL_INLINE bool next_intersection(RayMarchState* state) {
    bool stop = false;
    auto& s = *state;
    fp_t tmx = s.tm(0);
    fp_t tmy = s.tm(1);
    const bool equal = approx_equal(tmx, tmy, FP(1e-6));
    if ((tmx < tmy) || equal) {
        fp_t t = tmx;
        s.hit(0) = s.p0(0) + t * s.d(0);
        s.hit(1) = s.p0(1) + t * s.d(1);
        s.ds = t - s.prev_t;
        s.prev_t = t;
        s.tm(0) += s.td(0);
        s.p(0) += s.step(0);

        stop = (s.p(0) == s.end(0));
    } else {
        fp_t t = tmy;
        s.hit(0) = s.p0(0) + t * s.d(0);
        s.hit(1) = s.p0(1) + t * s.d(1);
        s.ds = t - s.prev_t;
        s.prev_t = t;
        s.tm(1) += s.td(1);
        s.p(1) += s.step(1);

        stop = (s.p(1) == s.end(1));
    } 

    if (equal) {
        // NOTE(cmo): If the two min steps are equal (e.g. traversing on a
        // diagonal), increment the position on both axes -- this is literally a
        // corner case :D
        // Stop condition should be consistent for either.
        s.tm(1) += s.td(1);
        s.p(1) += s.step(1);
        stop = stop || (s.p(1) == s.end(1));
    }

    if (stop) {
        // NOTE(cmo): Handle the (common) case where the final step isn't to the
        // next grid intersection, but to an arbitrary point in the cell.
        s.ds = std::sqrt(square(s.p1(0) - s.hit(0)) + square(s.p1(1) - s.hit(1)));
        s.hit(0) = s.p1(0);
        s.hit(1) = s.p1(1);
    }

    return !stop;
}

void init_state (State* state) {
    for (int l = 0; l < MAX_LEVEL + 1; ++l) {
        state->cascades.push_back(
            Fp5d(
                "cascade",
                PROBES_IN_CASCADE_0 / (1 << l),
                PROBES_IN_CASCADE_0 / (1 << l),
                PROBE0_NUM_RAYS * (1 << (l * CASCADE_BRANCHING_FACTOR)),
                NUM_COMPONENTS,
                NUM_AZ
            )
        );

        auto casc = state->cascades[l];
        auto dims = casc.get_dimensions();
        printf("[%d, %d, %d, %d]\n", dims(0), dims(1), dims(2), dims(3));
        parallel_for(
            SimpleBounds<3>(dims(0), dims(1), dims(2)),
            YAKL_LAMBDA (int i, int j, int k) {
                for (int m = 0; m < NUM_COMPONENTS; ++m) {
                    for (int p = 0; p < NUM_AZ; ++p) {
                        casc(i, j, k, m, p) = FP(0.0);
                    }
                }
            }
        );
    }

    state->emission = Fp3d("emission_map", CANVAS_X, CANVAS_Y, NUM_WAVELENGTHS);

    {
        auto emission = state->emission;
        parallel_for(
            SimpleBounds<2>(CANVAS_X, CANVAS_Y),
            YAKL_LAMBDA (int x, int y) {
                for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
                    emission(x, y, i) = FP(0.0);
                }

                // model_original(emission, x, y);
                // model_A(emission, x, y);
                // model_B(emission, x, y);
                LIGHT_MODEL(emission, x, y);
            }
        );
    }

#ifndef TRACE_OPAQUE_LIGHTS
    state->absorption = Fp3d("absorption_map", CANVAS_X, CANVAS_Y, NUM_WAVELENGTHS);
    {
        auto chi = state->absorption;
        parallel_for(
            SimpleBounds<2>(CANVAS_X, CANVAS_Y),
            YAKL_LAMBDA (int x, int y) {
#ifndef ABSORPTION_MODEL
                LIGHT_MODEL(chi, x, y);
                for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
                    if (chi(x, y, i) == FP(0.0)) {
                        chi(x, y, i) = FP(1e-10);
                    } else if (chi(x, y, i) == FP(1e-6)) {
                        chi(x, y, i) = FP(3.0);
                    } else {
                        chi(x, y, i) /= FP(2.0);
                    }
                }
#else
                ABSORPTION_MODEL(chi, x, y);
#endif
            }
        );

    }
#endif

    yakl::fence();
}

YAKL_INLINE yakl::SArray<fp_t, 1, NUM_COMPONENTS> dodgy_raymarch(const Fp3d& domain, vec2 ray_start, vec2 direction, fp_t distance) {
    int steps = distance * yakl::max(yakl::abs(direction(0)), yakl::abs(direction(1)));
    vec2 step;
    step(0) = distance * direction(0) / steps;
    step(1) = distance * direction(1) / steps;

    vec2 pos;
    pos(0) = ray_start(0);
    pos(1) = ray_start(1);

    ivec2 sample_coord;
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> sample;
    yakl::SArray<fp_t, 1, NUM_COMPONENTS> result;
    for (int i = 0; i < steps; ++i) {
        sample_coord(0) = int(std::round(pos(0)));
        sample_coord(1) = int(std::round(pos(1)));

        if (sample_coord(0) >= CANVAS_X || sample_coord(1) >= CANVAS_Y || sample_coord(0) < 0 || sample_coord(1) < 0) {
            for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
                result(i) = FP(0.0);
            }
            result(NUM_COMPONENTS-1) = FP(1.0);
            return result;
        }

        for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
            sample(i) = domain(sample_coord(0), sample_coord(1), i);
        }

        bool hits = false;
        for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
            hits |= bool(sample(i) != FP(0.0));
            if (hits) {
                break;
            }
        }

        if (hits) {
            for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
                result(i) = sample(i);
            }
            result(NUM_COMPONENTS - 1) = FP(0.0);
            return result;
        }
        
        pos(0) += step(0);
        pos(1) += step(1);
    }

    for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
        result(i) = FP(0.0);
    }
    result(NUM_COMPONENTS-1) = FP(1.0);
    return result;
}

YAKL_INLINE yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> empty_hit() {
    yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> result;
#ifdef TRACE_OPAQUE_LIGHTS
    for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
        result(i) = FP(0.0);
        result(NUM_WAVELENGTHS + i) = FP(1.0);
    }
#else
    result = FP(0.0);
#endif
    return result;

}

YAKL_INLINE yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> aw_raymarch(
    const Fp3d& domain, // eta in volumetric
    const Fp3d& chi,
    vec2 ray_start, 
    vec2 ray_end,
    yakl::SArray<fp_t, 1, NUM_AZ> az_rays
) {
    auto domain_dims = domain.get_dimensions();
    ivec2 domain_size;
    domain_size(0) = domain_dims(0);
    domain_size(1) = domain_dims(1);
    auto marcher = RayMarch_new(ray_start, ray_end, domain_size);
    if (!marcher) {
        return empty_hit();
    }

    RayMarchState s = *marcher;

    yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> result = empty_hit();
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> sample;
#ifndef TRACE_OPAQUE_LIGHTS
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> chi_sample;
#endif
    while (next_intersection(&s)) {
        auto sample_coord = s.p;
        if (sample_coord(0) < 0 || sample_coord(0) >= CANVAS_X) {
            printf("out x <%d, %d>, (%f, %f), [%f,%f] -> [%f,%f]\n", 
            sample_coord(0), sample_coord(1), s.hit(0), s.hit(1),
            s.p0(0), s.p0(1), s.p1(0), s.p1(1)
            );
            break;
        }
        if (sample_coord(1) < 0 || sample_coord(1) >= CANVAS_Y) {
            printf("out y <%d, %d>, (%f, %g), [%f,%f] -> [%f,%f]\n", 
            sample_coord(0), sample_coord(1), s.hit(0), s.hit(1),
            s.p0(0), s.p0(1), s.p1(0), s.p1(1)
            );
            break;
        }

        for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
            sample(i) = domain(sample_coord(0), sample_coord(1), i);
        }
#ifdef TRACE_OPAQUE_LIGHTS
        bool hits = false;
        for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
            hits |= bool(sample(i) != FP(0.0));
            if (hits) {
                break;
            }
        }

        if (hits) {
            for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
                result(i) = sample(i);
            }
            result(NUM_COMPONENTS - 1) = FP(0.0);
            return result;
        }
#else
        for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
            chi_sample(i) = chi(sample_coord(0), sample_coord(1), i);
        }

        for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
            fp_t tau = chi_sample(i) * s.ds;
            fp_t source_fn = sample(i) / chi_sample(i);

            for (int r = 0; r < NUM_AZ; ++r) {
                if (az_rays(r) == FP(0.0)) {
                    result(2*i, r) = source_fn;
                } else {
                    fp_t mu = az_rays(r);
                    result(2*i+1, r) += tau / mu;
                    fp_t edt = std::exp(-tau / mu);
                    result(2*i, r) = result(2*i, r) * edt + source_fn * (FP(1.0) - edt);
                }
            }
            if (sample_coord(0) == int(CANVAS_X / 2) && sample_coord(1) == int(CANVAS_X / 2) + 400) {
                printf("tau %g, sfn %g (%g/%g) \n", tau, source_fn, sample(i), chi_sample(i));
            }
        }
#endif
    }

    return result;
}

YAKL_INLINE yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> raymarch(
    const Fp3d& domain, 
    const Fp3d& chi, 
    vec2 ray_start, 
    vec2 direction, 
    fp_t distance,
    yakl::SArray<fp_t, 1, NUM_AZ> az_rays
) {
#ifdef OLD_RAYMARCH
    return dodgy_raymarch(domain, ray_start, direction, distance);
#else
    vec2 ray_end;
    ray_end(0) = ray_start(0) + direction(0) * distance;
    ray_end(1) = ray_start(1) + direction(1) * distance;
#ifdef TRACE_OPAQUE_LIGHTS
    return aw_raymarch(domain, chi, ray_start, ray_end);
#else
    // NOTE(cmo): Swap start/end to facilitate solution to RTE. Could reframe
    // and go the other way, dropping out of the march early if we have
    // traversed sufficient optical depth.
    return aw_raymarch(domain, chi, ray_end, ray_start, az_rays);
#endif
#endif
}

YAKL_INLINE yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> merge_intervals(
    yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> closer, 
    yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> further,
    yakl::SArray<fp_t, 1, NUM_AZ> az_rays
) {
#ifdef TRACE_OPAQUE_LIGHTS
    fp_t transmission = closer(NUM_COMPONENTS-1);
    if (transmission == FP(0.0)) {
        return closer;
    }

    for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
        closer(i) += transmission * further(i);
    }
    closer(NUM_COMPONENTS-1) = closer(NUM_COMPONENTS-1) * further(NUM_COMPONENTS-1);
    return closer;
#else
    // NOTE(cmo): Storage is interleaved [intensity_1, tau_1, intensity_2...] on the first axis
    for (int i = 0; i < NUM_COMPONENTS; i += 2) {
        for (int r = 0; r < NUM_AZ; ++r) {
            if (az_rays(r) == FP(0.0)) {
                continue;
            }
            fp_t transmission = std::exp(-closer(i+1, r));
            closer(i, r) += transmission * further(i, r);
            closer(i+1, r) += further(i+1, r);
        }
    }
    return closer;
#endif
}

void compute_cascade_i (
    State* state,
    int cascade_idx
) {
    const auto& emission = state->emission;
    const auto& chi = state->absorption;
    printf("Cascade idx: %d, vec size: %d\n", cascade_idx, state->cascades.size());
    auto& cascade_i = state->cascades[cascade_idx];
    auto cascade_ip = cascade_i;
    if (cascade_idx != MAX_LEVEL) {
        cascade_ip = state->cascades[cascade_idx + 1];
    }
    auto dims = cascade_i.get_dimensions();
    auto edims = emission.get_dimensions();
    auto upper_dims = cascade_ip.get_dimensions();

    auto az_rays = get_az_rays();

    printf("[%d, %d, %d, %d]\n", dims(0), dims(1), dims(2), dims(3));
    printf("[%d, %d, %d]\n", edims(0), edims(1), edims(2));
    parallel_for(
        SimpleBounds<3>(dims(2), dims(0), dims(1)),
        YAKL_LAMBDA (int ray_idx, int u, int v) {
            // NOTE(cmo): u, v probe indices
            // NOTE(cmo): x, y world coords
            for (int i = 0; i < NUM_COMPONENTS; ++i) {
                for (int j = 0; j < NUM_AZ; ++j) {
                    cascade_i(u, v, ray_idx, i, j) = FP(0.0);
                }
            }

            int upper_cascade_idx = cascade_idx + 1;
            fp_t spacing = PROBE0_SPACING * (1 << cascade_idx);
            fp_t upper_spacing = PROBE0_SPACING * (1 << upper_cascade_idx);
            int num_rays = PROBE0_NUM_RAYS * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            int upper_num_rays = PROBE0_NUM_RAYS * (1 << (upper_cascade_idx * CASCADE_BRANCHING_FACTOR));
            fp_t radius = PROBE0_LENGTH * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            fp_t prev_radius = FP(0.0);
            if (cascade_idx != 0) {
                prev_radius = PROBE0_LENGTH * (1 << ((cascade_idx-1) * CASCADE_BRANCHING_FACTOR));
            }

            fp_t cx = (u + FP(0.5)) * spacing;
            fp_t cy = (v + FP(0.5)) * spacing;
            fp_t distance = radius - prev_radius;

            // NOTE(cmo): bc: bottom corner of 2x2 used for bilinear interp on next cascade
            // uc: upper corner
            int upper_u_bc = int((u - 1) / 2);
            int upper_v_bc = int((v - 1) / 2);
            fp_t u_bc_weight = FP(0.25) + FP(0.5) * (u % 2);
            fp_t v_bc_weight = FP(0.25) + FP(0.5) * (v % 2);
            fp_t u_uc_weight = FP(1.0) - u_bc_weight;
            fp_t v_uc_weight = FP(1.0) - v_bc_weight;
            int u_bc = yakl::max(upper_u_bc, 0);
            int v_bc = yakl::max(upper_v_bc, 0);
            int u_uc = yakl::min(u_bc+1, (int)upper_dims(0));
            int v_uc = yakl::min(v_bc+1, (int)upper_dims(1));

            fp_t angle = FP(2.0) * FP(M_PI) / num_rays * (ray_idx + FP(0.5));
            vec2 direction;
            direction(0) = yakl::cos(angle);
            direction(1) = yakl::sin(angle);
            vec2 start;
            start(0) = cx + prev_radius * direction(0);
            start(1) = cy + prev_radius * direction(1);

            int upper_ray_start_idx = ray_idx * (1 << CASCADE_BRANCHING_FACTOR);
            int num_rays_per_ray = 1 << CASCADE_BRANCHING_FACTOR;
            fp_t ray_weight = FP(1.0) / num_rays_per_ray;

            if (BILINEAR_FIX) {
            } else {
                auto sample = raymarch(emission, chi, start, direction, distance, az_rays);
                decltype(sample) upper_sample(FP(0.0));
                // NOTE(cmo): Sample upper cascade.
                // if (cascade_idx != MAX_LEVEL && sample(NUM_COMPONENTS-1) > FP(0.0)) {
                if (cascade_idx != MAX_LEVEL) {
                    for (
                        int upper_ray_idx = upper_ray_start_idx;
                        upper_ray_idx < upper_ray_start_idx + num_rays_per_ray;
                        ++upper_ray_idx
                    ) {
                        for (int i = 0; i < NUM_COMPONENTS; ++i) {
                            for (int r = 0; r < NUM_AZ; ++r) {
                                if (az_rays(r) == FP(0.0)) {
                                    // NOTE(cmo): Can't merge the in-out of page ray.
                                    continue;
                                }
                                fp_t u_11 = cascade_ip(u_bc, v_bc, upper_ray_idx, i, r);
                                fp_t u_21 = cascade_ip(u_uc, v_bc, upper_ray_idx, i, r);
                                fp_t u_12 = cascade_ip(u_bc, v_uc, upper_ray_idx, i, r);
                                fp_t u_22 = cascade_ip(u_uc, v_uc, upper_ray_idx, i, r);
                                fp_t term = (
                                    v_bc_weight * (u_bc_weight * u_11 + u_uc_weight * u_21) +
                                    v_uc_weight * (u_bc_weight * u_12 + u_uc_weight * u_22)
                                );
                                upper_sample(i, r) += ray_weight * term;
                            }
                        }
                    }
                }
                auto merged = merge_intervals(sample, upper_sample, az_rays);
                for (int i = 0; i < NUM_COMPONENTS; ++i) {
                    for (int r = 0; r < NUM_AZ; ++r) {
                        cascade_i(u, v, ray_idx, i, r) = merged(i, r);
                    }
                }
            }
        }
    );
}

#ifdef HAVE_MPI
void save_results(Fp5d final_cascade) {
    printf("Writing\n");
    auto dims = final_cascade.get_dimensions();
    yakl::Array<double, 4, yakl::memDevice> fp64_copy("fp64_copy", dims(0), dims(1), dims(2), NUM_WAVELENGTHS);
    auto az_weights = get_az_weights();
    parallel_for(
        SimpleBounds<3>(dims(0), dims(1), dims(2)),
        YAKL_LAMBDA (int x, int y, int ray_idx) {
            for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
#ifdef TRACE_OPAQUE_LIGHTS
                fp64_copy(x, y, ray_idx, i)  = final_cascade(x, y, ray_idx, i);
#else
                fp64_copy(x, y, ray_idx, i) = 0.0;
                for (int r = 0; r < NUM_AZ; ++r) {
                    fp64_copy(x, y, ray_idx, i)  += az_weights(r) * final_cascade(x, y, ray_idx, 2*i, r);
                }
#endif
            }
        }
    );
    yakl::fence();
    auto host_copy = fp64_copy.createHostCopy();


    auto ncwrap = []( int ierr , int line ) {
        if (ierr != NC_NOERR) {
            printf("NetCDF Error at line: %d\n", line);
            printf("%s\n", ncmpi_strerror(ierr));
            exit(-1);
        }
    };

    auto out_dims = host_copy.get_dimensions();
    int ncid, x_dimid, y_dimid, ray_dimid, col_dimid, im_id;
    ncwrap(ncmpi_create(MPI_COMM_WORLD, "output.nc", NC_CLOBBER, MPI_INFO_NULL, &ncid), __LINE__);
    ncwrap(ncmpi_def_dim(ncid, "x", out_dims(0), &x_dimid), __LINE__);
    ncwrap(ncmpi_def_dim(ncid, "y", out_dims(1), &y_dimid), __LINE__);
    ncwrap(ncmpi_def_dim(ncid, "ray", out_dims(2), &ray_dimid), __LINE__);
    ncwrap(ncmpi_def_dim(ncid, "col", out_dims(3), &col_dimid), __LINE__);
    int dimids[4];
    dimids[0] = x_dimid;
    dimids[1] = y_dimid;
    dimids[2] = ray_dimid;
    dimids[3] = col_dimid;

    ncwrap(ncmpi_def_var(ncid, "image", NC_DOUBLE, 4, dimids, &im_id), __LINE__);
    ncwrap(ncmpi_enddef(ncid), __LINE__);
    MPI_Offset start[4] = {0};
    MPI_Offset count[4];
    count[0] = out_dims(0); 
    count[1] = out_dims(1); 
    count[2] = out_dims(2);
    count[3] = out_dims(3);
    ncwrap(ncmpi_put_vara_double_all(ncid, im_id, start, count, host_copy.data()), __LINE__);
    ncwrap(ncmpi_close(ncid), __LINE__);
}
#endif

int main(int argc, char** argv) {
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    yakl::init();
    {
        State state;
        init_state(&state);
        auto dims = state.cascades[0].get_dimensions();
        printf("[[%d, %d, %d, %d]]\n", dims(0), dims(1), dims(2), dims(3));


        for (int i = MAX_LEVEL; i >= 0; --i) {
            compute_cascade_i(&state, i);
            yakl::fence();
        }

#ifdef HAVE_MPI
        dims = state.cascades[0].get_dimensions();
        save_results(state.cascades[0]);
#endif
    }
    yakl::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
