#if !defined(DEXRT_RAY_MARCHING_2_HPP)
#define DEXRT_RAY_MARCHING_2_HPP
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "RayMarching.hpp"

YAKL_INLINE RadianceInterval merge_intervals(
    RadianceInterval closer,
    RadianceInterval further
) {
    fp_t transmission = std::exp(-closer.tau);
    closer.I += transmission * further.I;
    closer.tau += further.tau;
    return closer;
}

template <typename Bc>
struct Raymarch2dArgs {
    const CascadeStateAndBc<Bc>& casc_state_bc;
    RayProps ray;
    fp_t distance_scale = FP(1.0);
    fp_t incl;
    fp_t incl_weight;
    int wave;
    int la;
    vec3 offset;
};

template <int RcMode=0, typename Bc>
YAKL_INLINE RadianceInterval dda_raymarch_2d(
    const Raymarch2dArgs<Bc>& args
) {
    JasUnpack(args, casc_state_bc, ray, distance_scale, incl, incl_weight, wave, la, offset);
    JasUnpack(casc_state_bc, state, bc);

    auto domain_dims = state.eta.get_dimensions();
    ivec2 domain_size;
    // NOTE(cmo): This is swapped as the coord is still x,y,z, but the array is indexed (z,y,x)
    domain_size(0) = domain_dims(1);
    domain_size(1) = domain_dims(0);
    bool start_clipped;
    auto marcher = RayMarch2d_new(ray.start, ray.end, domain_size, &start_clipped);
    RadianceInterval result{};
    if ((RcMode & RC_SAMPLE_BC) && (!marcher || start_clipped)) {
        // NOTE(cmo): Check the ray is going up along z.
        if ((ray.dir(1) > FP(0.0)) && la != -1) {
            const fp_t cos_theta = incl;
            const fp_t sin_theta = std::sqrt(1.0 - square(cos_theta));
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

    fp_t eta_s, chi_s;
    do {
        const auto& sample_coord(s.curr_coord);

        if (sample_coord(0) < 0 || sample_coord(0) >= domain_size(0)) {
            break;
        }
        if (sample_coord(1) < 0 || sample_coord(1) >= domain_size(1)) {
            break;
        }

        eta_s = state.eta(sample_coord(1), sample_coord(0), wave);
        chi_s = state.chi(sample_coord(1), sample_coord(0), wave);

        const bool final_step = (s.t == s.max_t);

        fp_t tau = chi_s * s.dt * distance_scale;
        fp_t source_fn = eta_s / chi_s;

        // NOTE(cmo): implicit assumption muy != 1.0
        fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
        fp_t tau_mu = tau / sin_theta;
        fp_t edt, one_m_edt;
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

    return result;
}


#else
#endif