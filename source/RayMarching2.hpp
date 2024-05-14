#if !defined(DEXRT_RAY_MARCHING_2_HPP)
#define DEXRT_RAY_MARCHING_2_HPP
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "RayMarching.hpp"
#include "EmisOpac.hpp"

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
    yakl::Array<bool, 2, yakl::memDevice> active;
    DynamicState dyn_state;
};

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<RcMode & RC_COMPUTE_ALO, fp_t, DexEmpty>
>
YAKL_INLINE RadianceInterval<Alo> dda_raymarch_2d(
    const Raymarch2dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, casc_state_bc, ray, distance_scale, incl, incl_weight, wave, la, offset, dyn_state);
    JasUnpack(casc_state_bc, state, bc);
    constexpr bool dynamic = RcMode & RC_DYNAMIC;
    static_assert(!dynamic || (dynamic && std::is_same_v<DynamicState, Raymarch2dDynamicState>), "If dynamic must provide dynamic state");

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

    // NOTE(cmo): one_m_edt is also the ALO
    fp_t eta_s = FP(0.0), chi_s = FP(1e-20), one_m_edt = FP(0.0);
    do {
        const auto& sample_coord(s.curr_coord);
        const int u = sample_coord(0);
        const int v = sample_coord(1);

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
        if constexpr (dynamic) {
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
                EmisOpacState<fp_t> emis_opac_state{
                    .adata = dyn_state.adata,
                    .profile = dyn_state.profile,
                    .la = la,
                    .n = dyn_state.n,
                    .k = k,
                    .atmos = local_atmos,
                    .active_set = dyn_state.active_set,
                    .mode = EmisOpacMode::DynamicOnly
                };
                auto lines = emis_opac(emis_opac_state);
                eta_s += lines.eta;
                chi_s += lines.chi;
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