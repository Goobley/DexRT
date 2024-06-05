#if !defined(DEXRT_DYNAMIC_RAYMARCHING_HPP)
#define DEXRT_DYNAMIC_RAYMARCHING_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "JasPP.hpp"
#include "RayMarching.hpp"
#include "PromweaverBoundary.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "Voigt.hpp"
#include "GammaMatrix.hpp"
#include <optional>

struct DynamicRadianceInterval {
    fp_t I = FP(0.0);
    fp_t tau = FP(0.0);
};

YAKL_INLINE DynamicRadianceInterval merge_intervals(
    DynamicRadianceInterval closer,
    DynamicRadianceInterval further
) {
    // Assumes muy is never 1.0
    fp_t transmission = std::exp(-closer.tau);
    closer.I += transmission * further.I;
    closer.tau += further.tau;
    return closer;
}

struct Raymarch2dDynamicArgs {
    FpConst2d eta = Fp2d();
    FpConst2d chi = Fp2d();
    vec2 ray_start;
    vec2 ray_end;
    vec2 centre;
    fp_t mux;
    fp_t muy;
    fp_t muz;
    fp_t muy_weight;
    fp_t distance_scale = FP(1.0);
    vec3 offset = {};
    bool compute_rates = false;
    const Atmosphere& atmos;
    const CompAtom<fp_t>& atom;
    const yakl::Array<i32, 1, yakl::memDevice>& active_set;
    const VoigtProfile<fp_t, false>& phi;
    const Fp2d& nh0;
    const Fp2d& n;
    const Fp2d& n_star_scratch;
    const DynamicRadianceInterval& upper_sample = {};
    const PwBc<>& bc;
    const Fp3d& Gamma;
    fp_t wl_ray_weight;
    int la;
};

template <typename T=fp_t, int mem_space=yakl::memDevice>
struct AccumulateTransToGammaState {
    const EmisOpacState<T, mem_space>& emis_opac_state;
    const Fp3d& Gamma;
    i64 k;
    fp_t alo;
    fp_t I;
    fp_t wlamu;
};
template <typename T=fp_t, int mem_space=yakl::memDevice>
YAKL_INLINE void accumulate_transitions_to_Gamma(
    const AccumulateTransToGammaState<T, mem_space>& args
) {
    JasUnpack(args, emis_opac_state, Gamma, k, alo, I, wlamu);
    JasUnpack(emis_opac_state, atom, active_set, la, n);

    for (int kr = 0; kr < active_set.extent(0); ++kr) {
        const auto& l = atom.lines(active_set(kr));
        const UV uv = compute_uv_line(
            emis_opac_state,
            active_set(kr)
        );
        const fp_t nj = n(l.j, k);
        const fp_t ni = n(l.i, k);
        const fp_t eta = nj * uv.Uji;
        const fp_t chi = ni * uv.Vij - nj * uv.Vji;
        add_to_gamma<true>(GammaAccumState{
            .eta = eta,
            .chi = chi,
            .uv = uv,
            .I = I,
            .alo = alo,
            .wlamu = wlamu,
            .Gamma = Gamma,
            .i = l.i,
            .j = l.j,
            .k = k
        });
    }

    for (int kr = 0; kr < atom.continua.extent(0); ++kr) {
        const auto& cont = atom.continua(kr);
        if (!cont.is_active(la)) {
            continue;
        }

        const UV uv = compute_uv_cont(
            emis_opac_state,
            kr
        );
        const fp_t nj = n(cont.j, k);
        const fp_t ni = n(cont.i, k);
        const fp_t eta = nj * uv.Uji;
        const fp_t chi = ni * uv.Vij - nj * uv.Vji;
        add_to_gamma<true>(GammaAccumState{
            .eta = eta,
            .chi = chi,
            .uv = uv,
            .I = I,
            .alo = alo,
            .wlamu = wlamu,
            .Gamma = Gamma,
            .i = cont.i,
            .j = cont.j,
            .k = k
        });
    }
}

template <bool compute_alo=false, bool SampleBoundary=false>
YAKL_INLINE DynamicRadianceInterval dynamic_dda_raymarch_2d(
    const Raymarch2dDynamicArgs& args
) {
    JasUnpack(args, eta, chi, ray_start, ray_end, mux, muy, muz, muy_weight, wl_ray_weight, Gamma);
    JasUnpack(args, distance_scale, atom, phi, atmos, n, n_star_scratch, la, nh0, active_set);
    namespace Const = ConstantsFP;

    DynamicRadianceInterval ri{
        .I = FP(0.0),
        .tau = FP(0.0)
    };

    auto domain_dims = eta.get_dimensions();
    ivec2 domain_size;
    // NOTE(cmo): This is swapped as the coord is still x,y,z, but the array is indexed (z,y,x)
    domain_size(0) = domain_dims(1);
    domain_size(1) = domain_dims(0);
    bool start_clipped;
    auto marcher = RayMarch2d_new(ray_start, ray_end, domain_size, &start_clipped);
    if (SampleBoundary && USE_BC && (!marcher || start_clipped)) {
        // NOTE(cmo): Sample BC
        static_assert(!(USE_BC && USE_MIPMAPS), "BCs not currently supported with mipmaps");
        static_assert(!(USE_BC && !USE_ATMOSPHERE), "BCs not supported outside of atmosphere mode");

        // NOTE(cmo): Check the ray is going up along z.
        if (ray_start(1) < ray_end(1) && la != -1) {
            vec3 mu;
            // NOTE(cmo): Outgoing ray-direction -- inverted from traversal direction
            const fp_t flat_factor = std::sqrt(FP(1.0) - square(muy));
            mu(0) = -mux / flat_factor;
            mu(1) = FP(0.0);
            mu(2) = -muz / flat_factor;
            vec3 pos(args.offset);
            pos(0) += args.centre(0) * distance_scale;
            pos(2) += args.centre(1) * distance_scale;

            fp_t start_I = sample_boundary(args.bc, la, pos, mu);
            if constexpr (PWBC_SAMPLE_CONE) {
                constexpr fp_t cone_half_angle = FP(2.0) * Const::pi / FP(2048.0);
                constexpr fp_t edge_weight = FP(2.5) / FP(9.0);
                constexpr fp_t centre_weight = FP(4.0)/ FP(9.0);
                constexpr fp_t gl_sample =  FP(0.7745966692414834);
                const fp_t cos_cone = std::cos(cone_half_angle * gl_sample);
                const fp_t sin_cone = std::sin(cone_half_angle * gl_sample);
                start_I *= centre_weight;

                // cos(x+y) = cosx cosy - sinx siny
                // cos(x-y) = cosx cosy + sinx siny
                // sin(x+y) = sinx cosy + cosx siny
                // sin(x-y) = sinx cosy - cosx siny
                vec3 mu_cone;
                mu_cone(0) = mu(0) * cos_cone - mu(2) * sin_cone;
                mu_cone(1) = FP(0.0);
                mu_cone(2) = mu(2) * cos_cone + mu(0) * sin_cone;
                start_I += edge_weight * sample_boundary(args.bc, la, pos, mu_cone);

                mu_cone(0) = mu(0) * cos_cone + mu(2) * sin_cone;
                mu_cone(1) = FP(0.0);
                mu_cone(2) = mu(2) * cos_cone - mu(0) * sin_cone;
                start_I += edge_weight * sample_boundary(args.bc, la, pos, mu_cone);
            }
            // NOTE(cmo): The extra terms are correcting for solid angle so J is correct -- mux is cos(angle), but sin from the perspective of solid angle with z being our primary axis
            ri.I = (
                start_I
                * std::abs(mu(0))
                * FP(0.5)
                * Const::pi
            );
        }
    }
    if (!marcher) {
        return merge_intervals(ri, args.upper_sample);
    }

    RayMarchState2d s = *marcher;

    do {
        const auto& sample_coord(s.curr_coord);

        if (sample_coord(0) < 0 || sample_coord(0) >= domain_size(0)) {
            auto hit = s.p0 + s.t * s.direction;
            if (false) {
                printf("out x <%d, %d>, (%f, %f), [%f,%f] -> [%f,%f] (%s) [%d, %d] [%d, %d]\n",
                s.curr_coord(0), s.curr_coord(1), hit(0), hit(1),
                s.p0(0), s.p0(1), s.p1(0), s.p1(1), s.t == s.max_t ? "true" : "false",
                s.next_coord(0), s.next_coord(1), s.final_coord(0), s.final_coord(1)
                );
            }
            break;
        }
        if (sample_coord(1) < 0 || sample_coord(1) >= domain_size(1)) {
            auto hit = s.p0 + s.t * s.direction;
            if (false) {
                printf("out y <%d, %d>, (%f, %g), [%f,%f] -> [%f,%f] (%s)\n",
                s.curr_coord(0), s.curr_coord(1), hit(0), hit(1),
                s.p0(0), s.p0(1), s.p1(0), s.p1(1), s.t == s.max_t ?  "true" : "false"
                );
            }
            break;
        }

        // NOTE(cmo): Indexing (z, x) here
        const int64_t k = sample_coord(1) * atmos.temperature.extent(1) + sample_coord(0);
        fp_t eta_s = eta.get_data()[k];
        fp_t chi_s = chi.get_data()[k] + FP(1e-20);

        fp_t v_norm = std::sqrt(
                square(atmos.vx.get_data()[k])
                + square(atmos.vy.get_data()[k])
                + square(atmos.vz.get_data()[k])
        );
        fp_t vel = (
            atmos.vx.get_data()[k] * mux
            + atmos.vy.get_data()[k] * muy
            + atmos.vz.get_data()[k] * muz
        );
        AtmosPointParams local_atmos{
            .temperature = atmos.temperature.get_data()[k],
            .ne = atmos.ne.get_data()[k],
            .vturb = atmos.vturb.get_data()[k],
            .nhtot = atmos.nh_tot.get_data()[k],
            .vel = vel,
            .nh0 = nh0.get_data()[k]
        };
        auto emis_opac_state = EmisOpacState<fp_t>{
            .atom = atom,
            .profile = phi,
            .la = la,
            .n = n,
            .n_star_scratch = n_star_scratch,
            .k = k,
            .atmos = local_atmos,
            .active_set = active_set,
            .mode = EmisOpacMode::DynamicOnly
        };
        if (
            active_set.extent(0) > 0
            && v_norm > ANGLE_INVARIANT_THERMAL_VEL_FRAC * thermal_vel(atom.mass, local_atmos.temperature)
        ) {
            auto lines = emis_opac(emis_opac_state);
            eta_s += lines.eta;
            chi_s += lines.chi;
        }


        fp_t tau_s = chi_s * s.dt * distance_scale;
        // TODO(cmo): Add background scattering
        fp_t source_fn = eta_s / chi_s;

        if (muy == FP(0.0)) {
            ri.I = source_fn;
        } else {
            fp_t sinphi = std::sqrt(FP(1.0) - square(muy));
            fp_t tau_mu = tau_s / sinphi;
            fp_t edt, one_m_edt;
            if (tau_mu < FP(1e-2)) {
                edt = FP(1.0) + (-tau_mu) + FP(0.5) * square(tau_mu);
                one_m_edt = -std::expm1(-tau_mu);
            } else {
                edt = std::exp(-tau_mu);
                one_m_edt = -std::expm1(-tau_mu);
            }
            ri.tau += tau_mu;
            ri.I = ri.I * edt + source_fn * one_m_edt;
            const bool final_step = (s.t == s.max_t);
            if (final_step) {
                ri = merge_intervals(ri, args.upper_sample);
                if constexpr (compute_alo) {
                    const fp_t alo = one_m_edt;
                    accumulate_transitions_to_Gamma(
                        AccumulateTransToGammaState<>{
                            .emis_opac_state = emis_opac_state,
                            .Gamma = Gamma,
                            .k = k,
                            .alo = alo,
                            .I = ri.I,
                            .wlamu = wl_ray_weight * muy_weight
                        }
                    );
                }
            }
        }
    } while (next_intersection(&s));

    return ri;
}

template <bool compute_alo=false, bool SampleBoundary=false>
YAKL_INLINE  DynamicRadianceInterval dynamic_raymarch_2d(
    const Raymarch2dDynamicArgs& args
) {
    // NOTE(cmo): Swap start/end to facilitate solution to RTE. Could reframe
    // and go the other way, dropping out of the march early if we have
    // traversed sufficient optical depth.
    fp_t sx = args.ray_start(0);
    fp_t sy = args.ray_start(1);
    args.ray_start(0) = args.ray_end(0);
    args.ray_start(1) = args.ray_end(1);
    args.ray_end(0) = sx;
    args.ray_end(1) = sy;

    return dynamic_dda_raymarch_2d<compute_alo, SampleBoundary>(args);
}

#else
#endif