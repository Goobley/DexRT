#if !defined(DEXRT_DYNAMIC_RAYMARCHING_HPP)
#define DEXRT_DYNAMIC_RAYMARCHING_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "JasPP.hpp"
#include "RayMarching.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "Voigt.hpp"
#include <optional>

struct Raymarch2dDynamicArgs {
    FpConst2d eta = Fp2d();
    FpConst2d chi = Fp2d();
    vec2 ray_start;
    vec2 ray_end;
    fp_t mux;
    fp_t muy;
    fp_t muz;
    fp_t muy_weight;
    fp_t distance_scale = FP(1.0);
    bool compute_rates = false;
    const Atmosphere& atmos;
    const CompAtom<fp_t>& atom;
    const yakl::Array<i32, 1, yakl::memDevice>& active_set;
    const VoigtProfile<fp_t, false>& phi;
    const Fp2d& nh0;
    const Fp2d& n;
    const Fp2d& n_star_scratch;
    int la;
};

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

YAKL_INLINE DynamicRadianceInterval dynamic_dda_raymarch_2d(
    const Raymarch2dDynamicArgs& args
) {
    JasUnpack(args, eta, chi, ray_start, ray_end, mux, muy, muz, muy_weight);
    JasUnpack(args, distance_scale, atom, phi, atmos, n, n_star_scratch, la, nh0, active_set);

    DynamicRadianceInterval ri{
        .I = FP(0.0),
        .tau = FP(0.0)
    };

    auto domain_dims = eta.get_dimensions();
    ivec2 domain_size;
    // NOTE(cmo): This is swapped as the coord is still x,y,z, but the array is indexed (z,y,x)
    domain_size(0) = domain_dims(1);
    domain_size(1) = domain_dims(0);
    auto marcher = RayMarch2d_new(ray_start, ray_end, domain_size);
    if (!marcher) {
        return ri;
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
        fp_t temperature = atmos.temperature.get_data()[k];
        if (
            active_set.extent(0) > 0 
            && v_norm > ANGLE_INVARIANT_THERMAL_VEL_FRAC * thermal_vel(atom.mass, temperature)
        ) {
            fp_t vel = (
                atmos.vx.get_data()[k] * mux
                + atmos.vy.get_data()[k] * muy
                + atmos.vz.get_data()[k] * muz
            );
            AtmosPointParams local_atmos{
                .temperature = temperature,
                .ne = atmos.ne.get_data()[k],
                .vturb = atmos.vturb.get_data()[k],
                .nhtot = atmos.nh_tot.get_data()[k],
                .vel = vel,
                .nh0 = nh0.get_data()[k]
            };

            auto lines = emis_opac(EmisOpacState<fp_t>{
                .atom = atom,
                .profile = phi,
                .la = la,
                .n = n,
                .n_star_scratch = n_star_scratch,
                .k = k,
                .atmos = local_atmos,
                .active_set = active_set,
                .mode = EmisOpacMode::DynamicOnly
            });
            eta_s += lines.eta;
            chi_s += lines.chi;
        }

        const bool final_step = (s.t == s.max_t);
        const bool compute_rates = final_step && (args.compute_rates);

        fp_t tau_s = chi_s * s.dt * distance_scale;
        // TODO(cmo): Add background scattering
        fp_t source_fn = eta_s / chi_s;

        const fp_t weight = FP(1.0) / PROBE0_NUM_RAYS;
        if (muy == FP(0.0)) {
            ri.I = source_fn;
        } else {
            fp_t tau_mu = tau_s / muy;
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
            if (compute_rates) {

            }
        }
    } while (next_intersection(&s));

    return ri;
}

template <bool UseMipmaps=USE_MIPMAPS, int NumWavelengths=NUM_WAVELENGTHS, int NumAz=NUM_AZ, int NumComponents=NUM_COMPONENTS>
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

    return dynamic_dda_raymarch_2d(args);
}

#else
#endif