#if !defined(DEXRT_DIRECTIONAL_EMIS_OPAC_INTERP_HPP)
#define DEXRT_DIRECTIONAL_EMIS_OPAC_INTERP_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "EmisOpac.hpp"
#include "Atmosphere.hpp"

struct DirectionalEmisOpacInterp {
    Fp3d emis_vel; // [k_active, wave, vel]
    Fp3d opac_vel; // [k_active, wave, vel]
    // NOTE(cmo): Stacking the above in a 4d array may lead to better cache usage (emis/opac as last dim)
    Fp2d vel_start; // [k_active, wave]
    Fp2d vel_step; // [k_active, wave]

    void zero() {
        emis_vel = FP(0.0);
        opac_vel = FP(0.0);
        vel_start = FP(0.0);
        vel_step = FP(0.0);
        yakl::fence();
    }

    template <int RcMode>
    void fill(
        const State& state,
        const CascadeState& casc_state,
        const CascadeCalcSubset& subset,
        const Fp3d& n_star
    ) const {
        Fp1d max_vel("max_vel", emis_vel.extent(0));
        Fp1d min_vel("min_vel", emis_vel.extent(0));
        // NOTE(cmo): Relativistic flows should be fast enough!
        max_vel = -FP(1e8);
        min_vel = FP(1e8);
        yakl::fence();

        fmt::println("fill");

        // NOTE(cmo): Base this off the highest angular resolution cascade
        const int num_cascades = casc_state.num_cascades;
        const int cascade_idx = casc_state.num_cascades;
        CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, cascade_idx);
        CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset.subset_idx);

        i64 spatial_bounds = ray_subset.num_probes(1) * ray_subset.num_probes(0);
        yakl::Array<i32, 2, yakl::memDevice> probe_space_lookup;
        const bool sparse_calc = state.config.sparse_calculation;
        if (sparse_calc) {
            probe_space_lookup = casc_state.probes_to_compute[0];
            spatial_bounds = probe_space_lookup.extent(0);
        }
        assert(emis_vel.extent(0) == spatial_bounds && "Sparse sizes don't match");
        const auto& incl_quad = state.incl_quad;
        const auto& atmos = state.atmos;
        const auto& active_map = state.active_map;

        parallel_for(
            "Min/Max Vel",
            SimpleBounds<3>(
                spatial_bounds,
                ray_subset.num_flat_dirs,
                ray_subset.num_incl
            ),
            YAKL_LAMBDA (i64 k, int phi_idx, int theta_idx) {
                int u, v;
                if (sparse_calc) {
                    u = probe_space_lookup(k, 0);
                    v = probe_space_lookup(k, 1);
                } else {
                    // NOTE(cmo): As in the loop over probes we iterate as [v, u] (u
                    // fast-running), but index as [u, v], i.e. dims.num_probes(0) =
                    // dim(u). Typical definition of k = u * Nv + v, but here we do
                    // loop index k = v * Nu + u where Nu = dims.num_probes(0). This
                    // preserves our iteration ordering
                    u = k % ray_set.num_probes(0);
                    v = k / ray_set.num_probes(0);
                }
                ivec2 cell_coord;
                cell_coord(0) = u;
                cell_coord(1) = v;
                phi_idx += ray_subset.start_flat_dirs;
                theta_idx += ray_subset.start_incl;
                ProbeIndex probe_idx{
                    .coord=cell_coord,
                    .dir=phi_idx,
                    .incl=theta_idx,
                    .wave=0
                };
                RayProps ray = ray_props(ray_set, num_cascades, cascade_idx, probe_idx);
                const fp_t incl = incl_quad.muy(theta_idx);
                vec3 mu;
                const fp_t sin_theta = std::sqrt(FP(1.0) - square(incl));
                mu(0) = ray.dir(0) * sin_theta;
                mu(1) = incl;
                mu(2) = ray.dir(1) * sin_theta;
                // NOTE(cmo): The k in the for loop may be the sparse flat index, rather than the flat index.
                i64 k_full = v * atmos.vx.extent(1) + u;
                const fp_t vel = (
                    atmos.vx.get_data()[k_full] * mu(0)
                    + atmos.vy.get_data()[k_full] * mu(1)
                    + atmos.vz.get_data()[k_full] * mu(2)
                );

                const i64 storage_idx = active_map(v, u);
                // TODO(cmo): If the atomics are a bottleneck, we should be able
                // to do the reduction per cell in shared memory
                // TODO(cmo): yes this is a huge issue. Need to move to hierarchical parallelism
                yakl::atomicMin(
                    min_vel(storage_idx),
                    vel
                );
                yakl::atomicMax(
                    max_vel(storage_idx),
                    vel
                );
            }
        );
        yakl::fence();
        fmt::println("Min/max done");

        int wave_batch = subset.la_end - subset.la_start;
        wave_batch = std::min(wave_batch, ray_subset.wave_batch);

        JasUnpack((*this), emis_vel, opac_vel, vel_start, vel_step);
        JasUnpack(state, adata, dynamic_opac, phi, pops);
        JasUnpack(casc_state, eta, chi);
        const auto flatmos = flatten<const fp_t>(atmos);
        // yakl::ScalarLiveOut<fp_t> max_thermal_vel_frac(FP(0.0));
        // yakl::ScalarLiveOut<i32> thermal_vel_frac_over_count(0);
        auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
        auto flat_n_star = n_star.reshape<2>(Dims(n_star.extent(0), n_star.extent(1) * n_star.extent(2)));

        parallel_for(
            "Emis/Opac Samples",
            SimpleBounds<3>(
                spatial_bounds,
                wave_batch,
                emis_vel.extent(2)
            ),
            YAKL_LAMBDA (i64 ks, int wave, int vel_idx) {
                int u, v;
                if (sparse_calc) {
                    u = probe_space_lookup(ks, 0);
                    v = probe_space_lookup(ks, 1);
                } else {
                    u = ks % ray_set.num_probes(0);
                    v = ks / ray_set.num_probes(0);
                }
                if (!dynamic_opac(v, u, wave)) {
                    return;
                }

                const int kf = v * atmos.temperature.extent(1) + u;
                const fp_t vmin = min_vel(ks);
                const fp_t vmax = max_vel(ks);
                const fp_t dv = (vmax - vmin) / fp_t(INTERPOLATE_DIRECTIONAL_BINS - 1);
                const fp_t vel = vmin + dv * vel_idx;
                const int la = subset.la_start + wave;
                if (vel_idx == 0) {
                    vel_start(ks, wave) = vmin;
                    vel_step(ks, wave) = dv;
                    // NOTE(cmo): Compare with thermal vel, and have a warning
                    int governing_atom = adata.governing_trans(la).atom;
                    const fp_t vtherm = thermal_vel(adata.mass(governing_atom), flatmos.temperature(kf));
                    const fp_t vtherm_frac = dv / vtherm;

                    // if (vtherm_frac > INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH) {
                    //     yakl::atomicMax(max_thermal_vel_frac(), vtherm_frac);
                    //     yakl::atomicAdd(thermal_vel_frac_over_count(), 1);
                    // }
                }
                AtmosPointParams local_atmos;
                local_atmos.temperature = flatmos.temperature(kf);
                local_atmos.ne = flatmos.ne(kf);
                local_atmos.vturb = flatmos.vturb(kf);
                local_atmos.nhtot = flatmos.nh_tot(kf);
                local_atmos.nh0 = flatmos.nh0(kf);
                local_atmos.vel = vel;

                fp_t chi_s = chi(v, u, wave);
                fp_t eta_s = eta(v, u, wave);
                auto line_terms = emis_opac(
                    EmisOpacState<fp_t>{
                        .adata = adata,
                        .profile = phi,
                        .la = la,
                        .n = flat_pops,
                        .n_star_scratch = flat_n_star,
                        .k = kf,
                        .atmos = local_atmos,
                        .active_set = slice_active_set(adata, la),
                        .mode = EmisOpacMode::DynamicOnly
                    }
                );
                chi_s += line_terms.chi;
                eta_s += line_terms.eta;

                emis_vel(ks, wave, vel_idx) = eta_s;
                opac_vel(ks, wave, vel_idx) = chi_s;
            }
        );
        yakl::fence();
        fmt::println("Emis/opac done");
        // if (thermal_vel_frac_over_count.hostRead() > 0) {
        //     fmt::println(
        //         "{} cells with velocity sampling over {} thermal widths (max: {}), consider increasing INTERPOLATE_DIRECTIONAL_BINS",
        //         thermal_vel_frac_over_count.hostRead(),
        //         INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH,
        //         max_thermal_vel_frac.hostRead()
        //     );
        // }
    }

    YAKL_DEVICE_INLINE EmisOpac sample(i64 ks, int wave, fp_t vel) const {
        fp_t frac_v = (vel - vel_start(ks, wave)) / vel_step(ks, wave);
        int iv = int(frac_v);
        int ivp;
        fp_t tv, tvp;
        if (frac_v < FP(0.0) || frac_v >= (INTERPOLATE_DIRECTIONAL_BINS - 1)) {
            iv = std::min(std::max(iv, 0), INTERPOLATE_DIRECTIONAL_BINS-1);
            ivp = iv;
            tv = FP(1.0);
            tvp = FP(0.0);
        } else {
            ivp = iv + 1;
            tvp = frac_v - iv;
            tv = FP(1.0) - tvp;
        }
        const fp_t eta = tv * emis_vel(ks, wave, iv) + tvp * emis_vel(ks, wave, ivp);
        const fp_t chi = tv * opac_vel(ks, wave, iv) + tvp * opac_vel(ks, wave, ivp);
        return EmisOpac{
            .eta = eta,
            .chi = chi
        };
    }
};

inline
DirectionalEmisOpacInterp DirectionalEmisOpacInterp_new(i64 num_active_zones, int wave_batch) {
    DirectionalEmisOpacInterp result;
    result.emis_vel = Fp3d("emis_vel", num_active_zones, wave_batch, INTERPOLATE_DIRECTIONAL_BINS);
    result.opac_vel = Fp3d("opac_vel", num_active_zones, wave_batch, INTERPOLATE_DIRECTIONAL_BINS);
    result.vel_start = Fp2d("vel_start", num_active_zones, wave_batch);
    result.vel_step = Fp2d("vel_step", num_active_zones, wave_batch);
    result.zero();
    return result;
}



#else
#endif