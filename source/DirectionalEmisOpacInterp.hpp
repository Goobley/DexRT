#if !defined(DEXRT_DIRECTIONAL_EMIS_OPAC_INTERP_HPP)
#define DEXRT_DIRECTIONAL_EMIS_OPAC_INTERP_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "EmisOpac.hpp"
#include "Atmosphere.hpp"

struct FlatVelocity {
    Fp1d vx;
    Fp1d vy;
    Fp1d vz;
};

inline void compute_min_max_vel(
    const State& state,
    const CascadeCalcSubset& subset,
    i32 mip_level,
    const FlatVelocity& vels,
    const Fp1d& min_vel,
    const Fp1d& max_vel
) {
    min_vel = FP(1e8);
    max_vel = -FP(1e8);
    yakl::fence();

    // TODO(cmo): This needs to support flat mipped velocity arrays.
    // Handle min and max vel being the length of the mip, or of the mip chain.
    // Compare with mr_block_map.buffer_len and block_map.buffer_len(1 << mip_level)
    // Exception if neither

    constexpr i32 RcMode = RC_flags_storage();
    CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset.subset_idx);

    JasUnpack(state, incl_quad, atmos, block_map, mr_block_map);

    assert(min_vel.extent(0) == max_vel.extent(0));
    i64 vel_len = min_vel.extent(0);
    i64 vel_idx_offset = 0;
    // NOTE(cmo): Handle case of velocity arrays being length of requested mip
    if (vel_len == block_map.buffer_len(1 << mip_level)) {
        for (int i = 0; i < mip_level; ++i) {
            vel_idx_offset -= block_map.buffer_len(1 << i);
        }
    } else if (vel_len != mr_block_map.buffer_len()) {
        // NOTE(cmo): If not the length of current mip, or all mips, throw
        throw std::runtime_error("Unexpected size in min/max vel calculation");
    }

    parallel_for(
        "Min/Max Vel",
        block_map.loop_bounds(1 << mip_level),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            MultiLevelIndexGen<BLOCK_SIZE, ENTRY_SIZE> idx_gen(block_map, mr_block_map);
            i64 ks = idx_gen.loop_idx(mip_level, tile_idx, block_idx);

            auto vec_norm = [] (vec3 v) -> fp_t {
                return std::sqrt(square(v(0)) + square(v(1))  + square(v(2)));
            };
            auto dot_vecs = [] (vec3 a, vec3 b) -> fp_t {
                fp_t acc = FP(0.0);
                for (int i = 0; i < 3; ++i) {
                    acc += a(i) * b(i);
                }
                return acc;
            };

            vec3 vel;
            // NOTE(cmo): Because we invert the x and z of the view ray when
            // tracing, so invert those in the velocity for this
            // calculation.
            vel(0) = -vels.vx(ks);
            vel(1) = -vels.vy(ks);
            vel(2) = -vels.vz(ks);
            const fp_t vel_norm = vec_norm(vel);
            vec3 vel_dir = vel / vel_norm;
            vec3 opp_vel_dir;
            opp_vel_dir(0) = -vel_dir(0);
            opp_vel_dir(1) = -vel_dir(1);
            opp_vel_dir(2) = -vel_dir(2);
            vec2 vel_angs = dir_to_angs(vel_dir);
            vec2 opp_vel_angs = dir_to_angs(opp_vel_dir);

            fp_t vmin = FP(1e6);
            fp_t vmax = FP(-1e6);

            auto extra_tests = [&vmin, &vmax, &vel, dot_vecs] (vec2 angles, const SphericalAngularRange& range) {
                auto in_range = [] (fp_t ang, vec2 range) -> int {
                    bool r0 = (ang >= range(0));
                    bool r1 = (ang <= range(1));
                    return (r0 && r1);
                };
                const int theta_ir = in_range(angles(0), range.theta_range);
                const int phi_ir = in_range(angles(1), range.phi_range);

                if (theta_ir && phi_ir) {
                    vec3 d = angs_to_dir(angles);
                    const fp_t dot = dot_vecs(vel, d);
                    vmin = std::min(vmin, dot);
                    vmax = std::max(vmax, dot);
                    return;
                }

                vec3 d0, d1;
                if (theta_ir) {
                    vec2 a;
                    a(0) = angles(0);
                    a(1) = range.phi_range(0);
                    d0 = angs_to_dir(a);
                    a(1) = range.phi_range(1);
                    d1 = angs_to_dir(a);
                } else if (phi_ir) {
                    vec2 a;
                    a(0) = range.theta_range(0);
                    a(1) = angles(1);
                    d0 = angs_to_dir(a);
                    a(0) = range.theta_range(1);
                    d1 = angs_to_dir(a);
                } else {
                    return;
                }
                const fp_t dot0 = dot_vecs(vel, d0);
                const fp_t dot1 = dot_vecs(vel, d1);
                vmin = std::min(vmin, std::min(dot0, dot1));
                vmax = std::max(vmax, std::max(dot0, dot1));
            };

            for (
                int phi_idx = ray_subset.start_flat_dirs;
                phi_idx < ray_subset.num_flat_dirs + ray_subset.start_flat_dirs;
                ++phi_idx
            ) {
                SphericalAngularRange ang_range = c0_flat_dir_angular_range(ray_set, incl_quad, phi_idx);
                // NOTE(cmo): Check the 4 corners of the range
                for (int i = 0; i < 4; ++i) {
                    vec2 angs;
                    angs(0) = ang_range.theta_range(i / 2);
                    angs(1) = ang_range.phi_range(i % 2);

                    vec3 d = angs_to_dir(angs);
                    const fp_t dot = dot_vecs(vel, d);
                    vmin = std::min(vmin, dot);
                    vmax = std::max(vmax, dot);
                }

                // NOTE(cmo): Do the extra tests
                extra_tests(vel_angs, ang_range);
                extra_tests(opp_vel_angs, ang_range);
            }

            vmin -= FP(0.02) * std::abs(vmin);
            vmax += FP(0.02) * std::abs(vmax);
            const i64 storage_idx = ks + vel_idx_offset;
            min_vel(storage_idx) = vmin;
            max_vel(storage_idx) = vmax;
        }
    );
    yakl::fence();
}

struct DirectionalEmisOpacInterp {
    Fp4d emis_opac_vel; // [k_active, vel, eta(0)/chi(1), wave]
    Fp1d vel_start; // [k_active]
    Fp1d vel_step; // [k_active]

    void zero() {
        emis_opac_vel = FP(0.0);
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
        Fp1d max_vel("max_vel", emis_opac_vel.extent(0));
        Fp1d min_vel("min_vel", emis_opac_vel.extent(0));

        assert(emis_opac_vel.extent(0) == state.block_map.buffer_len() && "Sparse sizes don't match");
        const auto& incl_quad = state.incl_quad;
        const auto& atmos = state.atmos;
        // const auto& active_map = state.active_map;
        const auto& block_map = state.block_map;

        FlatVelocity vels{
            .vx = atmos.vx,
            .vy = atmos.vy,
            .vz = atmos.vz
        };
        compute_min_max_vel(state, subset, 0, vels, min_vel, max_vel);

        int wave_batch = subset.la_end - subset.la_start;
        CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
        CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset.subset_idx);
        wave_batch = std::min(wave_batch, ray_subset.wave_batch);

        JasUnpack((*this), emis_opac_vel, vel_start, vel_step);
        JasUnpack(state, adata, dynamic_opac, phi, pops);
        JasUnpack(casc_state, eta, chi);
        const auto flatmos = flatten<const fp_t>(atmos);
        // NOTE(cmo): Was getting segfaults with ScalarLiveOuts
        Fp1d max_thermal_vel_frac("max_thermal_vel_frac", 1);
        yakl::Array<i32, 1, yakl::memDevice> thermal_vel_frac_over_count("thermal_vel_frac_over_count", 1);
        max_thermal_vel_frac = FP(0.0);
        thermal_vel_frac_over_count = 0;
        yakl::fence();
        auto flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));
        auto flat_n_star = n_star.reshape<2>(Dims(n_star.extent(0), n_star.extent(1) * n_star.extent(2)));

        auto block_bounds = state.block_map.loop_bounds();
        parallel_for(
            "Emis/Opac Samples",
            SimpleBounds<4>(
                block_bounds.dim(0),
                block_bounds.dim(1),
                emis_opac_vel.extent(1),
                wave_batch
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, int vel_idx, int wave) {
                IndexGen<BLOCK_SIZE> idx_gen(block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
                int u = coord.x;
                int v = coord.z;

                const int kf = v * atmos.temperature.extent(1) + u;
                const fp_t vmin = min_vel(ks);
                const fp_t vmax = max_vel(ks);
                const fp_t dv = (vmax - vmin) / fp_t(INTERPOLATE_DIRECTIONAL_BINS - 1);
                const fp_t vel = vmin + dv * vel_idx;
                const int la = subset.la_start + wave;
                if (vel_idx == 0 && wave == 0) {
                    vel_start(ks) = vmin;
                    vel_step(ks) = dv;
                    // NOTE(cmo): Compare with thermal vel, and have a warning
                    int governing_atom = adata.governing_trans(la).atom;
                    const fp_t vtherm = thermal_vel(adata.mass(governing_atom), flatmos.temperature(kf));
                    const fp_t vtherm_frac = dv / vtherm;

                    if (vtherm_frac > INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH) {
                        yakl::atomicMax(max_thermal_vel_frac(0), vtherm_frac);
                        yakl::atomicAdd(thermal_vel_frac_over_count(0), 1);
                    }
                }
                AtmosPointParams local_atmos;
                local_atmos.temperature = flatmos.temperature(kf);
                local_atmos.ne = flatmos.ne(kf);
                local_atmos.vturb = flatmos.vturb(kf);
                local_atmos.nhtot = flatmos.nh_tot(kf);
                local_atmos.nh0 = flatmos.nh0(kf);
                local_atmos.vel = vel;

                // fp_t chi_s = chi(v, u, wave);
                // fp_t eta_s = eta(v, u, wave);
                fp_t chi_s = FP(0.0);
                fp_t eta_s = FP(0.0);
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
                        // .mode = EmisOpacMode::DynamicOnly
                        .mode = EmisOpacMode::All
                    }
                );
                chi_s += line_terms.chi;
                eta_s += line_terms.eta;

                emis_opac_vel(ks, vel_idx, 0, wave) = eta_s;
                emis_opac_vel(ks, vel_idx, 1, wave) = chi_s;
            }
        );
        yakl::fence();
        i32 count = thermal_vel_frac_over_count.createHostCopy()(0);
        if (count > 0) {
            fp_t max_frac = max_thermal_vel_frac.createHostCopy()(0);
            fmt::println(
                "{} cells with velocity sampling over {} thermal widths (max: {}), consider increasing INTERPOLATE_DIRECTIONAL_BINS",
                count,
                INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH,
                max_frac
            );
        }
    }

    YAKL_INLINE EmisOpac sample(i64 ks, int wave, fp_t vel) const {
        fp_t frac_v = (vel - vel_start(ks)) / vel_step(ks);
        if (vel_step(ks) == FP(0.0)) {
            frac_v = FP(0.0);
        }
        int iv = int(frac_v);
        int ivp;
        fp_t tv, tvp;
        if (frac_v < FP(0.0) || frac_v >= (INTERPOLATE_DIRECTIONAL_BINS - 1)) {
            // if (ks == 173641) {
                // printf("Clamping frac_v: %f (%d) -- %f\n", frac_v, ks, vel);
                // assert(false);
            // }
            iv = std::min(std::max(iv, 0), INTERPOLATE_DIRECTIONAL_BINS-1);
            ivp = iv;
            tv = FP(1.0);
            tvp = FP(0.0);
        } else {
            // printf("Not clamping frac_v: %f\n", frac_v);
            ivp = iv + 1;
            tvp = frac_v - iv;
            tv = FP(1.0) - tvp;
        }
        const fp_t eta = tv * emis_opac_vel(ks, iv, 0, wave) + tvp * emis_opac_vel(ks, ivp, 0, wave);
        const fp_t chi = tv * emis_opac_vel(ks, iv, 1, wave) + tvp * emis_opac_vel(ks, ivp, 1, wave);
        return EmisOpac{
            .eta = eta,
            .chi = chi
        };
    }
};

inline
DirectionalEmisOpacInterp DirectionalEmisOpacInterp_new(i64 num_active_zones, int wave_batch) {
    DirectionalEmisOpacInterp result;
    result.emis_opac_vel = Fp4d("emis_opac_vel", num_active_zones, INTERPOLATE_DIRECTIONAL_BINS, 2, wave_batch);
    result.vel_start = Fp1d("vel_start", num_active_zones);
    result.vel_step = Fp1d("vel_step", num_active_zones);
    result.zero();
    return result;
}



#else
#endif