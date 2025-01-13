#if !defined(DEXRT_DIRECTIONAL_EMIS_OPAC_INTERP_HPP)
#define DEXRT_DIRECTIONAL_EMIS_OPAC_INTERP_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "EmisOpac.hpp"
#include "Atmosphere.hpp"

struct FlatVelocity {
    Fp1d vx;
    Fp1d vy;
    Fp1d vz;
};

/// Compute the min/max velocity projection for an arbitrary ray in the region
/// associated with this subset of the cascade at each cell. Can write into
/// full-length mipped arrays.
void compute_min_max_vel(
    const State& state,
    const CascadeCalcSubset& subset,
    i32 mip_level,
    const FlatVelocity& vels,
    const Fp1d& min_vel,
    const Fp1d& max_vel
);

struct DirectionalEmisOpacInterp {
    Fp4d emis_opac_vel; // [k_active, vel, eta(0)/chi(1), wave]
    Fp1d vel_start; // [k_active]
    Fp1d vel_step; // [k_active]

    void init(i64 num_active_zones, i32 wave_batch);
    void zero() const;

    template <int RcMode>
    void fill(
        const State& state,
        const CascadeCalcSubset& subset,
        const FlatVelocity& vels,
        const Fp2d& n_star
    ) const {
        Fp1d max_vel("max_vel", emis_opac_vel.extent(0));
        Fp1d min_vel("min_vel", emis_opac_vel.extent(0));

        // assert(emis_opac_vel.extent(0) == state.block_map.buffer_len() && "Sparse sizes don't match");
        const auto& atmos = state.atmos;
        // const auto& active_map = state.active_map;
        const auto& block_map = state.mr_block_map.block_map;

        compute_min_max_vel(state, subset, 0, vels, min_vel, max_vel);

        int wave_batch = subset.la_end - subset.la_start;
        CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
        CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset.subset_idx);
        wave_batch = std::min(wave_batch, ray_subset.wave_batch);

        JasUnpack((*this), emis_opac_vel, vel_start, vel_step);
        JasUnpack(state, adata, phi, pops);
        const auto flatmos = flatten<const fp_t>(atmos);
        // NOTE(cmo): Was getting segfaults with ScalarLiveOuts
        Fp1d max_thermal_vel_frac("max_thermal_vel_frac", 1);
        yakl::Array<i32, 1, yakl::memDevice> thermal_vel_frac_over_count("thermal_vel_frac_over_count", 1);
        max_thermal_vel_frac = FP(0.0);
        thermal_vel_frac_over_count = 0;
        yakl::fence();

        auto block_bounds = block_map.loop_bounds();
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
                    const fp_t vtherm = thermal_vel(adata.mass(governing_atom), flatmos.temperature(ks));
                    const fp_t vtherm_frac = dv / vtherm;

                    if (vtherm_frac > INTERPOLATE_DIRECTIONAL_MAX_THERMAL_WIDTH) {
                        yakl::atomicMax(max_thermal_vel_frac(0), vtherm_frac);
                        yakl::atomicAdd(thermal_vel_frac_over_count(0), 1);
                    }
                }
                AtmosPointParams local_atmos;
                local_atmos.temperature = flatmos.temperature(ks);
                local_atmos.ne = flatmos.ne(ks);
                local_atmos.vturb = flatmos.vturb(ks);
                local_atmos.nhtot = flatmos.nh_tot(ks);
                local_atmos.nh0 = flatmos.nh0(ks);
                local_atmos.vel = vel;

                fp_t chi_s = FP(0.0);
                fp_t eta_s = FP(0.0);
                auto line_terms = emis_opac(
                    EmisOpacState<fp_t>{
                        .adata = adata,
                        .profile = phi,
                        .la = la,
                        .n = pops,
                        .n_star_scratch = n_star,
                        .k = ks,
                        .atmos = local_atmos,
                        .active_set = slice_active_set(adata, la),
                        .active_set_cont = slice_active_cont_set(adata, la),
                        .update_n_star = false,
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
            state.println(
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
            iv = std::min(std::max(iv, 0), INTERPOLATE_DIRECTIONAL_BINS-1);
            ivp = iv;
            tv = FP(1.0);
            tvp = FP(0.0);
        } else {
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

    void compute_mip_n(const State& state, const MipmapComputeState& mm_state, i32 level) const {
    }
    void compute_subset_mip_n(
        const State& state,
        const MipmapSubsetState& mm_state,
        const CascadeCalcSubset& subset,
        i32 level
    ) const;
};

#else
#endif