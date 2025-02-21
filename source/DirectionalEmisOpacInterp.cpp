#include "DirectionalEmisOpacInterp.hpp"
#include "RcUtilsModes.hpp"

void compute_min_max_vel(
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

    constexpr i32 RcMode = RC_flags_storage();
    CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset.subset_idx);

    JasUnpack(state, incl_quad, mr_block_map);
    const auto& block_map = mr_block_map.block_map;

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

    dex_parallel_for(
        "Min/Max Vel",
        block_map.loop_bounds(1 << mip_level),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            MRIdxGen idx_gen(mr_block_map);
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

void DirectionalEmisOpacInterp::init(i64 num_active_zones, i32 wave_batch) {
    emis_opac_vel = Fp4d("emis_opac_vel", num_active_zones, INTERPOLATE_DIRECTIONAL_BINS, 2, wave_batch);
    vel_start = Fp1d("vel_start", num_active_zones);
    vel_step = Fp1d("vel_step", num_active_zones);
    zero();
}

void DirectionalEmisOpacInterp::zero() const {
    emis_opac_vel = FP(0.0);
    vel_start = FP(0.0);
    vel_step = FP(0.0);
    yakl::fence();
}

void DirectionalEmisOpacInterp::compute_subset_mip_n(
    const State& state,
    const MipmapSubsetState& mm_state,
    const CascadeCalcSubset& subset,
    i32 level
) const {
    JasUnpack(state, mr_block_map);
    const auto& block_map = mr_block_map.block_map;
    JasUnpack(mm_state, vx, vy, vz);
    const i32 wave_batch = emis_opac_vel.extent(3);
    constexpr i32 mip_block = 4;
    const i32 level_m_1 = level - 1;

    Fp1d min_vel("min_vel_mips", mr_block_map.buffer_len());
    Fp1d max_vel("max_vel_mips", mr_block_map.buffer_len());
    FlatVelocity vels{
        .vx = vx,
        .vy = vy,
        .vz = vz
    };
    compute_min_max_vel(
        state,
        subset,
        level_m_1 + 1,
        vels,
        min_vel,
        max_vel
    );
    yakl::fence();

    // TODO(cmo): This isn't done properly. The min and max v should be
    // generated from the velocity mips, as in the setup for
    // DirectionalEmisOpacInterp. If _this_ is outside the range, we should
    // compute them again from scratch, through the full N levels of mips
    // (as we need the original atmospheric params)
    const int vox_size = 1 << (level_m_1 + 1);
    auto bounds = block_map.loop_bounds(vox_size);
    dex_parallel_for(
        "Compute mip (dir interp)",
        FlatLoop<4>(
            bounds.dim(0),
            bounds.dim(1),
            INTERPOLATE_DIRECTIONAL_BINS,
            wave_batch
        ),
        YAKL_CLASS_LAMBDA (i64 tile_idx, i32 block_idx, i32 vel_idx, i32 wave) {
            MRIdxGen idx_gen(mr_block_map);
            i64 ks = idx_gen.loop_idx(level, tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(level, tile_idx, block_idx);

            const i32 upper_vox_size = vox_size / 2;
            i64 idxs[mip_block] = {
                idx_gen.idx(level_m_1, coord),
                idx_gen.idx(level_m_1, Coord2{.x = coord.x+upper_vox_size, .z = coord.z}),
                idx_gen.idx(level_m_1, Coord2{.x = coord.x, .z = coord.z+upper_vox_size}),
                idx_gen.idx(level_m_1, Coord2{.x = coord.x+upper_vox_size, .z = coord.z+upper_vox_size})
            };

            fp_t min_vs[4];
            fp_t max_vs[4];
            auto compute_vels = [&] (int i) {
                i64 idx = idxs[i];
                const fp_t sample_min = vel_start(idx);
                const fp_t sample_max = sample_min + INTERPOLATE_DIRECTIONAL_BINS * vel_step(idx);
                min_vs[i] = sample_min;
                max_vs[i] = sample_max;
            };
            for (int i = 0; i < mip_block; ++i) {
                compute_vels(i);
            }

            fp_t min_v = min_vel(ks);
            fp_t max_v = max_vel(ks);
            fp_t v_start = min_v;
            fp_t v_step = (max_v - min_v) / fp_t(INTERPOLATE_DIRECTIONAL_BINS - 1);
            if (vel_idx == 0 && wave == 0) {
                vel_start(ks) = v_start;
                vel_step(ks) = v_step;
            }

            // NOTE(cmo): Need to take 4 samples of the upper level for each bin/wave
            const fp_t vel_sample = v_start + vel_idx * v_step;
            // NOTE(cmo): Clamp to the vel range of each pixel. Need to check how important this is.
            auto clamp_vel = [&] (int corner) {
                fp_t vel = vel_sample;
                if (vel < min_vs[corner]) {
                    vel = min_vs[corner];
                } else if (vel > max_vs[corner]) {
                    vel = max_vs[corner];
                }
                return vel;
            };

            fp_t emis_vel = FP(0.0);
            fp_t opac_vel = FP(0.0);
            for (int i = 0; i < mip_block; ++i) {
                i64 idx = idxs[i];
                const fp_t vel_i = clamp_vel(i);
                auto curr_sample = sample(idx, wave, vel_i);
                emis_vel += curr_sample.eta;
                opac_vel += curr_sample.chi;
            }
            emis_vel *= FP(0.25);
            opac_vel *= FP(0.25);

            emis_opac_vel(ks, vel_idx, 0, wave) = emis_vel;
            emis_opac_vel(ks, vel_idx, 1, wave) = opac_vel;
        }
    );
    yakl::fence();
}