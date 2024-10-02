#if !defined(DEXRT_MIPMAPS_HPP)
#define DEXRT_MIPMAPS_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "BlockMap.hpp"
#include "MiscSparseStorage.hpp"
#include "DirectionalEmisOpacInterp.hpp"

struct SparseMip {
    i32 vox_size;
    Fp2d emis; // [ks, wave_batch]
    Fp2d opac; // [ks, wave_batch]
    DirectionalEmisOpacInterp dir_data;
    Fp1d vx; // [ks]
    Fp1d vy; // [ks]
    Fp1d vz; // [ks]
    Fp3d aniso_emis; // [ks, dir, wave_batch]
    Fp3d aniso_opac; // [ks, dir, wave_batch]
};

struct SparseMips {
    std::vector<SparseMip> mips;
};

inline SparseMips compute_mips(
    const State& state,
    const CascadeState& casc_state,
    const SparseStores& full_res,
    const DirectionalEmisOpacInterp& dir_interp
) {
    assert(USE_MIPMAPS);
    JasUnpack(state, block_map);

    i32 max_mip_factor = 0;
    for (int i = 0; i < MAX_CASCADE + 1; ++i) {
        max_mip_factor += MIPMAP_FACTORS[i];
    }

    SparseMips result;
    result.mips.resize(max_mip_factor+1);
    auto& mip0 = result.mips[0];
    mip0.vox_size = 1;
    mip0.emis = full_res.emis;
    mip0.opac = full_res.opac;
    mip0.dir_data = dir_interp;
    mip0.vx = full_res.vx;
    mip0.vy = full_res.vy;
    mip0.vz = full_res.vz;

    if (max_mip_factor == 0) {
        return result;
    }

    for (int level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
        auto& prev_mip = result.mips[level_m_1];
        auto& mip = result.mips[level_m_1+1];
        i32 vox_size = (1 << (level_m_1 + 1));
        mip.vox_size = vox_size;
        i64 flat_size = mip0.emis.extent(0) / square(vox_size);
        mip.emis = Fp2d("emis mip", flat_size, state.c0_size.wave_batch);
        mip.opac = Fp2d("opac mip", flat_size, state.c0_size.wave_batch);
        mip.dir_data = DirectionalEmisOpacInterp_new(flat_size, state.c0_size.wave_batch);
        mip.vx = Fp1d("vx mip", flat_size);
        mip.vy = Fp1d("vy mip", flat_size);
        mip.vz = Fp1d("vz mip", flat_size);

        auto bound = block_map.loop_bounds(vox_size);
        parallel_for(
            "Compute mip (wave batch)",
            SimpleBounds<3>(
                bound.dim(0),
                bound.dim(1),
                state.c0_size.wave_batch
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
                IndexGen<BLOCK_SIZE> idx_gen(block_map, vox_size);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                const i32 upper_vox_size = vox_size / 2;
                IndexGen<BLOCK_SIZE> idx_gen_upper(block_map, upper_vox_size);
                fp_t emis = FP(0.0);
                fp_t opac = FP(0.0);
                i64 idx = idx_gen_upper.idx(coord.x, coord.z);
                emis += prev_mip.emis(idx, wave);
                opac += prev_mip.opac(idx, wave);

                idx = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z);
                emis += prev_mip.emis(idx, wave);
                opac += prev_mip.opac(idx, wave);

                idx = idx_gen_upper.idx(coord.x, coord.z+upper_vox_size);
                emis += prev_mip.emis(idx, wave);
                opac += prev_mip.opac(idx, wave);

                idx = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z+upper_vox_size);
                emis += prev_mip.emis(idx, wave);
                opac += prev_mip.opac(idx, wave);

                emis *= FP(0.25);
                opac *= FP(0.25);

                mip.emis(ks, wave) = emis;
                mip.opac(ks, wave) = opac;
            }
        );

        parallel_for(
            "Compute mip (no wave batch)",
            bound,
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                IndexGen<BLOCK_SIZE> idx_gen(block_map, vox_size);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                const i32 upper_vox_size = vox_size / 2;
                IndexGen<BLOCK_SIZE> idx_gen_upper(block_map, upper_vox_size);
                fp_t vx = FP(0.0);
                fp_t vy = FP(0.0);
                fp_t vz = FP(0.0);
                i64 idx = idx_gen_upper.idx(coord.x, coord.z);
                vx += prev_mip.vx(idx);
                vy += prev_mip.vy(idx);
                vz += prev_mip.vz(idx);

                idx = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z);
                vx += prev_mip.vx(idx);
                vy += prev_mip.vy(idx);
                vz += prev_mip.vz(idx);

                idx = idx_gen_upper.idx(coord.x, coord.z+upper_vox_size);
                vx += prev_mip.vx(idx);
                vy += prev_mip.vy(idx);
                vz += prev_mip.vz(idx);

                idx = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z+upper_vox_size);
                vx += prev_mip.vx(idx);
                vy += prev_mip.vy(idx);
                vz += prev_mip.vz(idx);

                vx *= FP(0.25);
                vy *= FP(0.25);
                vz *= FP(0.25);

                mip.vx(ks) = vx;
                mip.vy(ks) = vy;
                mip.vz(ks) = vz;
            }
        );

        // TODO(cmo): This isn't done properly. The min and max v should be
        // generated from the velocity mips, as in the setup for
        // DirectionalEmisOpacInterp. If _this_ is outside the range, we should
        // compute them again from scratch, through the full N levels of mips
        // (as we need the original atmospheric params)
        parallel_for(
            "Compute mip (dir interp)",
            SimpleBounds<4>(
                bound.dim(0),
                bound.dim(1),
                INTERPOLATE_DIRECTIONAL_BINS,
                state.c0_size.wave_batch
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 vel_idx, i32 wave) {
                IndexGen<BLOCK_SIZE> idx_gen(block_map, vox_size);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                const i32 upper_vox_size = vox_size / 2;
                IndexGen<BLOCK_SIZE> idx_gen_upper(block_map, upper_vox_size);
                fp_t min_vs[4] = {};
                fp_t max_vs[4] = {};

                auto compute_vels = [&] (int idx, int corner) {
                    const fp_t sample_min = prev_mip.dir_data.vel_start(idx);
                    const fp_t sample_max = sample_min + INTERPOLATE_DIRECTIONAL_BINS * prev_mip.dir_data.vel_step(idx);
                    min_vs[corner] = sample_min;
                    max_vs[corner] = sample_max;
                };

                i64 idx0 = idx_gen_upper.idx(coord.x, coord.z);
                compute_vels(idx0, 0);

                i64 idx1 = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z);
                compute_vels(idx1, 1);

                i64 idx2 = idx_gen_upper.idx(coord.x, coord.z+upper_vox_size);
                compute_vels(idx2, 2);

                i64 idx3 = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z+upper_vox_size);
                compute_vels(idx3, 3);

                const fp_t min_v = std::min(min_vs[0], std::min(min_vs[1], std::min(min_vs[2], min_vs[3])));
                const fp_t max_v = std::max(max_vs[0], std::max(max_vs[1], std::max(max_vs[2], max_vs[3])));

                fp_t vel_start = min_v;
                fp_t vel_step = (max_v - min_v) / fp_t(INTERPOLATE_DIRECTIONAL_BINS - 1);
                if (vel_idx == 0 && wave == 0) {
                    mip.dir_data.vel_start(ks) = vel_start;
                    mip.dir_data.vel_step(ks) = vel_step;
                }

                // NOTE(cmo): Need to take 4 samples of the upper level for each bin/wave
                const fp_t vel_sample = vel_start + vel_idx * vel_step;
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

                fp_t emis = FP(0.0);
                fp_t opac = FP(0.0);
                const fp_t vel0 = clamp_vel(0);
                auto sample = prev_mip.dir_data.sample(idx0, wave, vel0);
                emis += sample.eta;
                opac += sample.chi;

                const fp_t vel1 = clamp_vel(1);
                sample = prev_mip.dir_data.sample(idx1, wave, vel1);
                emis += sample.eta;
                opac += sample.chi;

                const fp_t vel2 = clamp_vel(2);
                sample = prev_mip.dir_data.sample(idx2, wave, vel2);
                emis += sample.eta;
                opac += sample.chi;

                const fp_t vel3 = clamp_vel(3);
                sample = prev_mip.dir_data.sample(idx3, wave, vel3);
                emis += sample.eta;
                opac += sample.chi;

                emis *= FP(0.25);
                opac *= FP(0.25);

                mip.dir_data.emis_opac_vel(ks, vel_idx, 0, wave) = emis;
                mip.dir_data.emis_opac_vel(ks, vel_idx, 1, wave) = opac;
            }
        );
        yakl::fence();
    }

    return result;
}


/// Used for static given fs case
inline SparseMips compute_mips(
    const State& state,
    const CascadeState& casc_state,
    const SparseStores& full_res
) {
    assert(USE_MIPMAPS);
    JasUnpack(state, block_map);

    i32 max_mip_factor = 0;
    for (int i = 0; i < MAX_CASCADE + 1; ++i) {
        max_mip_factor += MIPMAP_FACTORS[i];
    }

    SparseMips result;
    result.mips.resize(max_mip_factor+1);
    auto& mip0 = result.mips[0];
    mip0.vox_size = 1;
    mip0.emis = full_res.emis;
    mip0.opac = full_res.opac;
    mip0.vx = full_res.vx;
    mip0.vy = full_res.vy;
    mip0.vz = full_res.vz;

    if constexpr (MIP_MODE == MipMode::Anisotropic) {
        mip0.aniso_emis = Fp3d("aniso emis", mip0.emis.extent(0), 8, mip0.emis.extent(1));
        mip0.aniso_opac = Fp3d("aniso opac", mip0.emis.extent(0), 8, mip0.emis.extent(1));
        parallel_for(
            "Fill mip 0 aniso",
            SimpleBounds<3>(
                mip0.emis.extent(0),
                8,
                mip0.emis.extent(1)
            ),
            YAKL_LAMBDA (i64 ks, int dir_idx, int wave) {
                mip0.aniso_emis(ks, dir_idx, wave) = mip0.emis(ks, wave);
                mip0.aniso_opac(ks, dir_idx, wave) = mip0.opac(ks, wave);
            }
        );
        yakl::fence();
    }

    if (max_mip_factor == 0) {
        return result;
    }

    for (int level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
        auto& prev_mip = result.mips[level_m_1];
        auto& mip = result.mips[level_m_1+1];
        i32 vox_size = (1 << (level_m_1 + 1));
        mip.vox_size = vox_size;
        i64 flat_size = mip0.emis.extent(0) / square(vox_size);
        mip.emis = Fp2d("emis mip", flat_size, mip0.emis.extent(1));
        mip.opac = Fp2d("opac mip", flat_size, mip0.emis.extent(1));

        if constexpr (MIP_MODE == MipMode::Anisotropic) {
            mip.aniso_emis = Fp3d("aniso emis", flat_size, 8, mip0.emis.extent(1));
            mip.aniso_opac = Fp3d("aniso opac", flat_size, 8, mip0.emis.extent(1));
        }

        auto bound = block_map.loop_bounds(vox_size);
        parallel_for(
            "Compute mip (wave batch)",
            SimpleBounds<3>(
                bound.dim(0),
                bound.dim(1),
                mip0.emis.extent(1)
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
                IndexGen<BLOCK_SIZE> idx_gen(block_map, vox_size);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                const SparseMip& pm = prev_mip;

                const i32 upper_vox_size = vox_size / 2;
                IndexGen<BLOCK_SIZE> idx_gen_upper(block_map, upper_vox_size);
                fp_t emis = FP(0.0);
                fp_t opac = FP(0.0);
                i64 idx0 = idx_gen_upper.idx(coord.x, coord.z);
                i64 idx1 = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z);
                i64 idx2 = idx_gen_upper.idx(coord.x, coord.z+upper_vox_size);
                i64 idx3 = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z+upper_vox_size);
                if constexpr (MIP_MODE == MipMode::Classic) {
                    emis += pm.emis(idx0, wave);
                    opac += pm.opac(idx0, wave);

                    emis += pm.emis(idx1, wave);
                    opac += pm.opac(idx1, wave);

                    emis += pm.emis(idx2, wave);
                    opac += pm.opac(idx2, wave);

                    emis += pm.emis(idx3, wave);
                    opac += pm.opac(idx3, wave);

                    emis *= FP(0.25);
                    opac *= FP(0.25);
                } else if constexpr (MIP_MODE == MipMode::Perceptual) {
                    const fp_t ds = FP(0.7);
                    // TODO(cmo): get this value properly
                    const fp_t vox_scale = FP(1.0);
                    const fp_t opac0 = pm.opac(idx0, wave);
                    const fp_t opac1 = pm.opac(idx1, wave);
                    const fp_t opac2 = pm.opac(idx2, wave);
                    const fp_t opac3 = pm.opac(idx3, wave);
                    const fp_t emis0 = pm.emis(idx0, wave);
                    const fp_t emis1 = pm.emis(idx1, wave);
                    const fp_t emis2 = pm.emis(idx2, wave);
                    const fp_t emis3 = pm.emis(idx3, wave);
                    const fp_t trans0 = std::exp(-opac0 * ds * vox_scale);
                    const fp_t trans1 = std::exp(-opac1 * ds * vox_scale);
                    const fp_t trans2 = std::exp(-opac2 * ds * vox_scale);
                    const fp_t trans3 = std::exp(-opac3 * ds * vox_scale);

                    fp_t trans_pairs = FP(0.0);
                    trans_pairs += trans0 * trans1;
                    trans_pairs += trans0 * trans2;
                    trans_pairs += trans0 * trans3;
                    trans_pairs += trans1 * trans2;
                    trans_pairs += trans1 * trans3;
                    trans_pairs += trans2 * trans3;
                    trans_pairs /= FP(6.0);
                    fp_t effective_chi = -std::log(trans_pairs) / (FP(2.0) * ds * vox_scale);
                    if (std::isinf(effective_chi)) {
                        effective_chi = FP(0.25) * (opac0 + opac1 + opac2 + opac3);
                    }
                    // const fp_t mean_source_fn = FP(0.25) * (
                    //     emis0 / (opac0 + FP(1e-15)) +
                    //     emis1 / (opac1 + FP(1e-15)) +
                    //     emis2 / (opac2 + FP(1e-15)) +
                    //     emis3 / (opac3 + FP(1e-15))
                    // );
                    // const fp_t effective_eta = mean_source_fn * effective_chi;
                    fp_t trans_weighted_source = (
                        (FP(1.0) - trans0) * (emis0 / (opac0 + FP(1e-15))) +
                        (FP(1.0) - trans1) * (emis1 / (opac1 + FP(1e-15))) +
                        (FP(1.0) - trans2) * (emis2 / (opac2 + FP(1e-15))) +
                        (FP(1.0) - trans3) * (emis3 / (opac3 + FP(1e-15)))
                    ) / (
                        (FP(1.0) - trans0) +
                        (FP(1.0) - trans1) +
                        (FP(1.0) - trans2) +
                        (FP(1.0) - trans3) +
                        FP(1e-15)
                    );
                    const fp_t effective_eta = trans_weighted_source * effective_chi;

                    emis = effective_eta;
                    opac = effective_chi;
                }

                mip.emis(ks, wave) = emis;
                mip.opac(ks, wave) = opac;
            }
        );

        if constexpr (MIP_MODE == MipMode::Anisotropic) {
            // NOTE(cmo): Done in ray direction, not interval direction
            parallel_for(
                "Compute mip (aniso wave batch)",
                SimpleBounds<4>(
                    bound.dim(0),
                    bound.dim(1),
                    mip.aniso_emis.extent(1),
                    mip0.emis.extent(1)
                ),
                YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 dir_idx, i32 wave) {
                    namespace C = ConstantsFP;
                    IndexGen<BLOCK_SIZE> idx_gen(block_map, vox_size);
                    i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                    Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                    const SparseMip& pm = prev_mip;

                    const i32 upper_vox_size = vox_size / 2;
                    IndexGen<BLOCK_SIZE> idx_gen_upper(block_map, upper_vox_size);
                    i64 idx0 = idx_gen_upper.idx(coord.x, coord.z);
                    i64 idx1 = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z);
                    i64 idx2 = idx_gen_upper.idx(coord.x, coord.z+upper_vox_size);
                    i64 idx3 = idx_gen_upper.idx(coord.x+upper_vox_size, coord.z+upper_vox_size);
                    const fp_t emis0 = pm.aniso_emis(idx0, dir_idx, wave);
                    const fp_t emis1 = pm.aniso_emis(idx1, dir_idx, wave);
                    const fp_t emis2 = pm.aniso_emis(idx2, dir_idx, wave);
                    const fp_t emis3 = pm.aniso_emis(idx3, dir_idx, wave);
                    const fp_t opac0 = pm.aniso_opac(idx0, dir_idx, wave);
                    const fp_t opac1 = pm.aniso_opac(idx1, dir_idx, wave);
                    const fp_t opac2 = pm.aniso_opac(idx2, dir_idx, wave);
                    const fp_t opac3 = pm.aniso_opac(idx3, dir_idx, wave);
                    const fp_t source0 = emis0 / (opac0 + FP(1e-15));
                    const fp_t source1 = emis1 / (opac1 + FP(1e-15));
                    const fp_t source2 = emis2 / (opac2 + FP(1e-15));
                    const fp_t source3 = emis3 / (opac3 + FP(1e-15));


                    const fp_t vox_scale = FP(1.0);

                    const fp_t theta = fp_t(dir_idx) * C::pi / FP(4.0);
                    // Layout
                    // ---------
                    // | 0 | 1 |
                    // | 2 | 3 |
                    // ---------
                    if (dir_idx % 2 == 0) {
                        const fp_t trans0 = std::exp(-opac0 * vox_scale);
                        const fp_t trans1 = std::exp(-opac1 * vox_scale);
                        const fp_t trans2 = std::exp(-opac2 * vox_scale);
                        const fp_t trans3 = std::exp(-opac3 * vox_scale);

                        fp_t avg_trans = FP(0.0);
                        if (dir_idx % 4 == 0) {
                            // horizontal
                            avg_trans = FP(0.5) * (trans0 * trans1 + trans2 * trans3);
                        } else {
                            // vertical
                            avg_trans = FP(0.5) * (trans0 * trans2 + trans1 * trans3);
                        }
                        fp_t effective_chi = -std::log(avg_trans) / (FP(2.0) * vox_scale);
                        if (std::isinf(effective_chi)) {
                            effective_chi = FP(0.25) * (opac0 + opac1 + opac2 + opac3);
                        }

                        fp_t avg_I_gain = FP(0.0);

                        if (dir_idx == 0) {
                            // 0 -> 1, 2 -> 3
                            avg_I_gain += source0 * (FP(1.0) - trans0) * trans1 + (FP(1.0) - trans1) * source1;
                            avg_I_gain += source2 * (FP(1.0) - trans2) * trans3 + (FP(1.0) - trans3) * source3;
                        } else if (dir_idx == 2) {
                            // 2 -> 0, 3 -> 1
                            avg_I_gain += source2 * (FP(1.0) - trans2) * trans0 + (FP(1.0) - trans0) * source0;
                            avg_I_gain += source3 * (FP(1.0) - trans3) * trans1 + (FP(1.0) - trans1) * source1;
                        } else if (dir_idx == 4) {
                            // 1 -> 0, 3 -> 2
                            avg_I_gain += source1 * (FP(1.0) - trans1) * trans0 + (FP(1.0) - trans0) * source0;
                            avg_I_gain += source3 * (FP(1.0) - trans3) * trans2 + (FP(1.0) - trans2) * source2;
                        } else if (dir_idx == 6) {
                            // 0 -> 2, 1 -> 3
                            avg_I_gain += source0 * (FP(1.0) - trans0) * trans2 + (FP(1.0) - trans2) * source2;
                            avg_I_gain += source1 * (FP(1.0) - trans1) * trans3 + (FP(1.0) - trans3) * source3;
                        }
                        avg_I_gain *= FP(0.5);

                        fp_t effective_eta = avg_I_gain / (FP(1.0) - avg_trans) * effective_chi;
                        if (std::isnan(effective_eta)) {
                            effective_eta = FP(0.25) * (emis0 + emis1 + emis2 + emis3);
                        }
                        mip.aniso_emis(ks, dir_idx, wave) = effective_eta;
                        mip.aniso_opac(ks, dir_idx, wave) = effective_chi;
                    } else {
                        constexpr fp_t ds = FP(0.7071067);
                        const fp_t trans0 = std::exp(-opac0 * ds * vox_scale);
                        const fp_t trans1 = std::exp(-opac1 * ds * vox_scale);
                        const fp_t trans2 = std::exp(-opac2 * ds * vox_scale);
                        const fp_t trans3 = std::exp(-opac3 * ds * vox_scale);

                        fp_t avg_trans_short = FP(0.0);
                        fp_t avg_trans_long = FP(0.0);
                        if (dir_idx % 4 == 1) {
                            // Direction as / ...
                            avg_trans_short = FP(0.5) * (trans0 + trans3);
                            avg_trans_long =  trans1 * trans2;
                        } else {
                            // % 4 == 3
                            // Direction as \ ...
                            avg_trans_short = FP(0.5) * (trans1 + trans2);
                            avg_trans_long = trans0 * trans3;
                        }
                        // NOTE(cmo): Chi computed like this is unexpectedly low in uniform regions.
                        // Do the single and double separately?

                        // 2 paths with length sqrt(2), 1 path with length 2sqrt(2), but also double weighted. Gives average path length of 3/2 sqrt2
                        fp_t effective_chi_short = -std::log(avg_trans_short) / (ds * vox_scale);
                        fp_t effective_chi_long = -std::log(avg_trans_long) / (FP(2.0) * ds * vox_scale);
                        fp_t effective_chi = FP(0.5) * (effective_chi_long + effective_chi_short);
                        if (std::isinf(effective_chi)) {
                            effective_chi = FP(0.25) * (opac0 + opac1 + opac2 + opac3);
                        }

                        fp_t avg_I_gain_short = FP(0.0);
                        fp_t avg_I_gain_long = FP(0.0);
                        if (dir_idx == 1) {
                            // 0, 2 -> 1, 3
                            avg_I_gain_short += source0 * (FP(1.0) - trans0);
                            avg_I_gain_short += source3 * (FP(1.0) - trans3);
                            avg_I_gain_long+= source2 * (FP(1.0) - trans2) * trans1 + (FP(1.0) - trans1) * source1;
                        } else if (dir_idx == 3) {
                            // 1, 3 -> 0, 2
                            avg_I_gain_short += source1 * (FP(1.0) - trans1);
                            avg_I_gain_short += source2 * (FP(1.0) - trans2);
                            avg_I_gain_long += source3 * (FP(1.0) - trans3) * trans0 + (FP(1.0) - trans0) * source0;
                        } else if (dir_idx == 5) {
                            // 0, 1 -> 2, 3
                            avg_I_gain_short += source0 * (FP(1.0) - trans0);
                            avg_I_gain_short += source3 * (FP(1.0) - trans3);
                            avg_I_gain_long += source1 * (FP(1.0) - trans1) * trans2 + (FP(1.0) - trans2) * source2;
                        } else if (dir_idx == 7) {
                            // 1, 0 -> 3, 2
                            avg_I_gain_short += source1 * (FP(1.0) - trans1);
                            avg_I_gain_short += source2 * (FP(1.0) - trans2);
                            avg_I_gain_long += source0 * (FP(1.0) - trans0) * trans3 + (FP(1.0) - trans3) * source3;
                        }
                        avg_I_gain_short *= FP(0.5);

                        fp_t effective_eta_short = avg_I_gain_short / (FP(1.0) - avg_trans_short) * effective_chi;
                        fp_t effective_eta_long = avg_I_gain_long / (FP(1.0) - avg_trans_long) * effective_chi;
                        fp_t effective_eta = FP(0.5) * (effective_eta_short + effective_eta_long);
                        if (std::isnan(effective_eta)) {
                            effective_eta = FP(0.25) * (emis0 + emis1 + emis2 + emis3);
                        }
                        mip.aniso_emis(ks, dir_idx, wave) = effective_eta;
                        mip.aniso_opac(ks, dir_idx, wave) = effective_chi;
                    }
                }
            );
        }

        yakl::fence();
    }

    return result;
}

/// Needed for the MultiResBlockMap where all mips are stored in one flat array. This will also modify mr_block_map
inline SparseMip compute_flat_mips(
    const State& state,
    const CascadeState& casc_state,
    const SparseStores& full_res
) {
    assert(USE_MIPMAPS);
    JasUnpack(state, block_map, mr_block_map);

    i32 max_mip_factor = 0;
    for (int i = 0; i < MAX_CASCADE + 1; ++i) {
        max_mip_factor += MIPMAP_FACTORS[i];
    }


    // NOTE(cmo): mippable entries is used as an accumulator during each round
    // of mipping, and then divided by the number of sub blocks and placed into
    // max mip_level. max_mip_level then holds the max_mip_level + 1 s.t. 0 represents empty.
    yakl::Array<i32, 1, yakl::memDevice> mippable_entries("mippable entries", block_map.num_active_tiles);
    yakl::Array<i32, 1, yakl::memDevice> max_mip_level("max mip entries", block_map.num_z_tiles * block_map.num_x_tiles);

    max_mip_level = 0;
    mippable_entries = 0;
    yakl::fence();

    SparseMip result;
    i64 flat_size = mr_block_map.buffer_len();
    const i32 wave_batch = full_res.emis.extent(1);
    const i32 vox_scale = state.atmos.voxel_scale;
    result.emis = Fp2d("emis flat mip", flat_size, wave_batch);
    result.opac = Fp2d("opad flat mip", flat_size, wave_batch);

    auto bounds = block_map.loop_bounds();
    parallel_for(
        "Copy mip 0",
        SimpleBounds<3>(
            bounds.dim(0),
            bounds.dim(1),
            wave_batch
        ),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
            IndexGen<BLOCK_SIZE> idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

            if (block_idx == 0 && wave == 0) {
                Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);
                // NOTE(cmo): This is the first entry in an active tile, so
                // increment. No need for atomic here.
                max_mip_level(tile_coord.z * block_map.num_x_tiles + tile_coord.x) += 1;
            }

            result.emis(ks, wave) = full_res.emis(ks, wave);
            result.opac(ks, wave) = full_res.opac(ks, wave);
        }
    );
    yakl::fence();

    for (int level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
        i32 vox_size = (1 << (level_m_1 + 1));
        auto bounds = block_map.loop_bounds(vox_size);

        parallel_for(
            "Compute mip n (wave batch)",
            SimpleBounds<3>(
                bounds.dim(0),
                bounds.dim(1),
                wave_batch
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
                const i32 level = level_m_1 + 1;
                const fp_t ds = vox_scale;
                MultiLevelIndexGen<BLOCK_SIZE, ENTRY_SIZE> idx_gen(block_map, mr_block_map);
                const i64 ks = idx_gen.loop_idx(level, tile_idx, block_idx);
                const Coord2 coord = idx_gen.loop_coord(level, tile_idx, block_idx);
                const Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);

                const i32 upper_vox_size = vox_size / 2;
                fp_t emis = FP(0.0);
                fp_t opac = FP(0.0);
                i64 idx0 = idx_gen.idx(level_m_1, coord.x, coord.z);
                i64 idx1 = idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z);
                i64 idx2 = idx_gen.idx(level_m_1, coord.x, coord.z+upper_vox_size);
                i64 idx3 = idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z+upper_vox_size);

                // E[log(chi ds)]
                fp_t m1_chi = FP(0.0);
                // E[log(chi ds)^2]
                fp_t m2_chi = FP(0.0);
                // E[log(eta ds)]
                fp_t m1_eta = FP(0.0);
                // E[log(eta ds)^2]
                fp_t m2_eta = FP(0.0);

                bool consider_variance = false;
                constexpr fp_t opacity_threshold = FP(0.25);
                constexpr fp_t variance_threshold = FP(1.0);

                fp_t eta_s = result.emis(idx0, wave);
                fp_t chi_s = result.opac(idx0, wave);
                emis += eta_s;
                opac += chi_s;
                consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                chi_s += FP(1e-15);
                eta_s += FP(1e-15);
                m1_chi += std::log(chi_s * ds);
                m2_chi += square(std::log(chi_s * ds));
                m1_eta += std::log(eta_s * ds);
                m2_eta += square(std::log(eta_s * ds));

                eta_s = result.emis(idx1, wave);
                chi_s = result.opac(idx1, wave);
                emis += eta_s;
                opac += chi_s;
                consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                chi_s += FP(1e-15);
                eta_s += FP(1e-15);
                m1_chi += std::log(chi_s * ds);
                m2_chi += square(std::log(chi_s * ds));
                m1_eta += std::log(eta_s * ds);
                m2_eta += square(std::log(eta_s * ds));

                eta_s = result.emis(idx2, wave);
                chi_s = result.opac(idx2, wave);
                emis += eta_s;
                opac += chi_s;
                consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                chi_s += FP(1e-15);
                eta_s += FP(1e-15);
                m1_chi += std::log(chi_s * ds);
                m2_chi += square(std::log(chi_s * ds));
                m1_eta += std::log(eta_s * ds);
                m2_eta += square(std::log(eta_s * ds));

                eta_s = result.emis(idx3, wave);
                chi_s = result.opac(idx3, wave);
                emis += eta_s;
                opac += chi_s;
                consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                chi_s += FP(1e-15);
                eta_s += FP(1e-15);
                m1_chi += std::log(chi_s * ds);
                m2_chi += square(std::log(chi_s * ds));
                m1_eta += std::log(eta_s * ds);
                m2_eta += square(std::log(eta_s * ds));

                emis *= FP(0.25);
                opac *= FP(0.25);
                m1_chi *= FP(0.25);
                m2_chi *= FP(0.25);
                m1_eta *= FP(0.25);
                m2_eta *= FP(0.25);

                bool do_increment = true;
                if (consider_variance) {
                    // index of dispersion D[x] = Var[x] / Mean[x] = (M_2[x] - M_1[x]^2) / M_1[x] = M_2[x] / M_1[x] - M_1[x]
                    fp_t D_chi = std::abs(m2_chi / m1_chi - m1_chi); // due to the log, this often negative.
                    if (m2_chi == FP(0.0)) {
                        D_chi = FP(0.0);
                    }
                    fp_t D_eta = std::abs(m2_eta / m1_eta - m1_eta);
                    if (m2_eta == FP(0.0)) {
                        D_eta = FP(0.0);
                    }
                    // Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);
                    // printf("tile: %d, %d, vars: %e, %e\n %e, %e | %e, %e\n-------\n", tile_coord.x, tile_coord.z, D_chi, D_eta, m1_chi, m2_chi, m1_eta, m2_eta);
                    if (D_chi > variance_threshold || D_eta > variance_threshold) {
                        // printf("tile: %d, %d: false\n", tile_coord.x, tile_coord.z);
                        do_increment = false;
                    }
                }

                // NOTE(cmo): This is coming from many threads of a warp
                // simultaneously, which isn't great. If it's a bottleneck,
                // ballot across threads, do a popcount, and increment from one
                // thread.
                if (do_increment) {
                    yakl::atomicAdd(mippable_entries(tile_idx), 1);
                }

                result.emis(ks, wave) = emis;
                result.opac(ks, wave) = opac;
            }
        );
        yakl::fence();

        parallel_for(
            "Update mippable array",
            block_map.loop_bounds().dim(0),
            YAKL_LAMBDA (i64 tile_idx) {
                MultiLevelIndexGen<BLOCK_SIZE, ENTRY_SIZE> idx_gen(block_map, mr_block_map);
                Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);
                const int level = level_m_1 + 1;
                const i32 expected_entries = square(BLOCK_SIZE >> level) * wave_batch;
                i64 flat_entry = mr_block_map.lookup.flat_tile_index(tile_coord.x, tile_coord.z);
                i32 before = max_mip_level(flat_entry);
                if (max_mip_level(flat_entry) == level) {
                    max_mip_level(flat_entry) += mippable_entries(tile_idx) / expected_entries;
                }
                mippable_entries(tile_idx) = 0;
            }
        );
        yakl::fence();
    }

    mr_block_map.lookup.pack_entries(max_mip_level);
    return result;
}

#else
#endif