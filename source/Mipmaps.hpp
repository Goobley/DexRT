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

                IndexGen<BLOCK_SIZE> idx_gen_upper(block_map, vox_size / 2);
                fp_t emis = FP(0.0);
                fp_t opac = FP(0.0);
                i64 idx = idx_gen_upper.idx(coord.x, coord.z);
                emis += prev_mip.emis(idx, wave);
                opac += prev_mip.opac(idx, wave);

                idx = idx_gen_upper.idx(coord.x+1, coord.z);
                emis += prev_mip.emis(idx, wave);
                opac += prev_mip.opac(idx, wave);

                idx = idx_gen_upper.idx(coord.x, coord.z+1);
                emis += prev_mip.emis(idx, wave);
                opac += prev_mip.opac(idx, wave);

                idx = idx_gen_upper.idx(coord.x+1, coord.z+1);
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

                IndexGen<BLOCK_SIZE> idx_gen_upper(block_map, vox_size / 2);
                fp_t vx = FP(0.0);
                fp_t vy = FP(0.0);
                fp_t vz = FP(0.0);
                i64 idx = idx_gen_upper.idx(coord.x, coord.z);
                vx += prev_mip.vx(idx);
                vy += prev_mip.vy(idx);
                vz += prev_mip.vz(idx);

                idx = idx_gen_upper.idx(coord.x+1, coord.z);
                vx += prev_mip.vx(idx);
                vy += prev_mip.vy(idx);
                vz += prev_mip.vz(idx);

                idx = idx_gen_upper.idx(coord.x, coord.z+1);
                vx += prev_mip.vx(idx);
                vy += prev_mip.vy(idx);
                vz += prev_mip.vz(idx);

                idx = idx_gen_upper.idx(coord.x+1, coord.z+1);
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

                IndexGen<BLOCK_SIZE> idx_gen_upper(block_map, vox_size / 2);
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

                i64 idx1 = idx_gen_upper.idx(coord.x+1, coord.z);
                compute_vels(idx1, 1);

                i64 idx2 = idx_gen_upper.idx(coord.x, coord.z+1);
                compute_vels(idx2, 2);

                i64 idx3 = idx_gen_upper.idx(coord.x+1, coord.z+1);
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

#else
#endif