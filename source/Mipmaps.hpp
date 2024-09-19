#if !defined(DEXRT_MIPMAPS_HPP)
#define DEXRT_MIPMAPS_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "BlockMap.hpp"
#include "MiscSparseStorage.hpp"

struct SparseMip {
    Fp2d emis; // [ks, wave_batch]
    Fp2d opac; // [ks, wave_batch]
    // TODO(cmo): DirectionalEmisOpac
    Fp1d vx; // [ks]
    Fp1d vy; // [ks]
    Fp1d vz; // [ks]
};

struct SparseMips {
    std::vector<SparseMip> mips;
};

inline SparseMips compute_mips(const State& state, const CascadeState& casc_state, const SparseStores& full_res) {
    assert(USE_MIPMAPS);
    JasUnpack(state, block_map);

    i32 max_mip_factor = 0;
    for (int i = 0; i < MAX_CASCADE + 1; ++i) {
        max_mip_factor += MIPMAP_FACTORS[i];
    }

    SparseMips result;
    result.mips.reserve(max_mip_factor+1);
    SparseMip mip0 {
        .emis = full_res.emis,
        .opac = full_res.opac,
        .vx = full_res.vx,
        .vy = full_res.vy,
        .vz = full_res.vz
    };
    result.mips[0] = mip0;

    for (int level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
        auto& prev_mip = result.mips[level_m_1];
        i32 vox_size = (1 << (level_m_1 + 1));
        SparseMip mip;
        mip.emis = Fp2d("emis mip", mip0.emis.extent(0) / square(vox_size), state.c0_size.wave_batch);
        mip.opac = Fp2d("opac mip", mip0.emis.extent(0) / square(vox_size), state.c0_size.wave_batch);
        mip.vx = Fp1d("vx mip", mip0.emis.extent(0) / square(vox_size));
        mip.vy = Fp1d("vy mip", mip0.emis.extent(0) / square(vox_size));
        mip.vz = Fp1d("vz mip", mip0.emis.extent(0) / square(vox_size));

        auto bound = block_map.loop_bounds(level_m_1+1);
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
        yakl::fence();
    }

    return result;
}

#else
#endif