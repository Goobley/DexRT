#if !defined(DEXRT_MISC_SPARSE_STORAGE_HPP)
#define DEXRT_MISC_SPARSE_STORAGE_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "BlockMap.hpp"

struct SparseStores {
    Fp2d dynamic; // [ks, wave]
    Fp2d emis; // [ks, wave]
    Fp2d opac; // [ks, wave]
    Fp1d vx; // [ks]
    Fp1d vy; // [ks]
    Fp1d vz; // [ks]

    void init(i64 len, i32 wave_batch) {
        dynamic = Fp2d("dynamic_sparse", len, wave_batch);
        emis = Fp2d("emis_sparse", len, wave_batch);
        opac = Fp2d("opac_sparse", len, wave_batch);
        vx = Fp1d("vx_sparse", len);
        vy = Fp1d("vy_sparse", len);
        vz = Fp1d("vz_sparse", len);
    }

    void fill(const State& state, const CascadeState& casc_state) {
        JasUnpack(state, dynamic_opac, atmos, block_map);
        JasUnpack(casc_state, eta, chi);
        JasUnpack((*this), dynamic, emis, opac, vx, vy, vz);

        auto bounds = block_map.loop_bounds();
        parallel_for(
            "copy to sparse structure",
            SimpleBounds<3>(
                bounds.dim(0),
                bounds.dim(1),
                state.c0_size.wave_batch
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
                IndexGen<BLOCK_SIZE> idx_gen(block_map);
                const i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                const Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                dynamic(ks, wave) = dynamic_opac(coord.z, coord.x, wave);
                emis(ks, wave) = eta(coord.z, coord.x, wave);
                opac(ks, wave) = chi(coord.z, coord.x, wave);
                if (wave == 0) {
                    vx(ks) = atmos.vx(coord.z, coord.x);
                    vy(ks) = atmos.vy(coord.z, coord.x);
                    vz(ks) = atmos.vz(coord.z, coord.x);
                }

            }
        );
        yakl::fence();
    }

};



#else
#endif