#if !defined(DEXRT_MIPMAPS_HPP)
#define DEXRT_MIPMAPS_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "BlockMap.hpp"
#include "DirectionalEmisOpacInterp.hpp"

struct MultiResMipChain {
    Fp2d emis; // [ks, wave_batch]
    Fp2d opac; // [ks, wave_batch]
    DirectionalEmisOpacInterp dir_data;
    Fp1d vx; // [ks]
    Fp1d vy; // [ks]
    Fp1d vz; // [ks]

    /// buffer_len is expected to be the one from mr_block_map, i.e. to hold all the mips.
    void init(const State& state, i64 buffer_len, i32 wave_batch) {
        emis = Fp2d("emis_mips", buffer_len, wave_batch);
        opac = Fp2d("opac_mips", buffer_len, wave_batch);

        if (state.config.mode != DexrtMode::GivenFs) {
            dir_data = DirectionalEmisOpacInterp_new(buffer_len, wave_batch);
            vx = Fp1d("vx_mips", buffer_len);
            vy = Fp1d("vy_mips", buffer_len);
            vz = Fp1d("vz_mips", buffer_len);
        }
    }

    /// compute the mips from mip0 being stored in the start of these arrays. Will also update state.mr_block_map.
    void compute_mips(const State& state, const CascadeCalcSubset& subset) {
        JasUnpack(state, mr_block_map);
        const auto& block_map = mr_block_map.block_map;
        JasUnpack((*this), vx, vy, vz, emis, opac, dir_data);

        i32 max_mip_factor = 0;
        for (int i = 0; i < MAX_CASCADE + 1; ++i) {
            max_mip_factor += MIPMAP_FACTORS[i];
        }
        const fp_t vox_scale = state.atmos.voxel_scale;
        const i32 wave_batch = emis.extent(1);

        // NOTE(cmo): mippable entries is used as an accumulator during each round
        // of mipping, and then divided by the number of sub blocks and placed into
        // max mip_level. max_mip_level then holds the max_mip_level + 1 s.t. 0 represents empty.
        yakl::Array<i32, 1, yakl::memDevice> mippable_entries("mippable entries", block_map.num_active_tiles);
        yakl::Array<i32, 1, yakl::memDevice> max_mip_level("max mip entries", block_map.num_z_tiles * block_map.num_x_tiles);

        max_mip_level = 0;
        mippable_entries = 0;
        yakl::fence();

        parallel_for(
            "Set active blocks in mr_block_map",
            SimpleBounds<1>(block_map.loop_bounds().dim(0)),
            YAKL_LAMBDA (i64 tile_idx) {
                MRIdxGen idx_gen(mr_block_map);
                Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);
                i64 flat_entry = mr_block_map.lookup.flat_tile_index(tile_coord.x, tile_coord.z);
                max_mip_level(flat_entry) = 1;
            }
        );
        yakl::fence();

        // NOTE(cmo): First compute all the velocity and isotropic emis/opac mips
        for (int level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
            const i32 vox_size = (1 << (level_m_1 + 1));
            auto bounds = block_map.loop_bounds(vox_size);

            if (state.config.mode != DexrtMode::GivenFs) {
                parallel_for(
                    "Compute vel mip",
                    SimpleBounds<3>(bounds.dim(0), bounds.dim(1), wave_batch),
                    YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
                        MRIdxGen idx_gen(mr_block_map);
                        const i32 vox_size = (1 << (level_m_1 + 1));
                        const i32 level = level_m_1 + 1;

                        const i64 ks = idx_gen.loop_idx(level, tile_idx, block_idx);
                        const Coord2 coord = idx_gen.loop_coord(level, tile_idx, block_idx);
                        const Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);

                        const i32 upper_vox_size = vox_size / 2;
                        fp_t vel_x = FP(0.0);
                        fp_t vel_y = FP(0.0);
                        fp_t vel_z = FP(0.0);
                        i64 idx0 = idx_gen.idx(level_m_1, coord.x, coord.z);
                        i64 idx1 = idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z);
                        i64 idx2 = idx_gen.idx(level_m_1, coord.x, coord.z+upper_vox_size);
                        i64 idx3 = idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z+upper_vox_size);

                        vel_x += vx(idx0);
                        vel_y += vy(idx0);
                        vel_z += vz(idx0);

                        vel_x += vx(idx1);
                        vel_y += vy(idx1);
                        vel_z += vz(idx1);

                        vel_x += vx(idx2);
                        vel_y += vy(idx2);
                        vel_z += vz(idx2);

                        vel_x += vx(idx3);
                        vel_y += vy(idx3);
                        vel_z += vz(idx3);

                        vel_x *= FP(0.25);
                        vel_y *= FP(0.25);
                        vel_z *= FP(0.25);

                        vx(ks) = vel_x;
                        vy(ks) = vel_y;
                        vz(ks) = vel_z;
                    }
                );
            }

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
                    MRIdxGen idx_gen(mr_block_map);
                    const i64 ks = idx_gen.loop_idx(level, tile_idx, block_idx);
                    const Coord2 coord = idx_gen.loop_coord(level, tile_idx, block_idx);
                    const Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);

                    const i32 upper_vox_size = vox_size / 2;
                    fp_t emis_mip = FP(0.0);
                    fp_t opac_mip = FP(0.0);
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

                    fp_t eta_s = emis(idx0, wave);
                    fp_t chi_s = opac(idx0, wave);
                    emis_mip += eta_s;
                    opac_mip += chi_s;
                    consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                    chi_s += FP(1e-15);
                    eta_s += FP(1e-15);
                    m1_chi += std::log(chi_s * ds);
                    m2_chi += square(std::log(chi_s * ds));
                    m1_eta += std::log(eta_s * ds);
                    m2_eta += square(std::log(eta_s * ds));

                    eta_s = emis(idx1, wave);
                    chi_s = opac(idx1, wave);
                    emis_mip += eta_s;
                    opac_mip += chi_s;
                    consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                    chi_s += FP(1e-15);
                    eta_s += FP(1e-15);
                    m1_chi += std::log(chi_s * ds);
                    m2_chi += square(std::log(chi_s * ds));
                    m1_eta += std::log(eta_s * ds);
                    m2_eta += square(std::log(eta_s * ds));

                    eta_s = emis(idx2, wave);
                    chi_s = opac(idx2, wave);
                    emis_mip += eta_s;
                    opac_mip += chi_s;
                    consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                    chi_s += FP(1e-15);
                    eta_s += FP(1e-15);
                    m1_chi += std::log(chi_s * ds);
                    m2_chi += square(std::log(chi_s * ds));
                    m1_eta += std::log(eta_s * ds);
                    m2_eta += square(std::log(eta_s * ds));

                    eta_s = emis(idx3, wave);
                    chi_s = opac(idx3, wave);
                    emis_mip += eta_s;
                    opac_mip += chi_s;
                    consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                    chi_s += FP(1e-15);
                    eta_s += FP(1e-15);
                    m1_chi += std::log(chi_s * ds);
                    m2_chi += square(std::log(chi_s * ds));
                    m1_eta += std::log(eta_s * ds);
                    m2_eta += square(std::log(eta_s * ds));

                    emis_mip *= FP(0.25);
                    opac_mip *= FP(0.25);
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

                    emis(ks, wave) = emis_mip;
                    opac(ks, wave) = opac_mip;
                }
            );
            yakl::fence();

            parallel_for(
                "Update mippable array",
                SimpleBounds<1>(block_map.loop_bounds().dim(0)),
                YAKL_LAMBDA (i64 tile_idx) {
                    MRIdxGen idx_gen(mr_block_map);
                    Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);
                    const int level = level_m_1 + 1;
                    const i32 expected_entries = square(BLOCK_SIZE >> level) * wave_batch;
                    i64 flat_entry = mr_block_map.lookup.flat_tile_index(tile_coord.x, tile_coord.z);
                    if (max_mip_level(flat_entry) == level) {
                        max_mip_level(flat_entry) += mippable_entries(tile_idx) / expected_entries;
                    }
                    mippable_entries(tile_idx) = 0;
                }
            );
            yakl::fence();
        }
        mr_block_map.lookup.pack_entries(max_mip_level);

        if (state.config.mode == DexrtMode::GivenFs) {
            return;
        }

        Fp1d min_vel("min_vel_mips", mr_block_map.buffer_len());
        Fp1d max_vel("max_vel_mips", mr_block_map.buffer_len());
        FlatVelocity vels{
            .vx = vx,
            .vy = vy,
            .vz = vz
        };
        for (int i = 0; i < (max_mip_factor + 1); ++i) {
            compute_min_max_vel(
                state,
                subset,
                i,
                vels,
                min_vel,
                max_vel
            );
        }
        yakl::fence();

        // TODO(cmo): This isn't done properly. The min and max v should be
        // generated from the velocity mips, as in the setup for
        // DirectionalEmisOpacInterp. If _this_ is outside the range, we should
        // compute them again from scratch, through the full N levels of mips
        // (as we need the original atmospheric params)
        for (int level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
            const int vox_size = 1 << (level_m_1 + 1);
            auto bounds = block_map.loop_bounds(vox_size);
            const int level = level_m_1 + 1;
            parallel_for(
                "Compute mip (dir interp)",
                SimpleBounds<4>(
                    bounds.dim(0),
                    bounds.dim(1),
                    INTERPOLATE_DIRECTIONAL_BINS,
                    wave_batch
                ),
                YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 vel_idx, i32 wave) {
                    MRIdxGen idx_gen(mr_block_map);
                    i64 ks = idx_gen.loop_idx(level, tile_idx, block_idx);
                    Coord2 coord = idx_gen.loop_coord(level, tile_idx, block_idx);

                    const i32 upper_vox_size = vox_size / 2;
                    const i64 idx0 = idx_gen.idx(level_m_1, coord.x, coord.z);
                    const i64 idx1 = idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z);
                    const i64 idx2 = idx_gen.idx(level_m_1, coord.x, coord.z+upper_vox_size);
                    const i64 idx3 = idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z+upper_vox_size);

                    fp_t min_vs[4];
                    fp_t max_vs[4];
                    auto compute_vels = [&] (int idx, int corner) {
                        const fp_t sample_min = dir_data.vel_start(idx);
                        const fp_t sample_max = sample_min + INTERPOLATE_DIRECTIONAL_BINS * dir_data.vel_step(idx);
                        min_vs[corner] = sample_min;
                        max_vs[corner] = sample_max;
                    };
                    compute_vels(idx0, 0);
                    compute_vels(idx1, 1);
                    compute_vels(idx2, 2);
                    compute_vels(idx3, 3);

                    fp_t min_v = min_vel(ks);
                    fp_t max_v = max_vel(ks);
                    fp_t vel_start = min_v;
                    fp_t vel_step = (max_v - min_v) / fp_t(INTERPOLATE_DIRECTIONAL_BINS - 1);
                    if (vel_idx == 0 && wave == 0) {
                        dir_data.vel_start(ks) = vel_start;
                        dir_data.vel_step(ks) = vel_step;
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

                    fp_t emis_vel = FP(0.0);
                    fp_t opac_vel = FP(0.0);
                    const fp_t vel0 = clamp_vel(0);
                    auto sample = dir_data.sample(idx0, wave, vel0);
                    emis_vel += sample.eta;
                    opac_vel += sample.chi;

                    const fp_t vel1 = clamp_vel(1);
                    sample = dir_data.sample(idx1, wave, vel1);
                    emis_vel += sample.eta;
                    opac_vel += sample.chi;

                    const fp_t vel2 = clamp_vel(2);
                    sample = dir_data.sample(idx2, wave, vel2);
                    emis_vel += sample.eta;
                    opac_vel += sample.chi;

                    const fp_t vel3 = clamp_vel(3);
                    sample = dir_data.sample(idx3, wave, vel3);
                    emis_vel += sample.eta;
                    opac_vel += sample.chi;

                    emis_vel *= FP(0.25);
                    opac_vel *= FP(0.25);

                    dir_data.emis_opac_vel(ks, vel_idx, 0, wave) = emis_vel;
                    dir_data.emis_opac_vel(ks, vel_idx, 1, wave) = opac_vel;
                }
            );
            yakl::fence();
        }
    }
};

#else
#endif