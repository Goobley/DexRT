#if !defined(DEXRT_MIPMAPS_HPP)
#define DEXRT_MIPMAPS_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "BlockMap.hpp"
#include "DirectionalEmisOpacInterp.hpp"
#include "CoreAndVoigtEmisOpac.hpp"

struct ClassicEmisOpacData {
    Fp2d dynamic_opac; // [ks, wave_batch] -- whether this cell needs its lines to be computed separately.

    void init(i64 buffer_len, i32 wave_batch) {
        dynamic_opac = Fp2d("dynamic_opac", buffer_len, wave_batch);
    }
};

struct MultiResMipChain {
    i32 max_mip_factor;
    Fp2d emis; // [ks, wave_batch]
    Fp2d opac; // [ks, wave_batch]
    DirectionalEmisOpacInterp dir_data;
    CoreAndVoigtData cav_data;
    ClassicEmisOpacData classic_data;
    Fp1d vx; // [ks], present if not LineCoeffCalc::Classic
    Fp1d vy; // [ks], present if not LineCoeffCalc::Classic
    Fp1d vz; // [ks], present if not LineCoeffCalc::Classic

    /// buffer_len is expected to be the one from mr_block_map, i.e. to hold all the mips.
    void init(const State& state, i64 buffer_len, i32 wave_batch) {
        emis = Fp2d("emis_mips", buffer_len, wave_batch);
        opac = Fp2d("opac_mips", buffer_len, wave_batch);

        max_mip_factor = state.mr_block_map.max_mip_level;

        if (state.config.mode != DexrtMode::GivenFs) {
            // NOTE(cmo): Classic doesn't use mips
            if constexpr (LINE_SCHEME != LineCoeffCalc::Classic) {
                vx = Fp1d("vx_mips", buffer_len);
                vy = Fp1d("vy_mips", buffer_len);
                vz = Fp1d("vz_mips", buffer_len);
            }

            if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
                dir_data.init(buffer_len, wave_batch);
            } else if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
                cav_data.init(buffer_len, CORE_AND_VOIGT_MAX_LINES);
            } else if constexpr (LINE_SCHEME == LineCoeffCalc::Classic) {
                classic_data.init(buffer_len, wave_batch);
            } else {
                throw std::runtime_error("It appears you've added a LineCoeffCalc, but nothing to Mipmaps. Do you need to?");
            }
        }
    }

    void fill_mip0_atomic(const State& state, const Fp2d& lte_scratch, int la_start, int la_end) {
        JasUnpack(state, atmos, pops, phi, adata);
        int wave_batch = la_end - la_start;

        const auto& flat_dynamic_opac = classic_data.dynamic_opac;
        const bool fill_dynamic_opac = flat_dynamic_opac.initialized();
        const auto flatmos = flatten<const fp_t>(atmos);

        JasUnpack((*this), emis, opac);
        const auto& block_map = state.mr_block_map.block_map;
        auto bounds = block_map.loop_bounds();
        parallel_for(
            "Compute eta, chi",
            SimpleBounds<3>(bounds.dim(0), bounds.dim(1), wave_batch),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, int wave) {
                IndexGen<BLOCK_SIZE> idx_gen(block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

                AtmosPointParams local_atmos;
                local_atmos.temperature = flatmos.temperature(ks);
                local_atmos.ne = flatmos.ne(ks);
                local_atmos.vturb = flatmos.vturb(ks);
                local_atmos.nhtot = flatmos.nh_tot(ks);
                local_atmos.nh0 = flatmos.nh0(ks);
                const fp_t v_norm = std::sqrt(
                        square(flatmos.vx(ks))
                        + square(flatmos.vy(ks))
                        + square(flatmos.vz(ks))
                );
                const int la = la_start + wave;
                int governing_atom = adata.governing_trans(la).atom;

                const bool static_only = v_norm >= (ANGLE_INVARIANT_THERMAL_VEL_FRAC * thermal_vel(
                    adata.mass(governing_atom),
                    local_atmos.temperature
                ));
                auto active_set = slice_active_set(adata, la);
                if (fill_dynamic_opac) {
                    const bool no_lines = (active_set.extent(0) == 0);
                    flat_dynamic_opac(ks, wave) = static_only || no_lines;
                }
                EmisOpacMode mode = static_only ? EmisOpacMode::StaticOnly : EmisOpacMode::All;
                if constexpr (BASE_MIP_CONTAINS == BaseMipContents::Continua) {
                    mode = EmisOpacMode::StaticOnly;
                } else if constexpr (BASE_MIP_CONTAINS == BaseMipContents::LinesAtRest) {
                    mode = EmisOpacMode::All;
                }

                auto result = emis_opac(
                    EmisOpacState<fp_t>{
                        .adata = adata,
                        .profile = phi,
                        .la = la,
                        .n = pops,
                        .n_star_scratch = lte_scratch,
                        .k = ks,
                        .atmos = local_atmos,
                        .active_set = active_set,
                        .active_set_cont = slice_active_cont_set(adata, la),
                        .mode = mode
                    }
                );
                emis(ks, wave) = result.eta;
                opac(ks, wave) = result.chi;
            }
        );

        if (vx.initialized()) {
            JasUnpack((*this), vx, vy, vz);
            parallel_for(
                "Copy vels",
                SimpleBounds<2>(bounds),
                YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                    IndexGen<BLOCK_SIZE> idx_gen(block_map);
                    i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                    Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
                    i64 k = idx_gen.full_flat_idx(coord.x, coord.z);

                    vx(ks) = flatmos.vx(k);
                    vy(ks) = flatmos.vy(k);
                    vz(ks) = flatmos.vz(k);
                }
            );
        }

        if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
            cav_data.fill(state, la_start, la_end);
        }

        yakl::fence();
    }

    void fill_subset_mip0_atomic(const State& state, const CascadeCalcSubset& subset, const Fp2d& n_star) {
        if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
            FlatVelocity vels{
                .vx = vx,
                .vy = vy,
                .vz = vz
            };
            dir_data.fill<RC_flags_storage()>(state, subset, vels, n_star);
        }
    }

    /// compute the mips from mip0 (stored in the start of the arrays), for
    /// direction independent terms. Also update state.mr_block_map as
    /// necessary.
    void compute_mips(const State& state, int la_start, int la_end) {
        JasUnpack(state, mr_block_map);
        const auto& block_map = mr_block_map.block_map;
        JasUnpack((*this), vx, vy, vz, emis, opac, dir_data);

        constexpr i32 mip_block = 4;
        const fp_t vox_scale = state.atmos.voxel_scale;
        const i32 wave_batch = la_end - la_start;

        // NOTE(cmo): mippable entries is used as an accumulator during each round
        // of mipping, and then divided by the number of sub blocks and placed into
        // max mip_level. max_mip_level then holds the max_mip_level + 1 s.t. 0 represents empty.
        yakl::Array<i32, 1, yakl::memDevice> mippable_entries("mippable entries", block_map.num_active_tiles);
        yakl::Array<i32, 1, yakl::memDevice> max_mip_level("max mip entries", block_map.num_z_tiles * block_map.num_x_tiles);

        const bool compute_criteria_on_base_array = (state.config.mode == DexrtMode::GivenFs) || (BASE_MIP_CONTAINS == BaseMipContents::LinesAtRest);

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
                        i64 idxs[mip_block] = {
                            idx_gen.idx(level_m_1, coord.x, coord.z),
                            idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z),
                            idx_gen.idx(level_m_1, coord.x, coord.z+upper_vox_size),
                            idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z+upper_vox_size)
                        };

                        for (int i = 0; i < mip_block; ++i) {
                            i64 idx = idxs[i];
                            vel_x += vx(idx);
                            vel_y += vy(idx);
                            vel_z += vz(idx);
                        }

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
                    i64 idxs[mip_block] = {
                        idx_gen.idx(level_m_1, coord.x, coord.z),
                        idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z),
                        idx_gen.idx(level_m_1, coord.x, coord.z+upper_vox_size),
                        idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z+upper_vox_size)
                    };

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

                    for (int i = 0; i < mip_block; ++i) {
                        i64 idx = idxs[i];
                        fp_t eta_s = emis(idx, wave);
                        fp_t chi_s = opac(idx, wave);
                        emis_mip += eta_s;
                        opac_mip += chi_s;

                        if (compute_criteria_on_base_array) {
                            consider_variance = consider_variance || (chi_s * ds) > opacity_threshold;
                            chi_s += FP(1e-15);
                            eta_s += FP(1e-15);
                            m1_chi += std::log(chi_s * ds);
                            m2_chi += square(std::log(chi_s * ds));
                            m1_eta += std::log(eta_s * ds);
                            m2_eta += square(std::log(eta_s * ds));
                        }
                    }
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
                        if (D_chi > variance_threshold || D_eta > variance_threshold) {
                            do_increment = false;
                        }
                    }

                    // NOTE(cmo): This is coming from many threads of a warp
                    // simultaneously, which isn't great. If it's a bottleneck,
                    // ballot across threads, do a popcount, and increment from one
                    // thread.
                    if (compute_criteria_on_base_array && do_increment) {
                        yakl::atomicAdd(mippable_entries(tile_idx), 1);
                    }

                    emis(ks, wave) = emis_mip;
                    opac(ks, wave) = opac_mip;
                }
            );
            yakl::fence();


            MipmapComputeState mm_state{
                .max_mip_factor = max_mip_factor,
                .la_start = la_start,
                .la_end = la_end,
                .mippable_entries = mippable_entries,
                .emis = emis,
                .opac = opac,
                .vx = vx,
                .vy = vy,
                .vz = vz
            };

            if (state.config.mode != DexrtMode::GivenFs) {
                if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
                    dir_data.compute_mip_n(state, mm_state, level_m_1+1);
                } else if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
                    cav_data.compute_mip_n(state, mm_state, level_m_1+1);
                } else {
                    throw std::runtime_error("Add your mip handling here");
                }
            }

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
    }

    /// compute the mips from mip0 being stored in the start of these arrays.
    /// Currently these aren't allowed to modify state.mr_block_map.
    void compute_subset_mips(const State& state, const CascadeCalcSubset& subset) {
        if (state.config.mode == DexrtMode::GivenFs) {
            return;
        }
        for (i32 level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
            MipmapSubsetState mm_state{
                .max_mip_factor = max_mip_factor,
                .la_start = subset.la_start,
                .la_end = subset.la_end,
                .emis = emis,
                .opac = opac,
                .vx = vx,
                .vy = vy,
                .vz = vz
            };
            if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
                dir_data.compute_subset_mip_n(state, mm_state, subset, level_m_1 + 1);
            } else if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
                cav_data.compute_subset_mip_n(state, mm_state, subset, level_m_1 + 1);
            } else {
                throw std::runtime_error("Add your mip handling here");
            }
        }
    }
};

#else
#endif