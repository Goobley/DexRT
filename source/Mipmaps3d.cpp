#include "Mipmaps3d.hpp"
#include "RcUtilsModes.hpp"
#include "EmisOpac.hpp"

void ClassicEmisOpacData3d::init(i64 buffer_len) {
    dynamic_opac = Fp1d("dynamic_opac", buffer_len);
}

void MultiResMipChain3d::init(const State3d& state, i64 buffer_len) {
    emis = Fp1d("emis_mips", buffer_len);
    opac = Fp1d("opac_mips", buffer_len);

    max_mip_factor = state.mr_block_map.max_mip_level;

    if (state.config.mode != DexrtMode::GivenFs) {
        // NOTE(cmo): Classic doesn't use mips
        if constexpr (LINE_SCHEME_3D != LineCoeffCalc::Classic) {
            vx = Fp1d("vx_mips", buffer_len);
            vy = Fp1d("vy_mips", buffer_len);
            vz = Fp1d("vz_mips", buffer_len);
        }

        if constexpr (LINE_SCHEME_3D == LineCoeffCalc::VelocityInterp) {
            throw std::runtime_error("Not yet handling Velocity interp in 3D");
        } else if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
            cav_data.init(buffer_len, CORE_AND_VOIGT_MAX_LINES_3D);
        } else if constexpr (LINE_SCHEME == LineCoeffCalc::Classic) {
            classic_data.init(buffer_len);
        } else {
            throw std::runtime_error("It appears you've added a LineCoeffCalc, but nothing to Mipmaps. Do you need to?");
        }
    }
}

void MultiResMipChain3d::fill_mip0_atomic(
    const State3d& state,
    const Fp2d& lte_scratch,
    int la
) const {
    JasUnpack(state, atmos, pops, phi, adata);

    const auto& flat_dynamic_opac = classic_data.dynamic_opac;
    const bool fill_dynamic_opac = flat_dynamic_opac.initialized();
    const auto flatmos = flatten<const fp_t>(atmos);

    JasUnpack((*this), emis, opac);
    const auto& block_map = state.mr_block_map.block_map;
    auto bounds = block_map.loop_bounds();
    dex_parallel_for(
        "Compute eta, chi",
        FlatLoop<2>(bounds.dim(0), bounds.dim(1)),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen3d idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord3 coord = idx_gen.loop_coord(tile_idx, block_idx);

            AtmosPointParams local_atmos{};
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
            int governing_atom = adata.governing_trans(la).atom;

            const bool static_only = v_norm >= (ANGLE_INVARIANT_THERMAL_VEL_FRAC * thermal_vel(
                adata.mass(governing_atom),
                local_atmos.temperature
            ));
            auto active_set = slice_active_set(adata, la);
            if (fill_dynamic_opac) {
                const bool no_lines = (active_set.extent(0) == 0);
                flat_dynamic_opac(ks) = static_only || no_lines;
            }
            EmisOpacMode mode = static_only ? EmisOpacMode::StaticOnly : EmisOpacMode::All;
            if constexpr (BASE_MIP_CONTAINS_3D == BaseMipContents::Continua) {
                mode = EmisOpacMode::StaticOnly;
            } else if constexpr (BASE_MIP_CONTAINS_3D == BaseMipContents::LinesAtRest) {
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
            emis(ks) = result.eta;
            opac(ks) = result.chi;
        }
    );

    if (vx.initialized()) {
        JasUnpack((*this), vx, vy, vz);
        dex_parallel_for(
            "Copy vels",
            FlatLoop<2>(bounds),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen3d idx_gen(block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);

                vx(ks) = flatmos.vx(ks);
                vy(ks) = flatmos.vy(ks);
                vz(ks) = flatmos.vz(ks);
            }
        );
    }

    if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
        cav_data.fill(state, la);
    }

    yakl::fence();
}

// void MultiResMipChain3d::fill_subset_mip0_atomic(
//     const State& state,
//     const CascadeCalcSubset& subset,
//     const Fp2d& n_star
// ) const {
//     if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
//         FlatVelocity vels{
//             .vx = vx,
//             .vy = vy,
//             .vz = vz
//         };
//         dir_data.fill<RC_flags_storage_2d()>(state, subset, vels, n_star);
//     }
// }

/// compute the mips from mip0 (stored in the start of the arrays), for
/// direction independent terms. Also update state.mr_block_map as
/// necessary.
void MultiResMipChain3d::compute_mips(const State3d& state, int la) const {
    JasUnpack(state, mr_block_map);
    const auto& block_map = mr_block_map.block_map;
    JasUnpack((*this), vx, vy, vz, emis, opac);

    constexpr i32 mip_block = 8;
    const fp_t vox_scale = (state.config.mode == DexrtMode::GivenFs) ? state.given_state.voxel_scale : state.atmos.voxel_scale;

    // NOTE(cmo): mippable entries is used as an accumulator during each round
    // of mipping, and then divided by the number of sub blocks and placed into
    // max mip_level. max_mip_level then holds the max_mip_level + 1 s.t. 0 represents empty.
    yakl::Array<i32, 1, yakl::memDevice> mippable_entries("mippable entries", block_map.num_active_tiles);
    yakl::Array<i32, 1, yakl::memDevice> max_mip_level("max mip entries", block_map.num_z_tiles() * block_map.num_y_tiles() * block_map.num_x_tiles());

    const bool compute_criteria_on_base_array = (state.config.mode == DexrtMode::GivenFs) || (BASE_MIP_CONTAINS_3D == BaseMipContents::LinesAtRest);
    const MipmapTolerance mip_config = {
        .opacity_threshold = state.config.mip_config.opacity_threshold,
        .log_chi_mip_variance = state.config.mip_config.log_chi_mip_variance,
        .log_eta_mip_variance = state.config.mip_config.log_eta_mip_variance,
    };

    max_mip_level = 0;
    mippable_entries = 0;
    yakl::fence();

    dex_parallel_for(
        "Set active blocks in mr_block_map",
        FlatLoop<1>(block_map.loop_bounds().dim(0)),
        YAKL_LAMBDA (i64 tile_idx) {
            MRIdxGen3d idx_gen(mr_block_map);
            Coord3 tile_coord = idx_gen.compute_tile_coord(tile_idx);
            i64 flat_entry = mr_block_map.lookup.flat_tile_index(tile_coord);
            max_mip_level(flat_entry) = 1;
        }
    );
    yakl::fence();

    // NOTE(cmo): First compute all the velocity and isotropic emis/opac mips
    for (int level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
        const i32 vox_size = (1 << (level_m_1 + 1));
        auto bounds = block_map.loop_bounds(vox_size);

        if (state.config.mode != DexrtMode::GivenFs) {
            dex_parallel_for(
                "Compute vel mip",
                FlatLoop<2>(bounds.dim(0), bounds.dim(1)),
                YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                    MRIdxGen3d idx_gen(mr_block_map);
                    const i32 vox_size = (1 << (level_m_1 + 1));
                    const i32 level = level_m_1 + 1;

                    const i64 ks = idx_gen.loop_idx(level, tile_idx, block_idx);
                    const Coord3 coord = idx_gen.loop_coord(level, tile_idx, block_idx);
                    const Coord3 tile_coord = idx_gen.compute_tile_coord(tile_idx);

                    const i32 upper_vox_size = vox_size / 2;
                    fp_t vel_x = FP(0.0);
                    fp_t vel_y = FP(0.0);
                    fp_t vel_z = FP(0.0);
                    i64 idxs[mip_block] = {
                        idx_gen.idx(
                            level_m_1,
                            coord // 0 0 0
                        ),
                        idx_gen.idx(
                            level_m_1,
                            Coord3{.x = coord.x+upper_vox_size, .y = coord.y, .z = coord.z} // 0 0 1
                        ),
                        idx_gen.idx(
                            level_m_1,
                            Coord3{.x = coord.x, .y = coord.y+upper_vox_size, .z = coord.z} // 0 1 0
                        ),
                        idx_gen.idx(
                            level_m_1,
                            Coord3{.x = coord.x+upper_vox_size, .y = coord.y+upper_vox_size, .z = coord.z} // 0 1 1
                        ),
                        idx_gen.idx(
                            level_m_1,
                            Coord3{.x = coord.x, .y = coord.y, .z = coord.z+upper_vox_size} // 1 0 0
                        ),
                        idx_gen.idx(
                            level_m_1,
                            Coord3{.x = coord.x+upper_vox_size, .y = coord.y, .z = coord.z+upper_vox_size} // 1 0 1
                        ),
                        idx_gen.idx(
                            level_m_1,
                            Coord3{.x = coord.x, .y = coord.y+upper_vox_size, .z = coord.z+upper_vox_size} // 1 1 0
                        ),
                        idx_gen.idx(
                            level_m_1,
                            Coord3{.x = coord.x+upper_vox_size, .y = coord.y+upper_vox_size, .z = coord.z+upper_vox_size} // 1, 1, 1
                        )
                    };

                    for (int i = 0; i < mip_block; ++i) {
                        i64 idx = idxs[i];
                        vel_x += vx(idx);
                        vel_y += vy(idx);
                        vel_z += vz(idx);
                    }

                    vel_x *= FP(0.125);
                    vel_y *= FP(0.125);
                    vel_z *= FP(0.125);

                    vx(ks) = vel_x;
                    vy(ks) = vel_y;
                    vz(ks) = vel_z;
                }
            );
        }

        dex_parallel_for(
            "Compute mip n",
            FlatLoop<2>(
                bounds.dim(0),
                bounds.dim(1)
            ),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                const i32 level = level_m_1 + 1;
                MRIdxGen3d idx_gen(mr_block_map);
                const i64 ks = idx_gen.loop_idx(level, tile_idx, block_idx);
                const Coord3 coord = idx_gen.loop_coord(level, tile_idx, block_idx);
                const Coord3 tile_coord = idx_gen.compute_tile_coord(tile_idx);

                const i32 upper_vox_size = vox_size / 2;
                const fp_t ds = vox_scale * upper_vox_size;
                fp_t emis_mip = FP(0.0);
                fp_t opac_mip = FP(0.0);
                i64 idxs[mip_block] = {
                    idx_gen.idx(
                        level_m_1,
                        coord // 0 0 0
                    ),
                    idx_gen.idx(
                        level_m_1,
                        Coord3{.x = coord.x+upper_vox_size, .y = coord.y, .z = coord.z} // 0 0 1
                    ),
                    idx_gen.idx(
                        level_m_1,
                        Coord3{.x = coord.x, .y = coord.y+upper_vox_size, .z = coord.z} // 0 1 0
                    ),
                    idx_gen.idx(
                        level_m_1,
                        Coord3{.x = coord.x+upper_vox_size, .y = coord.y+upper_vox_size, .z = coord.z} // 0 1 1
                    ),
                    idx_gen.idx(
                        level_m_1,
                        Coord3{.x = coord.x, .y = coord.y, .z = coord.z+upper_vox_size} // 1 0 0
                    ),
                    idx_gen.idx(
                        level_m_1,
                        Coord3{.x = coord.x+upper_vox_size, .y = coord.y, .z = coord.z+upper_vox_size} // 1 0 1
                    ),
                    idx_gen.idx(
                        level_m_1,
                        Coord3{.x = coord.x, .y = coord.y+upper_vox_size, .z = coord.z+upper_vox_size} // 1 1 0
                    ),
                    idx_gen.idx(
                        level_m_1,
                        Coord3{.x = coord.x+upper_vox_size, .y = coord.y+upper_vox_size, .z = coord.z+upper_vox_size} // 1, 1, 1
                    )
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

                for (int i = 0; i < mip_block; ++i) {
                    i64 idx = idxs[i];
                    fp_t eta_s = emis(idx);
                    fp_t chi_s = opac(idx);
                    emis_mip += eta_s;
                    opac_mip += chi_s;

                    if (compute_criteria_on_base_array) {
                        consider_variance = consider_variance || (chi_s * ds) > mip_config.opacity_threshold;
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
                    if (
                        D_chi > mip_config.log_chi_mip_variance
                        || D_eta > mip_config.log_eta_mip_variance
                    ) {
                        do_increment = false;
                    }
                }

                // NOTE(cmo): This is coming from many threads of a warp
                // simultaneously, which isn't great. If it's a bottleneck,
                // ballot across threads, do a popcount, and increment from one
                // thread.
                if (compute_criteria_on_base_array && do_increment) {
                    Kokkos::atomic_add(&mippable_entries(tile_idx), 1);
                }

                emis(ks) = emis_mip;
                opac(ks) = opac_mip;
            }
        );
        yakl::fence();


        MipmapComputeState3d mm_state{
            .max_mip_factor = max_mip_factor,
            .la = la,
            .mippable_entries = mippable_entries,
            .emis = emis,
            .opac = opac,
            .vx = vx,
            .vy = vy,
            .vz = vz
        };

        if (state.config.mode != DexrtMode::GivenFs) {
            if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
                // dir_data.compute_mip_n(state, mm_state, level_m_1+1);
                throw std::runtime_error("NYI");
            } else if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
                cav_data.compute_mip_n(state, mm_state, level_m_1+1);
            } else {
                throw std::runtime_error("Add your mip handling here");
            }
        }

        dex_parallel_for(
            "Update mippable array",
            FlatLoop<1>(block_map.loop_bounds().dim(0)),
            YAKL_LAMBDA (i64 tile_idx) {
                MRIdxGen3d idx_gen(mr_block_map);
                Coord3 tile_coord = idx_gen.compute_tile_coord(tile_idx);
                const int level = level_m_1 + 1;
                const i32 expected_entries = cube(mr_block_map.block_size >> level);
                i64 flat_entry = mr_block_map.lookup.flat_tile_index(tile_coord);
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
// void MultiResMipChain3d::compute_subset_mips(const State& state, const CascadeCalcSubset& subset) const {
//     if (state.config.mode == DexrtMode::GivenFs) {
//         return;
//     }
//     for (i32 level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
//         MipmapSubsetState mm_state{
//             .max_mip_factor = max_mip_factor,
//             .la_start = subset.la_start,
//             .la_end = subset.la_end,
//             .emis = emis,
//             .opac = opac,
//             .vx = vx,
//             .vy = vy,
//             .vz = vz
//         };
//         if constexpr (LINE_SCHEME == LineCoeffCalc::VelocityInterp) {
//             dir_data.compute_subset_mip_n(state, mm_state, subset, level_m_1 + 1);
//         } else if constexpr (LINE_SCHEME == LineCoeffCalc::CoreAndVoigt) {
//             cav_data.compute_subset_mip_n(state, mm_state, subset, level_m_1 + 1);
//         } else {
//             throw std::runtime_error("Add your mip handling here");
//         }
//     }
// }
