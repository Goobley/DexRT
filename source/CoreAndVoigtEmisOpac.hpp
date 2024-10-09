#if !defined(DEXRT_CORE_AND_VOIGT_EMIS_OPAC_HPP)
#define DEXRT_CORE_AND_VOIGT_EMIS_OPAC_HPP

#include "Config.hpp"
#include "EmisOpac.hpp"
#include "State.hpp"

namespace CoreAndVoigt {
    struct CoreAndVoigtState {
        i64 k;
        const AtmosPointParams& atmos;
        const AtomicData<>& adata;
        Fp2d n;
    };
    struct CoreAndVoigtResult {
        fp_t eta_star;
        fp_t chi_star;
        fp_t a_damp;
        fp_t dop_width;
    };

    template <typename T=fp_t, int mem_space=yakl::memDevice>
    YAKL_INLINE CoreAndVoigtResult compute_core_and_voigt(
        const CoreAndVoigtState& args,
        int line_idx
    ) {
        JasUnpack(args, k, atmos, adata, n);
        const auto& l = adata.lines(line_idx);
        const int offset = adata.level_start(l.atom);
        const fp_t nj = n(offset + l.j, k);
        const fp_t ni = n(offset + l.i, k);

        LineParams params;
        params.dop_width = doppler_width(l.lambda0, adata.mass(l.atom), atmos.temperature, atmos.vturb);
        params.gamma = gamma_from_broadening(l, adata.broadening, atmos.temperature, atmos.ne, atmos.nh0);
        params.vel = atmos.vel;

        using namespace ConstantsFP;
        // [kJ]
        // at lambda0
        const fp_t hnu_4pi = hc_kJ_nm / (four_pi * l.lambda0);
        // a_damp at lambda0
        const fp_t a = damping_from_gamma(params.gamma, l.lambda0, params.dop_width);
        CoreAndVoigtResult result {
            .eta_star = nj * hnu_4pi * l.Aji,
            .chi_star = hnu_4pi * (ni * l.Bij - nj * l.Bji),
            .a_damp = a,
            .dop_width = params.dop_width
        };
        return result;
    }
}

struct CavEmisOpacState {
    i64 ks;
    i32 krl;
    i32 wave;
    fp_t lambda;
    fp_t vel;
    const VoigtProfile<fp_t, false>& phi;
};

struct CoreAndVoigtData {
    // NOTE(cmo): Right now these aren't unified over wave, in practice, they
    // probably can be, since most of the wavelength varying parameters are
    // pretty insignificant other than over massive ranges.
    // krl: local radiative transition index
    Fp2d eta_star; // [ks, krl]
    Fp2d chi_star; // [ks, krl]
    Fp2d a_damp; // [ks, krl]
    Fp2d inv_dop_width; // [ks, krl]
    yakl::SArray<i32, 1, CORE_AND_VOIGT_MAX_LINES> active_set_mapping; // maps from krl to kr
    yakl::SArray<fp_t, 2, CORE_AND_VOIGT_MAX_LINES, WAVE_BATCH> emis_opac_ratios; // [krl, wave] lambda0 / lambda
    yakl::SArray<fp_t, 2, CORE_AND_VOIGT_MAX_LINES, WAVE_BATCH> a_damp_ratios; // [krl, wave] (lambda / lambda0)**2
    yakl::SArray<fp_t, 1, CORE_AND_VOIGT_MAX_LINES> lambda0s; // [krl]

    void init(i64 buffer_len, i32 max_kr) {
        eta_star = Fp2d("eta_star", buffer_len, max_kr);
        chi_star = Fp2d("chi_star", buffer_len, max_kr);
        a_damp = Fp2d("a_damp", buffer_len, max_kr);
        inv_dop_width = Fp2d("1 / dop_width", buffer_len, max_kr);
    }

    /// Fills mip0
    void fill(const State& state, i32 la_start, i32 la_end) {
        JasUnpack(state, atmos, pops, adata, mr_block_map);
        JasUnpack((*this), eta_star, chi_star, a_damp, inv_dop_width);
        i32 wave_batch = la_end - la_start;
        auto& block_map = mr_block_map.block_map;
        const auto& flatmos = flatten<const fp_t>(atmos);
        const auto& flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));

        for (int i = 0; i < CORE_AND_VOIGT_MAX_LINES; ++i) {
            active_set_mapping(i) = -1;
        }

        // NOTE(cmo): Fill active_set_mapping
        int fill_idx = 0;
        for (int la = la_start; la < la_end; ++la) {
            auto a_set_la = slice_active_set(state.adata_host, la);
            for (int kri = 0; kri < a_set_la.extent(0); ++kri) {
                int kr = a_set_la(kri);
                bool found = false;

                for (int i = 0; i < fill_idx; ++i) {
                    if (active_set_mapping(i) == kr) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    continue;
                }

                if (fill_idx == CORE_AND_VOIGT_MAX_LINES) {
                    throw std::runtime_error(fmt::format("For wavelength range [{}, {}], more than {} lines appear active (CoreAndVoigt limitation). Consider increasing CORE_AND_VOIGT_MAX_LINES", la_start, la_end, CORE_AND_VOIGT_MAX_LINES));
                }
                active_set_mapping(fill_idx++) = kr;
            }
        }

        for (int kri = 0; kri < CORE_AND_VOIGT_MAX_LINES; ++kri) {
            i32 kr = active_set_mapping(kri);
            if (kr < 0) {
                break;
            }

            const fp_t lambda0 = state.adata_host.lines(kr).lambda0;
            lambda0s(kri) = lambda0;
            for (int wave = 0; wave < wave_batch; ++wave) {
                const i32 la = la_start + wave;
                const fp_t lambda = state.adata_host.wavelength(la);

                a_damp_ratios(kri, wave) = square(lambda / lambda0);
                emis_opac_ratios(kri, wave) = lambda0 / lambda;
            }
        }

        const auto& active_set_mapping = this->active_set_mapping;
        parallel_for(
            "fill core and voigt",
            block_map.loop_bounds(),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
                i64 kf = idx_gen.full_flat_idx(coord.x, coord.z);

                AtmosPointParams atmos_point {
                    .temperature = flatmos.temperature(kf),
                    .ne = flatmos.ne(kf),
                    .vturb = flatmos.vturb(kf),
                    .nhtot = flatmos.nh_tot(kf),
                    .nh0 = flatmos.nh0(kf)
                };

                CoreAndVoigt::CoreAndVoigtState line_state {
                    .k = kf,
                    .atmos = atmos_point,
                    .adata = adata,
                    .n = flat_pops
                };
                for (int krl = 0; krl < active_set_mapping.size(); ++krl) {
                    if (active_set_mapping(krl) < 0) {
                        break;
                    }
                    CoreAndVoigt::CoreAndVoigtResult line_data = CoreAndVoigt::compute_core_and_voigt(
                        line_state,
                        active_set_mapping(krl)
                    );
                    eta_star(ks, krl) = line_data.eta_star;
                    chi_star(ks, krl) = line_data.chi_star;
                    a_damp(ks, krl) = line_data.a_damp;
                    inv_dop_width(ks, krl) = FP(1.0) / line_data.dop_width;
                }
            }
        );
    }

    YAKL_INLINE EmisOpac emis_opac(
        const CavEmisOpacState& args
    ) const {
        JasUnpack(args, ks, krl, wave, lambda, vel, phi);

        using namespace ConstantsFP;
#ifdef DEXRT_DEBUG
        fp_t a = a_damp(ks, krl);
        const fp_t inv_dop = inv_dop_width(ks, krl);
        const fp_t eta_s = eta_star(ks, krl);
        const fp_t chi_s = chi_star(ks, krl);
#else
        const i64 idx = ks * a_damp.extent(1) + krl;

        fp_t a = a_damp.get_data()[idx];
        const fp_t inv_dop = inv_dop_width.get_data()[idx];
        const fp_t eta_s = eta_star.get_data()[idx];
        const fp_t chi_s = chi_star.get_data()[idx];
#endif
        constexpr fp_t inv_c = FP(1.0) / c;
        constexpr fp_t inv_sqrt_pi = FP(1.0) / sqrt_pi;

        // const fp_t a_damp_ratio = square(lambda / lambda0);
        // const fp_t emis_opac_ratio = lambda0 / lambda;
        const fp_t a_damp_ratio = a_damp_ratios(krl, wave);
        const fp_t emis_opac_ratio = emis_opac_ratios(krl, wave);
        const fp_t lambda0 = lambda0s(krl);
        a *= a_damp_ratio;

        const fp_t v = ((lambda - lambda0) + (vel * lambda0) * inv_c) * inv_dop;
        const fp_t p = phi(a, v) * inv_sqrt_pi * inv_dop;

        EmisOpac result {
            .eta = emis_opac_ratio * eta_s * p,
            .chi = emis_opac_ratio * chi_s * p
        };
        return result;
    }

    void compute_mips(const State& state, const MipmapComputeState& mm_state) {
        JasUnpack(state, mr_block_map, adata, phi);
        const auto& block_map = mr_block_map.block_map;
        JasUnpack(mm_state, max_mip_factor, mippable_entries, emis, opac);
        constexpr i32 mip_block = 4;
        const fp_t vox_scale = state.atmos.voxel_scale;
        const i32 wave_batch = emis.extent(1);

        for (int level_m_1 = 0; level_m_1 < max_mip_factor; ++level_m_1) {
            const i32 vox_size = (1 << (level_m_1 + 1));
            auto bounds = block_map.loop_bounds(vox_size);
            parallel_for(
                "Compute mip n (CoreAndVoigt)",
                SimpleBounds<3>(bounds.dim(0), bounds.dim(1), wave_batch),
                YAKL_CLASS_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
                    const i32 level = level_m_1 + 1;
                    const fp_t ds = vox_scale;
                    MRIdxGen idx_gen(mr_block_map);

                    const i64 ks = idx_gen.loop_idx(level, tile_idx, block_idx);
                    const Coord2 coord = idx_gen.loop_coord(level, tile_idx, block_idx);
                    const Coord2 tile_coord = idx_gen.compute_tile_coord(tile_idx);

                    const i32 upper_vox_size = vox_size / 2;
                    i64 idxs[mip_block] = {
                        idx_gen.idx(level_m_1, coord.x, coord.z),
                        idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z),
                        idx_gen.idx(level_m_1, coord.x, coord.z+upper_vox_size),
                        idx_gen.idx(level_m_1, coord.x+upper_vox_size, coord.z+upper_vox_size)
                    };
                    yakl::SArray<fp_t, 1, CORE_AND_VOIGT_MAX_LINES> eta_star_mip;
                    yakl::SArray<fp_t, 1, CORE_AND_VOIGT_MAX_LINES> chi_star_mip;
                    yakl::SArray<fp_t, 1, CORE_AND_VOIGT_MAX_LINES> a_damp_mip;
                    yakl::SArray<fp_t, 1, CORE_AND_VOIGT_MAX_LINES> inv_dop_width_mip;
                    if (wave == 0) {
                        for (int i = 0; i < CORE_AND_VOIGT_MAX_LINES; ++i) {
                            eta_star_mip(i) = FP(0.0);
                            chi_star_mip(i) = FP(0.0);
                            a_damp_mip(i) = FP(0.0);
                            inv_dop_width_mip(i) = FP(0.0);
                        }
                    }

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
                        fp_t tot_emis = emis(idx, wave) + FP(1e-15);
                        fp_t tot_opac = opac(idx, wave) + FP(1e-15);

                        for (int kri = 0; kri < CORE_AND_VOIGT_MAX_LINES; ++kri) {
                            i32 krl = active_set_mapping(kri);
                            if (krl < 0) {
                                break;
                            }
                            // NOTE(cmo): Evaluate at line core as an overestimate
                            EmisOpac line = emis_opac(CavEmisOpacState{
                                .ks = idx,
                                .krl = krl,
                                .wave = wave,
                                .lambda = lambda0s(kri),
                                .vel = FP(0.0),
                                .phi = phi
                            });
                            tot_emis += line.eta;
                            tot_opac += line.chi;

                            if (wave == 0) {
                                eta_star_mip(krl) += eta_star(idx, krl);
                                chi_star_mip(krl) += chi_star(idx, krl);
                                a_damp_mip(krl) += a_damp(idx, krl);
                                inv_dop_width_mip(krl) += inv_dop_width(idx, krl);
                            }
                        }

                        consider_variance = consider_variance || (tot_opac * ds) > opacity_threshold;
                        m1_chi += std::log(tot_opac * ds);
                        m2_chi += square(std::log(tot_opac * ds));
                        m1_eta += std::log(tot_emis * ds);
                        m2_eta += square(std::log(tot_emis * ds));
                    }
                    m1_chi *= FP(0.25);
                    m2_chi *= FP(0.25);
                    m1_eta *= FP(0.25);
                    m2_eta *= FP(0.25);
                    if (wave == 0) {
                        for (int i = 0; i < CORE_AND_VOIGT_MAX_LINES; ++i) {
                            eta_star_mip(i) *= FP(0.25);
                            chi_star_mip(i) *= FP(0.25);
                            a_damp_mip(i) *= FP(0.25);
                            inv_dop_width_mip(i) *= FP(0.25);
                        }
                    }

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
                    if (do_increment) {
                        yakl::atomicAdd(mippable_entries(tile_idx), 1);
                    }

                    if (wave == 0) {
                        for (int kri = 0; kri < CORE_AND_VOIGT_MAX_LINES; ++kri) {
                            i32 krl = active_set_mapping(kri);
                            if (krl < 0) {
                                break;
                            }
                            eta_star(ks, kri) = eta_star_mip(kri);
                            chi_star(ks, kri) = chi_star_mip(kri);
                            a_damp(ks, kri) = a_damp_mip(kri);
                            inv_dop_width(ks, kri) = inv_dop_width_mip(kri);
                        }
                    }
                }
            );
            yakl::fence();
        }

    }

    void compute_subset_mips(
        const State& state,
        const MipmapSubsetState& mm_state,
        const CascadeCalcSubset& subset
    ) {

    }

};


#else
#endif