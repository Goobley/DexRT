#if !defined(DEXRT_CORE_AND_VOIGT_EMIS_OPAC_HPP)
#define DEXRT_CORE_AND_VOIGT_EMIS_OPAC_HPP

#include "Config.hpp"
#include "EmisOpac.hpp"
#include "State.hpp"

namespace CoreAndVoigt {
    struct CoreAndVoigtState {
        i32 la;
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
        JasUnpack(args, la, k, atmos, adata, n);
        const fp_t lambda = adata.wavelength(la);
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
        const fp_t hnu_4pi = hc_kJ_nm / (four_pi * lambda);
        const fp_t a = damping_from_gamma(params.gamma, lambda, params.dop_width);
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
    i32 kr;
    i32 wave;
    fp_t lambda0;
    fp_t lambda;
    fp_t vel;
    const VoigtProfile<fp_t, false>& phi;
};

struct CoreAndVoigtData {
    // NOTE(cmo): Right now these aren't unified over wave, in practice, they
    // probably can be, since most of the wavelength varying parameters are
    // pretty insignificant other than over massive ranges.
    Fp3d eta_star; // [ks, kr, wave]
    Fp3d chi_star; // [ks, kr, wave]
    Fp3d a_damp; // [ks, kr, wave]
    Fp3d inv_dop_width; // [ks, kr, wave]

    void init(i64 buffer_len, i32 max_kr, i32 wave_batch) {
        eta_star = Fp3d("eta_star", buffer_len, max_kr, wave_batch);
        chi_star = Fp3d("chi_star", buffer_len, max_kr, wave_batch);
        a_damp = Fp3d("a_damp", buffer_len, max_kr, wave_batch);
        inv_dop_width = Fp3d("1 / dop_width", buffer_len, max_kr, wave_batch);
    }

    /// Fills mip0
    void fill(const State& state, i32 la_start, i32 la_end) {
        JasUnpack(state, atmos, pops, adata, mr_block_map);
        JasUnpack((*this), eta_star, chi_star, a_damp, inv_dop_width);
        i32 wave_batch = la_end - la_start;
        auto& block_map = mr_block_map.block_map;
        const auto& flatmos = flatten<const fp_t>(atmos);
        const auto& flat_pops = pops.reshape<2>(Dims(pops.extent(0), pops.extent(1) * pops.extent(2)));

        auto bounds = block_map.loop_bounds();
        parallel_for(
            "fill core and voigt",
            SimpleBounds<3>(bounds.dim(0), bounds.dim(1), wave_batch),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx, i32 wave) {
                IdxGen idx_gen(block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
                i64 kf = idx_gen.full_flat_idx(coord.x, coord.z);
                const i32 la = la_start + wave;

                auto a_set = slice_active_set(adata, la);

                AtmosPointParams atmos_point {
                    .temperature = flatmos.temperature(kf),
                    .ne = flatmos.ne(kf),
                    .vturb = flatmos.vturb(kf),
                    .nhtot = flatmos.nh_tot(kf),
                    .nh0 = flatmos.nh0(kf)
                };

                CoreAndVoigt::CoreAndVoigtState line_state {
                    .la = la,
                    .k = kf,
                    .atmos = atmos_point,
                    .adata = adata,
                    .n = flat_pops
                };
                for (int kra = 0; kra < a_set.extent(0); ++kra) {
                    CoreAndVoigt::CoreAndVoigtResult line_data = CoreAndVoigt::compute_core_and_voigt(
                        line_state,
                        a_set(kra)
                    );
                    eta_star(ks, kra, wave) = line_data.eta_star;
                    chi_star(ks, kra, wave) = line_data.chi_star;
                    a_damp(ks, kra, wave) = line_data.a_damp;
                    inv_dop_width(ks, kra, wave) = FP(1.0) / line_data.dop_width;
                }
            }
        );
    }

    YAKL_INLINE EmisOpac emis_opac(
        const CavEmisOpacState& args
    ) const {
        JasUnpack(args, ks, kr, wave, lambda, lambda0, vel, phi);

        using namespace ConstantsFP;
#ifdef DEXRT_DEBUG
        const fp_t a = a_damp(ks, kr, wave);
        const fp_t inv_dop = inv_dop_width(ks, kr, wave);
        const fp_t v = ((lambda - lambda0) + (vel * lambda0) / c) * inv_dop;
        const fp_t p = phi(a, v) / sqrt_pi * inv_dop;
        const fp_t eta_s = eta_star(ks, kr, wave);
        const fp_t chi_s = chi_star(ks, kr, wave);
#else
        constexpr fp_t inv_c = FP(1.0) / c;
        constexpr fp_t inv_sqrt_pi = FP(1.0) / sqrt_pi;
        const i64 idx = (ks * a_damp.extent(1) + kr) * a_damp.extent(2) + wave;
        const fp_t a = a_damp.get_data()[idx];
        const fp_t inv_dop = inv_dop_width.get_data()[idx];
        const fp_t v = ((lambda - lambda0) + (vel * lambda0) * inv_c) * inv_dop;
        const fp_t p = phi(a, v) * inv_sqrt_pi * inv_dop;
        const fp_t eta_s = eta_star.get_data()[idx];
        const fp_t chi_s = chi_star.get_data()[idx];
#endif

        EmisOpac result {
            .eta = eta_s * p,
            .chi = chi_s * p
        };
        return result;
    }

};


#else
#endif