#if !defined(DEXRT_CORE_AND_VOIGT_EMIS_OPAC_HPP)
#define DEXRT_CORE_AND_VOIGT_EMIS_OPAC_HPP

#include "Config.hpp"
#include "EmisOpac.hpp"
#include "State.hpp"

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

    void init(i64 buffer_len, i32 max_kr);
    /// Fills mip0
    void fill(const State& state, i32 la_start, i32 la_end) const;

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

    void compute_mip_n(const State& state, const MipmapComputeState& mm_state, i32 level) const;
    inline void compute_subset_mip_n(
        const State& state,
        const MipmapSubsetState& mm_state,
        const CascadeCalcSubset& subset,
        i32 level
    ) const {
    }
};


#else
#endif