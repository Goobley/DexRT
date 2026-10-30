#if !defined(DEXRT_SC_MEDIUM_COEFFICIENTS_HPP)
#define DEXRT_SC_MEDIUM_COEFFICIENTS_HPP

#include "Types.hpp"
#include "ScState.hpp"
#include "Voigt.hpp"
#include "Constants.hpp"
#include "EmisOpac.hpp"
#include "ScConfig.hpp"
#include "ScPrd.hpp"
#include <vector>

/// Per-cell emissivity/opacity data for a wavelength batch of any size, from
/// which the direction dependent (Doppler shifted) values are evaluated
/// cheaply: the continua at each wavelength, and per active line its
/// coefficients at line centre (as in CoreAndVoigt, without its compile-time
/// batch/line limits).
struct ScMediumCoefficients {
    i32 la_start = 0;
    i32 wave_batch = 0;
    i32 num_lines = 0; /// lines active anywhere in the batch
    i32 max_lines_per_wave = 0;
    // NOTE(cmo): Cell index last, so neighbouring threads (cells) read
    // contiguous memory.
    Fp2d cont_eta;                                /// [wave, ks]
    Fp2d cont_chi;                                /// [wave, ks]
    Fp2d eta_star;                                /// [line, ks]
    Fp2d chi_star;                                /// [line, ks]
    Fp2d a_damp;                                  /// [line, ks]
    Fp2d inv_dop_width;                           /// [line, ks]
    Fp1d lambda0;                                 /// [line]
    yakl::Array<i32, 1, yakl::memDevice> line_kr; /// [line] index into the atomic data
    yakl::Array<i32, 2, yakl::memDevice>
        wave_lines; /// [wave, max_lines_per_wave] local line indices, -1 terminated
    Fp2d n_star;    /// [level, ks] LTE populations (filled with the continua)
    ScPrd prd;
    yakl::Array<i32, 1, yakl::memDevice> line_prd; /// [line] PRD line index, -1: CRD

    void init(const ScState& state, i32 wave_batch_max);
    void fill(const ScState& state, const Rad16Array& previous_J, i32 la_start, i32 la_end);

    /// Line profile [nm-1] of local line l at the G cells ks0 + c, wavelength
    /// lambda, for line-of-sight velocities vel, into p (vectorises on CPU).
    template <int G>
    YAKL_INLINE void profile(
        const VoigtProfile<fp_t, false>& phi,
        i32 l,
        i64 ks0,
        fp_t lambda,
        const fp_t* vel,
        fp_t* p,
        int n = G
    ) const {
        using namespace ConstantsFP;
        constexpr fp_t inv_c = FP(1.0) / c;
        constexpr fp_t inv_sqrt_pi = FP(1.0) / sqrt_pi;
        const fp_t l0 = lambda0(l);
        const fp_t a_scale = square(lambda / l0);
        const fp_t* __restrict a_damp_l = &a_damp(l, ks0);
        const fp_t* __restrict inv_dop_l = &inv_dop_width(l, ks0);
        fp_t a[G];
        fp_t v[G];
        fp_t H[G];
        SC_GROUP_SIMD
        for (int c = 0; c < n; ++c) {
            a[c] = a_damp_l[c] * a_scale;
            v[c] = ((lambda - l0) + (vel[c] * l0) * inv_c) * inv_dop_l[c];
        }
        phi.template eval<G>(a, v, H, n);
        SC_GROUP_SIMD
        for (int c = 0; c < n; ++c) {
            p[c] = H[c] * inv_sqrt_pi * inv_dop_l[c];
        }
    }

    /// eta, chi at the G cells ks0 + c, batch wavelength wave (lambda), for
    /// line-of-sight velocities vel.
    template <int G>
    YAKL_INLINE void evaluate(
        const VoigtProfile<fp_t, false>& phi,
        i64 ks0,
        i32 wave,
        fp_t lambda,
        const fp_t* vel,
        fp_t* eta,
        fp_t* chi,
        int n = G
    ) const {
        const fp_t* __restrict cont_eta_w = &cont_eta(wave, ks0);
        const fp_t* __restrict cont_chi_w = &cont_chi(wave, ks0);
        SC_GROUP_SIMD
        for (int c = 0; c < n; ++c) {
            eta[c] = cont_eta_w[c];
            chi[c] = cont_chi_w[c];
        }
        for (int i = 0; i < max_lines_per_wave; ++i) {
            const i32 l = wave_lines(wave, i);
            if (l < 0) {
                break;
            }
            fp_t p[G];
            profile<G>(
                phi,
                l,
                ks0,
                lambda,
                vel,
                p,
                n
            );
            const fp_t ratio = lambda0(l) / lambda;
            const fp_t* __restrict eta_star_l = &eta_star(l, ks0);
            const fp_t* __restrict chi_star_l = &chi_star(l, ks0);
            const i32 pl = line_prd(l);
            if (pl >= 0) {
                // NOTE(cmo): psi = rho phi: emission and stimulated emission.
                const fp_t stim = prd.lines(pl).stim_ratio;
                for (int c = 0; c < n; ++c) {
                    const fp_t rho = prd.rho_lab(pl, ks0 + c, lambda, vel[c]);
                    eta[c] += ratio * eta_star_l[c] * rho * p[c];
                    chi[c] += ratio * (chi_star_l[c] + (FP(1.0) - rho) * stim * eta_star_l[c]) * p[c];
                }
                continue;
            }
            SC_GROUP_SIMD
            for (int c = 0; c < n; ++c) {
                eta[c] += ratio * eta_star_l[c] * p[c];
                chi[c] += ratio * chi_star_l[c] * p[c];
            }
        }
    }
};

/// Coefficient payloads have an independent memory bound: sizeof(fp_t) * cells
/// * (2 * width + 4 * lines + levels) per entry. Selection is valid through
/// sweep and accumulation; only the next task may evict it.
struct ScMediumCoefficientCache {
    struct Entry {
        ScMediumCoefficients coefficients;
        i32 la = -1, end = -1;
        i64 touched = 0;
    };
    std::vector<Entry> entries;
    i64 clock = 0;
    i64 builds = 0, hits = 0;
    i32 current_entry = -1;

    const ScMediumCoefficients& current() const {
        return entries.at(current_entry).coefficients;
    }
};

/// Allocate the coefficient budget once; GivenFs needs no entries.
ScMediumCoefficientCache sc_make_coefficient_cache(const ScState& state, i32 count, i32 width);
/// Populations/background changed: invalidate all entries at the solve boundary.
void sc_reset_coefficient_cache(ScMediumCoefficientCache& cache);
/// Select a matching entry or rebuild the least recently used one.
const ScMediumCoefficients& sc_get_medium_coefficients(
    ScMediumCoefficientCache& cache,
    const ScState& state,
    const Rad16Array& previous_J,
    i32 la,
    i32 end
);

#else
#endif
