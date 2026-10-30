#ifndef DEXRT_SC_GAMMA_ACCUMULATE_HPP
#define DEXRT_SC_GAMMA_ACCUMULATE_HPP
#include "GammaRayAccumulate.hpp"
#include "ScMediumCoefficients.hpp"
#include "ScConfig.hpp"

/// SC-specific storage/profile adapter. Transition algebra, sums, response
/// recovery, diagnostics and continuum accumulation are shared with RC.
struct ScGammaCell {
    const AtomicData<fp_t>& adata;
    const VoigtProfile<fp_t, false>& phi;
    const Fp2d& pops;
    const Fp2d& lte_scratch;
    const Rad16Array& wphi;
    const GammaAccumArrayState& out;
    const yakl::Array<const u16, 1, yakl::memDevice>& active_set;
    const yakl::Array<const u16, 1, yakl::memDevice>& active_cont_set;
    const AtmosPointParams& atmos; /// vel unused
    i64 ks;
    int la;
    fp_t wl_weight; /// gamma_wavelength_weight
    /// Per-cell line coefficients, used if SC_GAMMA_CACHED_LINES
    const ScMediumCoefficients* line_cache;
    int wave; /// index of la in line_cache's batch
    /// Background + all active continua (sc_gamma_static_terms)
    GammaStaticSums statics;
};


YAKL_INLINE GammaCell sc_gamma_cell(const ScGammaCell& cell, i64 ks, const GammaStaticSums& statics) {
    return {cell.adata, cell.phi, cell.pops, cell.lte_scratch, cell.wphi, cell.out,
            cell.active_set, cell.active_cont_set, ks, cell.la, cell.wl_weight, statics};
}
YAKL_INLINE void sc_gamma_static_terms(ScGammaCell* cell, fp_t eta_background, fp_t chi_background) {
    // SC transport has no RC raymarch floor; use the same summation machinery
    // with its actual transport convention rather than inventing a new floor.
    cell->statics = gamma_static_sums(sc_gamma_cell(*cell, cell->ks, {}), cell->atmos, FP(0.0));
    cell->statics.add({eta_background, chi_background});
}

template <int G>
struct ScGammaRays {
    fp_t weight;
    fp_t vel[G];
    fp_t I_ext[G];
    fp_t one_minus_alo[G];
};

/// U and V of a line for a group of G rays (structure of arrays).
template <int G>
struct ScLineUv {
    fp_t Uji[G];
    fp_t Vji[G];
    fp_t Vij[G];
};

/// Line kr's (the kra-th active line's) U and V for one direction's rays
/// through the cells ks = cell0.ks + c, with line-of-sight velocities vel
/// (atmos: the cells' parameters).
template <int G>
YAKL_INLINE void sc_line_uv(const ScGammaCell& cell0,
    const AtmosPointParams* atmos,
    const fp_t* vel,
    int kra,
    int kr,
    const UvOptions& opts,
    ScLineUv<G>* uv,
    int n = G) {
    using namespace ConstantsFP;
    const i64 ks0 = cell0.ks;
    const fp_t lambda = cell0.adata.wavelength(cell0.la);
    if constexpr (SC_GAMMA_CACHED_LINES) {
        const ScMediumCoefficients& lines = *cell0.line_cache;
        const int l = lines.wave_lines(cell0.wave, kra);
        fp_t hnu_4pi = FP(1.0) / lambda;
        if (!opts.divide_by_hc_4pi) {
            hnu_4pi *= hc_kJ_nm / four_pi;
        }
        const auto& line = cell0.adata.lines(kr);
        const fp_t Bij = hnu_4pi * line.Bij;
        const fp_t Bji = hnu_4pi * line.Bji;
        const fp_t Aji = hnu_4pi * line.Aji;
        fp_t p[G];
        lines.template profile<G>(cell0.phi, l, ks0, lambda, vel, p, n);
        SC_GROUP_SIMD
        for (int c = 0; c < n; ++c) {
            uv->Vij[c] = Bij * p[c];
            uv->Vji[c] = Bji * p[c];
            uv->Uji[c] = Aji * p[c];
        }
    } else {
        for (int c = 0; c < n; ++c) {
            AtmosPointParams ray_atmos = atmos[c];
            ray_atmos.vel = vel[c];
            const EmisOpacState<> eo{
                .adata = cell0.adata,
                .profile = cell0.phi,
                .la = cell0.la,
                .n = cell0.pops,
                .n_star_scratch = cell0.lte_scratch,
                .k = ks0 + c,
                .atmos = ray_atmos
            };
            const UV r = compute_uv_line(eo, kr, opts);
            uv->Uji[c] = r.Uji;
            uv->Vji[c] = r.Vji;
            uv->Vij[c] = r.Vij;
        }
    }
    const ScPrd& prd = cell0.line_cache->prd;
    const i32 pl = prd.enabled() ? prd.line_of_kr(kr) : -1;
    if (pl >= 0) {
        // NOTE(cmo): psi = rho phi in the emission terms.
        for (int c = 0; c < n; ++c) {
            const fp_t rho = prd.rho_lab(pl, ks0 + c, lambda, vel[c]);
            uv->Uji[c] *= rho;
            uv->Vji[c] *= rho;
        }
    }
}


/// A worker owns a cell group. Keep profile evaluation grouped for CPU SIMD;
/// use RC's numerical machinery directly for each cell's transition terms.
template <int G>
YAKL_INLINE void sc_accumulate_gamma_rays(const ScGammaCell& cell0,
    const AtmosPointParams* atmos, const GammaStaticSums* statics,
    const ScGammaRays<G>& rays, GammaRayAngularIntegral* moments, int n = G) {
    JasUnpack(cell0, adata, pops, active_set);
    const UvOptions options{.divide_by_hc_4pi = !GAMMA_INCLUDE_4PI_HC};
    GammaStaticSums totals[G];
    GammaRayDerived derived[G]{};
    for (int c = 0; c < n; ++c) {
        totals[c] = statics[c];
    }
    ScLineUv<G> uv;
    for (int kra = 0; kra < active_set.extent(0); ++kra) {
        const int kr = active_set(kra);
        const auto& line = adata.lines(kr);
        const int offset = adata.level_start(line.atom);
        sc_line_uv<G>(cell0, atmos, rays.vel, kra, kr, options, &uv, n);
        SC_GROUP_SIMD
        for (int c = 0; c < n; ++c) {
            const auto coefficients = gamma_eta_chi({uv.Uji[c], uv.Vji[c], uv.Vij[c]},
                pops(offset + line.i, cell0.ks + c), pops(offset + line.j, cell0.ks + c));
            totals[c].add(coefficients);
            derived[c].lines.eta += coefficients.eta;
            derived[c].lines.chi += coefficients.chi;
        }
    }
    for (int c = 0; c < n; ++c) {
        const GammaRay ray{rays.I_ext[c], rays.one_minus_alo[c], rays.weight};
        const auto recovered = gamma_recover_response(ray, totals[c].eta, totals[c].chi, totals[c].chi_neg);
        derived[c].psi_star = recovered.psi_star;
        derived[c].I = recovered.I;
        moments[c].add(ray, derived[c]);
    }
    for (int kra = 0; kra < active_set.extent(0); ++kra) {
        const int kr = active_set(kra);
        const auto& line = adata.lines(kr);
        const int offset = adata.level_start(line.atom);
        // A single line is still in uv from the totals pass.
        if (active_set.extent(0) > 1) {
            sc_line_uv<G>(cell0, atmos, rays.vel, kra, kr, options, &uv, n);
        }
        SC_GROUP_SIMD
        for (int c = 0; c < n; ++c) {
            const i64 ks = cell0.ks + c;
            const UV uv_cell{uv.Uji[c], uv.Vji[c], uv.Vij[c]};
            const auto coefficients = gamma_eta_chi(uv_cell, pops(offset + line.i, ks), pops(offset + line.j, ks));
            const GammaLineTerms terms{uv_cell, coefficients};
            accumulate_gamma_line(sc_gamma_cell(cell0, ks, statics[c]),
                {rays.I_ext[c], rays.one_minus_alo[c], rays.weight}, derived[c], kr, terms,
                totals[c].eta.minus(coefficients.eta), totals[c].chi.minus(coefficients.chi));
        }
    }
}
#endif
