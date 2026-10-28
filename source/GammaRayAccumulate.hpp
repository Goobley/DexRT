#if !defined(DEXRT_GAMMA_RAY_ACCUMULATE_HPP)
#define DEXRT_GAMMA_RAY_ACCUMULATE_HPP

#include "Types.hpp"
#include "Constants.hpp"
#include "EmisOpac.hpp"
#include "GammaMatrix.hpp"
#include "CompensatedSum.hpp"
#include "RateDiagnostics.hpp"
#include "Voigt.hpp"

/// The integral weight drops 4pi/hc, so U and V are divided by hc/4pi, and the
/// eta/chi built from them multiplied back up.
constexpr bool GAMMA_INCLUDE_4PI_HC = false;

/// Wavelength part of the Gamma integration weight (excluding the angular
/// weight) - wla.
inline fp_t gamma_wavelength_weight(const AtomicData<fp_t, yakl::memHost>& adata_host, int la) {
    using namespace ConstantsFP;
    constexpr fp_t hc_4pi = hc_kJ_nm / four_pi;
    fp_t hc_4pi_eff = FP(1.0);
    if (GAMMA_INCLUDE_4PI_HC) {
        hc_4pi_eff *= hc_4pi;
    }
    return adata_host.wavelength(la) / hc_4pi_eff * adata_host.wavelength_bin(la);
}

/// Repack Gamma (and diagnostics if requested) for each atom, so it can be
/// accumulated in one kernel
struct GammaAccumArrayState {
    PreconRMat precon_R[GAMMA_MAX_ATOMS];
    PreconRMat rad_rates[GAMMA_MAX_ATOMS];
    ContEnergyMat cont_absorb[GAMMA_MAX_ATOMS];
    ContEnergyMat cont_emit[GAMMA_MAX_ATOMS];
    yakl::Array<i32, 1, yakl::memDevice> level_stage_idx;
};

inline GammaAccumArrayState gamma_accum_array_state(const std::vector<PreconRMat>& precon_R, const RateDiagnostics* diag) {
    if (precon_R.size() > GAMMA_MAX_ATOMS) {
        throw std::runtime_error(fmt::format("At most {} active atoms are supported (GAMMA_MAX_ATOMS).", GAMMA_MAX_ATOMS));
    }
    GammaAccumArrayState result;
    for (int ia = 0; ia < precon_R.size(); ++ia) {
        result.precon_R[ia] = precon_R[ia];
        if (diag) {
            if (!diag->radiative_rates.empty()) {
                result.rad_rates[ia] = diag->radiative_rates[ia];
            }
            if (!diag->cont_energy_absorb.empty()) {
                result.cont_absorb[ia] = diag->cont_energy_absorb[ia];
                result.cont_emit[ia] = diag->cont_energy_emit[ia];
            }
        }
    }
    if (diag) {
        result.level_stage_idx = diag->level_stage_idx;
    }
    return result;
}

/// Static (direction independent) emissivity and opacity of a cell: the active
/// continua and raymarch's opacity floor, using compensated sums for precision
/// in fp32
struct GammaStaticSums {
    CompensatedSum eta;
    CompensatedSum chi;
    fp_t chi_neg = FP(0.0); /// sum of |chi| of the negative components -- used to keep psi* stable by dividing by sum |chi| = chi.value() + 2 * chi_neg
};

/// Data common to the different rays of a cell to build Gamma
struct GammaCell {
    const AtomicData<fp_t>& adata;
    const VoigtProfile<fp_t, false>& phi;
    const Fp2d& pops;
    const Fp2d& lte_scratch;
    const Rad16Array& wphi;
    const GammaAccumArrayState& store;
    const yakl::Array<const u16, 1, yakl::memDevice>& active_set;
    const yakl::Array<const u16, 1, yakl::memDevice>& active_cont_set;
    i64 ks;
    int la;
    fp_t wl_ray_weight; /// ray weight over wavelength and angle, component common to all rays (i.e. in RC everything but wmuy in 2D)
    GammaStaticSums statics; /// gamma_static_sums
};

/// A ray's terms from the formal solver, at the probe.
struct GammaRay {
    fp_t I_ext;
    fp_t one_minus_alo;
    fp_t weight; /// wmuy in 2D RC, 1.0 in 3D RC, varying in SC
};

/// A ray's terms derived in accumulate_gamma_ray_lines: psi* = (1 -
/// one_minus_alo) / chi and I = I_ext + psi* eta. As psi* is just part of the
/// preconditioning, it's fine for it to lose some precision in thin cells.
struct GammaRayDerived {
    fp_t psi_star;
    fp_t I;
    EmisOpac lines;
};

/// Angular integration for a set (or subset) of rays at a wavelength.
/// Pre-integrating these quantities for continua lets us have considerably
/// fewer writes into Gamma. If we're integrated over the whole quadrature w
/// sums to 1, otherwise, it's a fraction of 1. The integrated I is I_ext +
/// psi_star * eta_static + psi_eta_lines.
struct GammaRayAngularIntegral {
    fp_t w = FP(0.0);
    fp_t I_ext = FP(0.0);
    fp_t one_minus_alo = FP(0.0);
    fp_t psi_star = FP(0.0);
    fp_t psi_eta_lines = FP(0.0);
    fp_t psi_chi_lines = FP(0.0);

    YAKL_INLINE void add(const GammaRay& ray, const GammaRayDerived& d) {
        const fp_t w_psi = ray.weight * d.psi_star;
        w += ray.weight;
        I_ext += ray.weight * ray.I_ext;
        one_minus_alo += ray.weight * ray.one_minus_alo;
        psi_star += w_psi;
        psi_eta_lines += w_psi * d.lines.eta;
        psi_chi_lines += w_psi * d.lines.chi;
    }
};

YAKL_INLINE EmisOpac gamma_eta_chi(const UV& uv, fp_t ni, fp_t nj) {
    using namespace ConstantsFP;
    constexpr fp_t hc_4pi = hc_kJ_nm / four_pi;
    EmisOpac result{
        .eta = nj * uv.Uji,
        .chi = ni * uv.Vij - nj * uv.Vji
    };
    if (!GAMMA_INCLUDE_4PI_HC) {
        // NOTE(cmo): If we have dropped the hc/4pi from U and V, these still
        // need the dimensioning.
        result.eta *= hc_4pi;
        result.chi *= hc_4pi;
    }
    result.chi += FP(1e-20);
    return result;
}

/// Fill cell.statics (direction independent, so once per cell and wavelength).
YAKL_INLINE GammaStaticSums gamma_static_sums(const GammaCell& cell, const AtmosPointParams& atmos) {
    JasUnpack(cell, adata, pops, active_cont_set, ks);
    const EmisOpacState<> emis_opac{
        .adata = adata,
        .profile = cell.phi,
        .la = cell.la,
        .n = pops,
        .n_star_scratch = cell.lte_scratch,
        .k = ks,
        .atmos = atmos
    };
    GammaStaticSums result;
    // NOTE(cmo): The raymarch's opacity floor.
    result.chi.add(FP(1e-15));
    for (int kra = 0; kra < active_cont_set.extent(0); ++kra) {
        const int kr = active_cont_set(kra);
        const auto& cont = adata.continua(kr);
        const int offset = adata.level_start(cont.atom);
        const UV uv = compute_uv_cont(
            emis_opac,
            kr,
            UvOptions{.divide_by_hc_4pi = !GAMMA_INCLUDE_4PI_HC}
        );
        const EmisOpac eta_chi = gamma_eta_chi(uv, pops(offset + cont.i, ks), pops(offset + cont.j, ks));
        result.eta.add(eta_chi.eta);
        result.chi.add(eta_chi.chi);
        result.chi_neg += std::max(-eta_chi.chi, FP(0.0));
    }
    return result;
}

struct GammaLineTerms {
    UV uv;
    EmisOpac eta_chi;
};

/// Add line kr's contribution for one ray. I - psi* eta and 1 - psi* chi are
/// formed as I_ext + psi* eta_other and one_minus_alo + psi* chi_other, which
/// minimise cancellation in optically thick cells. This uses CompensatedSums on
/// f32.  Non-atomic.
YAKL_INLINE void accumulate_gamma_line(
    const GammaCell& cell,
    const GammaRay& ray,
    const GammaRayDerived& d,
    int kr,
    const GammaLineTerms& t,
    fp_t eta_other,
    fp_t chi_other
) {
    JasUnpack(cell, adata, wphi, store, ks);
    const auto& l = adata.lines(kr);
    const int trans = line_transition_idx(adata, kr);
    const fp_t wlamu = cell.wl_ray_weight * ray.weight * wphi(kr, ks);
    add_to_gamma_split<false>(
        GammaAccumState{
            .uv = t.uv,
            .psi_star = d.psi_star,
            .wlamu = wlamu,
            .R = store.precon_R[l.atom],
            .t = trans,
            .k = ks
        },
        ray.I_ext,
        ray.one_minus_alo,
        eta_other,
        chi_other
    );

    const PreconRMat& rad_rates = store.rad_rates[l.atom];
    if (rad_rates.initialized()) {
        // NOTE(cmo): psi_star = 0 => the true radiative rates.
        add_to_gamma<false>(GammaAccumState{
            .eta = t.eta_chi.eta,
            .chi = t.eta_chi.chi,
            .uv = t.uv,
            .I = d.I,
            .psi_star = FP(0.0),
            .wlamu = wlamu,
            .R = rad_rates,
            .t = trans,
            .k = ks
        });
    }
}

inline int gamma_max_active_lines(const AtomicData<fp_t, yakl::memHost>& adata, int la_start, int la_end) {
    int result = 0;
    for (int la = la_start; la < la_end; ++la) {
        result = std::max(result, int(adata.active_lines_end(la) - adata.active_lines_start(la)));
    }
    return result;
}

/// Add every active line's contribution (all atoms) for one ray. Non-atomic.
/// Each line's other terms, and psi*, need the ray's totals first, so the lines
/// are visited twice. To minimise this cost, we cache up to MaxLines profiles.
/// Setting MaxLines to 0 avoids the cache, for the case where there are many
/// lines. Anticipated to dispatch this on the same basis as CoreAndVoigt
/// Returns the derived ray quantities including total line opacity.
template <int MaxLines>
YAKL_INLINE GammaRayDerived accumulate_gamma_ray_lines(const GammaCell& cell, const AtmosPointParams& atmos, const GammaRay& ray) {
    JasUnpack(cell, adata, pops, active_set, ks);
    const EmisOpacState<> emis_opac{
        .adata = adata,
        .profile = cell.phi,
        .la = cell.la,
        .n = pops,
        .n_star_scratch = cell.lte_scratch,
        .k = ks,
        .atmos = atmos
    };
    const UvOptions uv_opts{.divide_by_hc_4pi = !GAMMA_INCLUDE_4PI_HC};
    const fp_t lambda = adata.wavelength(cell.la);
    const int num_lines = active_set.extent(0);
    auto line_terms_from_profile = [&](int kr, fp_t p) {
        const auto& l = adata.lines(kr);
        const int offset = adata.level_start(l.atom);
        GammaLineTerms t;
        t.uv = compute_uv_from_profile(l, p, lambda, uv_opts);
        t.eta_chi = gamma_eta_chi(t.uv, pops(offset + l.i, ks), pops(offset + l.j, ks));
        return t;
    };

    GammaRayDerived d;
    d.lines = EmisOpac{FP(0.0), FP(0.0)};
    CompensatedSum eta_total = cell.statics.eta;
    CompensatedSum chi_total = cell.statics.chi;
    fp_t chi_neg = cell.statics.chi_neg;
    auto add_to_totals = [&](const EmisOpac& eta_chi) {
        eta_total.add(eta_chi.eta);
        chi_total.add(eta_chi.chi);
        chi_neg += std::max(-eta_chi.chi, FP(0.0));
        d.lines.eta += eta_chi.eta;
        d.lines.chi += eta_chi.chi;
    };
    fp_t profiles[MaxLines > 0 ? MaxLines : 1];
    if constexpr (MaxLines > 0) {
        #pragma unroll
        for (int kra = 0; kra < MaxLines; ++kra) {
            if (kra < num_lines) {
                const int kr = active_set(kra);
                profiles[kra] = compute_line_profile(emis_opac, kr);
                add_to_totals(line_terms_from_profile(kr, profiles[kra]).eta_chi);
            }
        }
    } else {
        for (int kra = 0; kra < num_lines; ++kra) {
            const int kr = active_set(kra);
            add_to_totals(line_terms_from_profile(kr, compute_line_profile(emis_opac, kr)).eta_chi);
        }
    }
    // NOTE(cmo): psi* = ALO / sum |chi_k| (= ALO / chi_tot when no component is
    // negative): a laser line can cancel the rest of chi_tot, and this keeps
    // every psi* |chi_k| <= ALO (which is clamped to [0, 1]).
    const fp_t alo = std::min(std::max(FP(1.0) - ray.one_minus_alo, FP(0.0)), FP(1.0));
    d.psi_star = alo / (chi_total.value() + FP(2.0) * chi_neg);
    d.I = ray.I_ext + d.psi_star * eta_total.value();

    auto accumulate = [&](int kr, fp_t p) {
        const GammaLineTerms t = line_terms_from_profile(kr, p);
        accumulate_gamma_line(
            cell,
            ray,
            d,
            kr,
            t,
            eta_total.minus(t.eta_chi.eta),
            chi_total.minus(t.eta_chi.chi)
        );
    };
    if constexpr (MaxLines > 0) {
        #pragma unroll
        for (int kra = 0; kra < MaxLines; ++kra) {
            if (kra < num_lines) {
                accumulate(active_set(kra), profiles[kra]);
            }
        }
    } else {
        for (int kra = 0; kra < num_lines; ++kra) {
            const int kr = active_set(kra);
            accumulate(kr, compute_line_profile(emis_opac, kr));
        }
    }
    return d;
}

/// Add every active continuum's contribution (all atoms) from the preintegrated angular
/// integral of the rays through the cell Non-atomic.
YAKL_INLINE void accumulate_gamma_continua(const GammaCell& cell, const AtmosPointParams& atmos, const GammaRayAngularIntegral& pre_int) {
    using namespace ConstantsFP;
    JasUnpack(cell, adata, pops, store, active_cont_set, ks, la);
    const EmisOpacState<> emis_opac{
        .adata = adata,
        .profile = cell.phi,
        .la = la,
        .n = pops,
        .n_star_scratch = cell.lte_scratch,
        .k = ks,
        .atmos = atmos
    };
    for (int kra = 0; kra < active_cont_set.extent(0); ++kra) {
        const int kr = active_cont_set(kra);
        const auto& cont = adata.continua(kr);
        const int trans = cont_transition_idx(adata, kr);
        const int offset = adata.level_start(cont.atom);
        const UV uv = compute_uv_cont(
            emis_opac,
            kr,
            UvOptions{.divide_by_hc_4pi = !GAMMA_INCLUDE_4PI_HC}
        );
        const fp_t ni = pops(offset + cont.i, ks);
        const fp_t nj = pops(offset + cont.j, ks);
        const EmisOpac eta_chi = gamma_eta_chi(uv, ni, nj);
        const fp_t wlamu = cell.wl_ray_weight;
        const GammaAccumState acc{
            .uv = uv,
            .wlamu = wlamu,
            .R = store.precon_R[cont.atom],
            .t = trans,
            .k = ks
        };
        // I_eff = I - psi* eta_trans = (I_ext + psi* eta_tot) - psi* eta_trans
        //       = I_ext + psi* (eta_tot - eta_trans) = I_ext + psi* eta_other_cont + psi* eta_lines
        // eta_other formed by compensated sum to avoid roundoff issues
        // Now, psi* = Lambda* / chi_tot
        // one_m_psi_chi = 1 - chi_trans psi*  = 1 - (chi_tot - chi_other) psi*
        //               = 1 - chi_tot psi* + chi_other psi* = (1 - Lambda*) + chi_other psi*
        add_gamma_terms<false>(
            acc,
            pre_int.I_ext + pre_int.psi_star * cell.statics.eta.minus(eta_chi.eta) + pre_int.psi_eta_lines,
            pre_int.one_minus_alo + pre_int.psi_star * cell.statics.chi.minus(eta_chi.chi) + pre_int.psi_chi_lines
        );

        const fp_t I = pre_int.I_ext + pre_int.psi_star * cell.statics.eta.value() + pre_int.psi_eta_lines;
        const PreconRMat& rad_rates = store.rad_rates[cont.atom];
        if (rad_rates.initialized()) {
            // NOTE(cmo): psi_star = 0 => the true radiative rates.
            add_gamma_terms<false>(
                GammaAccumState{
                    .uv = uv,
                    .wlamu = wlamu,
                    .R = rad_rates,
                    .t = trans,
                    .k = ks
                },
                I,
                pre_int.w
            );
        }

        const ContEnergyMat& cont_absorb = store.cont_absorb[cont.atom];
        if (cont_absorb.initialized()) {
            // NOTE(claude): wlamu * uv.V is the number rate integrand, 4pi
            // sigma dlambda / (h nu), under either setting of
            // GAMMA_INCLUDE_4PI_HC: the 4pi/hc sits in V when it is dropped
            // from the weights, and in the weights when it is not. So
            // multiplying through by h nu leaves the energy rate, 4pi sigma I
            // dlambda, either way. Times the population to make it volumetric.
            const fp_t w_energy = wlamu * hc_kJ_nm / adata.wavelength(la);
            const int c = store.level_stage_idx(offset + cont.j);
            cont_absorb(c, cont.i, ks) += w_energy * ni * uv.Vij * I;
            store.cont_emit[cont.atom](c, cont.i, ks) += w_energy * nj * (uv.Uji * pre_int.w + uv.Vji * I);
        }
    }
}

#else
#endif
