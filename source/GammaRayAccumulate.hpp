#if !defined(DEXRT_GAMMA_RAY_ACCUMULATE_HPP)
#define DEXRT_GAMMA_RAY_ACCUMULATE_HPP

#include "Types.hpp"
#include "Constants.hpp"
#include "EmisOpac.hpp"
#include "GammaMatrix.hpp"
#include "RateDiagnostics.hpp"
#include "Voigt.hpp"

/// The integral weight drops 4pi/hc, so U and V are divided by hc/4pi, and the
/// eta/chi built from them multiplied back up.
constexpr bool GAMMA_INCLUDE_4PI_HC = false;

/// Avoid integrating continuum bins through their edges on the midpoint
/// integration if the bins get wide.
constexpr bool LIMIT_CONT_EDGE_BINS = true;

/// Correction for the global wavelength bin at la to the continuum's true bin.
YAKL_INLINE fp_t cont_edge_bin_factor(const AtomicData<fp_t>& adata, const CompCont<fp_t>& cont, int la) {
    if (!LIMIT_CONT_EDGE_BINS || (la != cont.blue_idx && la != cont.red_idx - 1) || cont.red_idx - cont.blue_idx < 2) {
        return FP(1.0);
    }
    const int lo = std::max(la - 1, int(cont.blue_idx));
    const int hi = std::min(la + 1, int(cont.red_idx) - 1);
    const fp_t local_bin = FP(0.5) * (adata.wavelength(hi) - adata.wavelength(lo));
    return local_bin / adata.wavelength_bin(la);
}

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

constexpr int GAMMA_MAX_ATOMS = 8;

/// Repack Gamma (and diagnostics if requested) for each atom, so it can be
/// accumulated in one kernel
struct GammaAccumArrayState {
    GammaMat Gamma[GAMMA_MAX_ATOMS];
    GammaMat rad_rates[GAMMA_MAX_ATOMS];
    ContEnergyMat cont_absorb[GAMMA_MAX_ATOMS];
    ContEnergyMat cont_emit[GAMMA_MAX_ATOMS];
    yakl::Array<i32, 1, yakl::memDevice> level_stage_idx;
};

inline GammaAccumArrayState gamma_accum_array_state(const std::vector<GammaMat>& Gamma, const RateDiagnostics* diag) {
    if (Gamma.size() > GAMMA_MAX_ATOMS) {
        throw std::runtime_error(fmt::format("At most {} active atoms are supported (GAMMA_MAX_ATOMS).", GAMMA_MAX_ATOMS));
    }
    GammaAccumArrayState result;
    for (int ia = 0; ia < Gamma.size(); ++ia) {
        result.Gamma[ia] = Gamma[ia];
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

/// All the data shared between different directional rays in a cell for the Gamma accumulation
struct GammaCell {
    const AtomicData<fp_t>& adata;
    const VoigtProfile<fp_t, false>& phi;
    const Fp2d& pops;
    const Fp2d& lte_scratch;
    const Fp2d& wphi;
    const GammaAccumArrayState& store;
    const yakl::Array<const u16, 1, yakl::memDevice>& active_set;
    const yakl::Array<const u16, 1, yakl::memDevice>& active_cont_set;
    i64 ks;
    int la;
    fp_t wl_ray_weight; /// ray weight over wavelength and angle, component common to all rays (i.e. in RC everything but wmuy in 2D)
};

struct GammaRay {
    fp_t I;
    fp_t psi_star;
    fp_t weight; /// wmuy in 2D RC
};

/// Angular integration for a set (or subset) of rays at a wavelength.
/// Pre-integrating these quantities for continua lets us have considerably
/// fewer writes into Gamma. If we're integrated over the whole quadrature w
/// sums to 1, otherwise, it's a fraction of 1.
struct GammaRayAngularIntegral {
    fp_t w = FP(0.0);
    fp_t I = FP(0.0);
    fp_t psi_star = FP(0.0);

    YAKL_INLINE void add(const GammaRay& ray) {
        w += ray.weight;
        I += ray.weight * ray.I;
        psi_star += ray.weight * ray.psi_star;
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

/// Add every active line's contribution (all atoms) for one ray. Non-atomic.
YAKL_INLINE void accumulate_gamma_ray_lines(const GammaCell& cell, const AtmosPointParams& atmos, const GammaRay& ray) {
    JasUnpack(cell, adata, pops, wphi, store, active_set, ks);
    const EmisOpacState<> emis_opac{
        .adata = adata,
        .profile = cell.phi,
        .la = cell.la,
        .n = pops,
        .n_star_scratch = cell.lte_scratch,
        .k = ks,
        .atmos = atmos
    };
    for (int kra = 0; kra < active_set.extent(0); ++kra) {
        const int kr = active_set(kra);
        const auto& l = adata.lines(kr);
        const int offset = adata.level_start(l.atom);
        const UV uv = compute_uv_line(
            emis_opac,
            kr,
            UvOptions{.divide_by_hc_4pi = !GAMMA_INCLUDE_4PI_HC}
        );
        const EmisOpac eta_chi = gamma_eta_chi(uv, pops(offset + l.i, ks), pops(offset + l.j, ks));
        const fp_t wlamu = cell.wl_ray_weight * ray.weight * wphi(kr, ks);
        add_to_gamma<false>(GammaAccumState{
            .eta = eta_chi.eta,
            .chi = eta_chi.chi,
            .uv = uv,
            .I = ray.I,
            .psi_star = ray.psi_star,
            .wlamu = wlamu,
            .Gamma = store.Gamma[l.atom],
            .i = l.i,
            .j = l.j,
            .k = ks
        });

        const GammaMat& rad_rates = store.rad_rates[l.atom];
        if (rad_rates.initialized()) {
            // NOTE(cmo): psi_star = 0 => the true radiative rates.
            add_to_gamma<false>(GammaAccumState{
                .eta = eta_chi.eta,
                .chi = eta_chi.chi,
                .uv = uv,
                .I = ray.I,
                .psi_star = FP(0.0),
                .wlamu = wlamu,
                .Gamma = rad_rates,
                .i = l.i,
                .j = l.j,
                .k = ks
            });
        }
    }
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
        const int offset = adata.level_start(cont.atom);
        const UV uv = compute_uv_cont(
            emis_opac,
            kr,
            UvOptions{.divide_by_hc_4pi = !GAMMA_INCLUDE_4PI_HC}
        );
        const fp_t ni = pops(offset + cont.i, ks);
        const fp_t nj = pops(offset + cont.j, ks);
        const EmisOpac eta_chi = gamma_eta_chi(uv, ni, nj);
        const fp_t wlamu = cell.wl_ray_weight * cont_edge_bin_factor(adata, cont, la);
        add_gamma_terms<false>(
            GammaAccumState{
                .uv = uv,
                .wlamu = wlamu,
                .Gamma = store.Gamma[cont.atom],
                .i = cont.i,
                .j = cont.j,
                .k = ks
            },
            pre_int.I - pre_int.psi_star * eta_chi.eta,
            pre_int.w - pre_int.psi_star * eta_chi.chi
        );

        const GammaMat& rad_rates = store.rad_rates[cont.atom];
        if (rad_rates.initialized()) {
            // NOTE(cmo): psi_star = 0 => the true radiative rates.
            add_gamma_terms<false>(
                GammaAccumState{
                    .uv = uv,
                    .wlamu = wlamu,
                    .Gamma = rad_rates,
                    .i = cont.i,
                    .j = cont.j,
                    .k = ks
                },
                pre_int.I,
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
            const fp_t w = wlamu * hc_kJ_nm / adata.wavelength(la);
            const int c = store.level_stage_idx(offset + cont.j);
            cont_absorb(c, cont.i, ks) += w * ni * uv.Vij * pre_int.I;
            store.cont_emit[cont.atom](c, cont.i, ks) += w * nj * (uv.Uji * pre_int.w + uv.Vji * pre_int.I);
        }
    }
}

#else
#endif
