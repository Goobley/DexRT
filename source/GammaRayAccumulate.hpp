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

/// Integrate each continuum with weights restricted to its own wavelength
/// range (trapezoidal half bins at its edges) instead of the global bins,
/// which extend past the threshold where the edge cross-section is largest.
constexpr bool LIMIT_CONT_EDGE_BINS = true;

/// Factor converting the global wavelength bin at la to the continuum's own.
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
/// weight). Matches the convention required by accumulate_gamma_ray.
inline fp_t gamma_wavelength_weight(const AtomicData<fp_t, yakl::memHost>& adata_host, int la) {
    using namespace ConstantsFP;
    constexpr fp_t hc_4pi = hc_kJ_nm / four_pi;
    fp_t hc_4pi_eff = FP(1.0);
    if (GAMMA_INCLUDE_4PI_HC) {
        hc_4pi_eff *= hc_4pi;
    }
    return adata_host.wavelength(la) / hc_4pi_eff * adata_host.wavelength_bin(la);
}

struct GammaRayArgs {
    int ia;
    i64 ks;
    int la;
    const AtomicData<fp_t>& adata;
    const VoigtProfile<fp_t, false>& phi;
    const Fp2d& pops;
    const Fp2d& lte_scratch;
    const Fp2d& wphi;
    const AtmosPointParams& local_atmos; /// with vel projected along this ray
    const yakl::Array<const u16, 1, yakl::memDevice>& active_set;
    const yakl::Array<const u16, 1, yakl::memDevice>& active_cont_set;
    fp_t intensity;
    fp_t psi_star;
    fp_t wl_ray_weight; /// wavelength weight (gamma_wavelength_weight) times ray weight
    fp_t dir_weight; /// any additional angular weight (e.g. wmuy)
    const GammaMat& Gamma;
    const GammaMat& rad_rates; /// optional
    const ContEnergyMat& cont_absorb; /// optional
    const ContEnergyMat& cont_emit; /// optional
    const yakl::Array<i32, 1, yakl::memDevice>& level_stage_idx;
};

/// Add the contribution of one ray (intensity, psi_star) at one wavelength and
/// cell to the Gamma of atom ia (and the optional rate diagnostics).
/// Non-atomic: the caller must own cell ks.
YAKL_INLINE void accumulate_gamma_ray(const GammaRayArgs& args) {
    JasUnpack(args, ia, ks, la, adata, phi, pops, lte_scratch, wphi, local_atmos);
    JasUnpack(args, active_set, active_cont_set, intensity, wl_ray_weight, dir_weight);
    JasUnpack(args, Gamma, rad_rates, cont_absorb, cont_emit, level_stage_idx);
    const fp_t psi_star_entry = args.psi_star;
    using namespace ConstantsFP;
    constexpr bool include_4pi_hc = GAMMA_INCLUDE_4PI_HC;
    constexpr fp_t hc_4pi = hc_kJ_nm / four_pi;

    const int offset = adata.level_start(ia);
    const int kr_base = adata.line_start(ia);
    const int max_line = kr_base + adata.num_line(ia);
    for (int kra = 0; kra < active_set.extent(0); ++kra) {
        const int kr = active_set(kra);
        if (!(kr >= kr_base && kr < max_line)) {
            continue;
        }
        const auto& l = adata.lines(kr);
        const UV uv = compute_uv_line(
            EmisOpacState<>{
                .adata = adata,
                .profile = phi,
                .la = la,
                .n = pops,
                .n_star_scratch = lte_scratch,
                .k = ks,
                .atmos = local_atmos
            },
            kr,
            UvOptions{
                .divide_by_hc_4pi = !include_4pi_hc
            }
        );

        fp_t eta = pops(offset + l.j, ks) * uv.Uji;
        fp_t chi = pops(offset + l.i, ks) * uv.Vij - pops(offset + l.j, ks) * uv.Vji;
        if (!include_4pi_hc) {
            // NOTE(cmo): If we have dropped the hc/4pi
            // from U and V, these still need the
            // dimensioning
            eta *= hc_4pi;
            chi *= hc_4pi;
        }
        chi += FP(1e-20);
        const fp_t wlamu = wl_ray_weight * dir_weight * wphi(kr, ks);

        add_to_gamma<false>(GammaAccumState{
            .eta = eta,
            .chi = chi,
            .uv = uv,
            .I = intensity,
            .psi_star = psi_star_entry,
            .wlamu = wlamu,
            .Gamma = Gamma,
            .i = l.i,
            .j = l.j,
            .k = ks
        });

        if (rad_rates.initialized()) {
            // NOTE(claude): psi_star = 0 removes
            // preconditioning leaving the true
            // radiative rates.
            add_to_gamma<false>(GammaAccumState{
                .eta = eta,
                .chi = chi,
                .uv = uv,
                .I = intensity,
                .psi_star = FP(0.0),
                .wlamu = wlamu,
                .Gamma = rad_rates,
                .i = l.i,
                .j = l.j,
                .k = ks
            });
        }
    }
    const int kr_base_c = adata.cont_start(ia);
    const int max_cont = kr_base_c + adata.num_cont(ia);
    for (int kra = 0; kra < active_cont_set.extent(0); ++kra) {
        const int kr = active_cont_set(kra);
        if (!(kr >= kr_base_c && kr < max_cont)) {
            continue;
        }
        const auto& cont = adata.continua(kr);

        const UV uv = compute_uv_cont(
            EmisOpacState<>{
                .adata = adata,
                .profile = phi,
                .la = la,
                .n = pops,
                .n_star_scratch = lte_scratch,
                .k = ks,
                .atmos = local_atmos
            },
            kr,
            UvOptions{
                .divide_by_hc_4pi = !include_4pi_hc
            }
        );

        fp_t eta = pops(offset + cont.j, ks) * uv.Uji;
        fp_t chi = pops(offset + cont.i, ks) * uv.Vij - pops(offset + cont.j, ks) * uv.Vji;
        if (!include_4pi_hc) {
            // NOTE(cmo): If we have dropped the hc/4pi
            // from U and V, these still need the
            // dimensioning
            eta *= hc_4pi;
            chi *= hc_4pi;
        }
        chi += FP(1e-20);

        const fp_t cont_wlamu = wl_ray_weight * dir_weight * cont_edge_bin_factor(adata, cont, la);
        add_to_gamma<false>(GammaAccumState{
            .eta = eta,
            .chi = chi,
            .uv = uv,
            .I = intensity,
            .psi_star = psi_star_entry,
            .wlamu = cont_wlamu,
            .Gamma = Gamma,
            .i = cont.i,
            .j = cont.j,
            .k = ks
        });

        if (rad_rates.initialized()) {
            add_to_gamma<false>(GammaAccumState{
                .eta = eta,
                .chi = chi,
                .uv = uv,
                .I = intensity,
                .psi_star = FP(0.0),
                .wlamu = cont_wlamu,
                .Gamma = rad_rates,
                .i = cont.i,
                .j = cont.j,
                .k = ks
            });
        }

        if (cont_absorb.initialized()) {
            // NOTE(claude): cont_wlamu * uv.V is the
            // number rate integrand, 4pi sigma dlambda
            // / (h nu), under either setting of
            // include_4pi_hc: the 4pi/hc sits in V when
            // it is dropped from the weights, and in
            // wl_ray_weights when it is not. So
            // multiplying through by h nu leaves the
            // energy rate, 4pi sigma I dlambda, either
            // way. Times the population to make it
            // volumetric.
            const fp_t hnu = hc_kJ_nm / adata.wavelength(la);
            const fp_t w = cont_wlamu * hnu;
            const int c = level_stage_idx(offset + cont.j);
            const fp_t ni = pops(offset + cont.i, ks);
            const fp_t nj = pops(offset + cont.j, ks);
            cont_absorb(c, cont.i, ks) += w * ni * uv.Vij * intensity;
            cont_emit(c, cont.i, ks) += w * nj * (uv.Uji + uv.Vji * intensity);
        }
    }
}

#else
#endif
