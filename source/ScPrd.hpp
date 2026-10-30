#if !defined(DEXRT_SC_PRD_HPP)
#define DEXRT_SC_PRD_HPP

#include "Types.hpp"
#include "Utils.hpp"
#include "Constants.hpp"
#include "Rad16BitStorage.hpp"

/// Resonance PRD in the hybrid approximation (Leenaarts et al. 2012,
/// Sukhorukov & Leenaarts 2016): the profile ratio rho* = psi/phi in the
/// comoving frame, on each PRD line's own wavelengths, from the comoving
/// angle-averaged J* and the angle-averaged R_II (no cross-redistribution).
/// The lab-frame rho is rho* at the comoving wavelength lambda + v_los lambda0 / c,
/// as the line profile.

/// Largest index b in [base, base + len) with values(b) <= x, for sorted
/// values, given values(base) <= x (or base is a sentinel below the range,
/// never read): a binary search of ceil(log2(len)) branchless steps.
/// TODO(cmo): The loop itself stays a (per-thread trip count) branch on GPU;
/// a fixed, unrolled count of predicated steps would make it branch-free.
template <class Values> YAKL_INLINE i32 sc_bracket(const Values& values, i32 base, i32 len, fp_t x) {
    while (len > 1) {
        const i32 half = len / 2;
        base = (values(base + half) <= x) ? base + half : base;
        len -= half;
    }
    return base;
}

struct ScPrdLine {
    i32 kr;
    i32 atom;
    i32 i;    /// lower level (within the atom)
    i32 j;    /// upper level (within the atom)
    i32 blue; /// first global wavelength
    i32 red;  /// one past the last
    i32 row0; /// first row of rho/J_rest
    fp_t lambda0;
    fp_t mass;
    fp_t stim_ratio; /// Bji / Aji: stimulated emission opacity per unit eta*
};

struct ScPrd {
    i32 num_lines = 0;
    i32 num_rows = 0;
    yakl::Array<ScPrdLine, 1, yakl::memDevice> lines;
    yakl::Array<i32, 1, yakl::memDevice> line_of_kr; /// -1: not PRD
    yakl::Array<fp_t const, 1, yakl::memDevice> wavelength;
    Rad16Array rho;    /// [row, ks] rho* (comoving)
    Rad16Array J_rest; /// [row, ks] J* (comoving)
    Fp2d C_out;        /// [prd line, ks] collisional depopulation rate of the upper level
    /// Lab wavelengths [reach_lo, reach_hi) of the PRD lines (the knots the
    /// transforms use).
    i32 reach_lo = 0;
    i32 reach_hi = 0;
    /// Largest |v| on any rank [m/s] (bounds the J* launch).
    fp_t max_speed = FP(0.0);

    YAKL_INLINE bool enabled() const { return num_lines > 0; }

    /// Largest index b in [lo, hi) with wavelength(b) <= x (lo - 1 if none).
    YAKL_INLINE i32 bracket(i32 lo, i32 hi, fp_t x) const {
        return sc_bracket(wavelength, lo - 1, hi - lo + 1, x);
    }

    /// Lab-frame rho of PRD line p at cell ks, wavelength lambda, line-of-sight
    /// velocity vel: linear in rho*, constant beyond the line's ends.
    YAKL_INLINE fp_t rho_lab(i32 p, i64 ks, fp_t lambda, fp_t vel) const {
        using ConstantsFP::c;
        const ScPrdLine& l = lines(p);
        const fp_t x = lambda + vel * l.lambda0 / c;
        const i32 b = bracket(l.blue, l.red, x);
        if (b < l.blue) {
            return rho(l.row0, ks);
        }
        if (b >= l.red - 1) {
            return rho(l.row0 + l.red - 1 - l.blue, ks);
        }
        const fp_t t = (x - wavelength(b)) / (wavelength(b + 1) - wavelength(b));
        const i32 r = l.row0 + b - l.blue;
        return (FP(1.0) - t) * rho(r, ks) + t * rho(r + 1, ks);
    }
};

/// Gouttebroze's (1986) fast approximation of g_II = R_II / phi(q_emit),
/// angle averaged, resonance lines (as Lightweaver's GII).
namespace ScPrdGii {
constexpr fp_t q_wing = FP(4.0);
constexpr fp_t q_core = FP(2.0);
constexpr fp_t q_spread = FP(5.0);
/// Fine absorption grid spacing for the redistribution integral [Doppler units].
constexpr fp_t dq = FP(0.15);

YAKL_INLINE fp_t g_zero(fp_t x) {
    return FP(1.0) / (std::abs(x) + std::sqrt(square(x) + FP(1.273239545)));
}

YAKL_INLINE fp_t gii(fp_t a_damp, fp_t q_emit, fp_t q_abs) {
    using ConstantsFP::sqrt_pi;
    if (q_emit < FP(0.0)) {
        q_emit = -q_emit;
        q_abs = -q_abs;
    }
    fp_t core = FP(0.0);
    fp_t core_factor = FP(0.0);
    if (q_emit < q_wing) {
        if (q_abs < -q_wing || q_abs > q_emit + q_spread) {
            return FP(0.0);
        }
        if (std::abs(q_abs) <= q_emit) {
            core = g_zero(q_emit);
        } else {
            core = std::exp(square(q_emit) - square(q_abs)) * g_zero(q_abs);
        }
        if (q_emit < q_core) {
            return core;
        }
        const fp_t phi_core = std::exp(-square(q_emit));
        const fp_t phi_wing = a_damp / (sqrt_pi * (square(a_damp) + square(q_emit)));
        core_factor = phi_core / (phi_core + phi_wing);
    } else if (std::abs(q_abs - q_emit) > q_spread) {
        return FP(0.0);
    }
    const fp_t u_min = FP(0.5) * std::abs(q_abs - q_emit);
    fp_t wing = (FP(1.0) - FP(2.0) * u_min * g_zero(u_min)) * std::exp(-square(u_min)) / sqrt_pi;
    const fp_t ratio = q_abs / q_emit;
    wing *= FP(2.75) - (FP(2.5) - FP(0.75) * ratio) * ratio;
    return core_factor * core + (FP(1.0) - core_factor) * wing;
}

/// Absorption range [q0, qN] where g_II(q_emit, .) is non-zero.
YAKL_INLINE void range(fp_t q_emit, fp_t* q0, fp_t* qN) {
    if (std::abs(q_emit) < q_core) {
        *q0 = -q_wing;
        *qN = q_wing;
    } else if (std::abs(q_emit) < q_wing) {
        *q0 = (q_emit > FP(0.0)) ? -q_wing : q_emit - q_spread;
        *qN = (q_emit > FP(0.0)) ? q_emit + q_spread : q_wing;
    } else {
        *q0 = q_emit - q_spread;
        *qN = q_emit + q_spread;
    }
}
} // namespace ScPrdGii

/// Elastic collision rate Q_E [s-1] of a line (the broadening terms flagged
/// elastic; natural broadening is not).
template <int mem_space = yakl::memDevice>
YAKL_INLINE fp_t sc_prd_elastic_rate(
    const CompLine<fp_t>& line,
    const yakl::Array<ScaledExponentsBroadening<fp_t> const, 1, mem_space>& broadeners,
    fp_t temperature,
    fp_t ne,
    fp_t nh0
) {
    fp_t Q = FP(0.0);
    for (int b_idx = line.broad_start; b_idx < line.broad_end; ++b_idx) {
        const auto& b = broadeners(b_idx);
        if (b.elastic) {
            Q += b.scaling * std::pow(temperature, b.temperature_exponent) *
                 std::pow(ne, b.electron_exponent) * std::pow(nh0, b.hydrogen_exponent);
        }
    }
    return Q;
}

struct ScState;
struct ScDirectionSet;
struct ScDirBatch;
struct ScMedium;
struct ScSolution;

/// The PRD lines of the active atoms (none if mode is Off), rho* = 1, J* = 0.
ScPrd sc_prd_init(const ScState& state, bool enabled);

/// Add the batch's intensity to J* at every PRD line's comoving wavelengths
/// whose lab wavelength the batch's knots bracket (linear interpolation on the
/// line's own lab knots, constant beyond its ends). Each batch adds its own
/// knots' share, so the sum over batches is the full interpolation.
void sc_prd_accumulate_J_rest(
    const ScState& state,
    const ScPrd& prd,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMedium& medium,
    const ScSolution& solution,
    i32 la_start,
    i32 wave_batch
);

/// Store each PRD upper level's collisional depopulation rate from the
/// (collisions only) Gamma.
void sc_prd_capture_collisions(const ScState& state, const ScPrd& prd);

/// rho* from J* and the radiative rates (state.rate_diag) for the current
/// populations. Returns the largest relative change of rho* (rank-local).
fp_t sc_prd_redistribute(const ScState& state, const ScPrd& prd);

#else
#endif
