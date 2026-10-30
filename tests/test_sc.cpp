#include "catch_amalgamated.hpp"
#include "ScConfig.hpp"
#include "ScCells.hpp"
#include "ScAngularQuadrature.hpp"
#include "ScGeometry.hpp"
#include "ScInterp.hpp"
#include "ScFormalSolver.hpp"
#include "ScRayMarch.hpp"
#include "ScRayEntry.hpp"
#include "GammaRayAccumulate.hpp"
#include "ScPreparedGeometry.hpp"
#include "ScBoundary.hpp"
#include "ScFormal.hpp"
#include "ScLongerGeometry.hpp"
#include "ScLongerIntegration.hpp"
#include "Healpix.hpp"
#include "ScPrd.hpp"
#include "Rad16BitStorage.hpp"
#include "LoopUtils.hpp"
#include <cmath>
#include <random>

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

struct Moments {
    f64 w;
    f64 m1[3];
    f64 m2[3][3];
};

Moments moments(const std::vector<ScHostDir>& dirs) {
    Moments m{};
    for (const auto& d : dirs) {
        m.w += d.weight;
        for (int i = 0; i < 3; ++i) {
            m.m1[i] += d.weight * d.mu[i];
            for (int j = 0; j < 3; ++j) {
                m.m2[i][j] += d.weight * d.mu[i] * d.mu[j];
            }
        }
    }
    return m;
}

}

TEST_CASE("SC Stepan quadratures", "[sc]") {
    for (int order : {3, 7, 9, 11, 13, 15}) {
        ScOptions opts{.quadrature = ScQuadratureType::Stepan, .quadrature_order = order};
        auto dirs = sc_full_sphere_quadrature(opts);
        Moments m = moments(dirs);
        REQUIRE_THAT(m.w, WithinRel(1.0, 1e-12));
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(m.m1[i], WithinAbs(0.0, 1e-12));
            for (int j = 0; j < 3; ++j) {
                REQUIRE_THAT(m.m2[i][j], WithinAbs(i == j ? 1.0 / 3.0 : 0.0, 1e-10));
            }
        }
        for (const auto& d : dirs) {
            REQUIRE_THAT(d.mu[0] * d.mu[0] + d.mu[1] * d.mu[1] + d.mu[2] * d.mu[2], WithinRel(1.0, 1e-12));
        }

        auto folded = sc_fold_y(dirs);
        REQUIRE(folded.size() <= dirs.size());
        Moments mf = moments(folded);
        REQUIRE_THAT(mf.w, WithinRel(1.0, 1e-12));
        REQUIRE_THAT(mf.m2[0][0], WithinAbs(1.0 / 3.0, 1e-10));
        REQUIRE_THAT(mf.m2[2][2], WithinAbs(1.0 / 3.0, 1e-10));
        REQUIRE_THAT(mf.m2[1][1], WithinAbs(1.0 / 3.0, 1e-10));
        for (const auto& d : folded) {
            REQUIRE(d.mu[1] >= 0.0);
        }
    }
    REQUIRE_THROWS(sc_full_sphere_quadrature(ScOptions{.quadrature = ScQuadratureType::Stepan, .quadrature_order = 5}));
}

TEST_CASE("SC mirrored Stepan quadratures (Lightweaver layout)", "[sc]") {
    for (int order : {3, 7, 9, 11, 13, 15}) {
        ScOptions opts{.quadrature = ScQuadratureType::StepanMirrored, .quadrature_order = order};
        auto dirs = sc_full_sphere_quadrature(opts);
        Moments m = moments(dirs);
        REQUIRE_THAT(m.w, WithinRel(1.0, 1e-12));
        REQUIRE_THAT(m.m2[2][2], WithinAbs(1.0 / 3.0, 1e-10));
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(m.m1[i], WithinAbs(0.0, 1e-12));
        }
        // NOTE(cmo): Symmetric in y, so folds to Lightweaver's 2D set of 4n rays.
        auto folded = sc_fold_y(dirs);
        REQUIRE(folded.size() * 2 == dirs.size());
    }
}

TEST_CASE("SC Healpix quadratures", "[sc]") {
    f64 prev_err = 1.0;
    for (int order = 0; order <= 4; ++order) {
        ScOptions opts{.quadrature = ScQuadratureType::Healpix, .quadrature_order = order};
        auto dirs = sc_full_sphere_quadrature(opts);
        REQUIRE(dirs.size() == 12 * (1 << (2 * order)));
        Moments m = moments(dirs);
        REQUIRE_THAT(m.w, WithinRel(1.0, 1e-12));
        f64 err = 0.0;
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(m.m1[i], WithinAbs(0.0, 1e-12));
            err = std::max(err, std::abs(m.m2[i][i] - 1.0 / 3.0));
        }
        REQUIRE(err < prev_err);
        prev_err = err;
        // NOTE(cmo): 2D: rotated about y, so still symmetric under y -> -y,
        // and off mu_z = 0 except along y.
        auto folded = sc_fold_y(dirs);
        REQUIRE_THAT(moments(folded).w, WithinRel(1.0, 1e-12));
        for (const auto& d : dirs) {
            REQUIRE((std::abs(d.mu[2]) > 1e-3 || std::abs(d.mu[0]) < 1e-6));
        }
        // NOTE(cmo): 3D: no direction on mu_z = 0, and still antipodal pairs.
        const auto dirs3 = sc_full_sphere_quadrature(opts, 3);
        f64 min_mu_z = 1.0;
        for (const auto& d : dirs3) {
            min_mu_z = std::min(min_mu_z, std::abs(d.mu[2]));
            bool pair = false;
            for (const auto& e : dirs3) {
                pair = pair || (std::abs(e.mu[0] + d.mu[0]) + std::abs(e.mu[1] + d.mu[1]) + std::abs(e.mu[2] + d.mu[2]) < 1e-10);
            }
            REQUIRE(pair);
        }
        REQUIRE(min_mu_z > 1e-3);
        REQUIRE_THAT(moments(dirs3).m1[2], WithinAbs(0.0, 1e-12));
    }
    REQUIRE(prev_err < 1e-3);

    // NOTE(cmo): Boundary cones: Gauss-Legendre on [0, 1] is exact to degree 2n - 1.
    for (int n = 1; n <= 8; ++n) {
        std::vector<f64> u, w;
        sc_gauss_legendre_unit(n, &u, &w);
        for (int k = 0; k < 2 * n; ++k) {
            f64 integral = 0.0;
            for (int i = 0; i < n; ++i) {
                REQUIRE(u[i] > 0.0);
                REQUIRE(u[i] < 1.0);
                integral += w[i] * std::pow(u[i], k);
            }
            REQUIRE_THAT(integral, WithinRel(1.0 / (k + 1), 1e-12));
        }
    }

    // NOTE(cmo): nside = 1 has rings at z = +-2/3 and 0
    for (int p = 0; p < 12; ++p) {
        const auto loc = healpix_nest_loc<f64, f64>(1, p);
        const f64 expected_z = (p < 4) ? 2.0 / 3.0 : (p < 8 ? 0.0 : -2.0 / 3.0);
        REQUIRE_THAT(loc.z, WithinAbs(expected_z, 1e-14));
        REQUIRE_THAT(loc.z * loc.z + loc.s * loc.s, WithinRel(1.0, 1e-14));
        const auto loc_f = healpix_nest_loc<fp_t, f32>(1, p);
        REQUIRE_THAT(f64(loc_f.phi), WithinAbs(loc.phi, 1e-5));
    }
}

TEST_CASE("SC linear formal solver coefficients", "[sc]") {
    for (f64 tau = 1e-7; tau < 200.0; tau *= 1.3) {
        const f64 edt = std::exp(-tau);
        const f64 a = (tau < 1e-3)
            ? tau * (0.5 + tau * (-1.0 / 3.0 + tau / 8.0))
            : (1.0 - edt * (1.0 + tau)) / tau;
        const f64 b = -std::expm1(-tau) - a;
        const ScLinearCoeffs c = sc_linear_coeffs(fp_t(tau));
        REQUIRE_THAT(f64(c.edt), WithinAbs(edt, 1e-6));
        REQUIRE_THAT(f64(c.a), WithinRel(a, 1e-5));
        REQUIRE_THAT(f64(c.alo), WithinRel(b, 1e-5));
        // NOTE(cmo): Constant S: I = edt I_uw + (1 - edt) S
        const fp_t S = FP(3.0);
        const fp_t chi = FP(2.0);
        const fp_t ds = fp_t(tau) / chi;
        const fp_t I_uw = FP(0.7);
        const f64 expected = edt * I_uw + (-std::expm1(-tau)) * S;
        auto r_lin = sc_formal_step<ScFormalSolver::PiecewiseLinear>(I_uw, S, chi, S, chi, ds);
        auto r_const = sc_formal_step<ScFormalSolver::PiecewiseConstant>(I_uw, S, chi, S, chi, ds);
        // Uniform opacity gives the same solution; renormalising the weights
        // can change the final few bits through rounding.
        auto r_eta = sc_formal_step<ScFormalSolver::PiecewiseLinearEmissivity>(I_uw, S, chi, S, chi, ds);
        REQUIRE_THAT(f64(r_eta.I), WithinRel(f64(r_lin.I), 1e-6));
        REQUIRE_THAT(f64(r_eta.I_ext), WithinRel(f64(r_lin.I_ext), 1e-6));
        REQUIRE_THAT(f64(r_eta.one_minus_alo), WithinRel(f64(r_lin.one_minus_alo), 1e-6));
        REQUIRE_THAT(f64(r_lin.I), WithinRel(expected, 1e-5));
        REQUIRE_THAT(f64(r_const.I), WithinRel(expected, 1e-5));
        REQUIRE_THAT(f64(r_lin.psi_star), WithinRel(b / f64(chi), 1e-5));
        // NOTE(cmo): Piecewise constant: the node's own material covers the half
        // of the step past the face.
        REQUIRE_THAT(f64(r_const.psi_star), WithinRel(-std::expm1(-0.5 * tau) / f64(chi), 1e-5));
        // NOTE(cmo): The non-local part and q = 1 - psi* chi keep their
        // relative precision when they are tiny (thick).
        REQUIRE_THAT(f64(r_lin.I_ext), WithinRel(edt * I_uw + a * S, 1e-5));
        REQUIRE_THAT(f64(r_lin.one_minus_alo), WithinRel(edt + a, 1e-5));
        REQUIRE_THAT(f64(r_const.one_minus_alo), WithinRel(std::exp(-0.5 * tau), 1e-5) || WithinAbs(std::exp(-0.5 * tau), 1e-37));
        REQUIRE_THAT(f64(r_lin.I_ext + r_lin.alo * S), WithinRel(f64(r_lin.I), 1e-6));
        REQUIRE_THAT(f64(r_const.I_ext + r_const.alo * S), WithinRel(f64(r_const.I), 1e-6));
    }
    // NOTE(cmo): Each source function comes with its own opacity: a step from
    // dense material into a hot, transparent node (enormous S, chi ~ 0) emits
    // only the dense half's eta ds / 2 in the thin limit, not that S.
    for (auto solver : {ScFormalSolver::PiecewiseConstant, ScFormalSolver::PiecewiseLinearEmissivity}) {
        const fp_t chi_uw = FP(1e-3), S_uw = FP(0.5), chi0 = FP(1e-15), S0 = FP(1e6), ds = FP(1.0);
        const ScStepResult r = (solver == ScFormalSolver::PiecewiseConstant)
            ? sc_formal_step<ScFormalSolver::PiecewiseConstant>(FP(0.0), S_uw, chi_uw, S0, chi0, ds)
            : sc_formal_step<ScFormalSolver::PiecewiseLinearEmissivity>(FP(0.0), S_uw, chi_uw, S0, chi0, ds);
        const f64 eta_half = 0.5 * (f64(chi_uw) * S_uw + f64(chi0) * S0) * ds;
        REQUIRE_THAT(f64(r.I), WithinRel(eta_half, 2e-3));
    }
    // NOTE(cmo): Void passes intensity through.
    auto r = sc_formal_step<ScFormalSolver::PiecewiseLinear>(FP(1.5), FP(0.0), FP(0.0), FP(0.0), FP(0.0), FP(10.0));
    REQUIRE(r.I == FP(1.5));
    REQUIRE(r.psi_star == FP(0.0));
}

TEST_CASE("SC preserves negative opacity and amplified radiation", "[sc][formal]") {
    const auto medium = sc_medium_from_eta(FP(-2.0), FP(3.0));
    REQUIRE(medium.chi == FP(-2.0));
    REQUIRE(medium.S == FP(-1.5));
    REQUIRE(sc_medium_from_eta(FP(0.0), FP(0.0)).S == FP(0.0));

    const fp_t chi = FP(-2.0);
    const fp_t source = FP(-1.5);
    const fp_t incoming = FP(0.7);
    for (fp_t tau : {
        FP(-1e-7),
        FP(-1e-3),
        FP(-0.049),
        FP(-0.05),
        FP(-0.2),
        FP(-1.0),
        FP(-5.0),
        FP(-20.0),
    }) {
        const fp_t ds = tau / chi;
        const f64 transmission = std::exp(-f64(tau));
        const f64 absorbed = -std::expm1(-f64(tau));
        const f64 expected = transmission * incoming + absorbed * source;
        const f64 a = std::abs(tau) < 1e-3
            ? f64(tau) * (0.5 + f64(tau) * (-1.0 / 3.0 + f64(tau) / 8.0))
            : absorbed / tau - transmission;
        const f64 alo = absorbed - a;
        const auto linear = sc_formal_step<ScFormalSolver::PiecewiseLinear>(
            incoming,
            source,
            chi,
            source,
            chi,
            ds
        );
        const auto emissivity = sc_formal_step<ScFormalSolver::PiecewiseLinearEmissivity>(
            incoming,
            source,
            chi,
            source,
            chi,
            ds
        );
        const auto constant = sc_formal_step<ScFormalSolver::PiecewiseConstant>(
            incoming,
            source,
            chi,
            source,
            chi,
            ds
        );
        for (const auto& result : {linear, emissivity, constant}) {
            REQUIRE_THAT(f64(result.I), WithinRel(expected, 1e-5));
            REQUIRE(result.I > incoming);
            REQUIRE(result.psi_star > FP(0.0));
            REQUIRE_THAT(f64(result.I_ext + result.alo * source), WithinRel(expected, 1e-5));
        }
        REQUIRE_THAT(f64(linear.psi_star), WithinRel(alo / chi, 1e-5));
        REQUIRE_THAT(f64(constant.psi_star), WithinRel(-std::expm1(-0.5 * tau) / chi, 1e-5));

        const f64 mean_response = absorbed / tau;
        const f64 expected_mean = mean_response * incoming + (1.0 - mean_response) * source;
        const auto longer = sc_integrate_longer_segment(incoming, source, tau, FP(1.0));
        REQUIRE_THAT(f64(longer.I), WithinRel(expected, 1e-5));
        REQUIRE_THAT(f64(longer.I_ext + longer.alo * source), WithinRel(expected_mean, 1e-5));
        REQUIRE(longer.one_minus_alo >= FP(1.0));
    }
}

TEST_CASE("SC interpolation", "[sc]") {
    // NOTE(cmo): Linear data is reproduced exactly by every variant.
    const fp_t a = FP(2.0), b = FP(-0.3);
    auto f = [&](int s) { return a + b * fp_t(s); };
    for (fp_t t : {FP(0.0), FP(0.25), FP(0.5), FP(0.8), FP(1.0)}) {
        const fp_t expected = f(5) - b * t;
        REQUIRE_THAT(sc_interp<ScInterp::Linear>(f(3), f(4), f(5), f(6), t), WithinAbs(expected, 1e-6));
        REQUIRE_THAT(sc_interp<ScInterp::VanLeer>(f(3), f(4), f(5), f(6), t), WithinAbs(expected, 1e-6));
        REQUIRE_THAT(sc_interp<ScInterp::MonotonizedCentral>(f(3), f(4), f(5), f(6), t), WithinAbs(expected, 1e-6));
    }
    // NOTE(cmo): No new extrema on step data.
    const fp_t step[6] = {FP(0.0), FP(0.0), FP(1.0), FP(1.0), FP(0.0), FP(0.0)};
    for (int s = 2; s < 5; ++s) {
        for (fp_t t = FP(0.0); t <= FP(1.0); t += FP(0.1)) {
            const fp_t vl = sc_interp<ScInterp::VanLeer>(step[s - 2], step[s - 1], step[s], step[s + 1], t);
            const fp_t mc = sc_interp<ScInterp::MonotonizedCentral>(step[s - 2], step[s - 1], step[s], step[s + 1], t);
            REQUIRE(vl >= FP(0.0));
            REQUIRE(vl <= FP(1.0));
            REQUIRE(mc >= FP(0.0));
            REQUIRE(mc <= FP(1.0));
        }
    }
}

TEST_CASE("SC plane interpolation", "[sc]") {
    // NOTE(cmo): Linear is bilinear: exact for a + bx + cy + dxy.
    auto bilinear = [](int i, int j) { return FP(2.0) - FP(0.3) * fp_t(i) + FP(0.7) * fp_t(j) + FP(0.2) * fp_t(i) * fp_t(j); };
    for (fp_t ta : {FP(0.0), FP(0.3), FP(1.0)}) {
        for (fp_t tb : {FP(0.0), FP(0.55), FP(0.9)}) {
            const fp_t expected = bilinear(0, 0) - FP(0.3) * (-ta) + FP(0.7) * (-tb) + FP(0.2) * ta * tb;
            REQUIRE_THAT(sc_interp_2d<ScInterp::Linear>(bilinear, ta, tb), WithinAbs(expected, 1e-5));
        }
    }
    // NOTE(cmo): Limited variants are exact for linear data.
    auto linear = [](int i, int j) { return FP(2.0) - FP(0.3) * fp_t(i) + FP(0.7) * fp_t(j); };
    for (fp_t ta : {FP(0.0), FP(0.3), FP(1.0)}) {
        for (fp_t tb : {FP(0.0), FP(0.55), FP(0.9)}) {
            const fp_t expected = linear(0, 0) + FP(0.3) * ta - FP(0.7) * tb;
            REQUIRE_THAT(sc_interp_2d<ScInterp::VanLeer>(linear, ta, tb), WithinAbs(expected, 1e-5));
            REQUIRE_THAT(sc_interp_2d<ScInterp::MonotonizedCentral>(linear, ta, tb), WithinAbs(expected, 1e-5));
        }
    }
    std::mt19937 rng(7);
    std::uniform_real_distribution<fp_t> uni(FP(0.0), FP(1.0));
    fp_t F[4][4];
    auto fetch = [&](int i, int j) { return F[i + 2][j + 2]; };
    for (int trial = 0; trial < 200; ++trial) {
        fp_t lo = FP(1.0), hi = FP(0.0);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                // NOTE(cmo): Steps and random data.
                F[i][j] = (trial % 2) ? uni(rng) : fp_t((i + 2 * j + trial / 2) % 3 == 0);
                lo = std::min(lo, F[i][j]);
                hi = std::max(hi, F[i][j]);
            }
        }
        const fp_t ta = uni(rng);
        const fp_t tb = uni(rng);
        // NOTE(cmo): tb = 0 is the 1D remap on line 0.
        REQUIRE(sc_interp_2d<ScInterp::MonotonizedCentral>(fetch, ta, FP(0.0)) ==
                sc_interp<ScInterp::MonotonizedCentral>(F[0][2], F[1][2], F[2][2], F[3][2], ta));
        // NOTE(cmo): Bounded by the stencil.
        for (fp_t v : {sc_interp_2d<ScInterp::VanLeer>(fetch, ta, tb), sc_interp_2d<ScInterp::MonotonizedCentral>(fetch, ta, tb)}) {
            REQUIRE(v >= lo - FP(1e-6));
            REQUIRE(v <= hi + FP(1e-6));
        }
    }
}

TEST_CASE("SC longer characteristics 3D plane tables", "[sc]") {
    std::mt19937 rng(11);
    std::uniform_real_distribution<f64> uni(0.0, 1.0);
    for (int trial = 0; trial < 500; ++trial) {
        const f64 Z0 = 10.0 * uni(rng);
        const f64 Z1 = Z0 + 0.1 + 2.0 * uni(rng);
        const f64 tau_x = (trial % 5 == 0) ? 0.0 : 4.0 * uni(rng);
        const f64 tau_y = (trial % 7 == 0) ? 0.0 : 4.0 * uni(rng);
        const f64 path_per_z = 1.0 + 3.0 * uni(rng);
        const ScLongerPlane pl = sc_longer_plane(Z0, Z1, tau_x, tau_y, path_per_z);
        const i32 n = pl.seglen.size();
        REQUIRE(n == pl.ox.size());
        REQUIRE(n == pl.oy.size());
        // NOTE(cmo): The segments cover the whole path, so each cell's path
        // (one segment per index, from the lattice) is (Z1 - Z0) path_per_z.
        f64 total = 0.0;
        f64 Z = Z0;
        for (int i = 0; i < n; ++i) {
            REQUIRE(pl.seglen[i] >= 0.0);
            total += pl.seglen[i];
            // NOTE(cmo): The segment's midpoint lies in its cell.
            const f64 Zm = Z + 0.5 * pl.seglen[i] / path_per_z;
            Z += pl.seglen[i] / path_per_z;
            if (pl.seglen[i] > 1e-9) {
                REQUIRE(i32(std::floor(Zm * tau_x) - std::floor(Z0 * tau_x)) == pl.ox[i]);
                REQUIRE(i32(std::floor(Zm * tau_y) - std::floor(Z0 * tau_y)) == pl.oy[i]);
            }
            if (i > 0) {
                REQUIRE(pl.ox[i] + pl.oy[i] == pl.ox[i - 1] + pl.oy[i - 1] + 1);
            }
        }
        REQUIRE_THAT(total, WithinRel((Z1 - Z0) * path_per_z, 1e-12));
        REQUIRE(pl.ox.back() <= pl.dkx);
        REQUIRE(pl.oy.back() <= pl.dky);
        REQUIRE(pl.dkx <= pl.ox.back() + 1);
        REQUIRE(pl.dky <= pl.oy.back() + 1);
        // NOTE(cmo): No y crossings: the 2D row.
        if (tau_y == 0.0) {
            const ScLongerRow row = sc_longer_row(Z0, Z1, tau_x, path_per_z);
            REQUIRE(row.seglen.size() == n);
            REQUIRE(row.dk == pl.dkx);
            REQUIRE_THAT(row.u0, WithinAbs(pl.u0x, 1e-12));
            for (int i = 0; i < n; ++i) {
                REQUIRE_THAT(row.seglen[i], WithinAbs(pl.seglen[i], 1e-9));
            }
        }
    }
    // NOTE(cmo): Side boundary indices are a bijection onto [0, side size).
    const std::array<std::array<i32, 4>, 3> cases{{{5, 4, 2, 3}, {6, 3, 0, 2}, {4, 7, 3, 0}}};
    for (const auto& [nx, ny, mx, my] : cases) {
        const auto layout = sc_longer_layout(nx, ny, 1, mx, my, false, false);
        std::vector<int> hits(layout.side_size, 0);
        for (i32 ky = -my; ky < ny; ++ky) {
            for (i32 kx = -mx; kx < nx; ++kx) {
                if (kx < 0 || ky < 0) {
                    hits[sc_longer_inflow_index(layout, {0, kx, ky}) - layout.plane_size] += 1;
                }
            }
        }
        for (int h : hits) {
            REQUIRE(h == 1);
        }
    }
}

TEST_CASE("SC generated directions propagate or are analytic along y", "[sc]") {
    const ScGrid grid{.num_x = 16, .num_z = 16, .dx = FP(1.0), .dz = FP(1.0)};
    for (auto type : {ScQuadratureType::Stepan, ScQuadratureType::StepanMirrored, ScQuadratureType::Healpix}) {
        const std::vector<int> orders = type == ScQuadratureType::Healpix
            ? std::vector<int>{0, 1, 2, 3, 4} : std::vector<int>{3, 7, 9, 11, 13, 15};
        for (int order : orders) {
            const auto dirs = sc_full_sphere_quadrature(ScOptions{.quadrature = type, .quadrature_order = order});
            int sweep = 0;
            int local = 0;
            for (const auto& d : sc_fold_y(dirs)) {
                const auto f = sc_frame_2d(grid, d.mu[0], d.mu[2], true);
                if (f.mode == ScRayMode::AlongY) {
                    ++local;
                    REQUIRE(std::abs(d.mu[1]) > 1.0 - 1e-10);
                } else {
                    ++sweep;
                    REQUIRE(std::abs(d.mu[2]) >= SC_DEGENERATE_MU);
                }
            }
            REQUIRE(sweep > 0);
            if (type != ScQuadratureType::Healpix) { REQUIRE(local == 0); }
            if (type == ScQuadratureType::Healpix && order == 0) { REQUIRE(local > 0); }
        }
    }
}

TEST_CASE("SC geometry", "[sc]") {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<f64> u(-1.0, 1.0);
    for (auto [dx, dz] : {std::pair{1.0, 1.0}, std::pair{2.0, 1.0}, std::pair{0.3, 5.0}}) {
        const ScGrid grid{.num_x = 48, .num_z = 32, .dx = fp_t(dx), .dz = fp_t(dz)};
        for (int i = 0; i < 500; ++i) {
            f64 mx = u(rng), my = u(rng), mz = u(rng);
            const f64 norm = std::sqrt(mx * mx + my * my + mz * mz);
            mx /= norm; my /= norm; mz /= norm;
            // NOTE(cmo): Rays crossing more than SC_MAX_ROW_CROSSINGS columns
            // per row are rejected.
            if (std::abs(mx) * dz > SC_MAX_ROW_CROSSINGS * std::abs(mz) * dx) {
                REQUIRE_THROWS(sc_frame_2d(grid, mx, mz, true));
                continue;
            }
            const ScFrame fp = sc_frame_2d(grid, mx, mz, true);
            const ScFrame f = sc_frame_2d(grid, mx, mz, false);
            REQUIRE(f.num_p == grid.num_z);
            REQUIRE(f.num_s == grid.num_x);
            // NOTE(cmo): Periodic laps are swept, up to the cost cap.
            if (fp.mode == ScRayMode::Sweep) {
                REQUIRE(fp.t <= fp_t(SC_MAX_ROW_CROSSINGS));
            }
            if (f.mode != ScRayMode::Sweep) {
                continue;
            }
            REQUIRE(f.t >= FP(0.0));
            // NOTE(cmo): Step back ds along mu from a node: must land on row p-1 at s - t.
            const i32 p = 5, s = 7;
            const Coord2 c = sc_physical_coord(f, p, s);
            const f64 x_uw = (c.x + 0.5) * dx - f.ds * mx;
            const f64 z_uw = (c.z + 0.5) * dz - f.ds * mz;
            const vec3 expected = sc_canonical_position(grid, f, fp_t(p - 1), fp_t(s) - f.t);
            REQUIRE_THAT(x_uw, WithinAbs(expected(0), 1e-3 * std::max(dx, dz) * (1.0 + f.t)));
            REQUIRE_THAT(z_uw, WithinAbs(expected(2), 1e-3 * std::max(dx, dz)));
            REQUIRE(sc_canonical_idx(f, c.x, 0, c.z) == i64(p) * f.num_s + s);
        }
    }
    // NOTE(cmo): Exactly horizontal and along-y directions
    const ScGrid grid{.num_x = 16, .num_z = 16, .dx = FP(1.0), .dz = FP(1.0)};
    // NOTE(cmo): mu_z = 0 and the cost cap are rejected; periodic laps are swept.
    REQUIRE_THROWS(sc_frame_2d(grid, 1.0, 0.0, false));
    REQUIRE(sc_frame_2d(grid, 0.0, 0.0, false).mode == ScRayMode::AlongY);
    REQUIRE(sc_frame_2d(grid, 0.999, 0.01, true).mode == ScRayMode::Sweep);
    REQUIRE(sc_frame_2d(grid, 0.999, 0.01, false).mode == ScRayMode::Sweep);
    REQUIRE_THROWS(sc_frame_2d(grid, 1.0, 0.5 / SC_MAX_ROW_CROSSINGS, true));
    const ScGrid grid3{.num_x = 16, .num_y = 8, .num_z = 16, .dx = FP(1.0), .dy = FP(1.0), .dz = FP(1.0), .dim = 3};
    REQUIRE_THROWS(sc_frame_3d(grid3, 0.6, 0.8, 0.0, 1.0));
    REQUIRE(sc_frame_3d(grid3, 0.6, 0.79, 0.1, 1.0).mode == ScRayMode::Sweep);
}

TEST_CASE("SC non-uniform z geometry", "[sc]") {
    const ScOptions opts{.quadrature = ScQuadratureType::Stepan, .quadrature_order = 7};
    const ScDirectionSet dirs = build_sc_direction_set(opts, false);
    // NOTE(cmo): Uniform: the per-row tables are exactly the frame's t and ds.
    {
        const ScGrid grid{.num_x = 32, .num_z = 16, .dx = FP(0.7), .dz = FP(1.3)};
        for (const auto& b : sc_make_dir_batches(grid, dirs, false)) {
            if (b.mode != ScRayMode::Sweep) {
                continue;
            }
            const auto t = b.row_t.createHostCopy();
            const auto ds = b.row_ds.createHostCopy();
            for (int i = 0; i < b.num_dirs; ++i) {
                for (int p = 0; p < grid.num_z; ++p) {
                    REQUIRE(t(i, p) == b.frames_host[i].t);
                    REQUIRE(ds(i, p) == b.frames_host[i].ds);
                }
            }
        }
    }

    // NOTE(cmo): Stretched (geometric, plus a 3x jump), as the middle slab
    // [8, 24) of 32 rows.
    std::vector<f64> z(32);
    f64 step = 0.5;
    z[0] = 10.0;
    for (int k = 1; k < z.size(); ++k) {
        step *= (k == 20) ? 3.0 : 1.08;
        z[k] = z[k - 1] + step;
    }
    REQUIRE_THAT(sc_global_row_height(z, -1), WithinAbs(2.0 * z[0] - z[1], 1e-12));
    REQUIRE_THAT(sc_global_row_height(z, 32), WithinAbs(2.0 * z[31] - z[30], 1e-12));
    const i32 z_start = 8;
    ScGrid grid{.num_x = 32, .num_z = 16, .dx = FP(0.7), .dz = FP(0.5)};
    Fp1dHost z_node("z_node", grid.num_z + 2);
    for (int k = -1; k <= grid.num_z; ++k) {
        z_node(k + 1) = fp_t(sc_global_row_height(z, z_start + k));
    }
    grid.z_node = z_node.createDeviceCopy();
    // NOTE(cmo): sc_canonical_position reads z_node: give it the host copy.
    ScGrid grid_h = grid;
    grid_h.z_node = Fp1d("z_node_h", z_node.data(), z_node.extent(0));
    const std::vector<f64> steps = sc_row_steps(grid, z, z_start);
    REQUIRE(steps.size() == grid.num_z + 1);
    REQUIRE_THAT(steps[0], WithinAbs(z[z_start] - z[z_start - 1], 1e-12));
    for (const auto& b : sc_make_dir_batches(grid, dirs, false, steps)) {
        if (b.mode != ScRayMode::Sweep) {
            continue;
        }
        const auto t = b.row_t.createHostCopy();
        const auto ds = b.row_ds.createHostCopy();
        for (int i = 0; i < b.num_dirs; ++i) {
            const ScFrame& f = b.frames_host[i];
            const f64 mx = dirs.mu_host(b.dir_idx_host[i], 0);
            const f64 mz = dirs.mu_host(b.dir_idx_host[i], 2);
            for (int p = 0; p < grid.num_z; ++p) {
                // NOTE(cmo): Step back ds along mu from node (p, s): lands on
                // row p - 1 at s - t.
                const i32 s = 20;
                const vec3 node = sc_canonical_position(grid_h, f, fp_t(p), fp_t(s));
                const vec3 uw = sc_canonical_position(grid_h, f, fp_t(p - 1), fp_t(s) - t(i, p));
                REQUIRE_THAT(node(0) - ds(i, p) * mx, WithinAbs(uw(0), 1e-4 * (1.0 + t(i, p))));
                REQUIRE_THAT(node(2) - ds(i, p) * mz, WithinAbs(uw(2), 1e-4));
                const i32 k = f.flip_z ? grid.num_z - 1 - p : p;
                REQUIRE_THAT(node(2), WithinAbs(z[z_start + k], 1e-5));
            }
        }
    }
    // NOTE(cmo): Domain face half a step below node 0 (global slab).
    ScGrid full = grid_h;
    full.num_z = 32;
    Fp1dHost z_full("z_full", 34);
    for (int k = -1; k <= 32; ++k) {
        z_full(k + 1) = fp_t(sc_global_row_height(z, k));
    }
    full.z_node = Fp1d("z_full_h", z_full.data(), z_full.extent(0));
    const ScFrame up = sc_frame_2d(full, 0.3, 0.9, false);
    REQUIRE_THAT(sc_canonical_position(full, up, -FP(0.5), FP(3.0))(2), WithinAbs(z[0] - 0.5 * (z[1] - z[0]), 1e-5));
    const ScFrame down = sc_frame_2d(full, 0.3, -0.9, false);
    REQUIRE_THAT(sc_canonical_position(full, down, -FP(0.5), FP(3.0))(2), WithinAbs(z[31] + 0.5 * (z[31] - z[30]), 1e-5));
}

TEST_CASE("SC thermalised boundary", "[sc]") {
    // NOTE(cmo): Thin, non-emitting: upward rays on the first row carry ~B(T).
    const ScGrid grid{.num_x = 32, .num_z = 32, .dx = FP(1.0), .dz = FP(1.0)};
    const ScOptions opts{.quadrature = ScQuadratureType::Stepan, .quadrature_order = 3};
    const ScDirectionSet dirs = build_sc_direction_set(opts, false);
    const auto batches = sc_make_dir_batches(grid, dirs, true);
    auto geometry_opts = opts;
    geometry_opts.characteristics = ScCharacteristics::Short;
    geometry_opts.x_boundary = ScXBoundary::Periodic;
    const ScPreparedGeometry geometry(grid, {}, dirs, geometry_opts, {}, {}, grid.dz);
    auto work = sc_make_buffers(geometry, 1);
    work.medium.chi = FP(1e-6);
    work.medium.S = FP(0.0);
    Fp1dHost T("T", grid.num_x);
    T = FP(6000.0);
    Fp1dHost wl("wl", 1);
    wl(0) = FP(500.0);
    const ThermalisedBc th_bc{
        .temperature = T.createDeviceCopy(),
        .wavelength = wl.createDeviceCopy(),
        .dx = grid.dx,
        .offset_x = FP(0.0),
        .z_face = FP(0.25)
    };
    const fp_t B = planck_wavelength(FP(500.0), FP(6000.0));
    for (const auto& batch : batches) {
        if (batch.mode != ScRayMode::Sweep) {
            continue;
        }
        ScSweepTask task{
            .batch = batch,
            .la_start = 0,
            .wave_batch = 1,
            .has_upstream = false,
            .has_downstream = false,



        };
        sc_prepare_short_boundaries(geometry, work, task, ScBoundary{th_bc});
        sc_solve_short(geometry, work, task, ScFormalSolver::PiecewiseLinear, ScInterp::Linear);
        auto I_ext = work.solution.I_ext.createHostCopy();
        for (int d = 0; d < batch.num_dirs; ++d) {
            const fp_t I0 = I_ext(d, 0, 5);
            if (batch.flip_z) {
                REQUIRE(I0 == FP(0.0));
            } else {
                REQUIRE_THAT(I0, WithinRel(B, FP(1e-4)));
            }
        }
    }
}

TEST_CASE("SC longer characteristics geometry", "[sc]") {
    std::mt19937 rng(4321);
    std::uniform_real_distribution<f64> u(0.0, 1.0);
    for (int trial = 0; trial < 2000; ++trial) {
        const f64 Z0 = 50.0 * u(rng);
        const f64 H_raw = (trial % 3 == 0) ? 1e-3 + 0.01 * u(rng) : 0.1 + 3.0 * u(rng);
        const f64 H = (Z0 + H_raw) - Z0;
        const f64 tau = (trial % 7 == 0) ? 0.0 : 4.0 * u(rng);
        const f64 path_per_z = 1.0 + 3.0 * u(rng);
        const ScLongerRow row = sc_longer_row(Z0, Z0 + H, tau, path_per_z);
        const f64 A0 = Z0 * tau;
        const f64 A1 = (Z0 + H) * tau;
        REQUIRE_THAT(row.u0, WithinAbs(A0 - std::floor(A0), 1e-12));
        REQUIRE(row.dk == i32(std::floor(A1) - std::floor(A0)));
        f64 total = 0.0;
        // NOTE(cmo): Clip the characteristic entering cell 0 (at a = u0, in
        // column units relative to the entry cell) to each cell column.
        for (int i = 0; i < row.seglen.size(); ++i) {
            const f64 a_lo = std::max(row.u0, f64(i));
            const f64 a_hi = std::min(row.u0 + (A1 - A0), f64(i + 1));
            const f64 expected = (tau > 0.0) ? std::max(a_hi - a_lo, 0.0) / (A1 - A0) * H * path_per_z : H * path_per_z;
            REQUIRE_THAT(row.seglen[i], WithinAbs(expected, 1e-9 * H * path_per_z));
            total += row.seglen[i];
        }
        REQUIRE_THAT(total, WithinRel(H * path_per_z, 1e-12));
        // NOTE(cmo): The segments cover the characteristic's whole column span.
        REQUIRE(row.seglen.size() == std::max(i32(std::ceil(A1) - std::floor(A0)) - 1, 0) + 1);
    }
    // NOTE(cmo): Faces from centres: mirrored ends, and canonical heights from
    // the start face in either direction.
    const std::vector<f64> z{1.0, 2.0, 4.0, 8.0};
    const auto up = sc_canonical_faces(z, false);
    const auto down = sc_canonical_faces(z, true);
    const std::vector<f64> expected_up{0.0, 1.0, 2.5, 5.5, 9.5};
    for (int j = 0; j < 5; ++j) {
        REQUIRE_THAT(up[j], WithinAbs(expected_up[j], 1e-12));
        REQUIRE_THAT(down[j], WithinAbs(up[4] - up[4 - j], 1e-12));
    }
}

TEST_CASE("SC Longer cell-mean response", "[sc]") {
    for (int i = 0; i <= 140; ++i) {
        const f64 tau = std::pow(10.0, -8.0 + 0.1 * i);
        const ScLongerResponse c = sc_longer_response(fp_t(tau));
        const f64 tau_f = fp_t(tau);
        const f64 w0 = -std::expm1(-tau_f);
        const f64 expected_one_minus_alo = w0 / tau_f;
        // NOTE(cmo): alo by series where it cancels in f64 too.
        const f64 expected_alo = (tau_f < 1e-2) ? tau_f / 2.0 - tau_f * tau_f / 6.0 + tau_f * tau_f * tau_f / 24.0 : (tau_f - w0) / tau_f;
        REQUIRE_THAT(f64(c.one_minus_alo), WithinRel(expected_one_minus_alo, 2e-6));
        REQUIRE_THAT(f64(c.alo), WithinRel(expected_alo, 2e-6));
    }
}

TEST_CASE("SC z decomposition", "[sc]") {
    for (int ranks : {1, 2, 3, 5}) {
        i32 next = 0;
        for (int r = 0; r < ranks; ++r) {
            const ScSlab slab = sc_decompose_z(160, MpiState{.rank = r, .num_ranks = ranks});
            REQUIRE(slab.z_start == next);
            REQUIRE(slab.num_z() >= 1);
            REQUIRE((slab.num_z() == 160 / ranks || slab.num_z() == 160 / ranks + 1));
            next = slab.z_end;
        }
        REQUIRE(next == 160);
    }
    REQUIRE(sc_decompose_z(64, MpiState{.rank = 0, .num_ranks = 3}).num_z() == 22);
    REQUIRE(sc_decompose_z(48, MpiState{.rank = 0, .num_ranks = 1}).num_z() == 48);
}

TEST_CASE("SC sweep dispatch", "[sc]") {
    // NOTE(cmo): Homogeneous, optically thick: every compiled variant gives I = S
    // away from the boundaries.
    const ScGrid grid{.num_x = 32, .num_z = 32, .dx = FP(1.0), .dz = FP(1.0)};
    const ScOptions opts{.quadrature = ScQuadratureType::Stepan, .quadrature_order = 3};
    const ScDirectionSet dirs = build_sc_direction_set(opts, false);
    const auto batches = sc_make_dir_batches(grid, dirs, false);
    constexpr i32 wave_batch = 2;
    auto geometry_opts = opts;
    geometry_opts.characteristics = ScCharacteristics::Short;
    const ScPreparedGeometry geometry(grid, {}, dirs, geometry_opts, {}, {}, grid.dz);
    auto work = sc_make_buffers(geometry, wave_batch);
    work.medium.chi = FP(20.0);
    work.medium.S = FP(1.5);
    for (auto solver : {ScFormalSolver::PiecewiseConstant, ScFormalSolver::PiecewiseLinear, ScFormalSolver::PiecewiseLinearEmissivity}) {
        for (auto interp : {ScInterp::Linear, ScInterp::VanLeer, ScInterp::MonotonizedCentral}) {
            for (const auto& batch : batches) {
                ScSweepTask task{
                    .batch = batch,
                    .la_start = 0,
                    .wave_batch = wave_batch,
                    .has_upstream = false,
                    .has_downstream = false,


                };
                sc_prepare_short_boundaries(geometry, work, task, ScBoundary{ZeroBc{}});
                sc_solve_short(geometry, work, task, solver, interp);
                auto I_ext = work.solution.I_ext.createHostCopy();
                auto b = work.solution.alo.createHostCopy();
                const ScFrame f = batch.frames_host[0];
                for (int w = 0; w < wave_batch; ++w) {
                    const i64 idx = i64(f.num_p - 1) * f.num_s + f.num_s / 2;
                    const fp_t b_local = b(0, w, idx);
                    REQUIRE_THAT(I_ext(0, w, idx) + b_local * FP(1.5), WithinRel(FP(1.5), FP(1e-5)));
                    const fp_t psi = b_local / FP(20.0);
                    REQUIRE(psi > FP(0.0));
                    REQUIRE(psi < FP(1.0) / FP(20.0) + FP(1e-6));
                    // NOTE(cmo): First node sees half a cell of material from the zero boundary.
                    const i64 idx0 = f.num_s / 2;
                    REQUIRE(I_ext(0, w, idx0) + b(0, w, idx0) * FP(1.5) <= FP(1.5));
                }
            }
        }
    }
}

TEST_CASE("SC cells cover each owned cell exactly once", "[sc][cells]") {
    // Rows contain a long unaligned run, a gap, short tails and an entirely
    // empty row.
    ScActiveMaskHost mask("test_active", 3, 13);
    mask = u8(0);
    for (i32 x = 1; x < 9; ++x) {
        mask(0, x) = 1;
    }
    for (i32 x = 10; x < 13; ++x) {
        mask(0, x) = 1;
    }
    mask(2, 1) = mask(2, 12) = 1;
    constexpr i32 H = SC_MEDIUM_HALO;
    for (i32 group : {1, 4, 8, 16}) {
        ScCells cells;
        cells.init(mask, 7, group);
        REQUIRE(cells.num_active == 13);
        REQUIRE(cells.z_start == 7);
        REQUIRE(cells.num_cells() == (3 + 2 * H) * 13);
        REQUIRE(cells.owned_offset() == H * 13);
        auto owned = cells.owned.createHostCopy();
        auto active = cells.active.createHostCopy();
        auto runs = cells.runs.createHostCopy();
        for (i32 z = -H; z < 3 + H; ++z) {
            for (i32 x = 0; x < 13; ++x) {
                const i64 ks = cells.ks(x, 0, z);
                REQUIRE(ks == (z + H) * 13 + x);
                REQUIRE(cells.coord(ks) == Coord3{.x = x, .y = 0, .z = z});
                const bool owned_row = z >= 0 && z < 3;
                REQUIRE(active(z + H, x) == (owned_row ? mask(z, x) : 0));
            }
        }
        i64 next = 0;
        for (i32 z = 0; z < 3; ++z) {
            for (i32 x = 0; x < 13; ++x) {
                if (mask(z, x)) {
                    REQUIRE(owned(next++) == cells.ks(x, 0, z));
                }
            }
        }
        std::vector<i32> visits(cells.num_cells(), 0);
        for (i64 r = 0; r < cells.num_runs; ++r) {
            const auto piece = runs(r);
            REQUIRE(piece.n > 0);
            REQUIRE(piece.n <= group);
            REQUIRE(piece.x0 / group == (piece.x0 + piece.n - 1) / group);
            const i64 ks0 = cells.ks(piece.x0, piece.y, piece.z);
            for (i32 c = 0; c < piece.n; ++c) {
                REQUIRE(mask(piece.z, piece.x0 + c));
                ++visits.at(ks0 + c);
            }
        }
        for (i64 k = 0; k < cells.num_cells(); ++k) {
            REQUIRE(visits[k] == (active.data()[k] ? 1 : 0));
        }
    }
}

TEST_CASE("SC cells list halo rows after the owned ones", "[sc][cells]") {
    constexpr i32 H = SC_MEDIUM_HALO;
    ScActiveMaskHost mask("test_owned", 2, 8);
    mask = u8(1);
    mask(1, 3) = 0;
    ScCells cells;
    cells.init(mask, 4, 4);
    REQUIRE(cells.num_active_with_halo == cells.num_active);
    REQUIRE(cells.num_runs_with_halo == cells.num_runs);
    // NOTE(cmo): As ScHalo leaves them: a neighbour's activity in the halo rows.
    auto act = cells.active.createHostCopy();
    for (i32 x = 2; x < 7; ++x) {
        act(0, x) = 1;
        act(H + 2 + 1, x) = 1;
    }
    act.deep_copy_to(cells.active);
    yakl::fence();
    cells.build_lists();
    REQUIRE(cells.num_active == 15);
    REQUIRE(cells.num_active_with_halo == 25);
    auto owned = cells.owned.createHostCopy();
    auto all = cells.owned_and_halo.createHostCopy();
    auto runs = cells.runs.createHostCopy();
    for (i64 i = 1; i < cells.num_active_with_halo; ++i) {
        REQUIRE(all(i) > all(i - 1));
    }
    REQUIRE(all(0) == cells.ks(2, 0, -H));
    REQUIRE(all(cells.num_active_with_halo - 1) == cells.ks(6, 0, 3));
    for (i64 i = 0; i < cells.num_active; ++i) {
        const Coord3 c = cells.coord(owned(i));
        REQUIRE(c.z >= 0);
        REQUIRE(c.z < 2);
    }
    i64 halo_cells = 0;
    for (i64 r = 0; r < cells.num_runs_with_halo; ++r) {
        const bool owned_row = runs(r).z >= 0 && runs(r).z < 2;
        REQUIRE(owned_row == (r < cells.num_runs));
        REQUIRE(runs(r).n <= 4);
        halo_cells += owned_row ? 0 : runs(r).n;
    }
    REQUIRE(halo_cells == 10);
}

TEST_CASE("SC cells index 3D planes and move fields in and out", "[sc][cells]") {
    // [z = 2, y = 3, x = 4], one void cell per row.
    ScActiveMaskHost mask("test_active_3d", 2, 12);
    mask = u8(1);
    mask(0, 5) = mask(1, 11) = 0;
    ScCells cells;
    cells.init(mask, 0, 1, 3);
    REQUIRE(cells.num_x == 4);
    REQUIRE(cells.plane_size() == 12);
    REQUIRE(cells.num_active == 22);
    REQUIRE(cells.coord(cells.ks(3, 2, 1)) == Coord3{.x = 3, .y = 2, .z = 1});
    yakl::Array<f32, 3, yakl::memHost> rows("test_rows", 2, 2, 12);
    for (i32 i = 0; i < rows.totElems(); ++i) {
        rows.data()[i] = f32(i + 1);
    }
    const auto dense = cells.from_owned_rows(rows);
    REQUIRE(dense.extent(1) == cells.num_cells());
    REQUIRE(dense(1, cells.ks(1, 1, 0)) == FP(0.0));
    REQUIRE(dense(1, cells.ks(2, 1, 0)) == rows(1, 0, 6));
    REQUIRE(dense(0, cells.ks(0, 0, -1)) == FP(0.0));
    const auto dense_d = dense.createDeviceCopy();
    const auto back = cells.owned_rows(dense_d);
    for (i32 l = 0; l < 2; ++l) {
        for (i32 z = 0; z < 2; ++z) {
            for (i32 xy = 0; xy < 12; ++xy) {
                REQUIRE(back(l, z, xy) == (mask(z, xy) ? rows(l, z, xy) : FP(0.0)));
            }
        }
    }
}

TEST_CASE("SC cells support empty masks and rebuilding", "[sc][cells]") {
    ScActiveMaskHost mask("test_empty", 1, 5);
    mask = u8(0);
    ScCells cells;
    cells.init(mask);
    REQUIRE(cells.num_active == 0);
    REQUIRE(cells.num_runs == 0);
    REQUIRE(cells.list().num() == 0);
    REQUIRE(cells.list().host().num() == 0);
    REQUIRE(cells.num_cells() == (1 + 2 * SC_MEDIUM_HALO) * 5);
    mask(0, 4) = 1;
    cells.init(mask);
    REQUIRE(cells.num_active == 1);
    REQUIRE(cells.list().host()(0) == cells.ks(4, 0, 0));
    mask = u8(0);
    cells.init(mask);
    REQUIRE(cells.num_active == 0);
    REQUIRE(cells.num_runs == 0);
    REQUIRE_THROWS(cells.init(mask, -1));
    REQUIRE_THROWS(cells.init(mask, 0, 0));
    mask(0, 4) = 2;
    REQUIRE_THROWS(cells.init(mask));
}

TEST_CASE("SC tile masks preserve global alignment across arbitrary slabs", "[sc][cells]") {
    ScActiveMaskHost tiles("global_activity", 2, 2);
    tiles(0, 0) = tiles(1, 1) = 1;
    tiles(0, 1) = tiles(1, 0) = 0;
    auto whole = sc_active_mask_from_tiles(tiles, 0, 2 * BLOCK_SIZE);
    // Include a one-row slab at a tile boundary and slabs cutting both tiles.
    const std::vector<i32> ends = {0, 5, BLOCK_SIZE, BLOCK_SIZE + 1, 2 * BLOCK_SIZE};
    for (size_t i = 1; i < ends.size(); ++i) {
        const i32 start = ends[i - 1];
        auto local = sc_active_mask_from_tiles(tiles, start, ends[i] - start);
        ScCells cells;
        cells.init(local, start);
        for (i32 z = 0; z < cells.num_z; ++z) {
            for (i32 x = 0; x < cells.num_x; ++x) {
                REQUIRE(local(z, x) == whole(start + z, x));
            }
        }
        REQUIRE(cells.num_active == i64(cells.num_z) * BLOCK_SIZE);
    }
    REQUIRE_THROWS(sc_active_mask_from_tiles(tiles, -1, 1));
    REQUIRE_THROWS(sc_active_mask_from_tiles(tiles, 0, 0));
    REQUIRE_THROWS(sc_active_mask_from_tiles(tiles, 2 * BLOCK_SIZE, 1));
}

TEST_CASE("SC slab partitions are whole units", "[sc][cells]") {
    MpiState mpi;
    mpi.num_ranks = 5;
    for (i32 r = 0; r < 5; ++r) {
        mpi.rank = r;
        const auto slab = sc_decompose_z(5, mpi);
        REQUIRE(slab.z_start == r);
        REQUIRE(slab.num_z() == 1);
    }
    mpi.num_ranks = 3;
    i32 last = 0;
    for (i32 r = 0; r < 3; ++r) {
        mpi.rank = r;
        const auto slab = sc_decompose_z(5 * BLOCK_SIZE, mpi, BLOCK_SIZE);
        REQUIRE(slab.z_start == last);
        REQUIRE(slab.z_start % BLOCK_SIZE == 0);
        REQUIRE(slab.num_z() == (r < 2 ? 2 : 1) * BLOCK_SIZE);
        last = slab.z_end;
    }
    REQUIRE(last == 5 * BLOCK_SIZE);
    REQUIRE_THROWS(sc_decompose_z(2 * BLOCK_SIZE, mpi, BLOCK_SIZE));
    REQUIRE_THROWS(sc_decompose_z(3 * BLOCK_SIZE + 1, mpi, BLOCK_SIZE));
    mpi.num_ranks = 4;
    REQUIRE_THROWS(sc_decompose_z(3, mpi));
}

namespace {

/// Lightweaver's GII (Prd.cpp, resonance lines), in f64.
f64 lw_gii(f64 a, f64 qe, f64 qa) {
    const auto g0 = [](f64 x) { return 1.0 / (std::abs(x) + std::sqrt(x * x + 1.273239545)); };
    if (qe < 0.0) {
        qe = -qe;
        qa = -qa;
    }
    f64 core = 0.0;
    f64 cf = 0.0;
    if (qe < 4.0) {
        if (qa < -4.0 || qa > qe + 5.0) {
            return 0.0;
        }
        core = (std::abs(qa) <= qe) ? g0(qe) : std::exp(qe * qe - qa * qa) * g0(qa);
        if (qe >= 2.0 && qe <= 4.0) {
            const f64 pc = std::exp(-qe * qe);
            const f64 pw = a / (std::sqrt(M_PI) * (a * a + qe * qe));
            cf = pc / (pc + pw);
        } else {
            return core;
        }
    }
    f64 gii = 0.0;
    if (qe >= 2.0) {
        if (qe >= 4.0 && std::abs(qa - qe) > 5.0) {
            return 0.0;
        }
        const f64 u = std::abs((qa - qe) / 2.0);
        f64 wing = 2.0 * (1.0 - 2.0 * u * g0(u)) * std::exp(-u * u) / (2.0 * std::sqrt(M_PI));
        const f64 ratio = qa / qe;
        wing *= 2.75 - (2.5 - 0.75 * ratio) * ratio;
        gii = cf * core + (1.0 - cf) * wing;
    }
    return gii;
}

}

TEST_CASE("SC PRD g_II matches Lightweaver", "[sc][prd]") {
    f64 max_err = 0.0;
    for (f64 a : {1e-4, 1e-2, 0.3}) {
        for (f64 qe = -9.0; qe <= 9.0; qe += 0.37) {
            fp_t q0, qN;
            ScPrdGii::range(fp_t(qe), &q0, &qN);
            for (f64 qa = qe - 10.0; qa <= qe + 10.0; qa += 0.11) {
                const f64 ref = lw_gii(a, qe, qa);
                const f64 g = ScPrdGii::gii(fp_t(a), fp_t(qe), fp_t(qa));
                max_err = std::max(max_err, std::abs(g - ref) / std::max(std::abs(ref), 1e-3));
                if (ref > 1e-5) {
                    // NOTE(cmo): The integration range covers the support (as
                    // Lightweaver, less the negligible core tail).
                    REQUIRE(qa >= q0 - 1e-5);
                    REQUIRE(qa <= qN + 1e-5);
                }
            }
        }
    }
    REQUIRE(max_err < 1e-4);
}

TEST_CASE("SC PRD lab-frame rho lookup", "[sc][prd]") {
    // NOTE(cmo): One line on knots 1..4 of a 6-knot grid, rho* = 10 + row.
    yakl::Array<fp_t, 1, yakl::memHost> wl_h("wl", 6);
    for (int i = 0; i < 6; ++i) {
        wl_h(i) = FP(100.0) + FP(0.01) * i * i;
    }
    ScPrd prd;
    prd.num_lines = 1;
    prd.num_rows = 4;
    yakl::Array<ScPrdLine, 1, yakl::memHost> lines_h("lines", 1);
    lines_h(0) = ScPrdLine{.kr = 0, .atom = 0, .i = 0, .j = 1, .blue = 1, .red = 5, .row0 = 0,
        .lambda0 = FP(100.05), .mass = FP(1.0), .stim_ratio = FP(1.0)};
    prd.lines = lines_h.createDeviceCopy();
    Fp1d wl = wl_h.createDeviceCopy();
    prd.wavelength = wl;
    Fp2dHost rho_h("rho", 4, 1);
    for (int r = 0; r < 4; ++r) {
        rho_h(r, 0) = FP(10.0) + r;
    }
    prd.rho = Rad16Array::from_fp(rho_h.createDeviceCopy());

    using ConstantsFP::c;
    // NOTE(cmo): (lab wavelength, velocity, expected): at v = 0 the knots
    // themselves, between them linear, beyond the line constant; a
    // velocity moves the comoving point by v lambda0 / c.
    const fp_t v_knot = (wl_h(3) - wl_h(2)) * c / FP(100.05);
    const std::vector<std::array<fp_t, 3>> cases{
        {wl_h(2), FP(0.0), FP(11.0)},
        {FP(0.5) * (wl_h(2) + wl_h(3)), FP(0.0), FP(11.5)},
        {wl_h(0), FP(0.0), FP(10.0)},
        {wl_h(5), FP(0.0), FP(13.0)},
        {wl_h(2), v_knot, FP(12.0)},
    };
    Fp1d out("out", cases.size());
    yakl::Array<fp_t, 2, yakl::memHost> in_h("in", cases.size(), 2);
    for (int i = 0; i < cases.size(); ++i) {
        in_h(i, 0) = cases[i][0];
        in_h(i, 1) = cases[i][1];
    }
    const auto in = in_h.createDeviceCopy();
    dex_parallel_for("rho lookup", FlatLoop<1>(cases.size()), YAKL_LAMBDA(i64 i) {
        out(i) = prd.rho_lab(0, 0, in(i, 0), in(i, 1));
    });
    const auto out_h = out.createHostCopy();
    for (int i = 0; i < cases.size(); ++i) {
        CHECK_THAT(out_h(i), WithinAbs(cases[i][2], 2e-3));
    }
}

TEST_CASE("SC 16-bit radiation storage", "[sc][half]") {
    // NOTE(cmo): Every value round trips exactly.
    for (u32 h = 0; h < 0x10000; ++h) {
        REQUIRE(float_to_rad16(rad16_to_float(u16(h))) == u16(h));
    }
    REQUIRE(rad16_to_float(float_to_rad16(1.0f)) == 1.0f);
    REQUIRE(rad16_to_float(0x0001) == std::ldexp(1.0f, -49));
    REQUIRE(rad16_to_float(0x0400) == std::ldexp(1.0f, -39));
    // NOTE(cmo): Ties to even, saturation, negative/NaN to 0.
    REQUIRE(float_to_rad16(1.0f + std::ldexp(1.0f, -11)) == float_to_rad16(1.0f));
    REQUIRE(float_to_rad16(1.0f + 3.0f * std::ldexp(1.0f, -11)) == float_to_rad16(1.0f) + 2);
    REQUIRE(float_to_rad16(1e30f) == RAD16_MAX_BITS);
    REQUIRE(float_to_rad16(-1.0f) == 0);
    REQUIRE(float_to_rad16(std::nanf("")) == 0);
    // NOTE(cmo): Nearest: neither neighbouring value is closer (ties: even).
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> e(-52, 23);
    std::uniform_real_distribution<f32> m(1.0f, 2.0f);
    for (int i = 0; i < 200000; ++i) {
        const f32 f = std::ldexp(m(rng), e(rng));
        const u16 h = float_to_rad16(f);
        const f64 d = std::abs(f64(rad16_to_float(h)) - f);
        for (int s : {-1, 1}) {
            const i32 n = i32(h) + s;
            if (n < 0 || n > RAD16_MAX_BITS) {
                continue;
            }
            const f64 dn = std::abs(f64(rad16_to_float(u16(n))) - f);
            REQUIRE((d < dn || (d == dn && (h & 1) == 0)));
        }
        if (f >= std::ldexp(1.0f, -39) && f < 1.6e7f) {
            REQUIRE(d <= std::ldexp(f64(f), -11));
        }
    }
}

TEST_CASE("SC DDA has bounded state and preserves clipped path lengths", "[sc][march]") {
    for (fp_t tx : {FP(0.0), FP(0.2), FP(1.0), FP(32.5), FP(1000.25)}) {
        for (fp_t ty : {FP(0.0), FP(0.3), FP(1.0), FP(27.5)}) {
            for (fp_t start : {FP(0.0), FP(0.5), FP(0.93)}) {
                ScRayMarch<2> march(FP(0.25), FP(0.25), tx, ty, start);
                double sum = 0;
                int count = 0;
                do {
                    const auto seg = march.segment();
                    REQUIRE(seg.fraction() >= FP(0.0));
                    REQUIRE(seg.end <= FP(1.0));
                    REQUIRE(seg.end >= seg.begin);
                    if (seg.axis == 0) REQUIRE(seg.x == fp_t(seg.face));
                    if (seg.axis == 1) REQUIRE(seg.y == fp_t(seg.face));
                    sum += seg.fraction();
                    REQUIRE(++count < 1100);
                } while (march.step());
                REQUIRE_THAT(sum, WithinAbs(1.0 - start, 2e-6));
                REQUIRE_FALSE(march.valid());
            }
        }
    }
    ScRayMarch<2> tied(FP(0.0), FP(0.0), FP(2.0), FP(2.0));
    REQUIRE(tied.segment().axis == 0);
    REQUIRE(tied.step());
    REQUIRE(tied.segment().axis == 1);
    REQUIRE(tied.segment().fraction() == FP(0.0));
    // A node endpoint must not also be treated as an interior crossing.
    ScRayMarch<1> node(FP(-0.7), FP(0.0), FP(1.7), FP(0.0), FP(0.0), 1);
    REQUIRE(node.segment().face == 0);
    REQUIRE(node.step());
    REQUIRE(node.segment().last());
    REQUIRE_FALSE(node.step());
    ScRayMarch<1> empty(FP(0.0), FP(0.0), FP(1.0), FP(0.0), FP(1.0));
    REQUIRE_FALSE(empty.valid());
}

TEST_CASE("RC and SC recover the split response with absolute component opacity", "[sc][gamma]") {
    for (fp_t complement : {FP(0.0), FP(1e-7), FP(0.25), FP(1.0), FP(1.1), FP(-0.1)}) {
        GammaStaticSums totals;
        totals.add({FP(2.0), FP(1e8)});
        totals.add({FP(3.0), -FP(1e8) + FP(8.0)});
        totals.add({FP(1e-4), FP(1e-4)});
        const GammaRay ray{FP(0.125), complement, FP(1.0)};
        const auto recovered = gamma_recover_response(ray, totals.eta, totals.chi, totals.chi_neg);
        REQUIRE(std::isfinite(recovered.I));
        REQUIRE(recovered.psi_star >= FP(0.0));
        REQUIRE(recovered.psi_star * FP(1e8) <= FP(1.0));
        const double alo = std::clamp(1.0 - double(complement), 0.0, 1.0);
        const double denominator = double(FP(1e8)) + double(FP(1e8) - FP(8.0)) + double(FP(1e-4));
        REQUIRE_THAT(double(recovered.psi_star), WithinAbs(alo / denominator, 1e-14));
    }
    GammaStaticSums positive;
    positive.add({FP(3.0), FP(2.0)});
    const auto r = gamma_recover_response({FP(0.5), FP(0.25), FP(1.0)}, positive.eta, positive.chi, positive.chi_neg);
    REQUIRE_THAT(r.psi_star, WithinAbs(0.75 / 2.0, 1e-7));
    REQUIRE_THAT(r.I, WithinAbs(0.5 + 0.75 / 2.0 * 3.0, 1e-7));
    const GammaStaticSums vacuum;
    const auto empty = gamma_recover_response({FP(2.0), FP(1.0), FP(1.0)}, vacuum.eta, vacuum.chi, vacuum.chi_neg);
    REQUIRE(empty.psi_star == FP(0.0));
    REQUIRE(empty.I == FP(2.0));
}

TEST_CASE("SC periodic indexing handles interior edges and multiple laps", "[sc][march]") {
    for (i32 n : {1, 2, 7, 16, 31, 128}) {
        for (i32 s = -17 * n; s < 17 * n; ++s) {
            const i32 expected = ((s % n) + n) % n;
            REQUIRE(sc_wrap(s, n) == expected);
        }
    }
}

TEST_CASE("SC buffer requirements match the selected method allocation", "[sc][storage]") {
    for (int dim : {2, 3}) {
        for (auto method : {ScCharacteristics::Short, ScCharacteristics::LongerCharacteristics}) {
            for (auto boundary : {ScXBoundary::Fixed, ScXBoundary::Periodic}) {
                const ScGrid grid{.num_x = 8, .num_y = dim == 3 ? 8 : 1, .num_z = 8,
                    .dx = FP(0.5), .dy = FP(0.75), .dz = FP(1.0), .dim = dim};
                ScOptions opts;
                opts.quadrature_order = 3;
                opts.characteristics = method;
                opts.x_boundary = opts.y_boundary = boundary;
                const auto dirs = build_sc_direction_set(opts, false, dim);
                std::vector<f64> z(8);
                for (int k = 0; k < 8; ++k) z[k] = k + 0.5;
                const ScPreparedGeometry geometry(grid, {.global_num_z = 8, .z_end = 8}, dirs, opts, z, {}, 1.0);
                const auto buffers = sc_make_buffers(geometry, 3);
                i64 actual = buffers.medium.chi.totElems() + buffers.medium.S.totElems()
                    + buffers.solution.I_ext.totElems() + buffers.solution.alo.totElems()
                    + buffers.solution.one_minus_alo.totElems() + buffers.incoming.totElems() + buffers.outgoing.totElems();
                if (method == ScCharacteristics::Short) {
                    const auto& scratch = buffers.short_storage();
                    for (const auto& array : {scratch.bottom, scratch.edge, scratch.ghost, scratch.side,
                            scratch.xface, scratch.yface, scratch.ghost_x, scratch.ghost_y}) {
                        if (array.initialized()) {
                            actual += array.totElems();
                        }
                    }
                    // The named arrays retain the exact old packed storage total.
                    const i64 nx = grid.num_x, ny = grid.num_y, np = grid.num_z;
                    const i64 sx = geometry.storage.side_x, sy = geometry.storage.side_y;
                    const i64 old_total = dim == 2 ? nx + np + 2 * (np + 1) + np * sx
                        : nx * ny + np * (ny + 1) + np * nx + np * (2 * (ny + 2) + 2 * nx)
                            + np * (ny * sx + sy * nx);
                    REQUIRE(geometry.storage.short_inflow.total() == old_total);
                    actual += scratch.intensity_planes.totElems();
                } else {
                    const auto& scratch = buffers.longer_storage();
                    for (const auto& array : {scratch.bottom, scratch.side_x, scratch.side_y}) {
                        if (array.initialized()) {
                            actual += array.totElems();
                        }
                    }
                    REQUIRE(geometry.storage.longer_inflow.total() == geometry.longer_layout.inflow_count);
                    if constexpr (HostDevSameSpace) actual += scratch.host_lines.totElems();
                }
                REQUIRE(actual == 3 * sc_max_batch_dirs(geometry.batches) * geometry.buffer_values_per_pair());
            }
        }
    }
}

TEST_CASE("SC Longer integration is a scalar formal solution", "[sc][formal]") {
    const fp_t I_in = FP(1.5), source = FP(2.25);
    for (fp_t tau : {FP(-1e-4), FP(0.0), FP(1e-8), FP(1e-3), FP(0.2), FP(1.0), FP(10.0), FP(1e6)}) {
        for (fp_t weight : {FP(0.0), FP(0.25), FP(1.0)}) {
            const auto result = sc_integrate_longer_segment(I_in, source, tau, weight);
            const double absorbed = -std::expm1(-double(tau));
            const double expected_one_minus_alo = tau == FP(0.0) ? 1.0 : absorbed / double(tau);
            const double expected_I = double(I_in) + (double(source) - I_in) * absorbed;
            const double expected_mean = weight * (expected_one_minus_alo * I_in + (1.0 - expected_one_minus_alo) * source);
            REQUIRE_THAT(result.I, WithinAbs(expected_I, 2e-6));
            REQUIRE_THAT(result.delta_I, WithinAbs(expected_I - I_in, 2e-6));
            REQUIRE_THAT(result.I_ext + result.alo * source, WithinAbs(expected_mean, 2e-6));
            REQUIRE_THAT(result.alo + result.one_minus_alo, WithinAbs(weight, 2e-6));
        }
    }
}
