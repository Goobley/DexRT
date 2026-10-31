#include "ScAngularQuadrature.hpp"
#include "ScStepanQuadratures.hpp"
#include "Healpix.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <fmt/core.h>

namespace {

std::vector<ScHostDir> stepan_full_sphere(int order, bool mirrored) {
    const ScStepanTable* table = nullptr;
    for (const auto& t : SC_STEPAN_TABLES) {
        if (t.order == order) {
            table = &t;
        }
    }
    if (!table) {
        throw std::runtime_error(fmt::format("No Stepan quadrature of order {} (expected 3, 7, 9, 11, 13, 15)", order));
    }

    constexpr f64 deg2rad = 3.14159265358979323846 / 180.0;
    std::vector<ScHostDir> result;
    result.reserve(8 * table->num_points);
    // NOTE(cmo): The octant is completed by rotations of pi/2 about z and
    // reflection in z; mirroring in x and y instead (as Lightweaver does)
    // loses the exactness to order L (by a few % already in the second
    // moments).
    constexpr f64 half_pi = 0.5 * 3.14159265358979323846;
    for (int hemi = 0; hemi < 2; ++hemi) {
        const f64 sz = hemi ? -1.0 : 1.0;
        for (int quad = 0; quad < 4; ++quad) {
            for (int i = 0; i < table->num_points; ++i) {
                const auto& e = table->data[i];
                const f64 theta = e.theta_deg * deg2rad;
                f64 chi = e.chi_deg * deg2rad;
                f64 sx = 1.0, sy = 1.0;
                if (mirrored) {
                    sx = (quad == 1 || quad == 2) ? -1.0 : 1.0;
                    sy = (quad >= 2) ? -1.0 : 1.0;
                } else {
                    chi += quad * half_pi;
                }
                result.push_back(ScHostDir{
                    .mu = {
                        sx * std::sin(theta) * std::cos(chi),
                        sy * std::sin(theta) * std::sin(chi),
                        sz * std::cos(theta)
                    },
                    .weight = e.weight
                });
            }
        }
    }
    return result;
}

std::vector<ScHostDir> healpix_full_sphere(int order) {
    if (order < 0 || order > 8) {
        throw std::runtime_error(fmt::format("Healpix order {} outside expected range [0, 8]", order));
    }
    const i32 nside = 1 << order;
    const i32 npix = 12 * nside * nside;
    std::vector<ScHostDir> result;
    result.reserve(npix);
    for (int p = 0; p < npix; ++p) {
        const auto loc = healpix_nest_loc<f64, f64>(nside, p);
        result.push_back(ScHostDir{
            .mu = {
                std::cos(loc.phi) * loc.s,
                std::sin(loc.phi) * loc.s,
                loc.z
            },
            .weight = 1.0 / f64(npix)
        });
    }
    return result;
}

/// Rotate Healpix off the mu_z = 0 ring (in-plane rays see an endless path in
/// a plane-parallel atmosphere, I = S, over-weighting grazing radiation): the
/// rotation maximising min |mu_z| from a fixed (Kronecker) sequence, so
/// deterministic. 3D: any rotation (Euler angles z-x-z). 2D: about y only (the
/// set stays symmetric in y for the fold; rays along y are the 2.5D case).
void healpix_rotate(std::vector<ScHostDir>* dirs, i32 dim) {
    constexpr int tries = 4096;
    constexpr f64 g[3] = {0.7548776662466927, 0.5698402909980532, 0.6180339887498949};
    auto rotation = [&] (int i, f64 R[3][3]) {
        f64 a = 0.0, b, c = 0.0;
        if (dim == 3) {
            a = 2.0 * M_PI * std::fmod(i * g[0], 1.0);
            b = std::acos(1.0 - 2.0 * std::fmod(i * g[1], 1.0));
            c = 2.0 * M_PI * std::fmod(i * g[2], 1.0);
            const f64 ca = std::cos(a), sa = std::sin(a), cb = std::cos(b), sb = std::sin(b), cc = std::cos(c), sc = std::sin(c);
            const f64 Rz1[3][3] = {{ca, -sa, 0.0}, {sa, ca, 0.0}, {0.0, 0.0, 1.0}};
            const f64 Rx[3][3] = {{1.0, 0.0, 0.0}, {0.0, cb, -sb}, {0.0, sb, cb}};
            const f64 Rz2[3][3] = {{cc, -sc, 0.0}, {sc, cc, 0.0}, {0.0, 0.0, 1.0}};
            f64 T[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int k = 0; k < 3; ++k) {
                    T[r][k] = Rx[r][0] * Rz1[0][k] + Rx[r][1] * Rz1[1][k] + Rx[r][2] * Rz1[2][k];
                }
            }
            for (int r = 0; r < 3; ++r) {
                for (int k = 0; k < 3; ++k) {
                    R[r][k] = Rz2[r][0] * T[0][k] + Rz2[r][1] * T[1][k] + Rz2[r][2] * T[2][k];
                }
            }
        } else {
            b = 0.5 * M_PI * std::fmod(i * g[2], 1.0);
            const f64 cb = std::cos(b), sb = std::sin(b);
            const f64 Ry[3][3] = {{cb, 0.0, sb}, {0.0, 1.0, 0.0}, {-sb, 0.0, cb}};
            for (int r = 0; r < 3; ++r) {
                for (int k = 0; k < 3; ++k) {
                    R[r][k] = Ry[r][k];
                }
            }
        }
    };
    auto min_mu_z = [&] (const f64 R[3][3]) {
        f64 result = 1.0;
        for (const auto& d : *dirs) {
            const f64 x = R[0][0] * d.mu[0] + R[0][1] * d.mu[1] + R[0][2] * d.mu[2];
            const f64 z = R[2][0] * d.mu[0] + R[2][1] * d.mu[1] + R[2][2] * d.mu[2];
            if (dim == 2 && x * x + z * z < 1e-12) {
                continue;
            }
            result = std::min(result, std::abs(z));
        }
        return result;
    };
    f64 best[3][3];
    f64 best_mz = -1.0;
    for (int i = 1; i <= tries; ++i) {
        f64 R[3][3];
        rotation(i, R);
        const f64 mz = min_mu_z(R);
        if (mz > best_mz) {
            best_mz = mz;
            std::copy(&R[0][0], &R[0][0] + 9, &best[0][0]);
        }
    }
    for (auto& d : *dirs) {
        const std::array<f64, 3> m = d.mu;
        for (int r = 0; r < 3; ++r) {
            d.mu[r] = best[r][0] * m[0] + best[r][1] * m[1] + best[r][2] * m[2];
        }
    }
}

void normalise(std::vector<ScHostDir>* dirs) {
    f64 total = 0.0;
    for (const auto& d : *dirs) {
        total += d.weight;
    }
    for (auto& d : *dirs) {
        d.weight /= total;
    }
}

} // namespace

std::vector<ScHostDir> sc_full_sphere_quadrature(const ScOptions& opts, i32 dim) {
    std::vector<ScHostDir> result;
    if (opts.quadrature == ScQuadratureType::Stepan || opts.quadrature == ScQuadratureType::StepanMirrored) {
        result = stepan_full_sphere(opts.quadrature_order, opts.quadrature == ScQuadratureType::StepanMirrored);
    } else {
        result = healpix_full_sphere(opts.quadrature_order);
        healpix_rotate(&result, dim);
    }
    normalise(&result);
    return result;
}

std::vector<ScHostDir> sc_fold_y(const std::vector<ScHostDir>& dirs) {
    constexpr f64 tol = 1e-10;
    std::vector<ScHostDir> result;
    for (const auto& d : dirs) {
        ScHostDir r = d;
        r.mu[1] = std::abs(r.mu[1]);
        bool merged = false;
        for (auto& e : result) {
            if (
                std::abs(e.mu[0] - r.mu[0]) < tol
                && std::abs(e.mu[1] - r.mu[1]) < tol
                && std::abs(e.mu[2] - r.mu[2]) < tol
            ) {
                e.weight += r.weight;
                merged = true;
                break;
            }
        }
        if (!merged) {
            result.push_back(r);
        }
    }
    return result;
}

namespace {
/// Upload directions, output indices and boundary-cone geometry for either solve type.
ScDirectionSet sc_upload_direction_set(
    const ScOptions& opts,
    const std::vector<ScHostDir>& dirs,
    bool fold_y
) {
    ScDirectionSet result;
    result.num_dirs = dirs.size();
    result.folded_y = fold_y;
    result.mu_host = Fp2dHost("sc_mu", result.num_dirs, 3);
    result.wmu_host = Fp1dHost("sc_wmu", result.num_dirs);
    for (int d = 0; d < result.num_dirs; ++d) {
        for (int i = 0; i < 3; ++i) {
            result.mu_host(d, i) = dirs[d].mu[i];
        }
        result.wmu_host(d) = dirs[d].weight;
    }
    result.mu = result.mu_host.createDeviceCopy();
    result.wmu = result.wmu_host.createDeviceCopy();

    std::vector<i32> upward;
    for (i32 d = 0; d < result.num_dirs; ++d) {
        if (dirs[d].mu[2] > 0.0) {
            upward.push_back(d);
        }
    }
    result.emergent_dir_host = decltype(result.emergent_dir_host)("sc_emergent_dir", upward.size());
    yakl::Array<i32, 1, yakl::memHost> index("sc_emergent_index", result.num_dirs);
    index = -1;
    for (i32 e = 0; e < upward.size(); ++e) {
        result.emergent_dir_host(e) = upward[e];
        index(upward[e]) = e;
    }
    result.emergent_index = index.createDeviceCopy();

    Fp1dHost cone_cos("sc_cone_cos", result.num_dirs);
    for (int d = 0; d < result.num_dirs; ++d) {
        // NOTE(cmo): A folded direction stands for the +-mu_y pair.
        const bool pair = fold_y && std::abs(dirs[d].mu[1]) > 1e-10;
        const f64 w = pair ? 0.5 * dirs[d].weight : dirs[d].weight;
        cone_cos(d) = std::max(1.0 - 2.0 * w, -1.0);
    }
    result.cone_cos = cone_cos.createDeviceCopy();
    result.cone_num_mu = opts.boundary_cone_mu;
    result.cone_num_phi = opts.boundary_cone_phi;
    std::vector<f64> u, wu;
    sc_gauss_legendre_unit(std::max(opts.boundary_cone_mu, 1), &u, &wu);
    Fp1dHost cone_u("sc_cone_u", u.size());
    Fp1dHost cone_w("sc_cone_w", u.size());
    for (int i = 0; i < u.size(); ++i) {
        cone_u(i) = u[i];
        cone_w(i) = wu[i];
    }
    result.cone_u = cone_u.createDeviceCopy();
    result.cone_w = cone_w.createDeviceCopy();
    return result;
}
} // namespace

ScDirectionSet build_sc_direction_set(const ScOptions& opts, bool fold_y, i32 dim) {
    auto dirs = sc_full_sphere_quadrature(opts, dim);
    if (fold_y) {
        dirs = sc_fold_y(dirs);
    }
    return sc_upload_direction_set(opts, dirs, fold_y);
}

ScDirectionSet build_sc_final_direction_set(const ScOptions& opts) {
    std::vector<ScHostDir> dirs;
    for (const auto& mu : opts.final_mus) {
        dirs.push_back({.mu = mu, .weight = 0.0});
    }
    return sc_upload_direction_set(opts, dirs, false);
}

void sc_gauss_legendre_unit(int n, std::vector<f64>* nodes, std::vector<f64>* weights) {
    nodes->resize(n);
    weights->resize(n);
    for (int i = 0; i < n; ++i) {
        // NOTE(cmo): Newton iteration on P_n from the Chebyshev guess.
        f64 x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        f64 dp = 1.0;
        for (int it = 0; it < 100; ++it) {
            f64 p0 = 1.0;
            f64 p1 = x;
            for (int k = 2; k <= n; ++k) {
                const f64 p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            const f64 pn = (n == 1) ? x : p1;
            const f64 pm1 = (n == 1) ? 1.0 : p0;
            dp = n * (x * pn - pm1) / (x * x - 1.0);
            const f64 dx = pn / dp;
            x -= dx;
            if (std::abs(dx) < 1e-15) {
                break;
            }
        }
        (*nodes)[i] = 0.5 * (1.0 - x);
        (*weights)[i] = 1.0 / ((1.0 - x * x) * dp * dp);
    }
}
