#if !defined(DEXRT_SC_ANGULAR_QUADRATURE_HPP)
#define DEXRT_SC_ANGULAR_QUADRATURE_HPP

#include "Types.hpp"
#include "ScConfig.hpp"
#include <array>
#include <vector>

/// A direction on the unit sphere (propagation direction) and its weight.
struct ScHostDir {
    std::array<f64, 3> mu;
    f64 weight;
};

/// Full-sphere quadrature, weights normalised to 1 (Healpix rotated off the
/// mu_z = 0 ring for dim, 2 or 3).
std::vector<ScHostDir> sc_full_sphere_quadrature(const ScOptions& opts, i32 dim = 2);
/// Reflect every direction to my >= 0, merging coincident directions (valid
/// when the problem is symmetric in y, i.e. vy == 0 in 2D).
std::vector<ScHostDir> sc_fold_y(const std::vector<ScHostDir>& dirs);

struct ScDirectionSet {
    i32 num_dirs;
    bool folded_y;
    Fp2d mu; /// [dir, 3] propagation direction
    Fp1d wmu; /// [dir], sums to 1
    Fp2dHost mu_host;
    Fp1dHost wmu_host;
    /// Boundary cones (sc.boundary_cone_*): cos of each direction's cone
    /// half-angle (solid angle 4 pi w of one unfolded direction), and the
    /// Gauss-Legendre nodes/weights (sum 1) in u = (1 - cos a) / (1 - cos_cone).
    Fp1d cone_cos;
    Fp1d cone_u;
    Fp1d cone_w;
    i32 cone_num_mu = 0;
    i32 cone_num_phi = 1;
};

/// Gauss-Legendre nodes and weights on [0, 1] (weights sum to 1).
void sc_gauss_legendre_unit(int n, std::vector<f64>* nodes, std::vector<f64>* weights);

ScDirectionSet build_sc_direction_set(const ScOptions& opts, bool fold_y, i32 dim = 2);

#else
#endif
