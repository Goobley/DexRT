#if !defined(DEXRT_SC_BOUNDARY_CONE_HPP)
#define DEXRT_SC_BOUNDARY_CONE_HPP

#include "Types.hpp"
#include "ScAngularQuadrature.hpp"
#include "ScGeometry.hpp"
#include "PromweaverBoundary.hpp"
#include <type_traits>

/// The direction set's boundary cone quadrature, for use in kernels.
struct ScConeQuad {
    Fp1d cos_cone; /// [dir]
    Fp1d u;
    Fp1d w;
    i32 num_mu;
    i32 num_phi;

    explicit ScConeQuad(const ScDirectionSet& dirs)
        : cos_cone(dirs.cone_cos), u(dirs.cone_u), w(dirs.cone_w), num_mu(dirs.cone_num_mu),
          num_phi(dirs.cone_num_phi) {}
};

/// sample_boundary for viewing direction dir (direction dir_idx of the set). For
/// the Promweaver boundary with a cone quadrature, the mean over dir's
/// equal-area cone (as Promweaver's ConePromBc averages over each ray's cone);
/// otherwise the value along dir.
template <class Bc>
YAKL_INLINE fp_t sc_sample_boundary_cone(
    const Bc& bc,
    int la,
    vec3 at,
    vec3 dir,
    i32 dir_idx,
    const ScConeQuad& cone
) {
    if constexpr (std::is_same_v<Bc, PwBc<>>) {
        if (cone.num_mu > 0) {
            // NOTE(cmo): Orthonormal basis (e1, e2) perpendicular to dir.
            vec3 a;
            a(0) = (std::abs(dir(0)) < FP(0.9)) ? FP(1.0) : FP(0.0);
            a(1) = FP(1.0) - a(0);
            a(2) = FP(0.0);
            vec3 e1;
            e1(0) = a(1) * dir(2) - a(2) * dir(1);
            e1(1) = a(2) * dir(0) - a(0) * dir(2);
            e1(2) = a(0) * dir(1) - a(1) * dir(0);
            const fp_t inv_len = FP(1.0) / std::sqrt(square(e1(0)) + square(e1(1)) + square(e1(2)));
            for (int i = 0; i < 3; ++i) {
                e1(i) *= inv_len;
            }
            vec3 e2;
            e2(0) = dir(1) * e1(2) - dir(2) * e1(1);
            e2(1) = dir(2) * e1(0) - dir(0) * e1(2);
            e2(2) = dir(0) * e1(1) - dir(1) * e1(0);

            const fp_t one_minus_cos = FP(1.0) - cone.cos_cone(dir_idx);
            const fp_t w_phi = FP(1.0) / fp_t(cone.num_phi);
            fp_t result = FP(0.0);
            for (int i = 0; i < cone.num_mu; ++i) {
                const fp_t ca = FP(1.0) - one_minus_cos * cone.u(i);
                const fp_t sa = std::sqrt(std::max(FP(1.0) - square(ca), FP(0.0)));
                fp_t acc = FP(0.0);
                for (int j = 0; j < cone.num_phi; ++j) {
                    const fp_t psi = FP(2.0) * ConstantsFP::pi * (fp_t(j) + FP(0.5)) * w_phi;
                    const fp_t cp = std::cos(psi);
                    const fp_t sp = std::sin(psi);
                    vec3 sub;
                    for (int k = 0; k < 3; ++k) {
                        sub(k) = ca * dir(k) + sa * (cp * e1(k) + sp * e2(k));
                    }
                    acc += sample_boundary(bc, la, at, sub);
                }
                result += cone.w(i) * w_phi * acc;
            }
            return result;
        }
    }
    return sample_boundary(bc, la, at, dir);
}

/// A single direction/wavelength bound to a boundary condition. Algorithms
/// supply their actual entry locations; cone and viewing-direction handling
/// are identical in Short/Longer and in 2D/3D. Borrow captured kernel state:
/// copying array handles for every boundary entry is expensive on the CPU.
template <int Dim, class Bc> struct ScBoundarySampler {
    const ScGrid& grid;
    const ScFrame& frame;
    const Bc& bc;
    const ScConeQuad& cone;
    vec3 direction;
    i32 dir;
    i32 wavelength;

    YAKL_INLINE fp_t operator()(fp_t p, fp_t s, fp_t sy = FP(0.0)) const {
        vec3 at;
        if constexpr (Dim == 2) {
            at = sc_canonical_position(grid, frame, p, s);
        } else {
            at = sc_canonical_position_3d(
                grid,
                frame,
                p,
                s,
                sy
            );
        }
        return sc_sample_boundary_cone(
            bc,
            wavelength,
            at,
            direction,
            dir,
            cone
        );
    }
};

template <int Dim, class Bc>
YAKL_INLINE ScBoundarySampler<Dim, Bc> sc_boundary_sampler(
    const ScGrid& grid,
    const ScFrame& frame,
    const Bc& bc,
    const ScConeQuad& cone,
    const Fp2d& mu,
    i32 dir,
    i32 wavelength
) {
    vec3 direction;
    for (i32 i = 0; i < 3; ++i) {
        direction(i) = -mu(dir, i);
    }
    return {grid, frame, bc, cone, direction, dir, wavelength};
}

#else
#endif
