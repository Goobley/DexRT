#if !defined(DEXRT_COLLISIONS_HPP)
#define DEXRT_COLLISIONS_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "LteHPops.hpp"

YAKL_INLINE fp_t interp_rates(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    int x,
    int y
) {
    yakl::Array<fp_t const, 1, yakl::memDevice> temp_view(
        "temp_view",
        &atom.temperature(coll.start_idx),
        coll.end_idx - coll.start_idx
    );
    yakl::Array<fp_t const, 1, yakl::memDevice> rate_view(
        "rate_view",
        &atom.coll_rates(coll.start_idx),
        coll.end_idx - coll.start_idx
    );
    return interp(atmos.temperature(x, y), temp_view, rate_view);
}

YAKL_INLINE void collision_omega(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp4d& C,
    const FpConst3d& n_star,
    const HPartFn<>& nh_lte,
    int x,
    int y
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, x, y);
    const fp_t Cdown = seaton_c0 * atmos.ne(x, y) * rate / (atom.g(coll.j) * std::sqrt(atmos.temperature(x, y)));
    const fp_t Cup = Cdown * n_star(coll.j, x, y) / n_star(coll.i, x, y);
    C(coll.i, coll.j, x, y) += Cdown;
    C(coll.j, coll.i, x, y) += Cup;
}

YAKL_INLINE void collision_ci(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp4d& C,
    const FpConst3d& n_star,
    const HPartFn<>& nh_lte,
    int x,
    int y
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, x, y);
    const fp_t dE = atom.energy(coll.j) - atom.energy(coll.i);
    const fp_t Cup = (
        (rate * atmos.ne(x, y))
        * (
            std::exp(-dE / (k_B_eV * atmos.temperature(x, y)))
            * std::sqrt(atmos.temperature(x, y))
        )
    );
    const fp_t Cdown = Cup * n_star(coll.i, x, y) / n_star(coll.j, x, y);
    C(coll.i, coll.j, x, y) += Cdown;
    C(coll.j, coll.i, x, y) += Cup;
}

YAKL_INLINE void collision_ce(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp4d& C,
    const FpConst3d& n_star,
    const HPartFn<>& nh_lte,
    int x,
    int y
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, x, y);
    const fp_t gij = atom.g(coll.i) / atom.g(coll.j);
    const fp_t Cdown = (rate * atmos.ne(x, y)) * gij * std::sqrt(atmos.temperature(x, y));
    const fp_t Cup = Cdown * n_star(coll.j, x, y) / n_star(coll.i, x, y);
    C(coll.i, coll.j, x, y) += Cdown;
    C(coll.j, coll.i, x, y) += Cup;
}

YAKL_INLINE void collision_cp(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp4d& C,
    const FpConst3d& n_star,
    const HPartFn<>& nh_lte,
    int x,
    int y
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, x, y);
    // TODO(cmo): This is only LTE!
    fp_t n_hii;
    nh_lte(atmos.temperature(x, y), atmos.ne(x, y), atmos.nh_tot(x, y), &n_hii);
    const fp_t Cdown = rate * n_hii;
    const fp_t Cup = Cdown * n_star(coll.j, x, y) / n_star(coll.i, x, y);
    C(coll.i, coll.j, x, y) += Cdown;
    C(coll.j, coll.i, x, y) += Cup;
}

YAKL_INLINE void collision_ch(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp4d& C,
    const FpConst3d& n_star,
    const HPartFn<>& nh_lte,
    int x,
    int y
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, x, y);
    // TODO(cmo): This is only LTE!
    const fp_t nh0 = nh_lte(atmos.temperature(x, y), atmos.ne(x, y), atmos.nh_tot(x, y));
    const fp_t Cup = rate * nh0;
    const fp_t Cdown = Cup * n_star(coll.i, x, y) / n_star(coll.j, x, y);
    C(coll.i, coll.j, x, y) += Cdown;
    C(coll.j, coll.i, x, y) += Cup;
}

YAKL_INLINE void collision_charge_exc_h(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp4d& C,
    const FpConst3d& n_star,
    const HPartFn<>& nh_lte,
    int x,
    int y
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, x, y);
    // TODO(cmo): This is only LTE!
    const fp_t nh0 = nh_lte(atmos.temperature(x, y), atmos.ne(x, y), atmos.nh_tot(x, y));
    const fp_t Cdown = rate * nh0;
    C(coll.i, coll.j, x, y) += Cdown;
}

YAKL_INLINE void collision_charge_exc_p(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp4d& C,
    const FpConst3d& n_star,
    const HPartFn<>& nh_lte,
    int x,
    int y
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, x, y);
    // TODO(cmo): This is only LTE!
    fp_t n_hii;
    nh_lte(atmos.temperature(x, y), atmos.ne(x, y), atmos.nh_tot(x, y), &n_hii);
    const fp_t Cup = rate * n_hii;
    C(coll.j, coll.i, x, y) += Cup;
}

YAKL_INLINE void compute_collisions(
    const Atmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const Fp4d& C,
    const FpConst3d& n_star,
    const HPartFn<>& nh_lte,
    int x,
    int y
) {
    for (int i = 0; i < atom.collisions.extent(0); ++i) {
        const auto& coll = atom.collisions(i);
        switch (coll.type) {
            case CollRateType::Omega: {
                collision_omega(atmos, atom, coll, C, n_star, nh_lte, x, y);
            } break;
            case CollRateType::CI: {
                collision_ci(atmos, atom, coll, C, n_star, nh_lte, x, y);
            } break;
            case CollRateType::CE: {
                collision_ce(atmos, atom, coll, C, n_star, nh_lte, x, y);
            } break;
            case CollRateType::CP: {
                collision_cp(atmos, atom, coll, C, n_star, nh_lte, x, y);
            } break;
            case CollRateType::CH: {
                collision_ch(atmos, atom, coll, C, n_star, nh_lte, x, y);
            } break;
            case CollRateType::ChargeExcH: {
                collision_charge_exc_h(atmos, atom, coll, C, n_star, nh_lte, x, y);
            } break;
            case CollRateType::ChargeExcP: {
                collision_charge_exc_p(atmos, atom, coll, C, n_star, nh_lte, x, y);
            } break;
        }
    }
}

inline void compute_collisions_to_gamma(State* state) {
    const auto& atmos = state->atmos;
    const auto& Gamma = state->Gamma;
    const auto& atom = state->atom;
    const auto& nh_lte = state->nh_lte;
    const auto atmos_dims = atmos.temperature.get_dimensions();

    // TODO(cmo): Get rid of this!
    auto n_star = state->pops.createDeviceObject();
    auto n_star_flat = n_star.reshape<2>(Dims(n_star.extent(0), n_star.extent(1) * n_star.extent(2)));
    // NOTE(cmo): Zero Gamma before we start to refill it.
    Gamma = FP(0.0);
    compute_lte_pops_flat(atom, atmos, n_star_flat);
    yakl::fence();

    parallel_for(
        "collisions",
        SimpleBounds<2>(atmos_dims(0), atmos_dims(1)),
        YAKL_LAMBDA (int x, int y) {
            compute_collisions(
                atmos,
                atom,
                Gamma,
                n_star,
                nh_lte,
                x,
                y
            );
        }
    );
}

#else
#endif