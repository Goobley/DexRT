#if !defined(DEXRT_COLLISIONS_HPP)
#define DEXRT_COLLISIONS_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "Populations.hpp"
#include "LteHPops.hpp"

YAKL_INLINE fp_t interp_rates(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    i64 ks
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
    return interp(atmos.temperature(ks), temp_view, rate_view);
}

YAKL_INLINE void collision_omega(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp3d& C,
    const FpConst2d& n_star,
    const HPartFn<>& nh_lte,
    i64 ks
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, ks);
    const fp_t Cdown = seaton_c0 * atmos.ne(ks) * rate / (atom.g(coll.j) * std::sqrt(atmos.temperature(ks)));
    const fp_t Cup = Cdown * n_star(coll.j, ks) / n_star(coll.i, ks);
    C(coll.i, coll.j, ks) += Cdown;
    C(coll.j, coll.i, ks) += Cup;
}

YAKL_INLINE void collision_ci(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp3d& C,
    const FpConst2d& n_star,
    const HPartFn<>& nh_lte,
    i64 ks
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, ks);
    const fp_t dE = atom.energy(coll.j) - atom.energy(coll.i);
    const fp_t Cup = (
        (rate * atmos.ne(ks))
        * (
            std::exp(-dE / (k_B_eV * atmos.temperature(ks)))
            * std::sqrt(atmos.temperature(ks))
        )
    );
    const fp_t Cdown = Cup * n_star(coll.i, ks) / n_star(coll.j, ks);
    C(coll.i, coll.j, ks) += Cdown;
    C(coll.j, coll.i, ks) += Cup;
}

YAKL_INLINE void collision_ce(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp3d& C,
    const FpConst2d& n_star,
    const HPartFn<>& nh_lte,
    i64 ks
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, ks);
    const fp_t gij = atom.g(coll.i) / atom.g(coll.j);
    const fp_t Cdown = (rate * atmos.ne(ks)) * gij * std::sqrt(atmos.temperature(ks));
    const fp_t Cup = Cdown * n_star(coll.j, ks) / n_star(coll.i, ks);
    C(coll.i, coll.j, ks) += Cdown;
    C(coll.j, coll.i, ks) += Cup;
}

YAKL_INLINE void collision_cp(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp3d& C,
    const FpConst2d& n_star,
    const HPartFn<>& nh_lte,
    i64 ks
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, ks);
    // TODO(cmo): This is only LTE!
    fp_t n_hii;
    nh_lte(atmos.temperature(ks), atmos.ne(ks), atmos.nh_tot(ks), &n_hii);
    const fp_t Cdown = rate * n_hii;
    const fp_t Cup = Cdown * n_star(coll.j, ks) / n_star(coll.i, ks);
    C(coll.i, coll.j, ks) += Cdown;
    C(coll.j, coll.i, ks) += Cup;
}

YAKL_INLINE void collision_ch(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp3d& C,
    const FpConst2d& n_star,
    const HPartFn<>& nh_lte,
    i64 ks
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, ks);
    const fp_t nh0 = atmos.nh0(ks);
    const fp_t Cup = rate * nh0;
    const fp_t Cdown = Cup * n_star(coll.i, ks) / n_star(coll.j, ks);
    C(coll.i, coll.j, ks) += Cdown;
    C(coll.j, coll.i, ks) += Cup;
}

YAKL_INLINE void collision_charge_exc_h(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp3d& C,
    const FpConst2d& n_star,
    const HPartFn<>& nh_lte,
    i64 ks
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, ks);
    const fp_t nh0 = atmos.nh0(ks);
    const fp_t Cdown = rate * nh0;
    C(coll.i, coll.j, ks) += Cdown;
}

YAKL_INLINE void collision_charge_exc_p(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const Fp3d& C,
    const FpConst2d& n_star,
    const HPartFn<>& nh_lte,
    i64 ks
) {
    using namespace ConstantsFP;
    const fp_t rate = interp_rates(atmos, atom, coll, ks);
    // TODO(cmo): This is only LTE!
    fp_t n_hii;
    nh_lte(atmos.temperature(ks), atmos.ne(ks), atmos.nh_tot(ks), &n_hii);
    const fp_t Cup = rate * n_hii;
    C(coll.j, coll.i, ks) += Cup;
}

YAKL_INLINE void compute_collisions(
    const SparseAtmosphere& atmos,
    const CompAtom<fp_t>& atom,
    const Fp3d& C,
    const FpConst2d& n_star,
    const HPartFn<>& nh_lte,
    i64 ks
) {
    for (int i = 0; i < atom.collisions.extent(0); ++i) {
        const auto& coll = atom.collisions(i);
        switch (coll.type) {
            case CollRateType::Omega: {
                collision_omega(atmos, atom, coll, C, n_star, nh_lte, ks);
            } break;
            case CollRateType::CI: {
                collision_ci(atmos, atom, coll, C, n_star, nh_lte, ks);
            } break;
            case CollRateType::CE: {
                collision_ce(atmos, atom, coll, C, n_star, nh_lte, ks);
            } break;
            case CollRateType::CP: {
                collision_cp(atmos, atom, coll, C, n_star, nh_lte, ks);
            } break;
            case CollRateType::CH: {
                collision_ch(atmos, atom, coll, C, n_star, nh_lte, ks);
            } break;
            case CollRateType::ChargeExcH: {
                collision_charge_exc_h(atmos, atom, coll, C, n_star, nh_lte, ks);
            } break;
            case CollRateType::ChargeExcP: {
                collision_charge_exc_p(atmos, atom, coll, C, n_star, nh_lte, ks);
            } break;
        }
    }
}

void compute_collisions_to_gamma(State* state);

#else
#endif