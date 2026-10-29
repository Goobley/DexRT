#if !defined(DEXRT_COLLISIONS_HPP)
#define DEXRT_COLLISIONS_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "State.hpp"
#include "Populations.hpp"
#include "LteHPops.hpp"
#include "EmisOpac.hpp"

/// Local data inputs to collision functions
template <typename NStar>
struct CollisionCell {
    const AtmosPointParams& atmos;
    const NStar& n_star;
    const HPartFn<>& nh_lte;
};

YAKL_INLINE fp_t interp_rates(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    fp_t temperature
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
    return interp(temperature, temp_view, rate_view);
}

/// The upward (i -> j, C(j, i)) and downward (j -> i, C(i, j)) values of one rate.
struct CollisionRates {
    fp_t up;
    fp_t down;
};

template <typename NStar>
YAKL_INLINE CollisionRates collision_omega(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const CollisionCell<NStar>& cell
) {
    using namespace ConstantsFP;
    JasUnpack(cell, atmos, n_star);
    const fp_t rate = interp_rates(atom, coll, atmos.temperature);
    CollisionRates result;
    result.down = seaton_c0 * atmos.ne * rate / (atom.g(coll.j) * std::sqrt(atmos.temperature));
    result.up = result.down * n_star(coll.j) / n_star(coll.i);
    return result;
}

template <typename NStar>
YAKL_INLINE CollisionRates collision_ci(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const CollisionCell<NStar>& cell
) {
    using namespace ConstantsFP;
    JasUnpack(cell, atmos, n_star);
    const fp_t rate = interp_rates(atom, coll, atmos.temperature);
    const fp_t dE = atom.energy(coll.j) - atom.energy(coll.i);
    CollisionRates result;
    result.up = (
        (rate * atmos.ne)
        * (
            std::exp(-dE / (k_B_eV * atmos.temperature))
            * std::sqrt(atmos.temperature)
        )
    );
    result.down = result.up * n_star(coll.i) / n_star(coll.j);
    return result;
}

template <typename NStar>
YAKL_INLINE CollisionRates collision_ce(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const CollisionCell<NStar>& cell
) {
    using namespace ConstantsFP;
    JasUnpack(cell, atmos, n_star);
    const fp_t rate = interp_rates(atom, coll, atmos.temperature);
    const fp_t gij = atom.g(coll.i) / atom.g(coll.j);
    CollisionRates result;
    result.down = (rate * atmos.ne) * gij * std::sqrt(atmos.temperature);
    result.up = result.down * n_star(coll.j) / n_star(coll.i);
    return result;
}

template <typename NStar>
YAKL_INLINE CollisionRates collision_cp(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const CollisionCell<NStar>& cell
) {
    JasUnpack(cell, atmos, n_star, nh_lte);
    const fp_t rate = interp_rates(atom, coll, atmos.temperature);
    // TODO(cmo): This is only LTE!
    fp_t n_hii;
    nh_lte(atmos.temperature, atmos.ne, atmos.nhtot, &n_hii);
    CollisionRates result;
    result.down = rate * n_hii;
    result.up = result.down * n_star(coll.j) / n_star(coll.i);
    return result;
}

template <typename NStar>
YAKL_INLINE CollisionRates collision_ch(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const CollisionCell<NStar>& cell
) {
    JasUnpack(cell, atmos, n_star);
    const fp_t rate = interp_rates(atom, coll, atmos.temperature);
    CollisionRates result;
    result.up = rate * atmos.nh0;
    result.down = result.up * n_star(coll.i) / n_star(coll.j);
    return result;
}

template <typename NStar>
YAKL_INLINE CollisionRates collision_charge_exc_h(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const CollisionCell<NStar>& cell
) {
    JasUnpack(cell, atmos);
    const fp_t rate = interp_rates(atom, coll, atmos.temperature);
    return CollisionRates{.up = FP(0.0), .down = rate * atmos.nh0};
}

template <typename NStar>
YAKL_INLINE CollisionRates collision_charge_exc_p(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const CollisionCell<NStar>& cell
) {
    JasUnpack(cell, atmos, nh_lte);
    const fp_t rate = interp_rates(atom, coll, atmos.temperature);
    // TODO(cmo): This is only LTE!
    fp_t n_hii;
    nh_lte(atmos.temperature, atmos.ne, atmos.nhtot, &n_hii);
    return CollisionRates{.up = rate * n_hii, .down = FP(0.0)};
}

template <typename NStar>
YAKL_INLINE CollisionRates collision_rates(
    const CompAtom<fp_t>& atom,
    const CompColl<fp_t>& coll,
    const CollisionCell<NStar>& cell
) {
    switch (coll.type) {
        case CollRateType::Omega: {
            return collision_omega(atom, coll, cell);
        }
        case CollRateType::CI: {
            return collision_ci(atom, coll, cell);
        }
        case CollRateType::CE: {
            return collision_ce(atom, coll, cell);
        }
        case CollRateType::CP: {
            return collision_cp(atom, coll, cell);
        }
        case CollRateType::CH: {
            return collision_ch(atom, coll, cell);
        }
        case CollRateType::ChargeExcH: {
            return collision_charge_exc_h(atom, coll, cell);
        }
        case CollRateType::ChargeExcP: {
            return collision_charge_exc_p(atom, coll, cell);
        }
    }
    return CollisionRates{.up = FP(0.0), .down = FP(0.0)};
}

/// Non-atomic.
YAKL_INLINE void lte_pops_cell(
    const CompAtom<fp_t>& atom,
    const AtmosPointParams& atmos,
    const yakl::Array<fp_t, 1, yakl::memDevice>& n_star
) {
    const yakl::Array<fp_t, 2, yakl::memDevice> n_star_view("n_star", n_star.data(), n_star.extent(0), 1);
    lte_pops(
        atom.energy,
        atom.g,
        atom.stage,
        atmos.temperature,
        atmos.ne,
        atom.abundance * atmos.nhtot,
        n_star_view,
        0
    );
}

/// Add collisional rates to a 2D (i, j) (for rate j -> i) matrix. Anticipated
/// to be transiently stored in scratch/shared memory (same as n_star). Team
/// barrier at the end. Needs to be called before fixing the diagonal of the
/// rate matrix.
template <typename Team, typename Mat>
KOKKOS_INLINE_FUNCTION void scatter_add_collisions(
    const Team& team,
    const CompAtom<fp_t>& atom,
    const AtmosPointParams& atmos,
    const HPartFn<>& nh_lte,
    fp_t* n_star,
    const Mat& C
) {
    const yakl::Array<fp_t, 1, yakl::memDevice> n_star_k("n_star", n_star, atom.energy.extent(0));
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
        lte_pops_cell(atom, atmos, n_star_k);
    });
    team.team_barrier();
    const CollisionCell<decltype(n_star_k)> cell{
        .atmos = atmos,
        .n_star = n_star_k,
        .nh_lte = nh_lte
    };
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team, atom.collisions.extent(0)),
        [&] (const int c) {
            const auto& coll = atom.collisions(c);
            const CollisionRates rates = collision_rates(atom, coll, cell);
            using CT = std::remove_cvref_t<decltype(C(0, 0))>;
            Kokkos::atomic_add(&C(coll.i, coll.j), CT(rates.down));
            Kokkos::atomic_add(&C(coll.j, coll.i), CT(rates.up));
        }
    );
    team.team_barrier();
}

/// The atmosphere at cell ks, for the collisional rates.
template <typename Atmos>
YAKL_INLINE AtmosPointParams collision_atmos(const Atmos& atmos, i64 ks) {
    AtmosPointParams result;
    result.temperature = atmos.temperature(ks);
    result.ne = atmos.ne(ks);
    result.nhtot = atmos.nh_tot(ks);
    result.nh0 = atmos.nh0(ks);
    return result;
}

template <typename State>
CollisionalRatesOutput compute_collisional_rates_output(const State& state);

#else
#endif