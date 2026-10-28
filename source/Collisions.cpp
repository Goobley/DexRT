#include "Collisions.hpp"
#include "State3d.hpp"
#include <algorithm>

template <typename State>
CollisionalRatesOutput compute_collisional_rates_output(const State& state) {
    CollisionalRatesOutput result;
    const auto& atmos = state.atmos;
    const auto& nh_lte = state.nh_lte;
    const auto& adata_host = state.adata_host;
    const i64 num_space = state.pops.extent(1);

    for (int ia = 0; ia < adata_host.num_level.extent(0); ++ia) {
        std::vector<std::pair<int, int>> pair_list;
        for (int c = adata_host.coll_start(ia); c < adata_host.coll_start(ia) + adata_host.num_coll(ia); ++c) {
            const auto& coll = adata_host.collisions(c);
            const std::pair<int, int> p(std::min(coll.i, coll.j), std::max(coll.i, coll.j));
            if (std::find(pair_list.begin(), pair_list.end(), p) == pair_list.end()) {
                pair_list.emplace_back(p);
            }
        }
        const int num_pair = pair_list.size();
        yakl::Array<i32, 2, yakl::memHost> pairs_host("collision_pairs", std::max(num_pair, 1), 2);
        for (int p = 0; p < num_pair; ++p) {
            pairs_host(p, 0) = pair_list[p].first;
            pairs_host(p, 1) = pair_list[p].second;
        }
        const auto pairs = pairs_host.createDeviceCopy();
        Fp3d rates("collisional_rates", std::max(num_pair, 1), 2, num_space);
        rates = FP(0.0);

        const auto atom = extract_atom(state.adata, adata_host, ia);
        const int num_level = adata_host.num_level(ia);
        size_t scratch_size = ScratchView<fp_t**>::shmem_size(num_level, num_level);
        scratch_size += ScratchView<fp_t*>::shmem_size(num_level);
        FlatLoop<2> nxn_loop(num_level, num_level);
        Kokkos::parallel_for(
            "Collisional rates output",
            TeamPolicy(num_space, dex_team_size(std::min(square(num_level), 128))).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA (const KTeam& team) {
                const i64 ks = team.league_rank();
                ScratchView<fp_t**> Ck(team.team_scratch(0), num_level, num_level);
                ScratchView<fp_t*> n_star(team.team_scratch(0), num_level);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, nxn_loop.num_iter),
                    [&] (const int x) {
                        const auto args = nxn_loop.unpack(x);
                        Ck(args[0], args[1]) = FP(0.0);
                    }
                );
                team.team_barrier();
                scatter_add_collisions(team, atom, collision_atmos(atmos, ks), nh_lte, n_star.data(), Ck);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, num_pair),
                    [&] (const int p) {
                        const int i = pairs(p, 0);
                        const int j = pairs(p, 1);
                        rates(p, 0, ks) = Ck(j, i);
                        rates(p, 1, ks) = Ck(i, j);
                    }
                );
            }
        );
        Kokkos::fence();
        result.rates.emplace_back(rates);
        result.pairs.emplace_back(pairs_host);
    }
    return result;
}

template CollisionalRatesOutput compute_collisional_rates_output<State>(const State& state);
template CollisionalRatesOutput compute_collisional_rates_output<State3d>(const State3d& state);
