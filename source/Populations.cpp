#include "Populations.hpp"
#include "KokkosBatched_Gesv.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"
#include "KokkosBlas.hpp"

void compute_lte_pops_flat(
    const CompAtom<fp_t>& atom,
    const SparseAtmosphere& atmos,
    const Fp2d& pops
) {
    const auto& temperature = atmos.temperature;
    const auto& ne = atmos.ne;
    const auto& nhtot = atmos.nh_tot;
    dex_parallel_for(
        "LTE Pops",
        FlatLoop<1>(pops.extent(1)),
        KOKKOS_LAMBDA (int64_t ks) {
            lte_pops<fp_t, fp_t>(
                atom.energy,
                atom.g,
                atom.stage,
                temperature(ks),
                ne(ks),
                atom.abundance * nhtot(ks),
                pops,
                ks
            );
        }
    );
}

/**
 * Computes the LTE populations in state. Assumes state->pops is already allocated.
*/
void compute_lte_pops(State* state) {
    for (int ia = 0; ia < state->atoms.size(); ++ia) {
        const auto& atom = state->atoms[ia];
        const auto pops = slice_pops(
            state->pops,
            state->adata_host,
            ia
        );
        compute_lte_pops_flat(atom, state->atmos, pops);
    }
}

void compute_lte_pops(const State* state, const Fp2d& shared_pops) {
    for (int ia = 0; ia < state->atoms.size(); ++ia) {
        const auto& atom = state->atoms[ia];
        const auto flat_pops = slice_pops(
            shared_pops,
            state->adata_host,
            ia
        );
        compute_lte_pops_flat(atom, state->atmos, flat_pops);
    }
}

void compute_nh0(const State& state) {
    const auto& nh0 = state.atmos.nh0;
    const auto& nh_lte = state.nh_lte;

    if (state.have_h) {
        // NOTE(cmo): This could just be a pointer shuffle...
        const auto& pops = state.pops;
        dex_parallel_for(
            "Copy nh0",
            FlatLoop<1>(nh0.extent(0)),
            KOKKOS_LAMBDA (i64 ks) {
                nh0(ks) = pops(0, ks);
            }
        );
    } else {
        const auto& atmos = state.atmos;
        dex_parallel_for(
            "Compute nh0 in LTE",
            FlatLoop<1>(nh0.extent(0)),
            KOKKOS_LAMBDA (i64 ks) {
                const fp_t temperature = atmos.temperature(ks);
                const fp_t ne = atmos.ne(ks);
                const fp_t nh_tot = atmos.nh_tot(ks);
                nh0(ks) = nh_lte(temperature, ne, nh_tot);
            }
        );
    }

    yakl::fence();
}

#ifdef DEXRT_USE_MAGMA

template <typename T=fp_t>
fp_t stat_eq_impl(State* state, const StatEqOptions& args = StatEqOptions()) {
    yakl::timer_start("Stat eq");
    JasUnpack(args, ignore_change_below_ntot_frac);
    fp_t global_max_change = FP(0.0);
    for (int ia = 0; ia < state->adata_host.num_level.extent(0); ++ia) {
        JasUnpack((*state), pops);
        const auto& Gamma = state->Gamma[ia];
        // GammaT has shape [ks, Nlevel, Nlevel]
        const fp_t abundance = state->adata_host.abundance(ia);
        const auto nh_tot = state->atmos.nh_tot;

        constexpr bool fractional_pops = true;
        constexpr bool iterative_improvement = true;
        constexpr int num_refinement_passes = 2;

        const i64 Nspace = Gamma.extent(2);
        const int pops_start = state->adata_host.level_start(ia);
        const int num_level = state->adata_host.num_level(ia);

        // NOTE(cmo): This allocation could be avoided, but we would need to fuse everything into one kernel.
        KView<T**> new_pops("new_pops", Nspace, num_level);

        size_t scratch_size = KView<T**>::shmem_size(num_level, num_level);
        if (iterative_improvement) {
            scratch_size *= 2;
            scratch_size += 2 * KView<T*>::shmem_size(num_level);
        }

        FlatLoop<2> nxn_loop(num_level, num_level);

        Kokkos::parallel_for(
            "Stat Eq",
            TeamPolicy(Nspace, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA (const KTeam& team) {
                const i64 ks = team.league_rank();

                const fp_t n_total_k = nh_tot(ks) * abundance;

                typedef Kokkos::MaxLoc<fp_t, int, DefaultExecutionSpace> ReducerDev;
                typedef ReducerDev::value_type ReductionVal;
                // Compute i_elim
                ReductionVal max_pop_loc;
                Kokkos::parallel_reduce(
                    Kokkos::TeamVectorRange(team, num_level),
                    [&] (const int i, ReductionVal& rval) {
                        const fp_t n = pops(i + pops_start, ks);
                        if (n > rval.val) {
                            rval.val = n;
                            rval.loc = i;
                        }
                    },
                    ReducerDev(max_pop_loc)
                );
                const int i_elim = max_pop_loc.loc;

                ScratchView<T**> Gammak(team.team_scratch(0), num_level, num_level);
                auto new_popsk = Kokkos::subview(new_pops, ks, Kokkos::ALL);

                // Copy over Gamma chunk
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, nxn_loop.num_iter),
                    [&] (const int x) {
                        const auto args = nxn_loop.unpack(x);
                        const int i = args[0];
                        const int j = args[1];

                        Gammak(i, j) = Gamma(i, j, ks);
                    }
                );
                team.team_barrier();

                // Fixup gamma
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, num_level),
                    [&] (const int i) {
                        T diag = T(FP(0.0));
                        Gammak(i, i) = diag;
                        for (int j = 0; j < num_level; ++j) {
                            diag += Gammak(j, i);
                        }
                        Gammak(i, i) = -diag;
                    }
                );

                // Setup rhs
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, num_level),
                    [&] (const int i) {
                        if (i_elim == i) {
                            if (fractional_pops) {
                                new_popsk(i) = T(FP(1.0));
                            } else {
                                new_popsk(i) = n_total_k;
                            }
                        } else {
                            new_popsk(i) = T(FP(0.0));
                        }
                    }
                );
                team.team_barrier();

                // Population conservation equation
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, nxn_loop.num_iter),
                    [&] (const int x) {
                        const auto args = nxn_loop.unpack(x);
                        const int i = args[0];
                        const int j = args[1];

                        if (i == i_elim) {
                            Gammak(i, j) = T(FP(1.0));
                        }
                    }
                );
                team.team_barrier();

                ScratchView<T**> Gamma_copy;
                ScratchView<T*> lhs;
                ScratchView<T*> residuals;
                if (iterative_improvement) {
                    Gamma_copy = ScratchView<T**>(team.team_scratch(0), num_level, num_level);
                    lhs = ScratchView<T*>(team.team_scratch(0), num_level);
                    residuals = ScratchView<T*>(team.team_scratch(0), num_level);

                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, nxn_loop.num_iter),
                        [&] (const int x) {
                            const auto args = nxn_loop.unpack(x);
                            const int i = args[0];
                            const int j = args[1];

                            if (i == 0) {
                                lhs(j) = new_popsk(j);
                                residuals(j) = new_popsk(j);
                            }
                            Gamma_copy(i, j) = Gammak(i, j);
                        }
                    );
                }

                team.team_barrier();
                // LU factorise
                KokkosBatched::LU<KTeam, KokkosBatched::Mode::Team, KokkosBatched::Algo::LU::Unblocked>::invoke(
                    team, Gammak
                );
                team.team_barrier();
                // LU Solve
                KokkosBatched::TeamSolveLU<
                    KTeam,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Trsm::Unblocked
                >::invoke(
                    team,
                    Gammak,
                    new_popsk
                );
                team.team_barrier();

                if (iterative_improvement) {
                    for (int refinement = 0; refinement < num_refinement_passes; ++refinement) {
                        // r_i = b_i
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(team, residuals.extent(0)),
                            [&] (int i) {
                                residuals(i) = lhs(i);
                            }
                        );
                        team.team_barrier();
                        // r -= Gamma x
                        KokkosBlas::Experimental::Gemv<KokkosBlas::Mode::TeamVector, KokkosBlas::Algo::Gemv::Default>::invoke(
                            team,
                            'n',
                            T(-1),
                            Gamma_copy,
                            new_popsk,
                            T(1),
                            residuals
                        );
                        team.team_barrier();
                        // Solve Gamma x' = r (already factorised)
                        KokkosBatched::TeamSolveLU<
                            KTeam,
                            KokkosBatched::Trans::NoTranspose,
                            KokkosBatched::Algo::Trsm::Unblocked
                        >::invoke(
                            team,
                            Gammak,
                            residuals
                        );
                        team.team_barrier();
                        // x += x'
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(
                                team,
                                new_popsk.extent(0)
                            ),
                            [&] (int i) {
                                new_popsk(i) += residuals(i);
                            }
                        );
                        team.team_barrier();
                    }
                }
            }
        );
        Kokkos::fence();

        typedef Kokkos::MaxLoc<fp_t, Kokkos::pair<int, int>> Reducer;
        typedef Reducer::value_type ReductionVal;
        ReductionVal max_change_loc;
        dex_parallel_reduce(
            "Update pops and compute max change",
            FlatLoop<2>(Nspace, num_level),
            KOKKOS_LAMBDA (int ks, int i, ReductionVal& rval) {
                fp_t change = FP(0.0);
                const fp_t n_total_k = nh_tot(ks) * abundance;
                fp_t new_pop_scaled = new_pops(ks, i);
                if (fractional_pops) {
                    new_pop_scaled *= n_total_k;
                }

                // compute change
                if (pops(pops_start + i, ks) < ignore_change_below_ntot_frac * n_total_k) {
                    change = FP(0.0);
                } else {
                    change = std::abs(FP(1.0) - pops(pops_start + i, ks) / new_pop_scaled);
                }

                // update
                pops(pops_start + i, ks) = new_pop_scaled;

                // reduce update
                if (change > rval.val) {
                    rval.val = change;
                    rval.loc = Kokkos::make_pair(ks, i);
                }
            },
            Reducer(max_change_loc)
        );

        const fp_t max_change = max_change_loc.val;
        auto temp_val = Kokkos::subview(state->atmos.temperature, max_change_loc.loc.first);
        auto temp_h = Kokkos::create_mirror_view_and_copy(HostSpace{}, temp_val);

        state->println(
            "     Max Change (ele: {}, Z={}): {} (@ l={}, ks={}) [T={}]",
            ia,
            state->adata_host.Z(ia),
            max_change,
            max_change_loc.loc.second,
            max_change_loc.loc.first,
            temp_h()
        );
        global_max_change = std::max(max_change, global_max_change);

    }
    yakl::timer_stop("Stat eq");
    return global_max_change;
}
#else
template <typename T=fp_t>
fp_t stat_eq_impl(State* state, const StatEqOptions& args = StatEqOptions()) { return FP(0.0); }
#endif

fp_t stat_eq(State* state, const StatEqOptions& args) {
#ifdef HAVE_MPI
    fp_t max_rel_change;
    if (state->mpi_state.rank == 0) {
        max_rel_change = stat_eq_impl<StatEqPrecision>(state, args);
    }
    MPI_Bcast(&max_rel_change, 1, get_FpMpi(), 0, state->mpi_state.comm);
    return max_rel_change;
#else
    return stat_eq_impl<StatEqPrecision>(state, args);
#endif
}
