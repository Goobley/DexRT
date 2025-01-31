#include "ChargeConservation.hpp"
#include "KokkosBatched_Gesv.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"
#include "KokkosBlas.hpp"

#if 1
// #if 0
template <typename T=fp_t>
fp_t nr_post_update_impl(State* state, const NrPostUpdateOptions& args = NrPostUpdateOptions()) {
    yakl::timer_start("Charge conservation");
    JasUnpack(args, ignore_change_below_ntot_frac, conserve_pressure);
    JasUnpack((*state), atmos, nh_lte);
    // TODO(cmo): Add background n_e term like in Lw.
    // NOTE(cmo): Only considers H for now
    // TODO(cmo): He contribution?
    assert(state->have_h && "Need to have H active for non-lte EOS");
    const auto& pops = state->pops;
    const auto& GammaH = state->Gamma[0];
    const int num_level = GammaH.extent(0);
    const int num_eqn = GammaH.extent(0) + 1;
    const int num_space = GammaH.extent(2);
    JasUnpack(state->atmos, ne, nh_tot, pressure, temperature);
    // NOTE(cmo): GammaH_flat is how we access Gamma/C in the following
    const auto& GammaH_flat = state->Gamma[0];

    fp_t total_abund = FP(0.0);
    if constexpr (false) {
        for (int ia = 0; ia < state->adata_host.num_level.extent(0); ++ia) {
            total_abund += state->adata_host.abundance(ia);
        }
    } else {
        // NOTE(cmo): From Asplund 2009/Lw calc
        total_abund = FP(1.0861550335264554);
        // NOTE(cmo): 1.1 is traditionally used to account for He, but it's all much of a muchness
    }
    constexpr fp_t ne_pert_size = FP(1e-2);
    constexpr bool iterative_improvement = true;
    constexpr int num_refinement_passes = 2;

    const auto& H_atom = extract_atom(state->adata, state->adata_host, 0);

    size_t scratch_size = ScratchView<T**>::shmem_size(num_level, num_level); // Gammak
    scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // C
    scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // C_ne_pert
    scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // dC
    scratch_size += ScratchView<T**>::shmem_size(num_eqn, num_eqn); // dF
    scratch_size += ScratchView<T*>::shmem_size(num_eqn); // F
    if (iterative_improvement) {
        scratch_size += ScratchView<T**>::shmem_size(num_eqn, num_eqn); // dF copy
        scratch_size += 2 * ScratchView<T*>::shmem_size(num_eqn); // lhs_copy/residuals
    }
    KView<T**> new_pops("new_pops", num_space, num_level);
    KView<T**> new_F("new_pops", num_space, num_eqn);
    FlatLoop<2> nlxnl_loop(num_level, num_level);
    FlatLoop<2> nexne_loop(num_eqn, num_eqn);

    KView<fp_t**> n_star("n_star", state->pops.layout());
    compute_lte_pops(state, n_star);
    Kokkos::fence();
    const auto n_star_slice = slice_pops(
        n_star,
        state->adata_host,
        state->atoms_with_gamma_mapping[0]
    );

    Kokkos::parallel_for(
        "Charge Conservation",
        TeamPolicy(num_space, Kokkos::AUTO()).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA (const KTeam& team) {
            const i64 ks = team.league_rank();

            ScratchView<fp_t**> C(team.team_scratch(0), num_level, num_level);
            ScratchView<fp_t**> C_ne_pert(team.team_scratch(0), num_level, num_level);
            ScratchView<fp_t**> dC(team.team_scratch(0), num_level, num_level);

            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
                [&] (const int x) {
                    const auto args = nlxnl_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];
                    C(i, j) = FP(0.0);
                    C_ne_pert(i, j) = FP(0.0);
                }
            );
            team.team_barrier();

            // Compute changes in C
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                compute_C_ne_pert(
                    atmos,
                    H_atom,
                    n_star_slice,
                    nh_lte,
                    ks,
                    C,
                    C_ne_pert,
                    ne_pert_size
                );
            });

            // compute dCdne
            const fp_t ne_k = atmos.ne(ks);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
                [&] (const int x) {
                    const auto args = nlxnl_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];

                    dC(i, j) = (C_ne_pert(i, j) - C(i, j)) / (ne_pert_size * ne_k);
                }
            );
            team.team_barrier();
            // Fixup C and dC
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, num_level),
                [&] (const int i) {
                    fp_t diag = FP(0.0);
                    fp_t diag_dC = FP(0.0);
                    C(i, i) = diag;
                    dC(i, i) = diag_dC;
                    for (int j = 0; j < num_level; ++j) {
                        diag += C(j, i);
                        diag_dC += dC(j, i);
                    }
                    C(i, i) = -diag;
                    dC(i, i) = -diag_dC;
                }
            );
            team.team_barrier();

            ScratchView<T**> Gammak(team.team_scratch(0), num_level, num_level);
            auto new_popsk = Kokkos::subview(new_pops, ks, Kokkos::ALL);
            // Copy over Gamma and new_pops chunks
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
                [&] (const int x) {
                    const auto args = nlxnl_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];

                    Gammak(i, j) = GammaH_flat(i, j, ks);
                    new_popsk(j) = pops(j, ks);
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
            team.team_barrier();

            ScratchView<T**> dF(team.team_scratch(0), num_eqn, num_eqn);
            ScratchView<T*> F(team.team_scratch(0), num_eqn);
            // Compute LHS, based on Lightspinner impl
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, num_eqn),
                [&] (const int i) {
                    if (i < (num_level - 1)) {
                        T Fi = FP(0.0);
                        for (int j = 0; j < num_level; ++j) {
                            Fi += Gammak(i, j) * new_popsk(j);
                        }
                        F(i) = Fi;
                    } else if (i == (num_level - 1)) {
                        if (conserve_pressure) {
                            using ConstantsFP::k_B;
                            T N = pressure(ks) / (k_B * temperature(ks));
                            T dntot = N;
                            for (int j = 0; j < num_level; ++j) {
                                dntot -= total_abund * new_popsk(j);
                            }
                            dntot -= ne_k;
                            F(i) = dntot;
                        } else {
                            T dntot = H_atom.abundance * nh_tot(ks);
                            for (int j = 0; j < num_level; ++j) {
                                dntot -= new_popsk(j);
                            }
                            F(i) = dntot;
                        }
                    } else if (i == (num_eqn - 1)) {
                        T charge = FP(0.0);
                        for (int j = 0; j < num_level; ++j) {
                            charge += (H_atom.stage(j) - FP(1.0)) * new_popsk(j);
                        }
                        charge -= ne_k;
                        F(i) = charge;
                    }
                }
            );

            // Compute Jacobian dF
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
                [&] (const int x) {
                    const auto args = nlxnl_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];
                    dF(i, j) = -Gammak(i, j);
                    if (j == 0) {
                        dF(i, num_eqn-1) = FP(0.0);
                    }
                }
            );
            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
                [&] (const int x) {
                    const auto args = nlxnl_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];
                    const int num_cont = H_atom.continua.extent(0);
                    if (x < num_cont) {
                        const int kr = x;
                        const auto& cont = H_atom.continua(kr);
                        const T precon_Rji = Gammak(cont.i, cont.j) - C(cont.i, cont.j);
                        const T entry = -(precon_Rji / ne_k) * new_popsk(cont.j);
                        Kokkos::atomic_add(&dF(cont.i, num_eqn-1), entry);
                    }

                    Kokkos::atomic_add(&dF(i, num_eqn-1), -dC(i, j) * new_popsk(j));
                }
            );
            team.team_barrier();

            // Setup conservation equations
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nexne_loop.num_iter),
                [&] (const int x) {
                    const auto args = nexne_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];

                    if (i == (num_level-1) && j < num_level) {
                        if (conserve_pressure) {
                            // NOTE(cmo): Pressure conservation eqn
                            dF(i, j) = total_abund;
                        } else {
                            // NOTE(cmo): Number conservation eqn for H
                            dF(i, j) = FP(1.0);
                        }
                    } else if (i == (num_level-1) && j == num_level) {
                        if (conserve_pressure) {
                            // NOTE(cmo): Pressure conservation eqn (ne term)
                            dF(i, j) = FP(1.0);
                        } else {
                            // NOTE(cmo): Number conservation eqn for H
                            dF(i, j) = FP(0.0);
                        }
                    } else if (i == (num_eqn - 1) && j < num_level) {
                        dF(i, j) = -(H_atom.stage(j) - FP(1.0));
                    } else if (i == (num_eqn - 1) && j == (num_eqn - 1)) {
                        dF(i, j) = FP(1.0);
                    }
                }
            );
            team.team_barrier();

            // LU factorise
            KokkosBatched::LU<KTeam, KokkosBatched::Mode::Team, KokkosBatched::Algo::LU::Unblocked>::invoke(
                team, dF
            );
            team.team_barrier();
            // LU Solve
            KokkosBatched::TeamSolveLU<
                KTeam,
                KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Algo::Trsm::Unblocked
            >::invoke(
                team,
                dF,
                F
            );
            team.team_barrier();

            // Store result
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, num_eqn),
                [&] (const int i) {
                    new_F(ks, i) = F(i);
                }
            );
            team.team_barrier();
        }
    );
    Kokkos::fence();

    const auto& F = new_F;
    typedef Kokkos::MaxLoc<fp_t, Kokkos::pair<int, int>> Reducer;
    typedef Reducer::value_type ReductionVal;
    ReductionVal max_change_loc;
    dex_parallel_reduce(
        "Update pops and compute max change",
        FlatLoop<2>(num_space, num_eqn),
        KOKKOS_LAMBDA (int ks, int i, ReductionVal& rval) {
            fp_t change = FP(0.0);
            fp_t step_size = FP(1.0);

            constexpr bool clamp_step_size = true;
            if (i < num_level) {
                fp_t update = F(ks, i);
                fp_t updated = pops(i, ks) + update;
                if (clamp_step_size && updated < FP(0.0)) {
                    fp_t ne_update = F(ks, num_eqn-1);
                    step_size = std::max(FP(0.95) * ne(ks) / std::abs(ne_update), FP(1e-4));
                    update *= step_size;
                }
                if (pops(i, ks) > (ignore_change_below_ntot_frac * nh_tot(ks))) {
                    change = std::abs(update / (pops(i, ks)));
                }
                pops(i, ks) += update;
            } else {
                fp_t ne_update = F(ks, num_eqn-1);
                fp_t updated = ne(ks) + ne_update;
                if (clamp_step_size && updated < FP(0.0)) {
                    step_size = std::max(FP(0.95) * ne(ks) / std::abs(ne_update), FP(1e-4));
                    ne_update *= step_size;
                }
                change = std::abs(ne_update / ne(ks));
                ne(ks) += ne_update;
            }

            // reduce update
            if (change > rval.val) {
                rval.val = change;
                rval.loc = Kokkos::make_pair(ks, i);
            }
        },
        Reducer(max_change_loc)
    );
    Kokkos::fence();

    if (conserve_pressure) {
        const auto& full_pops = state->pops;
        dex_parallel_for(
            "Update and rescale pops (pressure)",
            FlatLoop<1>(num_space),
            KOKKOS_LAMBDA (i64 k) {
                fp_t pops_sum = FP(0.0);
                for (int i = 0; i < num_level; ++i) {
                    pops_sum += pops(i, k);
                }
                fp_t nh_tot_ratio = pops_sum / nh_tot(k);
                nh_tot(k) = pops_sum;

                for (int i = 0; i < full_pops.extent(0); ++i) {
                    full_pops(i, k) *= nh_tot_ratio;
                }
            }
        );
        Kokkos::fence();
    }

    fp_t max_change = max_change_loc.val;

    int max_change_level = max_change_loc.loc.second;
    i64 max_change_ks = max_change_loc.loc.first;
    state->println(
        "     NR Update Max Change (level: {}): {} (@ {})",
        max_change_level == (num_eqn - 1) ? "n_e": std::to_string(max_change_level),
        max_change,
        max_change_ks
    );
    yakl::timer_stop("Charge conservation");
    return max_change;
}
#else
template <typename T=fp_t>
fp_t nr_post_update_impl(State* state, const NrPostUpdateOptions& args = NrPostUpdateOptions()) { return FP(0.0); };
#endif

fp_t nr_post_update(State* state, const NrPostUpdateOptions& args) {
#ifdef HAVE_MPI
    fp_t max_rel_change;
    if (state->mpi_state.rank == 0) {
        max_rel_change = nr_post_update_impl<StatEqPrecision>(state, args);
    }
    MPI_Bcast(&max_rel_change, 1, get_FpMpi(), 0, state->mpi_state.comm);
    return max_rel_change;
#else
    return nr_post_update_impl<StatEqPrecision>(state, args);
#endif
}