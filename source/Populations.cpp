#include "Populations.hpp"
#include "State3d.hpp"
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
        YAKL_LAMBDA (int64_t ks) {
            lte_pops(
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
template <typename State>
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
template void compute_lte_pops<State>(State* state);
template void compute_lte_pops<State3d>(State3d* state);

template <typename State>
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
template void compute_lte_pops<State>(const State* state, const Fp2d& shared_pops);
template void compute_lte_pops<State3d>(const State3d* state, const Fp2d& shared_pops);

template <typename State>
void compute_nh0(const State& state) {
    const auto& nh0 = state.atmos.nh0;
    const auto& nh_lte = state.nh_lte;

    if (state.have_h) {
        // NOTE(cmo): This could just be a pointer shuffle...
        const auto& pops = state.pops;
        dex_parallel_for(
            "Copy nh0",
            FlatLoop<1>(nh0.extent(0)),
            YAKL_LAMBDA (i64 ks) {
                nh0(ks) = pops(0, ks);
            }
        );
    } else {
        const auto& atmos = state.atmos;
        dex_parallel_for(
            "Compute nh0 in LTE",
            FlatLoop<1>(nh0.extent(0)),
            YAKL_LAMBDA (i64 ks) {
                const fp_t temperature = atmos.temperature(ks);
                const fp_t ne = atmos.ne(ks);
                const fp_t nh_tot = atmos.nh_tot(ks);
                nh0(ks) = nh_lte(temperature, ne, nh_tot);
            }
        );
    }

    yakl::fence();
}

template void compute_nh0<State>(const State& state);
template void compute_nh0<State3d>(const State3d& state);

#ifdef DEXRT_USE_MAGMA

template <typename T=fp_t>
fp_t stat_eq_impl(State* state, const StatEqOptions& args = StatEqOptions()) {
    yakl::timer_start("Stat eq");
    JasUnpack(args, ignore_change_below_ntot_frac);
    fp_t global_max_change = FP(0.0);
    for (int ia = 0; ia < state->adata_host.num_level.extent(0); ++ia) {
        JasUnpack((*state), pops);
        const auto& Gamma = state->Gamma[ia];
        // GammaT has shape [ks, Nlevel, Nlevel]. When using magma we set up
        // each plane as column-major, but using our standard arrays. With
        // kokkos-kernels, it's row-major.
        const fp_t abundance = state->adata_host.abundance(ia);
        const auto nh_tot = state->atmos.nh_tot;
        yakl::Array<T, 3, yakl::memDevice> GammaT("GammaT", Gamma.extent(2), Gamma.extent(0), Gamma.extent(1));
        yakl::Array<T*, 1, yakl::memDevice> GammaT_ptrs("GammaT_ptrs", GammaT.extent(0));
        yakl::Array<T, 2, yakl::memDevice> new_pops("new_pops", GammaT.extent(0), GammaT.extent(1));
        yakl::Array<T*, 1, yakl::memDevice> new_pops_ptrs("new_pops_ptrs", GammaT.extent(0));
        yakl::Array<i32, 1, yakl::memDevice> i_elim("i_elim", GammaT.extent(0));
        yakl::Array<i32, 2, yakl::memDevice> ipivs("ipivs", new_pops.extent(0), new_pops.extent(1));
        yakl::Array<i32*, 1, yakl::memDevice> ipiv_ptrs("ipiv_ptrs", new_pops.extent(0));
        yakl::Array<i32, 1, yakl::memDevice> info("info", new_pops.extent(0));

        constexpr bool fractional_pops = true;

        const int pops_start = state->adata_host.level_start(ia);
        const int num_level = state->adata_host.num_level(ia);
        dex_parallel_for(
            "Max Pops",
            FlatLoop<1>(pops.extent(1)),
            YAKL_LAMBDA (int64_t k) {
                fp_t n_max = FP(0.0);
                i_elim(k) = 0;
                for (int i = pops_start; i < pops_start + num_level; ++i) {
                    fp_t n = pops(i, k);
                    if (n > n_max) {
                        i_elim(k) = i - pops_start;
                        n_max = n;
                    }
                }
            }
        );
        yakl::fence();

        dex_parallel_for(
            "Transpose Gamma",
            FlatLoop<3>(Gamma.extent(2), Gamma.extent(1), Gamma.extent(0)),
            YAKL_LAMBDA (int k, int i, int j) {
                GammaT(k, j, i) = Gamma(i, j, k);
            }
        );
        yakl::fence();

        dex_parallel_for(
            "Gamma fixup",
            FlatLoop<2>(GammaT.extent(0), GammaT.extent(1)),
            YAKL_LAMBDA (i64 k, int i) {
                // NOTE(cmo): This isn't ideal, as it's not coalesced.
                T diag = FP(0.0);
                GammaT(k, i, i) = FP(0.0);
                for (int j = 0; j < GammaT.extent(2); ++j) {
                    diag += GammaT(k, i, j);
                }
                GammaT(k, i, i) = -diag;
            }
        );
        dex_parallel_for(
            "Transpose Pops",
            FlatLoop<2>(new_pops.extent(0), new_pops.extent(1)),
            YAKL_LAMBDA (i64 k, int i) {
                if (i_elim(k) == i) {
                    if (fractional_pops) {
                        new_pops(k, i) = FP(1.0);
                    } else {
                        T n_total = nh_tot(k) * abundance;
                        new_pops(k, i) = n_total;
                    }
                } else {
                    new_pops(k, i) = FP(0.0);
                }
            }
        );
        dex_parallel_for(
            "Setup pointers",
            FlatLoop<1>(GammaT_ptrs.extent(0)),
            YAKL_LAMBDA (i64 k) {
                GammaT_ptrs(k) = &GammaT(k, 0, 0);
                new_pops_ptrs(k) = &new_pops(k, 0);
                ipiv_ptrs(k) = &ipivs(k, 0);
            }
        );
        yakl::fence();

        dex_parallel_for(
            "Conservation eqn",
            FlatLoop<3>(GammaT.extent(0), GammaT.extent(1), GammaT.extent(2)),
            YAKL_LAMBDA (i64 k, int i, int j) {
                if (i_elim(k) == i) {
                    GammaT(k, j, i) = FP(1.0);
                }
            }
        );

        yakl::fence();

        static_assert(
            std::is_same_v<T, f32> || std::is_same_v<T, f64>,
            "What type are you asking the poor stat_eq function to use internally?"
        );
        if constexpr (std::is_same_v<T, f32>) {
            magma_sgesv_batched(
                GammaT.extent(1),
                1,
                GammaT_ptrs.get_data(),
                GammaT.extent(1),
                ipiv_ptrs.get_data(),
                new_pops_ptrs.get_data(),
                new_pops.extent(1),
                info.get_data(),
                GammaT.extent(0),
                state->magma_queue
            );
        } else if constexpr (std::is_same_v<T, f64>) {
            constexpr bool iterative_improvement = true;
            constexpr int num_refinement_passes = 2;
            yakl::Array<T, 3, yakl::memDevice> gamma_copy;
            yakl::Array<T, 2, yakl::memDevice> lhs_copy;
            yakl::Array<T, 2, yakl::memDevice> residuals;
            yakl::Array<T*, 1, yakl::memDevice> residuals_ptrs;
            if constexpr (iterative_improvement) {
                gamma_copy = GammaT.createDeviceCopy();
                lhs_copy = new_pops.createDeviceCopy();
                residuals = new_pops.createDeviceCopy();
                residuals_ptrs = decltype(residuals_ptrs)("residuals_ptrs", residuals.extent(0));
            }

            magma_dgesv_batched(
                GammaT.extent(1),
                1,
                GammaT_ptrs.get_data(),
                GammaT.extent(1),
                ipiv_ptrs.get_data(),
                new_pops_ptrs.get_data(),
                new_pops.extent(1),
                info.get_data(),
                GammaT.extent(0),
                state->magma_queue
            );
            magma_queue_sync(state->magma_queue);

            if constexpr (iterative_improvement) {
                for (int refinement = 0; refinement < num_refinement_passes; ++refinement) {
                    // r_i = b_i
                    dex_parallel_for(
                        "Copy residual",
                        FlatLoop<2>(residuals.extent(0), residuals.extent(1)),
                        YAKL_LAMBDA (i64 ks, i32 i) {
                            if (i == 0) {
                                residuals_ptrs(ks) = &residuals(ks, 0);
                            }
                            residuals(ks, i) = lhs_copy(ks, i);
                        }
                    );
                    yakl::fence();
                    // r -= A x
                    magmablas_dgemv_batched_strided(
                        MagmaNoTrans,
                        residuals.extent(1),
                        residuals.extent(1),
                        -1,
                        gamma_copy.get_data(),
                        gamma_copy.extent(1),
                        square(gamma_copy.extent(1)),
                        new_pops.get_data(),
                        1,
                        new_pops.extent(1),
                        1,
                        residuals.get_data(),
                        1,
                        residuals.extent(1),
                        residuals.extent(0),
                        state->magma_queue
                    );
                    magma_queue_sync(state->magma_queue);

                    // Solve A x' = r
                    magma_dgetrs_batched(
                        MagmaNoTrans,
                        GammaT.extent(1),
                        1,
                        GammaT_ptrs.get_data(),
                        GammaT.extent(1),
                        ipiv_ptrs.get_data(),
                        residuals_ptrs.get_data(),
                        residuals.extent(1),
                        GammaT.extent(0),
                        state->magma_queue
                    );
                    magma_queue_sync(state->magma_queue);

                    // x += x'
                    dex_parallel_for(
                        "Apply residual",
                        FlatLoop<2>(new_pops.extent(0), new_pops.extent(1)),
                        YAKL_LAMBDA (i64 ks, i32 i) {
                            new_pops(ks, i) += residuals(ks, i);
                        }
                    );
                    yakl::fence();
                }
            }
        }

        magma_queue_sync(state->magma_queue);
        dex_parallel_for(
            "info check",
            FlatLoop<1>(info.extent(0)),
            YAKL_LAMBDA (int k) {
                if (info(k) != 0) {
                    printf("LINEAR SOLVER PROBLEM k: %d, info: %d\n", k, info(k));
                }
            }
        );

        typedef Kokkos::MaxLoc<fp_t, Kokkos::pair<int, int>> Reducer;
        typedef Reducer::value_type ReductionVal;
        ReductionVal max_change_loc;
        dex_parallel_reduce(
            "Update pops and compute max change",
            FlatLoop<2>(new_pops.extent(0), num_level),
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
        const i64 max_change_k = max_change_loc.loc.first;
        const int max_change_level = max_change_loc.loc.second;
        auto temp_h = Fp1d("temp_host_readback", &state->atmos.temperature(max_change_k), 1).createHostCopy();
        state->println(
            "     Max Change (ele: {}, Z={}): {} (@ l={}, ks={}) [T={}]",
            ia,
            state->adata_host.Z(ia),
            max_change,
            max_change_level,
            max_change_k,
            temp_h(0)
        );
        global_max_change = std::max(max_change, global_max_change);

    }
    yakl::timer_stop("Stat eq");
    return global_max_change;
}
#else
template <typename T=fp_t, typename State>
fp_t stat_eq_impl(State* state, const StatEqOptions& args = StatEqOptions()) {
    // NOTE(cmo): This implementation works, and uses less memory, however it's
    // not very computationally efficient, as we're processing a small matrix
    // per entire thread team. That said, it profiles faster
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
        yakl::Array<T, 2, yakl::memDevice> new_pops("new_pops", Nspace, num_level);

        size_t scratch_size = ScratchView<T**>::shmem_size(num_level, num_level);
        if (iterative_improvement) {
            scratch_size *= 2;
            scratch_size += 2 * ScratchView<T*>::shmem_size(num_level);
        }

        FlatLoop<2> nxn_loop(num_level, num_level);

        Kokkos::parallel_for(
            "Stat Eq",
            TeamPolicy(Nspace, std::min(square(num_level), 128)).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
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
                KView<T*> new_popsk(&new_pops(ks, 0), new_pops.extent(1));

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
                    team.team_barrier();
                }

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
        auto temp_h = Fp1d("temp_readback", &state->atmos.temperature(max_change_loc.loc.first), 1).createHostCopy();

        state->println(
            "     Max Change (ele: {}, Z={}): {} (@ l={}, ks={}) [T={}]",
            ia,
            state->adata_host.Z(ia),
            max_change,
            max_change_loc.loc.second,
            max_change_loc.loc.first,
            temp_h(0)
        );
        global_max_change = std::max(max_change, global_max_change);

    }
    yakl::timer_stop("Stat eq");
    return global_max_change;
}
#endif

template <typename State>
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

template fp_t stat_eq<State>(State* state, const StatEqOptions& args);
template fp_t stat_eq<State3d>(State3d* state, const StatEqOptions& args);
