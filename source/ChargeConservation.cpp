#include "ChargeConservation.hpp"
#include "State3d.hpp"
#include "KokkosBatched_Gesv.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"
#include "KokkosBlas.hpp"

#ifdef DEXRT_USE_MAGMA
template <typename T=fp_t>
fp_t nr_post_update_impl(State* state, const NrPostUpdateOptions& args = NrPostUpdateOptions()) {
    yakl::timer_start("Charge conservation");
    JasUnpack(args, ignore_change_below_ntot_frac, conserve_pressure);
    // TODO(cmo): Add background n_e term like in Lw.
    // NOTE(cmo): Only considers H for now
    // TODO(cmo): He contribution?
    assert(state->have_h && "Need to have H active for non-lte EOS");
    const auto& pops = state->pops;
    const auto& GammaH = state->Gamma[0];
    const int num_level = GammaH.extent(0);
    const int num_eqn = GammaH.extent(0) + 1;
    JasUnpack(state->atmos, ne, nh_tot, pressure, temperature);
    // NOTE(cmo): GammaH_flat is how we access Gamma/C in the following
    const auto& GammaH_flat = state->Gamma[0];
    // NOTE(cmo): Sort out the diagonal before we copy into the transpose (full Gamma)
    fixup_gamma(GammaH_flat);
    yakl::fence();

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

    const auto& H_atom = extract_atom(state->adata, state->adata_host, 0);
    // NOTE(cmo): Immediately transposed -- we can reduce the memory requirements here if really needed
    yakl::Array<T, 2, yakl::memDevice> F("F", GammaH_flat.extent(2), num_eqn);
    yakl::Array<T, 3, yakl::memDevice> GammaT("GammaH^T", GammaH_flat.extent(2), num_level, num_level);
    yakl::Array<T, 3, yakl::memDevice> dF("dF", GammaH_flat.extent(2), num_eqn, num_eqn);
    F = FP(0.0);
    dF = FP(0.0);
    yakl::Array<T, 2, yakl::memDevice> new_pops("new_pops", GammaT.extent(0), num_level);
    yakl::fence();

    dex_parallel_for(
        "Transpose Gamma",
        FlatLoop<3>(GammaH_flat.extent(2), GammaH_flat.extent(1), GammaH_flat.extent(0)),
        YAKL_LAMBDA (i64 k, int i, int j) {
            GammaT(k, j, i) = GammaH_flat(i, j, k);
        }
    );
    yakl::fence();
    // NOTE(cmo): Ensure the fixup is done in double precision for the full rate matrix
    dex_parallel_for(
        "Gamma fixup",
        FlatLoop<1>(GammaT.extent(0)),
        YAKL_LAMBDA (i64 k) {
            for (int i = 0; i < GammaT.extent(1); ++i) {
                T diag = FP(0.0);
                GammaT(k, i, i) = FP(0.0);
                for (int j = 0; j < GammaT.extent(2); ++j) {
                    diag += GammaT(k, i, j);
                }
                GammaT(k, i, i) = -diag;
            }
        }
    );
    dex_parallel_for(
        "Tranpose Pops",
        FlatLoop<2>(new_pops.extent(0), new_pops.extent(1)),
        YAKL_LAMBDA (i64 k, int i) {
            new_pops(k, i) = pops(i, k);
        }
    );
    yakl::fence();

    // NOTE(cmo): Overwrite Gamma with C
    compute_collisions_to_gamma(state);
    fixup_gamma(GammaH_flat);
    yakl::fence();
    const auto C = GammaH.createDeviceCopy();
    const auto C_flat = C.createDeviceCopy();
    auto ne_copy = state->atmos.ne.createDeviceCopy();
    auto ne_pert = state->atmos.ne.createDeviceCopy();
    yakl::fence();


    // NOTE(cmo): Perturb ne, compute C, compute dC/dne, restore ne
    constexpr fp_t pert_size = FP(1e-2);
    dex_parallel_for(
        "Perturb ne",
        FlatLoop<1>(ne.extent(0)),
        YAKL_LAMBDA (i64 ks) {
            ne_pert(ks) = ne(ks) * pert_size;
            ne(ks) += ne_pert(ks);
        }
    );
    yakl::fence();
    compute_collisions_to_gamma(state);
    fixup_gamma(GammaH_flat);
    yakl::fence();
    dex_parallel_for(
        "Compute dC",
        FlatLoop<3>(C.extent(0), C.extent(1), C.extent(2)),
        YAKL_LAMBDA (int i, int j, i64 ks) {
            C(i, j, ks) = (GammaH(i, j, ks) - C(i, j, ks)) / ne_pert(ks);
        }
    );
    // NOTE(cmo): Rename for clarity
    const auto& dC = C;
    dex_parallel_for(
        "Restore n_e",
        FlatLoop<1>(ne.extent(0)),
        YAKL_LAMBDA (i64 ks) {
            ne(ks) = ne_copy(ks);
        }
    );
    yakl::fence();

    // NOTE(cmo): Compute LHS, based on Lightspinner impl
    dex_parallel_for(
        "Compute F",
        FlatLoop<2>(F.extent(0), F.extent(1)),
        YAKL_LAMBDA (i64 k, int i) {
            if (i < (num_level - 1)) {
                T Fi = FP(0.0);
                for (int j = 0; j < num_level; ++j) {
                    Fi += GammaT(k, j, i) * new_pops(k, j);
                }
                F(k, i) = Fi;
            } else if (i == (num_level - 1)) {
                if (conserve_pressure) {
                    using ConstantsFP::k_B;
                    T N = pressure(k) / (k_B * temperature(k));
                    T dntot = N;
                    for (int j = 0; j < num_level; ++j) {
                        dntot -= total_abund * new_pops(k, j);
                    }
                    dntot -= ne(k);
                } else {
                    T dntot = H_atom.abundance * nh_tot(k);
                    for (int j = 0; j < num_level; ++j) {
                        dntot -= new_pops(k, j);
                    }
                    F(k, i) = dntot;
                }
            } else if (i == (num_eqn - 1)) {
                T charge = FP(0.0);
                for (int j = 0; j < num_level; ++j) {
                    charge += (H_atom.stage(j) - FP(1.0)) * new_pops(k, j);
                }
                charge -= ne(k);
                F(k, i) = charge;
            }
        }
    );
    // NOTE(cmo): Compute matrix system -- very messy.
    dex_parallel_for(
        "Compute dF",
        FlatLoop<3>(dF.extent(0), dF.extent(1), dF.extent(2)),
        YAKL_LAMBDA (i64 k, int i, int j) {
            if (i < num_level && j < num_level) {
                dF(k, i, j) = -GammaT(k, i, j);
            }
            if (j == 0) {
                if (i == 0) {
                    for (int kr = 0; kr < H_atom.continua.extent(0); ++kr) {
                        const auto& cont = H_atom.continua(kr);
                        const T precon_Rji = GammaT(k, cont.j, cont.i) - C_flat(cont.i, cont.j, k);
                        const T entry = -(precon_Rji / ne(k)) * new_pops(k, cont.j);
                        Kokkos::atomic_add(
                            &dF(k, num_eqn-1, cont.i),
                            entry
                        );
                    }
                }
                if (i < num_level) {
                    // TODO(cmo): This can be atomicised and done over j, but it works
                    for (int jj = 0; jj < num_level; ++jj) {
                        Kokkos::atomic_add(
                            &dF(k, num_eqn-1, i),
                            - dC(i, jj, k) * new_pops(k, jj)
                        );
                    }
                }
            }
            if (i < num_level && j == (num_level-1)) {
                if (conserve_pressure) {
                    // NOTE(cmo): Pressure conservation eqn
                    dF(k, i, j) = total_abund;
                } else {
                    // NOTE(cmo): Number conservation eqn for H
                    dF(k, i, j) = FP(1.0);
                }
            }
            if (i == num_level && j == (num_level-1)) {
                if (conserve_pressure) {
                    // NOTE(cmo): Pressure conservation eqn (ne term)
                    dF(k, i, j) = FP(1.0);
                } else {
                    // NOTE(cmo): Number conservation eqn for H
                    dF(k, i, j) = FP(0.0);
                }
            }
            if (i < num_level && j == (num_eqn - 1)) {
                dF(k, i, j) = -(H_atom.stage(i) - FP(1.0));
            }
            if (i == (num_eqn - 1) && j == (num_eqn - 1)) {
                dF(k, i, j) = FP(1.0);
            }
        }
    );
    yakl::fence();

    yakl::Array<i32, 2, yakl::memDevice> ipivs("ipivs", F.extent(0), F.extent(1));
    yakl::Array<i32*, 1, yakl::memDevice> ipiv_ptrs("ipiv_ptrs", F.extent(0));
    yakl::Array<i32, 1, yakl::memDevice> info("info", F.extent(0));
    yakl::Array<T*, 1, yakl::memDevice> dF_ptrs("dF_ptrs", dF.extent(0));
    yakl::Array<T*, 1, yakl::memDevice> F_ptrs("F_ptrs", F.extent(0));
    ipivs = FP(0.0);
    info = 0;

    dex_parallel_for(
        "Setup pointers",
        FlatLoop<1>(dF_ptrs.extent(0)),
        YAKL_LAMBDA (i64 k) {
            F_ptrs(k) = &F(k, 0);
            dF_ptrs(k) = &dF(k, 0, 0);
            ipiv_ptrs(k) = &ipivs(k, 0);
        }
    );
    yakl::fence();
    static_assert(
        std::is_same_v<T, f32> || std::is_same_v<T, f64>,
        "What type are you asking the poor charge conservation function to use internally?"
    );
    if constexpr (std::is_same_v<T, f32>) {
        magma_sgesv_batched(
            dF.extent(1),
            1,
            dF_ptrs.data(),
            dF.extent(1),
            ipiv_ptrs.data(),
            F_ptrs.data(),
            F.extent(1),
            info.data(),
            dF.extent(0),
            state->magma_queue
        );
    } else if constexpr (std::is_same_v<T, f64>) {

        constexpr bool iterative_improvement = true;
        constexpr int num_refinement_passes = 2;
        yakl::Array<T, 3, yakl::memDevice> dF_copy;
        yakl::Array<T, 2, yakl::memDevice> F_copy;
        yakl::Array<T, 2, yakl::memDevice> residuals;
        yakl::Array<T*, 1, yakl::memDevice> residuals_ptrs;
        if constexpr (iterative_improvement) {
            dF_copy = dF.createDeviceCopy();
            F_copy = F.createDeviceCopy();
            residuals = F.createDeviceCopy();
            residuals_ptrs = decltype(residuals_ptrs)("residuals_ptrs", residuals.extent(0));
        }

        magma_dgesv_batched(
            dF.extent(1),
            1,
            dF_ptrs.data(),
            dF.extent(1),
            ipiv_ptrs.data(),
            F_ptrs.data(),
            F.extent(1),
            info.data(),
            dF.extent(0),
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
                        residuals(ks, i) = F_copy(ks, i);
                    }
                );
                yakl::fence();
                // r -= A x
                magmablas_dgemv_batched_strided(
                    MagmaNoTrans,
                    residuals.extent(1),
                    residuals.extent(1),
                    -1,
                    dF_copy.get_data(),
                    dF_copy.extent(1),
                    square(dF_copy.extent(1)),
                    F.get_data(),
                    1,
                    F.extent(1),
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
                    dF.extent(1),
                    1,
                    dF_ptrs.get_data(),
                    dF.extent(1),
                    ipiv_ptrs.get_data(),
                    residuals_ptrs.get_data(),
                    residuals.extent(1),
                    dF.extent(0),
                    state->magma_queue
                );
                magma_queue_sync(state->magma_queue);

                // x += x'
                dex_parallel_for(
                    "Apply residual",
                    FlatLoop<2>(F.extent(0), F.extent(1)),
                    YAKL_LAMBDA (i64 ks, i32 i) {
                        F(ks, i) += residuals(ks, i);
                    }
                );
                yakl::fence();
            }
        }
    }

    magma_queue_sync(state->magma_queue);
    // NOTE(cmo): F is now the absolute update to apply to Hpops and ne
    dex_parallel_for(
        "info check",
        FlatLoop<1>(info.extent(0)),
        YAKL_LAMBDA (int k) {
            if (info(k) != 0) {
                printf("LINEAR SOLVER PROBLEM (charge conservation) ks: %d, info: %d\n", k, info(k));
            }
        }
    );

    Fp2d max_rel_change("max rel change", F.extent(0), F.extent(1));
    Fp1d nr_step_size("step size", F.extent(0));
    max_rel_change = FP(0.0);
    nr_step_size = FP(0.0);
    yakl::fence();
    dex_parallel_for(
        "Update & transpose pops",
        FlatLoop<1>(F.extent(0)),
        YAKL_LAMBDA (int64_t k) {
            fp_t step_size = FP(1.0);
            constexpr bool clamp_step_size = true;
            if (clamp_step_size) {
                for (int i = 0; i < num_level; ++i) {
                    fp_t update = F(k, i);
                    fp_t updated = pops(i, k) + update;
                    if (updated < FP(0.0)) {
                        fp_t local_step_size = FP(0.99) * pops(i, k) / std::abs(update);
                        step_size = std::max(std::min(step_size, local_step_size), FP(1e-4));
                    }
                }
                fp_t ne_update = F(k, num_eqn-1);
                fp_t ne_updated = ne(k) + ne_update;
                if (ne_updated < FP(0.0)) {
                    fp_t local_step_size = FP(0.95) * ne(k) / std::abs(ne_update);
                    step_size = std::max(std::min(step_size, local_step_size), FP(1e-4));
                }
            }
            for (int i = 0; i < num_level; ++i) {
                fp_t update = step_size * F(k, i);
                if (pops(i, k) > ignore_change_below_ntot_frac * nh_tot(k)) {
                    max_rel_change(k, i) = std::abs(update / (pops(i, k)));
                }
                pops(i, k) += update;
            }
            fp_t ne_update = step_size * F(k, num_eqn-1);
            max_rel_change(k, num_eqn-1) = std::abs(ne_update / (ne(k)));
            ne(k) += ne_update;
            nr_step_size(k) = step_size;
        }
    );
    yakl::fence();

    if (conserve_pressure) {
        Fp1d nh_tot_ratio("nh_tot_ratio", nh_tot.extent(0));
        dex_parallel_for(
            "Update nh_tot (pressure)",
            FlatLoop<1>(nh_tot_ratio.extent(0)),
            YAKL_LAMBDA (i64 k) {
                fp_t pops_sum = FP(0.0);
                for (int i = 0; i < num_level; ++i) {
                    pops_sum += pops(i, k);
                }
                nh_tot_ratio(k) = pops_sum / nh_tot(k);
                nh_tot(k) = pops_sum;
            }
        );
        yakl::fence();

        const auto& full_pops = state->pops;
        dex_parallel_for(
            "Rescale pops (presure)",
            FlatLoop<2>(full_pops.extent(0), full_pops.extent(1)),
            YAKL_LAMBDA (int i, i64 k) {
                full_pops(i, k) *= nh_tot_ratio(k);
            }
        );
    }


    fp_t max_change = yakl::intrinsics::maxval(max_rel_change);
    auto step_size_host = nr_step_size.createHostCopy();
    const i64 max_change_loc = yakl::intrinsics::maxloc(max_rel_change.collapse());
    yakl::fence();
    i64 max_change_acc = max_change_loc;

    int max_change_level = max_change_acc % F.extent(1);
    max_change_acc /= F.extent(1);
    i64 max_change_ks = max_change_acc;
    state->println(
        "     NR Update Max Change (level: {}): {} (@ {}), step_size: {}",
        max_change_level == (num_eqn - 1) ? "n_e": std::to_string(max_change_level),
        max_change,
        max_change_ks,
        step_size_host(max_change_loc / F.extent(1))
    );
    yakl::timer_stop("Charge conservation");
    return max_change;
}
#else
template <typename T=fp_t, typename State>
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
        if (args.total_abund <= FP(0.0)) {
            total_abund = FP(1.0861550335264554);
            // NOTE(cmo): 1.1 is traditionally used to account for He, but it's all much of a muchness
        } else {
            total_abund = args.total_abund;
        }
    }
    constexpr fp_t ne_pert_size = FP(1e-2);
    constexpr bool iterative_improvement = true;
    constexpr int num_refinement_passes = 2;

    const auto& H_atom = extract_atom(state->adata, state->adata_host, 0);

    size_t scratch_size = ScratchView<T**>::shmem_size(num_level, num_level); // Gammak
    scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // C
    // scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // C_ne_pert
    scratch_size += ScratchView<fp_t**>::shmem_size(num_level, num_level); // dC
    scratch_size += ScratchView<T**>::shmem_size(num_eqn, num_eqn); // dF
    scratch_size += ScratchView<T*>::shmem_size(num_eqn); // F
    scratch_size += ScratchView<T*>::shmem_size(num_level); // new_popsk
    if (iterative_improvement) {
        scratch_size += ScratchView<T**>::shmem_size(num_eqn, num_eqn); // dF copy
        scratch_size += 2 * ScratchView<T*>::shmem_size(num_eqn); // lhs_copy/residuals
    }
    yakl::Array<T, 2, yakl::memDevice> new_F("new_F", num_space, num_eqn);
    Fp3d C("C", num_level, num_level, num_space);
    Fp3d dC("dC", num_level, num_level, num_space);
    C = FP(0.0);
    dC = FP(0.0);
    FlatLoop<2> nlxnl_loop(num_level, num_level);
    FlatLoop<2> nexne_loop(num_eqn, num_eqn);

    Fp2d n_star = state->pops.createDeviceObject();
    compute_lte_pops(state, n_star);
    Kokkos::fence();
    const auto n_star_slice = slice_pops(
        n_star,
        state->adata_host,
        state->atoms_with_gamma_mapping[0]
    );

    // NOTE(cmo): These terms can be computed in shared memory (see the
    // commented blocks), but it's much slower because we're only doing one set
    // of rates per thread team.
    dex_parallel_for(
        "Compute C, dC",
        FlatLoop<1>(num_space),
        KOKKOS_LAMBDA (i64 ks) {
            for (int i = 0; i < num_level; ++i) {
                for (int j = 0; j < num_level; ++j) {
                    C(i, j, ks) = FP(0.0);
                    dC(i, j, ks) = FP(0.0);
                }
            }

            compute_C_ne_pert(
                atmos,
                H_atom,
                n_star_slice,
                nh_lte,
                ks,
                C,
                dC, // This contains C_ne_pert
                ne_pert_size
            );

            const fp_t ne_k = atmos.ne(ks);
            const fp_t recip_dNe = FP(1.0) / (ne_pert_size * ne_k);
            for (int i = 0; i < num_level; ++i) {
                for (int j = 0; j < num_level; ++j) {
                    fp_t dCdNe = (dC(i, j, ks) - C(i, j, ks)) * recip_dNe;
                    dC(i, j, ks) = dCdNe;
                }
            }

            for (int i = 0; i < num_level; ++i) {
                fp_t diag = FP(0.0);
                fp_t diag_dC = FP(0.0);
                C(i, i, ks) = diag;
                dC(i, i, ks) = diag_dC;
                for (int j = 0; j < num_level; ++j) {
                    diag += C(j, i, ks);
                    diag_dC += dC(j, i, ks);
                }
                C(i, i, ks) = -diag;
                dC(i, i, ks) = -diag_dC;
            }
        }
    );
    Kokkos::fence();

    Kokkos::parallel_for(
        "Charge Conservation",
        TeamPolicy(num_space, std::min(square(num_eqn), 128)).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA (const KTeam& team) {
            const i64 ks = team.league_rank();

            ScratchView<fp_t**> Ck(team.team_scratch(0), num_level, num_level);
            ScratchView<fp_t**> dCk(team.team_scratch(0), num_level, num_level);
            const fp_t ne_k = atmos.ne(ks);
            // ScratchView<fp_t**> C_ne_pert(team.team_scratch(0), num_level, num_level);

            // Kokkos::parallel_for(
            //     Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
            //     [&] (const int x) {
            //         const auto args = nlxnl_loop.unpack(x);
            //         const int i = args[0];
            //         const int j = args[1];
            //         Ck(i, j) = FP(0.0);
            //         C_ne_pert(i, j) = FP(0.0);
            //     }
            // );
            // team.team_barrier();

            // // Compute changes in C
            // Kokkos::single(Kokkos::PerTeam(team), [&]() {
            //     compute_C_ne_pert(
            //         atmos,
            //         H_atom,
            //         n_star_slice,
            //         nh_lte,
            //         ks,
            //         Ck,
            //         C_ne_pert,
            //         ne_pert_size
            //     );
            // });
            // team.team_barrier();

            // // compute dCdne
            // const fp_t recip_dNe = FP(1.0) / (ne_pert_size * ne_k);
            // Kokkos::parallel_for(
            //     Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
            //     [&] (const int x) {
            //         const auto args = nlxnl_loop.unpack(x);
            //         const int i = args[0];
            //         const int j = args[1];

            //         dCk(i, j) = (C_ne_pert(i, j) - Ck(i, j)) * recip_dNe;
            //         if (ks == 82452 && dCk(i, j) != dC(i, j, ks)) {
            //             printf("%.4e, %.4e (%d, %d)\n", dCk(i, j), dC(i, j, ks), i, j);
            //         }
            //     }
            // );
            // team.team_barrier();
            // // Fixup C and dC
            // Kokkos::parallel_for(
            //     Kokkos::TeamVectorRange(team, num_level),
            //     [&] (const int i) {
            //         fp_t diag = FP(0.0);
            //         fp_t diag_dC = FP(0.0);
            //         Ck(i, i) = diag;
            //         dCk(i, i) = diag_dC;
            //         for (int j = 0; j < num_level; ++j) {
            //             diag += Ck(j, i);
            //             diag_dC += dCk(j, i);
            //         }
            //         Ck(i, i) = -diag;
            //         dCk(i, i) = -diag_dC;
            //     }
            // );
            // team.team_barrier();

            ScratchView<T**> Gammak(team.team_scratch(0), num_level, num_level);
            ScratchView<T*> new_popsk(team.team_scratch(0), num_level);
            // Copy over Gamma and new_pops chunks
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, nlxnl_loop.num_iter),
                [&] (const int x) {
                    const auto args = nlxnl_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];

                    Gammak(i, j) = GammaH_flat(i, j, ks);
                    Ck(i, j) = C(i, j, ks);
                    dCk(i, j) = dC(i, j, ks);
                    if (i == 0) {
                        new_popsk(j) = pops(j, ks);
                    }
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
                Kokkos::TeamVectorRange(team, nexne_loop.num_iter),
                [&] (const int x) {
                    const auto args = nexne_loop.unpack(x);
                    const int i = args[0];
                    const int j = args[1];
                    if (i < num_level && j < num_level) {
                        dF(i, j) = -Gammak(i, j);
                    } else {
                        dF(i, j) = T(0);
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
                        const T precon_Rji = Gammak(cont.i, cont.j) - Ck(cont.i, cont.j);
                        const T entry = -(precon_Rji / ne_k) * new_popsk(cont.j);
                        Kokkos::atomic_add(&dF(cont.i, num_eqn-1), entry);
                    }

                    Kokkos::atomic_add(&dF(i, num_eqn-1), -dCk(i, j) * new_popsk(j));
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

            ScratchView<T**> dF_copy;
            ScratchView<T*> lhs;
            ScratchView<T*> residuals;
            if (iterative_improvement) {
                dF_copy = ScratchView<T**>(team.team_scratch(0), num_eqn, num_eqn);
                lhs = ScratchView<T*>(team.team_scratch(0), num_eqn);
                residuals = ScratchView<T*>(team.team_scratch(0), num_eqn);

                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, nexne_loop.num_iter),
                    [&] (const int x) {
                        const auto args = nexne_loop.unpack(x);
                        const int i = args[0];
                        const int j = args[1];

                        if (i == 0) {
                            lhs(j) = F(j);
                            residuals(j) = F(j);
                        }
                        dF_copy(i, j) = dF(i, j);
                    }
                );
                team.team_barrier();
            }


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
                    // r -= dF @ x
                    KokkosBlas::Experimental::Gemv<KokkosBlas::Mode::TeamVector, KokkosBlas::Algo::Gemv::Default>::invoke(
                        team,
                        'n',
                        T(-1),
                        dF_copy,
                        F,
                        T(1),
                        residuals
                    );
                    team.team_barrier();
                    // Solve dF F' = r (already factorised)
                    KokkosBatched::TeamSolveLU<
                        KTeam,
                        KokkosBatched::Trans::NoTranspose,
                        KokkosBatched::Algo::Trsm::Unblocked
                    >::invoke(
                        team,
                        dF,
                        residuals
                    );
                    team.team_barrier();
                    // F += F'
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(
                            team,
                            new_popsk.extent(0)
                        ),
                        [&] (int i) {
                            F(i) += residuals(i);
                        }
                    );
                    team.team_barrier();
                }
            }

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
                // if (pops(i, ks) < FP(1e-3) || std::isnan(pops(i, ks))) {
                //     pops(i, ks) = FP(1e-3);
                // }
            } else {
                fp_t ne_update = F(ks, num_eqn-1);
                fp_t updated = ne(ks) + ne_update;
                if (clamp_step_size && updated < FP(0.0)) {
                    step_size = std::max(FP(0.95) * ne(ks) / std::abs(ne_update), FP(1e-4));
                    ne_update *= step_size;
                }
                change = std::abs(ne_update / ne(ks));
                ne(ks) += ne_update;
                // if (ne(ks) < FP(1e-3) || std::isnan(ne(ks))) {
                //     ne(ks) = FP(1e-3);
                // }
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
#endif

template <typename State>
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

template fp_t nr_post_update<State>(State* state, const NrPostUpdateOptions& args);
template fp_t nr_post_update<State3d>(State3d* state, const NrPostUpdateOptions& args);