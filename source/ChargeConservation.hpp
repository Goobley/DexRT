#if !defined(DEXRT_CHARGE_CONSERVATION_HPP)
#define DEXRT_CHARGE_CONSERVATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "Collisions.hpp"
#include "GammaMatrix.hpp"

template <typename T=fp_t>
inline fp_t nr_post_update(State* state) {
    constexpr bool print_debug = false;
    // NOTE(cmo): Here be big angry dragons. The implementation is disgusting and hard to follow. This needs to be redone.
    // TODO(cmo): There's too many fences in here!
    // TODO(cmo): Add background n_e term like in Lw.
    // NOTE(cmo): Only considers H for now
    // TODO(cmo): He contribution?
    assert(state->have_h && "Need to have H active for non-lte EOS");
    const auto& pops = state->pops.reshape<2>(Dims(state->pops.extent(0), state->pops.extent(1) * state->pops.extent(2)));
    const auto& GammaH = state->Gamma[0];
    const int num_level = GammaH.extent(0);
    const int num_eqn = GammaH.extent(0) + 1;
    const auto& ne_flat = state->atmos.ne.collapse();
    const auto& nhtot = state->atmos.nh_tot.collapse();
    // NOTE(cmo): GammaH_flat is how we access Gamma/C in the following
    const auto& GammaH_flat = state->Gamma[0].reshape<3>(Dims(
        GammaH.extent(0),
        GammaH.extent(1),
        GammaH.extent(2) * GammaH.extent(3)
    ));
    // NOTE(cmo): Sort out the diagonal before we copy into the tranpose (full Gamma)
    fixup_gamma(GammaH_flat);
    yakl::fence();
    auto FullGammaH = GammaH.createDeviceCopy();
    const auto& H_atom = extract_atom(state->adata, state->adata_host, 0);
    // NOTE(cmo): Immediately transposed
    yakl::Array<T, 2, yakl::memDevice> F("F", GammaH_flat.extent(2), num_eqn);
    yakl::Array<T, 3, yakl::memDevice> GammaT("GammaH^T", GammaH_flat.extent(2), num_level, num_level);
    yakl::Array<T, 3, yakl::memDevice> dF("dF", GammaH_flat.extent(2), num_eqn, num_eqn);
    dF = FP(0.0);
    yakl::Array<T, 2, yakl::memDevice> new_pops("new_pops", GammaT.extent(0), num_level);
    yakl::fence();

    parallel_for(
        "Transpose Gamma",
        SimpleBounds<3>(GammaH_flat.extent(2), GammaH_flat.extent(1), GammaH_flat.extent(0)),
        YAKL_LAMBDA (i64 k, int i, int j) {
            GammaT(k, j, i) = GammaH_flat(i, j, k);
        }
    );
    yakl::fence();
    // NOTE(cmo): Ensure the fixup is done in double precision for the full rate matrix
    parallel_for(
        "Gamma fixup",
        SimpleBounds<1>(GammaT.extent(0)),
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
    parallel_for(
        "Tranpose Pops",
        SimpleBounds<2>(new_pops.extent(0), new_pops.extent(1)),
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
    const auto C_flat = C.createDeviceCopy().reshape<3>(Dims(C.extent(0), C.extent(1), C.extent(2) * C.extent(3)));
    auto ne_copy = state->atmos.ne.createDeviceCopy();
    auto ne_pert = state->atmos.ne.createDeviceCopy();
    yakl::fence();


    // NOTE(cmo): Perturb ne, compute C, compute dC/dne, restore ne
    auto& ne = state->atmos.ne;
    constexpr fp_t pert_size = FP(1e-2);
    parallel_for(
        "Perturb ne",
        SimpleBounds<2>(ne.extent(0), ne.extent(1)),
        YAKL_LAMBDA (int z, int x) {
            ne_pert(z, x) = ne(z, x) * pert_size;
            ne(z, x) += ne_pert(z, x);
        }
    );
    yakl::fence();
    compute_collisions_to_gamma(state);
    fixup_gamma(GammaH_flat);
    yakl::fence();
    parallel_for(
        "Compute dC",
        SimpleBounds<4>(C.extent(0), C.extent(1), C.extent(2), C.extent(3)),
        YAKL_LAMBDA (int i, int j, int z, int x) {
            C(i, j, z, x) = (GammaH(i, j, z, x) - C(i, j, z, x)) / ne_pert(z, x);
        }
    );
    const auto& dC = C.reshape<3>(Dims(C.extent(0), C.extent(1), C.extent(2) * C.extent(3)));
    parallel_for(
        "Restore n_e",
        SimpleBounds<2>(ne.extent(0), ne.extent(1)),
        YAKL_LAMBDA (int z, int x) {
            ne(z, x) = ne_copy(z, x);
        }
    );
    yakl::fence();

    // NOTE(cmo): Compute LHS, based on Lightspinner impl
    parallel_for(
        "Compute F",
        SimpleBounds<2>(F.extent(0), F.extent(1)),
        YAKL_LAMBDA (i64 k, int i) {
            if (i < num_level - 1) {
                T Fi = FP(0.0);
                for (int j = 0; j < num_level; ++j) {
                    Fi += GammaT(k, j, i) * new_pops(k, j);
                }
                F(k, i) = Fi;
            } else if (i == (num_level - 1)) {
                T dntot = H_atom.abundance * nhtot(k);
                for (int j = 0; j < num_level; ++j) {
                    dntot -= new_pops(k, j);
                }
                F(k, i) = dntot;
            } else {
                T charge = FP(0.0);
                for (int j = 0; j < num_level; ++j) {
                    charge += (H_atom.stage(j) - 1) * new_pops(k, j);
                }
                charge -= ne_flat(k);
                F(k, i) = charge;
            }
        }
    );
    // NOTE(cmo): Compute matrix system -- very messy.
    parallel_for(
        "Compute dF",
        SimpleBounds<3>(dF.extent(0), dF.extent(1), dF.extent(2)),
        YAKL_LAMBDA (i64 k, int i, int j) {
            if (i < num_level && j < num_level) {
                dF(k, i, j) = -GammaT(k, i, j);
            }
            if (j == 0) {
                if (i == 0) {
                    for (int kr = 0; kr < H_atom.continua.extent(0); ++kr) {
                        const auto& cont = H_atom.continua(kr);
                        const fp_t precon_Rji = GammaT(k, cont.j, cont.i) - C_flat(cont.i, cont.j, k);
                        dF(k, num_eqn-1, cont.i) += -(precon_Rji / ne_flat(k)) * new_pops(k, cont.j);
                    }
                }
                if (i < num_level) {
                    // TODO(cmo): This can be atomicised and done over j.
                    for (int jj = 0; jj < num_level; ++jj) {
                        dF(k, num_eqn-1, i) -= dC(i, jj, k) * new_pops(k, jj);
                    }
                }
            }
            if (i < num_level && j == (num_level-1)) {
                // NOTE(cmo): Number conservation eqn for H
                dF(k, i, j) = FP(1.0);
            }
            if (i == num_level && j == (num_level-1)) {
                // NOTE(cmo): Number conservation eqn for H
                dF(k, i, j) = FP(0.0);
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

    parallel_for(
        "Setup pointers",
        SimpleBounds<1>(dF_ptrs.extent(0)),
        YAKL_LAMBDA (i64 k) {
            F_ptrs(k) = &F(k, 0);
            dF_ptrs(k) = &dF(k, 0, 0);
            ipiv_ptrs(k) = &ipivs(k, 0);
        }
    );
    int print_idx = std::min(128 * state->pops.extent(1) + 128, F.extent(0)-1);
    if (print_debug) {
        const auto dF_host = dF.createHostCopy();
        const auto dC_host = dC.createHostCopy();
        const auto F_host = F.createHostCopy();

        fmt::println("-------- dF ----------");
        for (int i = 0; i < dF_host.extent(2); ++i) {
            for (int j = 0; j < dF_host.extent(1); ++j) {
                fmt::print("{:e}, ", dF_host(print_idx, j, i));
            }
            fmt::print("\n");
        }
        fmt::println("-------- dC ----------");
        for (int i = 0; i < dC_host.extent(0); ++i) {
            for (int j = 0; j < dC_host.extent(1); ++j) {
                fmt::print("{:e}, ", dC_host(i, j, print_idx));
            }
            fmt::print("\n");
        }
        fmt::print("F pre ");
        for (int i = 0; i < F.extent(1); ++i) {
            fmt::print("{:e}, ", F_host(print_idx, i));
        }
        fmt::print("\n");
    }
    yakl::fence();
    static_assert(
        std::is_same_v<T, f32> || std::is_same_v<T, f64>,
        "What type are you asking the poor charge conservation function to use internally?"
    );
    if constexpr (std::is_same_v<T, f32>) {
        magma_sgesv_batched_small(
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
        magma_dgesv_batched_small(
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
    }
    magma_queue_sync(state->magma_queue);
    // NOTE(cmo): F is now the absolute update to apply to Hpops and ne
    if (print_debug) {
        const auto F_host = F.createHostCopy();

        fmt::print("F post ");
        for (int i = 0; i < F.extent(1); ++i) {
            fmt::print("{:e}, ", F_host(print_idx, i));
        }
        fmt::print("\n");
    }

    parallel_for(
        "info check",
        SimpleBounds<1>(info.extent(0)),
        YAKL_LAMBDA (int k) {
            if (info(k) != 0) {
                printf("%d: %d\n", k, info(k));
            }
        }
    );

    Fp2d max_rel_change("max rel change", F.extent(0), F.extent(1));
    parallel_for(
        "Compute max change",
        SimpleBounds<2>(F.extent(0), F.extent(1)),
        YAKL_LAMBDA (int64_t k, int i) {
            fp_t change = FP(0.0);
            if (i < num_level) {
                change = std::abs(F(k, i) / (pops(i, k) + F(k, i)));
            } else {
                change = std::abs(F(k, i) / (ne_flat(k) + F(k, i)));
            }
            max_rel_change(k, i) = change;
        }
    );
    parallel_for(
        "Update & transpose pops",
        SimpleBounds<1>(F.extent(0)),
        YAKL_LAMBDA (int64_t k) {
            fp_t step_size = FP(1.0);
            // for (int i = 0; i < num_level; ++i) {
            //     fp_t update = F(k, i);
            //     fp_t updated = pops(i, k) + update;
            //     if (updated < FP(0.0)) {
            //         fp_t local_step_size = FP(1.0) - updated / update;
            //         step_size = std::max(std::min(step_size, local_step_size), FP(0.0));
            //     }
            // }
            // fp_t ne_update = F(k, num_eqn-1);
            // fp_t ne_updated = ne_flat(k) + ne_update;
            // if (ne_updated < FP(0.0)) {
            //     fp_t local_step_size = FP(1.0) - ne_updated / ne_update;
            //     step_size = std::max(std::min(step_size, local_step_size), FP(0.0));
            // }
            for (int i = 0; i < num_level; ++i) {
                fp_t update = F(k, i);
                pops(i, k) += step_size * update;
            }
            ne_flat(k) += step_size * F(k, num_eqn-1);
        }
    );
    yakl::fence();


    if (print_debug) {
        const auto F_host = F.createHostCopy();
        const auto ne_host = ne_flat.createHostCopy();
        const auto pops_host = pops.createHostCopy();
        fmt::print("Updated ");
        for (int i = 0; i < num_level; ++i) {
            fmt::print("{:e}, ", pops_host(i, print_idx) + F_host(print_idx, i));
        }
        fmt::print("{:e}, ", ne_host(print_idx) + F_host(print_idx, num_eqn-1));
        fmt::print("\n");
    }
    fp_t max_change = yakl::intrinsics::maxval(max_rel_change);
    int max_change_loc = yakl::intrinsics::maxloc(max_rel_change.collapse());

    int max_change_level = max_change_loc % F.extent(1);
    max_change_loc /= F.extent(1);
    int max_change_x = max_change_loc % state->pops.extent(2);
    max_change_loc /= state->pops.extent(2);
    int max_change_z = max_change_loc;
    fmt::println(
        "NR Update Max Change (level: {}): {} (@ {}, {})",
        max_change_level == (num_eqn - 1) ? "n_e": std::to_string(max_change_level),
        max_change,
        max_change_z,
        max_change_x
    );
    return max_change;
}



#else
#endif