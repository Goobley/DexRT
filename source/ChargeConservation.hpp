#if !defined(DEXRT_CHARGE_CONSERVATION_HPP)
#define DEXRT_CHARGE_CONSERVATION_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "Collisions.hpp"
#include "GammaMatrix.hpp"

struct NrPostUpdateOptions {
    /// When computing relative change, ignore the change in populations with a
    /// starting fraction lower than this
    fp_t ignore_change_below_ntot_frac = FP(0.0);
};

#ifdef DEXRT_USE_MAGMA
template <typename T=fp_t>
inline fp_t nr_post_update(State* state, const NrPostUpdateOptions& args = NrPostUpdateOptions()) {
    yakl::timer_start("Charge conservation");
    JasUnpack(args, ignore_change_below_ntot_frac);
    // TODO(cmo): Add background n_e term like in Lw.
    // NOTE(cmo): Only considers H for now
    // TODO(cmo): He contribution?
    assert(state->have_h && "Need to have H active for non-lte EOS");
    const auto& pops = state->pops;
    const auto& GammaH = state->Gamma[0];
    const int num_level = GammaH.extent(0);
    const int num_eqn = GammaH.extent(0) + 1;
    const auto& ne_flat = state->atmos.ne;
    const auto& nhtot = state->atmos.nh_tot;
    // NOTE(cmo): GammaH_flat is how we access Gamma/C in the following
    const auto& GammaH_flat = state->Gamma[0];
    // NOTE(cmo): Sort out the diagonal before we copy into the transpose (full Gamma)
    fixup_gamma(GammaH_flat);
    yakl::fence();

    const auto& H_atom = extract_atom(state->adata, state->adata_host, 0);
    // NOTE(cmo): Immediately transposed -- we can reduce the memory requirements here if really needed
    yakl::Array<T, 2, yakl::memDevice> F("F", GammaH_flat.extent(2), num_eqn);
    yakl::Array<T, 3, yakl::memDevice> GammaT("GammaH^T", GammaH_flat.extent(2), num_level, num_level);
    yakl::Array<T, 3, yakl::memDevice> dF("dF", GammaH_flat.extent(2), num_eqn, num_eqn);
    F = FP(0.0);
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
    const auto C_flat = C.createDeviceCopy();
    auto ne_copy = state->atmos.ne.createDeviceCopy();
    auto ne_pert = state->atmos.ne.createDeviceCopy();
    yakl::fence();


    // NOTE(cmo): Perturb ne, compute C, compute dC/dne, restore ne
    auto& ne = state->atmos.ne;
    constexpr fp_t pert_size = FP(1e-2);
    parallel_for(
        "Perturb ne",
        SimpleBounds<1>(ne.extent(0)),
        YAKL_LAMBDA (i64 ks) {
            ne_pert(ks) = ne(ks) * pert_size;
            ne(ks) += ne_pert(ks);
        }
    );
    yakl::fence();
    compute_collisions_to_gamma(state);
    fixup_gamma(GammaH_flat);
    yakl::fence();
    parallel_for(
        "Compute dC",
        SimpleBounds<3>(C.extent(0), C.extent(1), C.extent(2)),
        YAKL_LAMBDA (int i, int j, i64 ks) {
            C(i, j, ks) = (GammaH(i, j, ks) - C(i, j, ks)) / ne_pert(ks);
        }
    );
    // NOTE(cmo): Rename for clarity
    const auto& dC = C;
    parallel_for(
        "Restore n_e",
        SimpleBounds<1>(ne.extent(0)),
        YAKL_LAMBDA (i64 ks) {
            ne(ks) = ne_copy(ks);
        }
    );
    yakl::fence();

    // NOTE(cmo): Compute LHS, based on Lightspinner impl
    parallel_for(
        "Compute F",
        SimpleBounds<2>(F.extent(0), F.extent(1)),
        YAKL_LAMBDA (i64 k, int i) {
            if (i < (num_level - 1)) {
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
            } else if (i == (num_eqn - 1)) {
                T charge = FP(0.0);
                for (int j = 0; j < num_level; ++j) {
                    charge += (H_atom.stage(j) - FP(1.0)) * new_pops(k, j);
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
                        const T precon_Rji = GammaT(k, cont.j, cont.i) - C_flat(cont.i, cont.j, k);
                        const T entry = -(precon_Rji / ne_flat(k)) * new_pops(k, cont.j);
                        yakl::atomicAdd(
                            dF(k, num_eqn-1, cont.i),
                            entry
                        );
                    }
                }
                if (i < num_level) {
                    // TODO(cmo): This can be atomicised and done over j, but it works
                    for (int jj = 0; jj < num_level; ++jj) {
                        yakl::atomicAdd(
                            dF(k, num_eqn-1, i),
                            - dC(i, jj, k) * new_pops(k, jj)
                        );
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
    ipivs = FP(0.0);
    info = 0;

    parallel_for(
        "Setup pointers",
        SimpleBounds<1>(dF_ptrs.extent(0)),
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
    }
    magma_queue_sync(state->magma_queue);
    yakl::fence();
    // NOTE(cmo): F is now the absolute update to apply to Hpops and ne
    parallel_for(
        "info check",
        SimpleBounds<1>(info.extent(0)),
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
    parallel_for(
        "Update & transpose pops",
        SimpleBounds<1>(F.extent(0)),
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
                fp_t ne_updated = ne_flat(k) + ne_update;
                if (ne_updated < FP(0.0)) {
                    fp_t local_step_size = FP(0.95) * ne_flat(k) / std::abs(ne_update);
                    step_size = std::max(std::min(step_size, local_step_size), FP(1e-4));
                }
            }
            for (int i = 0; i < num_level; ++i) {
                fp_t update = step_size * F(k, i);
                if (pops(i, k) > ignore_change_below_ntot_frac * nhtot(k)) {
                    max_rel_change(k, i) = std::abs(update / (pops(i, k)));
                }
                pops(i, k) += update;
            }
            fp_t ne_update = step_size * F(k, num_eqn-1);
            max_rel_change(k, num_eqn-1) = std::abs(ne_update / (ne_flat(k)));
            ne_flat(k) += ne_update;
            nr_step_size(k) = step_size;
        }
    );
    yakl::fence();


    fp_t max_change = yakl::intrinsics::maxval(max_rel_change);
    auto step_size_host = nr_step_size.createHostCopy();
    const i64 max_change_loc = yakl::intrinsics::maxloc(max_rel_change.collapse());
    yakl::fence();
    i64 max_change_acc = max_change_loc;

    int max_change_level = max_change_acc % F.extent(1);
    max_change_acc /= F.extent(1);
    i64 max_change_ks = max_change_acc;
    fmt::println(
        "NR Update Max Change (level: {}): {} (@ {}), step_size: {}",
        max_change_level == (num_eqn - 1) ? "n_e": std::to_string(max_change_level),
        max_change,
        max_change_ks,
        step_size_host(max_change_loc / F.extent(1))
    );
    yakl::timer_stop("Charge conservation");
    return max_change;
}
#else
template <typename T=fp_t>
inline fp_t nr_post_update(State* state, const NrPostUpdateOptions& args = NrPostUpdateOptions()) { return FP(0.0); };
#endif



#else
#endif