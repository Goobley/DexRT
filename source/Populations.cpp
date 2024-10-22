#include "Populations.hpp"

void compute_lte_pops_flat(
    const CompAtom<fp_t>& atom,
    const SparseAtmosphere& atmos,
    const Fp2d& pops
) {
    const auto& temperature = atmos.temperature;
    const auto& ne = atmos.ne;
    const auto& nhtot = atmos.nh_tot;
    parallel_for(
        "LTE Pops",
        SimpleBounds<1>(pops.extent(1)),
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
        parallel_for(
            "Copy nh0",
            SimpleBounds<1>(nh0.extent(0)),
            YAKL_LAMBDA (i64 ks) {
                nh0(ks) = pops(0, ks);
            }
        );
    } else {
        const auto& atmos = state.atmos;
        parallel_for(
            "Compute nh0 in LTE",
            SimpleBounds<1>(nh0.extent(0)),
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
        yakl::Array<T, 3, yakl::memDevice> GammaT("GammaT", Gamma.extent(2), Gamma.extent(0), Gamma.extent(1));
        yakl::Array<T*, 1, yakl::memDevice> GammaT_ptrs("GammaT_ptrs", GammaT.extent(0));
        yakl::Array<T, 2, yakl::memDevice> new_pops("new_pops", GammaT.extent(0), GammaT.extent(1));
        yakl::Array<T, 1, yakl::memDevice> n_total("n_total", GammaT.extent(0));
        yakl::Array<T*, 1, yakl::memDevice> new_pops_ptrs("new_pops_ptrs", GammaT.extent(0));
        yakl::Array<i32, 1, yakl::memDevice> i_elim("i_elim", GammaT.extent(0));
        yakl::Array<i32, 2, yakl::memDevice> ipivs("ipivs", new_pops.extent(0), new_pops.extent(1));
        yakl::Array<i32*, 1, yakl::memDevice> ipiv_ptrs("ipiv_ptrs", new_pops.extent(0));
        yakl::Array<i32, 1, yakl::memDevice> info("info", new_pops.extent(0));

        const int pops_start = state->adata_host.level_start(ia);
        const int num_level = state->adata_host.num_level(ia);
        parallel_for(
            "Max Pops",
            SimpleBounds<1>(pops.extent(1)),
            YAKL_LAMBDA (int64_t k) {
                fp_t n_max = FP(0.0);
                i_elim(k) = 0;
                n_total(k) = FP(0.0);
                for (int i = pops_start; i < pops_start + num_level; ++i) {
                    fp_t n = pops(i, k);
                    n_total(k) += n;
                    if (n > n_max) {
                        i_elim(k) = i - pops_start;
                    }
                }
            }
        );
        yakl::fence();

        parallel_for(
            "Transpose Gamma",
            SimpleBounds<3>(Gamma.extent(2), Gamma.extent(1), Gamma.extent(0)),
            YAKL_LAMBDA (int k, int i, int j) {
                GammaT(k, j, i) = Gamma(i, j, k);
            }
        );
        yakl::fence();

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
            "Transpose Pops",
            SimpleBounds<2>(new_pops.extent(0), new_pops.extent(1)),
            YAKL_LAMBDA (i64 k, int i) {
                if (i_elim(k) == i) {
                    new_pops(k, i) = n_total(k);
                } else {
                    new_pops(k, i) = FP(0.0);
                }
            }
        );
        parallel_for(
            "Setup pointers",
            SimpleBounds<1>(GammaT_ptrs.extent(0)),
            YAKL_LAMBDA (i64 k) {
                GammaT_ptrs(k) = &GammaT(k, 0, 0);
                new_pops_ptrs(k) = &new_pops(k, 0);
                ipiv_ptrs(k) = &ipivs(k, 0);
            }
        );
        yakl::fence();

        parallel_for(
            "Conservation eqn",
            SimpleBounds<3>(GammaT.extent(0), GammaT.extent(1), GammaT.extent(2)),
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
            magma_sgesv_batched_small(
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
            magma_dgesv_batched_small(
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
        }

        magma_queue_sync(state->magma_queue);
        yakl::fence();
        parallel_for(
            "info check",
            SimpleBounds<1>(info.extent(0)),
            YAKL_LAMBDA (int k) {
                if (info(k) != 0) {
                    printf("LINEAR SOLVER PROBLEM k: %d, info: %d\n", k, info(k));
                }
            }
        );

        Fp2d max_rel_change("max rel change", new_pops.extent(0), new_pops.extent(1));
        parallel_for(
            "Compute max change",
            SimpleBounds<2>(new_pops.extent(0), new_pops.extent(1)),
            YAKL_LAMBDA (int64_t k, int i) {
                fp_t change = FP(0.0);
                if (pops(pops_start + i, k) < ignore_change_below_ntot_frac * n_total(k)) {
                    change = FP(0.0);
                } else {
                    change = std::abs(FP(1.0) - pops(pops_start + i, k) / new_pops(k, i));
                }
                max_rel_change(k, i) = change;
            }
        );
        yakl::fence();
        parallel_for(
            "Copy & transpose pops",
            SimpleBounds<2>(new_pops.extent(1), new_pops.extent(0)),
            YAKL_LAMBDA (int i, int64_t k) {
                pops(pops_start + i, k) = new_pops(k, i);
            }
        );

        fp_t max_change = yakl::intrinsics::maxval(max_rel_change);
        int max_change_loc = yakl::intrinsics::maxloc(max_rel_change.collapse());
        auto temp_h = state->atmos.temperature.createHostCopy();

        yakl::fence();
        int max_change_level = max_change_loc % new_pops.extent(1);
        max_change_loc /= new_pops.extent(1);
        fmt::println(
            "     Max Change (ele: {}, Z={}): {} (@ l={}, ks={}) [T={}]",
            ia,
            state->adata_host.Z(ia),
            max_change,
            max_change_level,
            max_change_loc,
            temp_h(max_change_loc)
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
    return stat_eq_impl<StatEqPrecision>(state, args);
}
