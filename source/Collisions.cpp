#include "Collisions.hpp"

void compute_collisions_to_gamma(State* state) {
    const auto& atmos = state->atmos;
    const auto& Gamma_store = state->Gamma;
    const auto& nh_lte = state->nh_lte;
    const auto atmos_dims = atmos.temperature.get_dimensions();
    const auto& mr_block_map = state->mr_block_map;

    // TODO(cmo): Get rid of this!
    auto n_star = state->pops.createDeviceObject();
    auto n_star_flat = n_star.reshape<2>(Dims(n_star.extent(0), n_star.extent(1) * n_star.extent(2)));
    // NOTE(cmo): Zero Gamma before we start to refill it.
    for (int i = 0; i < Gamma_store.size(); ++i) {
        Gamma_store[i] = FP(0.0);
    }
    compute_lte_pops(state, n_star);
    yakl::fence();

    for (int ia = 0; ia < state->atoms_with_gamma.size(); ++ia) {
        auto& Gamma = Gamma_store[ia];
        auto& atom = state->atoms_with_gamma[ia];
        const auto n_star_slice = slice_pops(
            n_star,
            state->adata_host,
            state->atoms_with_gamma_mapping[ia]
        );

        parallel_for(
            "collisions",
            mr_block_map.block_map.loop_bounds(),
            YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);

                compute_collisions(
                    atmos,
                    atom,
                    Gamma,
                    n_star_slice,
                    nh_lte,
                    ks
                );
            }
        );
        yakl::fence();
    }
}