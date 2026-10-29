#if !defined(DEXRT_TRANSITION_INDEX_HPP)
#define DEXRT_TRANSITION_INDEX_HPP

#include "Config.hpp"
#include "Types.hpp"

/// precon_R[t, slot, ks]: transition t (each atom's lines, then its continua, in
/// a contiguous block per atom), the upward rate i -> j (Gamma(j, i)) and the
/// downward rate j -> i (Gamma(i, j)), preconditioned. Assembled into Gamma with
/// the collisions in stat_eq and friends.
constexpr int PRECON_R_UP = 0;
constexpr int PRECON_R_DOWN = 1;

/// Index of the first transition of atom in precon_R.
template <int mem_space>
YAKL_INLINE int trans_start(const AtomicData<fp_t, mem_space>& adata, int atom) {
    return adata.line_start(atom) + adata.cont_start(atom);
}

/// Index of line kr in precon_R.
template <int mem_space>
YAKL_INLINE int line_transition_idx(const AtomicData<fp_t, mem_space>& adata, int kr) {
    const int atom = adata.lines(kr).atom;
    return trans_start(adata, atom) + kr - adata.line_start(atom);
}

/// Index of continuum kr in precon_R.
template <int mem_space>
YAKL_INLINE int cont_transition_idx(const AtomicData<fp_t, mem_space>& adata, int kr) {
    const int atom = adata.continua(kr).atom;
    return trans_start(adata, atom) + adata.num_line(atom) + kr - adata.cont_start(atom);
}

template <int mem_space>
inline int num_transitions(const AtomicData<fp_t, mem_space>& adata, int atom) {
    return adata.num_line(atom) + adata.num_cont(atom);
}

/// Number of transitions over all atoms, i.e. the leading extent of precon_R.
template <int mem_space>
inline int num_transitions(const AtomicData<fp_t, mem_space>& adata) {
    return adata.lines.extent(0) + adata.continua.extent(0);
}

/// View of atom's [num_transitions(atom), 2, ks] block of precon_R (or
/// radiative_rates), sharing its storage, so it is indexed by the atom's own t.
template <int mem_space>
inline PreconRMat atom_precon_R(const PreconRMat& R, const AtomicData<fp_t, mem_space>& adata, int atom) {
    const i64 stride = R.extent(1) * R.extent(2);
    return PreconRMat(
        "atom_precon_R",
        R.data() + trans_start(adata, atom) * stride,
        num_transitions(adata, atom),
        R.extent(1),
        R.extent(2)
    );
}

struct TransitionLevels {
    int i;
    int j;
};

/// Lower (i) and upper (j) level of transition t of atom. Index relative to atom, not global precon_R.
template <int mem_space>
YAKL_INLINE TransitionLevels transition_levels(const AtomicData<fp_t, mem_space>& adata, int atom, int t) {
    const int num_line = adata.num_line(atom);
    if (t < num_line) {
        const auto& l = adata.lines(adata.line_start(atom) + t);
        return TransitionLevels{.i = l.i, .j = l.j};
    }
    const auto& c = adata.continua(adata.cont_start(atom) + t - num_line);
    return TransitionLevels{.i = c.i, .j = c.j};
}

#else
#endif
