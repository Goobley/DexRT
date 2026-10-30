#if !defined(DEXRT_CELL_LIST_HPP)
#define DEXRT_CELL_LIST_HPP

#include "Types.hpp"
#include <type_traits>

struct ScState;

/// The cells a per-cell routine visits: i in [0, num()), ks = cells(i).
/// RC visits every (active) cell of its compact storage; dexrt_sc overloads
/// cell_list for its dense slab (ScState.hpp).
struct AllCells {
    i64 n = 0;

    YAKL_INLINE i64 num() const {
        return n;
    }
    YAKL_INLINE i64 operator()(i64 i) const {
        return i;
    }
    AllCells host() const {
        return *this;
    }
};

template <typename StateT>
AllCells cell_list(const StateT& state) {
    static_assert(!std::is_same_v<StateT, ScState>, "Include ScState.hpp for its cell_list");
    return AllCells{.n = i64(state.pops.extent(1))};
}

#else
#endif
