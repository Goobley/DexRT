#if !defined(DEXRT_BOUNDARY_DISPATCH_HPP)
#define DEXRT_BOUNDARY_DISPATCH_HPP

#include "State.hpp"

template <typename Bc=void>
YAKL_INLINE
CascadeStateAndBc<Bc> get_bc(const DeviceCascadeState& casc_state, const State& state) {
    static_assert(std::is_same_v<Bc, void>, "Need a specialisation for each BoundaryType in Boundary.hpp");
}

template <>
YAKL_INLINE
CascadeStateAndBc<ZeroBc> get_bc<ZeroBc>(
    const DeviceCascadeState& casc_state,
    const State& state
) {
    return CascadeStateAndBc<ZeroBc> {
        .state = casc_state,
        .bc = state.zero_bc
    };
}

template <>
YAKL_INLINE
CascadeStateAndBc<PwBc<>> get_bc<PwBc<>>(
    const DeviceCascadeState& casc_state,
    const State& state
) {
    return CascadeStateAndBc<PwBc<>> {
        .state = casc_state,
        .bc = state.pw_bc
    };
}

#else
#endif
