#if !defined(DEXRT_BOUNDARY_DISPATCH_HPP)
#define DEXRT_BOUNDARY_DISPATCH_HPP

#include "State.hpp"
#include "CascadeState.hpp"

struct DeviceBoundaries {
    BoundaryType boundary;
    ZeroBc zero_bc;
    PwBc<> pw_bc;
};

template <typename Bc=void>
YAKL_INLINE
CascadeStateAndBc<Bc> get_bc(
    const DeviceCascadeState& casc_state,
    const DeviceBoundaries& boundaries
) {
    static_assert(std::is_same_v<Bc, void>, "Need a specialisation for each BoundaryType in Boundary.hpp");
}

template <>
YAKL_INLINE
CascadeStateAndBc<ZeroBc> get_bc<ZeroBc>(
    const DeviceCascadeState& casc_state,
    const DeviceBoundaries& boundaries
) {
    return CascadeStateAndBc<ZeroBc> {
        .state = casc_state,
        .bc = boundaries.zero_bc
    };
}

template <>
YAKL_INLINE
CascadeStateAndBc<PwBc<>> get_bc<PwBc<>>(
    const DeviceCascadeState& casc_state,
    const DeviceBoundaries& boundaries
) {
    return CascadeStateAndBc<PwBc<>> {
        .state = casc_state,
        .bc = boundaries.pw_bc
    };
}

#else
#endif
