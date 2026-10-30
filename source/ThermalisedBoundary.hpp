#if !defined(DEXRT_THERMALISED_BOUNDARY_HPP)
#define DEXRT_THERMALISED_BOUNDARY_HPP
#include "Types.hpp"
#include "BackgroundOpacity.hpp"
#include <cmath>

/// Thermalised lower boundary: upward rays entering through the bottom face
/// carry B_lambda(T) of the bottom row (per column); zero elsewhere.
struct ThermalisedBc {
    Fp1d temperature; /// [y * num_x + x] bottom plane (2D: row) temperature
    yakl::Array<const fp_t, 1, yakl::memDevice> wavelength; /// [la] nm
    fp_t dx;
    fp_t offset_x;
    /// Positions at or below this height are on the bottom face (it lies just
    /// above the face, inside the first half cell).
    fp_t z_face;
    /// 3D: the bottom plane's y extent and spacing (2D: one row).
    i32 num_y = 1;
    fp_t dy = FP(1.0);
    fp_t offset_y = FP(0.0);
};

/// dir: the viewing direction (towards the boundary, against the
/// propagation), as for the other boundaries.
YAKL_INLINE fp_t sample_boundary(
    const ThermalisedBc& bc,
    int la,
    vec3 at,
    vec3 dir
) {
    if (dir(2) >= FP(0.0) || at(2) > bc.z_face) {
        return FP(0.0);
    }
    const i32 nx = bc.temperature.extent(0) / bc.num_y;
    const i32 x = std::min(std::max(i32(std::floor((at(0) - bc.offset_x) / bc.dx)), 0), nx - 1);
    i32 y = 0;
    if (bc.num_y > 1) {
        y = std::min(std::max(i32(std::floor((at(1) - bc.offset_y) / bc.dy)), 0), bc.num_y - 1);
    }
    return planck_wavelength(bc.wavelength(la), bc.temperature(i64(y) * nx + x));
}

#else
#endif
