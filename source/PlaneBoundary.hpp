#if !defined(DEXRT_PLANE_BOUNDARY)
#define DEXRT_PLANE_BOUNDARY

#include "Types.hpp"
#include "BoundaryType.hpp"
#include "Utils.hpp"
#include "YAKL_netcdf.h"

enum class PlaneResampleType {
    FluxConserving,
    Interpolation
};

template <int mem_space=yakl::memDevice>
struct PlaneBc {
    fp_t mu_min;
    fp_t mu_max;
    fp_t mu_step;
    yakl::Array<fp_t, 2, mem_space> I; // [wl, mu]
};
PlaneBc<> load_bc(
    const std::string& path,
    const FpConst1d& wavelength,
    BoundaryType type,
    PlaneResampleType resample = PlaneResampleType::FluxConserving
);

YAKL_INLINE fp_t sample_boundary(
    const PlaneBc<>& bc,
    int la,
    vec3 at,
    vec3 dir
) {
    fp_t mu_sample = dir(2);

    fp_t result;
    if (mu_sample < FP(0.0)) {
        result = FP(0.0);
    } else if (mu_sample <= bc.mu_min) {
        result = bc.I(la, 0);
    } else if (mu_sample >= bc.mu_max) {
        result = bc.I(la, bc.I.extent(1) - 1);
    } else {
        fp_t frac_idx = (mu_sample - bc.mu_min) / bc.mu_step;
        int idx = int(frac_idx);
        fp_t t = frac_idx - fp_t(idx);

        result = (FP(1.0) - t) * bc.I(la, idx) + t * bc.I(la, idx + 1);
    }
    return result;
}

#else
#endif