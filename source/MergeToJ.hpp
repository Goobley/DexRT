#if !defined(DEXRT_MERGE_TO_J_HPP)
#define DEXRT_MERGE_TO_J_HPP

#include "Config.hpp"
#include "Constants.hpp"
#include "Types.hpp"
#include "RcUtilsModes.hpp"

inline FpConst3d merge_c0_to_J(
    const FpConst1d& c0,
    const CascadeStorage& c0_dims,
    const Fp3d& J,
    const InclQuadrature incl_quad,
    int la_start,
    int la_end
) {
    constexpr int RcMode = RC_flags_storage();
    const fp_t phi_weight = FP(1.0) / fp_t(c0_dirs_to_average<RcMode>());
    int wave_batch = la_end - la_start;

    // NOTE(cmo): Handle case of J on the GPU only being a single plane (config.store_J_on_cpu)
    if (c0_dims.wave_batch == J.extent(0)) {
        la_start = 0;
    }

    parallel_for(
        "final_cascade_to_J",
        SimpleBounds<5>(
            c0_dims.num_probes(1),
            c0_dims.num_probes(0),
            c0_dims.num_flat_dirs,
            wave_batch,
            c0_dims.num_incl),
        YAKL_LAMBDA (int z, int x, int phi_idx, int wave, int theta_idx) {
            fp_t ray_weight = phi_weight * incl_quad.wmuy(theta_idx);
            int la = la_start + wave;
            ivec2 coord;
            coord(0) = x;
            coord(1) = z;
            ProbeStorageIndex idx{
                .coord=coord,
                .dir=phi_idx,
                .incl=theta_idx,
                .wave=wave
            };
            const fp_t sample = probe_fetch<RcMode>(c0, c0_dims, idx);
            yakl::atomicAdd(J(la, z, x), ray_weight * sample);
        }
    );
    return J;
}


#else
#endif