#if !defined(DEXRT_RAY_MARCHING_2_HPP)
#define DEXRT_RAY_MARCHING_2_HPP
#include "Types.hpp"
#include "Utils.hpp"
#include "RcUtilsModes.hpp"
#include "JasPP.hpp"
#include "State.hpp"
#include "EmisOpac.hpp"
#include "BlockMap.hpp"
#include "DirectionalEmisOpacInterp.hpp"
#include "Mipmaps.hpp"
#include <optional>

struct RayStartEnd {
    yakl::SArray<fp_t, 1, NUM_DIM> start;
    yakl::SArray<fp_t, 1, NUM_DIM> end;
};

struct Box {
    vec2 dims[NUM_DIM];
};

struct RayMarchState2d;
YAKL_INLINE bool next_intersection(RayMarchState2d* state);

/** Clips a ray to the specified box
 * \param ray the start/end points of the ray
 * \param box the box dimensions
 * \param start_clipped an out param specifying whether the start point was
 * clipped (i.e. may need to sample BC). Can be nullptr if not interest
 * \returns An optional ray start/end, nullopt if the ray is entirely outside.
*/
template <int NumDim=NUM_DIM>
YAKL_INLINE std::optional<RayStartEnd> clip_ray_to_box(RayStartEnd ray, Box box, bool* start_clipped=nullptr) {
    RayStartEnd result(ray);
    yakl::SArray<fp_t, 1, NumDim> length;
    fp_t clip_t_start = FP(0.0);
    fp_t clip_t_end = FP(0.0);
    if (start_clipped) {
        *start_clipped = false;
    }
    for (int d = 0; d < NumDim; ++d) {
        length(d) = ray.end(d) - ray.start(d);
        int clip_idx = -1;
        if (ray.start(d) < box.dims[d](0)) {
            clip_idx = 0;
        } else if (ray.start(d) > box.dims[d](1)) {
            clip_idx = 1;
        }
        if (clip_idx != -1) {
            fp_t clip_t = (box.dims[d](clip_idx) - ray.start(d)) / length(d);
            if (clip_t > clip_t_start) {
                clip_t_start = clip_t;
            }
        }

        clip_idx = -1;
        if (ray.end(d) < box.dims[d](0)) {
            clip_idx = 0;
        } else if (ray.end(d) > box.dims[d](1)) {
            clip_idx = 1;
        }
        if (clip_idx != -1) {
            fp_t clip_t = (box.dims[d](clip_idx) - ray.end(d)) / -length(d);
            if (clip_t > clip_t_end) {
                clip_t_end = clip_t;
            }
        }
    }

    if (
        clip_t_start < 0 ||
        clip_t_end < 0 ||
        clip_t_start + clip_t_end >= FP(1.0)
    ) {
        // NOTE(cmo): We've moved forwards from start enough, and back from end
        // enough that there's none of the original ray actually intersecting
        // the clip planes! Or numerical precision issues accidentally forced a
        // fake intersection.
        return std::nullopt;
    }

    if (clip_t_start > FP(0.0)) {
        for (int d = 0; d < NumDim; ++d) {
            result.start(d) += clip_t_start * length(d);
        }
        if (start_clipped) {
            *start_clipped = true;
        }
    }
    if (clip_t_end > FP(0.0)) {
        for (int d = 0; d < NumDim; ++d) {
            result.end(d) -= clip_t_end * length(d);
        }
    }
    // NOTE(cmo): Catch precision errors with a clamp -- without this we will
    // stop the ray at the edge of the box to floating point precision, but it's
    // better for these to line up perfectly.
    // for (int d = 0; d < NumDim; ++d) {
    //     if (result.start(d) < box.dims[d](0)) {
    //         result.start(d) = box.dims[d](0);
    //     } else if (result.start(d) > box.dims[d](1)) {
    //         result.start(d) = box.dims[d](1);
    //     }
    //     if (result.end(d) < box.dims[d](0)) {
    //         result.end(d) = box.dims[d](0);
    //     } else if (result.end(d) > box.dims[d](1)) {
    //         result.end(d) = box.dims[d](1);
    //     }
    // }

    return result;
}


struct RayMarchState2d {
    /// Start pos
    vec2 p0;
    /// end pos
    vec2 p1;

    /// Current cell coordinate
    ivec2 curr_coord;
    /// Next cell coordinate
    ivec2 next_coord;
    /// Final cell coordinate -- for intersections with the outer edge of the
    /// box, this isn't floor(p1), but inside it is.
    ivec2 final_coord;
    /// Integer step dir
    ivec2 step;

    /// t to next hit per axis
    vec2 next_hit;
    /// t increment per step per axis
    vec2 delta;
    /// t to stop at
    fp_t max_t;

    /// axis increment
    vec2 direction;
    /// value of t at current intersection (far side of curr_coord, just before entering next_coord)
    fp_t t = FP(0.0);
    /// length of step
    fp_t dt = FP(0.0);

    /**
     * Create a new state for grid traversal using DDA. The ray is first clipped to
     * the grid, and if it is outside, nullopt is returned.
     * \param start_pos The start position of the ray
     * \param end_pos The start position of the ray
     * \param domain_size The domain size
     * \param start_clipped whether the start position was clipped; i.e. sample the BC.
    */
    template <int NumDim=NUM_DIM>
    YAKL_INLINE bool init(
        vec2 start_pos,
        vec2 end_pos,
        ivec2 domain_size,
        bool* start_clipped=nullptr
    ) {
        Box box;
        for (int d = 0; d < NumDim; ++d) {
            box.dims[d](0) = FP(0.0);
            box.dims[d](1) = domain_size(d);
        }
        auto clipped = clip_ray_to_box({start_pos, end_pos}, box, start_clipped);
        if (!clipped) {
            return false;
        }

        start_pos = clipped->start;
        end_pos = clipped->end;

        this->p0 = start_pos;
        this->p1 = end_pos;

        fp_t length = FP(0.0);
        for (int d = 0; d < NumDim; ++d) {
            // r.curr_coord(d) = std::min(int(std::floor(start_pos(d))), domain_size(d)-1);
            this->curr_coord(d) = int(std::floor(start_pos(d)));
            this->direction(d) = end_pos(d) - start_pos(d);
            length += square(end_pos(d) - start_pos(d));
            // r.final_coord(d) = std::min(int(std::floor(end_pos(d))), domain_size(d)-1);
            this->final_coord(d) = int(std::floor(end_pos(d)));;
        }
        this->next_coord = this->curr_coord;
        length = std::sqrt(length);
        this->max_t = length;

        fp_t inv_length = FP(1.0) / length;
        for (int d = 0; d < NumDim; ++d) {
            this->direction(d) *= inv_length;
            if (this->direction(d) > FP(0.0)) {
                this->next_hit(d) = fp_t(this->curr_coord(d) + 1 - this->p0(d)) / this->direction(d);
                this->step(d) = 1;
            } else if (this->direction(d) == FP(0.0)) {
                this->step(d) = 0;
                this->next_hit(d) = FP(1e24);
            } else {
                this->step(d) = -1;
                this->next_hit(d) = (this->curr_coord(d) - this->p0(d)) / this->direction(d);
            }
            this->delta(d) = fp_t(this->step(d)) / this->direction(d);
        }

        this->t = FP(0.0);
        this->dt = FP(0.0);

        auto in_bounds = [&]() {
            return (
                this->curr_coord(0) >= 0
                && this->curr_coord(0) < domain_size(0)
                && this->curr_coord(1) >= 0
                && this->curr_coord(1) < domain_size(1)
            );
        };
        // NOTE(cmo): Initialise to the first intersection so dt != 0
        while (!in_bounds() || (in_bounds() && this->dt < FP(1e-6))) {
            next_intersection(this);
        }

        return true;
    }
};

// NOTE(cmo): Based on Nanovdb templated implementation
template <int axis>
YAKL_INLINE fp_t step_marcher(RayMarchState2d* state) {
    auto& s = *state;
    fp_t new_t = s.next_hit(axis);
    s.next_hit(axis) += s.delta(axis);
    s.next_coord(axis) += s.step(axis);
    return new_t;
}

YAKL_INLINE fp_t step_marcher(RayMarchState2d* state) {
    auto& s = *state;
    int axis = 0;
    if (s.next_hit(1) < s.next_hit(0)) {
        axis = 1;
    }
    switch (axis) {
        case 0: {
            return step_marcher<0>(state);
        } break;
        case 1: {
            return step_marcher<1>(state);
        } break;
    }
    return step_marcher<0>(state);
}

YAKL_INLINE bool next_intersection(RayMarchState2d* state) {
    using namespace yakl::componentwise;
    using yakl::intrinsics::sum;

    auto& s = *state;
    const fp_t prev_t = s.t;
    for (int d = 0; d < NUM_DIM; ++d) {
        s.curr_coord(d) = s.next_coord(d);
    }

    fp_t new_t = step_marcher(state);

    if (new_t > s.max_t && prev_t < s.max_t) {
        // NOTE(cmo): The end point is in the box we have just stepped through
        decltype(s.p1) prev_hit = s.p0 +  prev_t * s.direction;
        s.dt = std::sqrt(sum(square(s.p1 - prev_hit)));
        new_t = s.max_t;
        // NOTE(cmo): Set curr_coord to a value we know we clamped inside the
        // grid: minimise accumulated error
        for (int d = 0; d < NUM_DIM; ++d) {
            s.curr_coord(d) = s.final_coord(d);
        }
    } else {
        // NOTE(cmo): Progress as normal
        s.dt = new_t - prev_t;
    }

    s.t = new_t;
    return new_t <= s.max_t;
}

template <typename Alo>
YAKL_INLINE RadianceInterval<Alo> merge_intervals(
    RadianceInterval<Alo> closer,
    RadianceInterval<Alo> further
) {
    fp_t transmission = std::exp(-closer.tau);
    closer.I += transmission * further.I;
    if constexpr (STORE_TAU_CASCADES) {
        // NOTE(cmo): There will be nonsense in further.tau, so it's just not useful -- we're not storing anyway
        closer.tau += further.tau;
    }
    return closer;
}

struct Raymarch2dDynamicState {
    const yakl::Array<const u16, 1, yakl::memDevice> active_set;
    const SparseAtmosphere& atmos;
    const AtomicData<fp_t>& adata;
    const VoigtProfile<fp_t, false>& profile;
    const Fp1d& nh0;
    const Fp2d& n; // flattened
};

struct Raymarch2dDynamicInterpState {
};

struct Raymarch2dDynamicCoreAndVoigtState {
    const yakl::SArray<i32, 1, CORE_AND_VOIGT_MAX_LINES> active_set;
    const VoigtProfile<fp_t, false>& profile;
    const AtomicData<fp_t>& adata;
};

template <typename Bc, class DynamicState=DexEmpty>
struct Raymarch2dArgs {
    const CascadeStateAndBc<Bc>& casc_state_bc;
    RayProps ray;
    fp_t distance_scale = FP(1.0);
    vec3 mu;
    fp_t incl;
    fp_t incl_weight;
    int wave;
    int la;
    vec3 offset;
    int max_mip_to_sample;
    const BlockMap<BLOCK_SIZE>& block_map;
    const MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE>& mr_block_map;
    const MultiResMipChain& mip_chain;
    const DynamicState& dyn_state;
};

YAKL_INLINE RaySegment ray_seg_from_ray_props(const RayProps& ray) {
    fp_t t1 = (ray.end(0) - ray.start(0)) / ray.dir(0);

    return RaySegment(ray.start, ray.dir, FP(0.0), t1);
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
YAKL_INLINE RadianceInterval<Alo> multi_level_dda_raymarch_2d(
    const Raymarch2dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, casc_state_bc, ray, distance_scale, mu, incl, incl_weight, wave, la, offset, dyn_state);
    JasUnpack(args, mip_chain);
    JasUnpack(casc_state_bc, state, bc);
    constexpr bool dynamic = (RcMode & RC_DYNAMIC);
    constexpr bool dynamic_interp = dynamic && std::is_same_v<DynamicState, Raymarch2dDynamicInterpState>;
    constexpr bool dynmaic_cav = dynamic && std::is_same_v<DynamicState, Raymarch2dDynamicCoreAndVoigtState>;

    RaySegment ray_seg = ray_seg_from_ray_props(ray);
    bool start_clipped;
    MRIdxGen idx_gen(args.mr_block_map);
    auto s = MultiLevelDDA<BLOCK_SIZE, ENTRY_SIZE>(idx_gen);
    const bool marcher = s.init(ray_seg, args.max_mip_to_sample, &start_clipped);
    RadianceInterval<Alo> result{
        .I = FP(0.0),
        .tau = FP(0.0)
    };
    if ((RcMode & RC_SAMPLE_BC) && (!marcher || start_clipped)) {
        // NOTE(cmo): Check the ray is going up along z.
        if ((ray.dir(1) > FP(0.0)) && la != -1) {
            vec3 pos;
            pos(0) = ray.centre(0) * distance_scale + offset(0);
            pos(1) = offset(1);
            pos(2) = ray.centre(1) * distance_scale + offset(2);

            fp_t I_sample = sample_boundary(bc, la, pos, mu);
            result.I = I_sample;
        }
    }
    if (!marcher) {
        return result;
    }

    // NOTE(cmo): one_m_edt is also the ALO
    fp_t eta_s = FP(0.0), chi_s = FP(1e-20), one_m_edt = FP(0.0);
    // NOTE(cmo): implicit assumption muy != 1.0
    const fp_t inv_sin_theta = FP(1.0) / std::sqrt(FP(1.0) - square(incl));
    fp_t lambda;
    if constexpr (dynamic && std::is_same_v<DynamicState, Raymarch2dDynamicCoreAndVoigtState>) {
        lambda = dyn_state.adata.wavelength(la);
    }
    do {
        one_m_edt = FP(0.0);
        if (s.can_sample()) {
            i32 u = s.curr_coord(0);
            i32 v = s.curr_coord(1);
            i64 ks = idx_gen.idx(s.current_mip_level, u, v);

            if constexpr (dynamic_interp) {
                const fp_t vel = (
                    mip_chain.vx(ks) * mu(0)
                    + mip_chain.vy(ks) * mu(1)
                    + mip_chain.vz(ks) * mu(2)
                );
                auto contrib = mip_chain.dir_data.sample(ks, wave, vel);
                eta_s = contrib.eta;
                chi_s = contrib.chi + FP(1e-15);
            } else if constexpr (dynmaic_cav) {
                JasUnpack(dyn_state, active_set, profile, adata);
                i64 ks_wave = ks * mip_chain.emis.extent(1) + wave;
                eta_s = mip_chain.emis.get_data()[ks_wave];
                chi_s = mip_chain.opac.get_data()[ks_wave] + FP(1e-15);

                const fp_t vel = (
                    mip_chain.vx.get_data()[ks] * mu(0)
                    + mip_chain.vy.get_data()[ks] * mu(1)
                    + mip_chain.vz.get_data()[ks] * mu(2)
                );
                CavEmisOpacState emis_opac_state {
                    .ks = ks,
                    .krl = 0,
                    .lambda = lambda,
                    .vel = vel,
                    .phi = profile
                };

                // pls do this in registers
                #pragma unroll
                for (int kri = 0; kri < CORE_AND_VOIGT_MAX_LINES; ++kri) {
                    i32 krl = active_set(kri);
                    if (krl < 0) {
                        break;
                    }

                    emis_opac_state.krl = krl;
                    i32 kr = mip_chain.cav_data.active_set_mapping(krl);
                    EmisOpac eta_chi = mip_chain.cav_data.emis_opac(
                        emis_opac_state
                    );
                    eta_s += eta_chi.eta;
                    chi_s += eta_chi.chi;
                }
            } else {
                eta_s = mip_chain.emis(ks, wave);
                chi_s = mip_chain.opac(ks, wave) + FP(1e-15);
                if constexpr (dynamic) {
                    const SparseAtmosphere& atmos = dyn_state.atmos;
                    if (
                        mip_chain.classic_data.dynamic_opac(ks, wave)
                        && dyn_state.active_set.extent(0) > 0
                    ) {
                        fp_t vel = (
                            atmos.vx.get_data()[ks] * mu(0)
                            + atmos.vy.get_data()[ks] * mu(1)
                            + atmos.vz.get_data()[ks] * mu(2)
                        );
                        AtmosPointParams local_atmos{
                            .temperature = atmos.temperature.get_data()[ks],
                            .ne = atmos.ne.get_data()[ks],
                            .vturb = atmos.vturb.get_data()[ks],
                            .nhtot = atmos.nh_tot.get_data()[ks],
                            .vel = vel,
                            .nh0 = dyn_state.nh0.get_data()[ks]
                        };
                        auto lines = emis_opac(
                            EmisOpacState<fp_t>{
                                .adata = dyn_state.adata,
                                .profile = dyn_state.profile,
                                .la = la,
                                .n = dyn_state.n,
                                .k = ks,
                                .atmos = local_atmos,
                                .active_set = dyn_state.active_set,
                                .mode = EmisOpacMode::DynamicOnly
                            }
                        );

                        eta_s += lines.eta;
                        chi_s += lines.chi;
                    }
                }
            }

            fp_t tau = chi_s * s.dt * distance_scale;
            fp_t source_fn = eta_s / chi_s;

            fp_t tau_mu = tau * inv_sin_theta;
            fp_t edt;
            if (tau_mu < FP(1e-2)) {
                edt = FP(1.0) + (-tau_mu) + FP(0.5) * square(tau_mu);
                one_m_edt = -std::expm1(-tau_mu);
            } else {
                edt = std::exp(-tau_mu);
                one_m_edt = -std::expm1(-tau_mu);
            }
            result.tau += tau_mu;
            result.I = result.I * edt + source_fn * one_m_edt;
        }
    } while(s.step_through_grid());

    if constexpr ((RcMode & RC_COMPUTE_ALO) && !std::is_same_v<Alo, DexEmpty>) {
        result.alo = std::max(one_m_edt, FP(0.0));
    }
    return result;
}

#else
#endif